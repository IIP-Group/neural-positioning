import numpy as np
import cvxpy as cp

# Global variables
G = None # grid points
grid_gran = None
# epsilon: max. allowed offset of G@p' from x for solution p'. Depends on granularity of grid.
epsilon = None

def make_grid(positions,non_square_area=False):
    global G
    global grid_gran
    global epsilon

    # Target number of grid points per dimension (used to determine spacing)
    K = 22
    
    if non_square_area:
        # If the area is non-square, we need to calculate the grid spacing based on the smaller range
        # Calculate ranges with padding
        x_min = np.min(positions[:,0]) - 0.1
        x_max = np.max(positions[:,0]) + 0.1
        y_min = np.min(positions[:,1]) - 0.1
        y_max = np.max(positions[:,1]) + 0.1
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Use the smaller range to determine grid spacing, ensuring at least K points
        # This ensures equal spacing in both dimensions
        spacing = min(x_range, y_range) / (K - 1)
        
        # Calculate number of points needed for each dimension
        K_x = int(np.ceil(x_range / spacing)) + 1
        K_y = int(np.ceil(y_range / spacing)) + 1
        
        # Create grid with equal spacing
        x = np.linspace(x_min, x_max, K_x)
        y = np.linspace(y_min, y_max, K_y)
    else:
        # If the area is square, we can use the same grid spacing for both dimensions
        x = np.linspace(np.min(positions[:,0])-0.1, np.max(positions[:,0])+0.1, K)
        y = np.linspace(np.min(positions[:,1])-0.1, np.max(positions[:,1])+0.1, K)
    
    xv, yv = np.meshgrid(x, y)
    G = np.concatenate((xv.reshape(1,-1),yv.reshape(1,-1)),axis=0)

    # Grid granularity for x and y axis
    grid_gran = np.array([[np.abs(x[1]-x[0])],[np.abs(y[1]-y[0])]])

    epsilon = np.linalg.norm(grid_gran)/2

def learn_ref_probability_map(x,K):
    """
    Compute reference prob. for 4 nearest grid points
    x: A 2 x 1 two-dimensional position
    G: 2 x K matrix of grid positions
    grid_gran: granularity of position grid for x and y axis (2-D)
    """
    global G
    global grid_gran
    global epsilon

    # Find 4 nearest grid points via there distances to x
    distances = np.linalg.norm(G-x,axis=0)
    indices_4nearest = np.argsort(distances)[:4]
    G_4nearest = G[:,indices_4nearest]

    # Optimization problem for refernce prob. map p_hat
    v = np.linalg.norm((G_4nearest-x),axis=0)**2
    p = cp.Variable(4)
    objective = cp.Minimize(p.T@v)
    constraints = [0 <= p, p <= 1, cp.sum(p) == 1, cp.atoms.norm(G_4nearest@p - x,p=2) <= epsilon]
    prob = cp.Problem(objective, constraints)

    result = prob.solve()
    p_hat_4nearest = p.value
    p_hat = np.zeros((G.shape[1],))
    p_hat[indices_4nearest] = p_hat_4nearest[:]

    return p_hat, result

def prob_conflation(prob_maps):
    # fuse all APs: axis=0
    sub_pmf = np.prod(prob_maps,axis=0)
    # Normalize over all grid points
    return sub_pmf/np.sum(np.abs(sub_pmf),axis=1).reshape(-1,1)

def prob_conflation_gaussian(prob_maps, G):
    # fuse all APs: axis=0
    mu_x = np.squeeze(np.matmul(G, np.expand_dims(prob_maps, axis=3)), axis=-1)
    var_x = np.squeeze(np.matmul(np.square(G), np.expand_dims(prob_maps, axis=3)), axis=-1) - np.square(mu_x)
    x_hat = np.sum(mu_x / var_x, axis=0) / np.sum(1/var_x, axis=0)
    x_hat_var = 1 / np.sum(1/var_x, axis=0)
    return x_hat.T, x_hat_var.T

def sub_samp_by_rate(H_original,UE_pos,timestamps,rate=1):
    H = H_original[::rate,:,:,:]
    UE_pos = UE_pos[::rate,:]
    timestamps = timestamps[::rate]
    return H, UE_pos, timestamps

def sub_samp_by_AP(H_original,UE_pos,timestamps,ap=0):
    non_zero_indices = np.nonzero(H_original[:,0,0,0])[0] # Find the non-zero entries
    H = H_original[non_zero_indices,ap,:,:]
    UE_pos = UE_pos[non_zero_indices,:]
    timestamps = timestamps[non_zero_indices]
    return H, UE_pos, timestamps

def moving_average_over_N(x: np.ndarray, m: int) -> np.ndarray:
    """
    Windowed average along the first axis (N) with edge handling.
    
    Input:
        x : np.ndarray of shape (N, B, A, W)
        m : window size (int). For each n, average values from a window
            centered at n. At the edges the window is truncated to fit.
            The mean is computed per (B, A, W) entry, i.e., independently for
            each W index; only the N axis is averaged.

    Output:
        y : np.ndarray of shape (N, B, A, W)
            y[n, ...] is the average over x[s:e, ...], where:
              s = max(0, n - left)
              e = min(N, n + right)
            with left = m // 2 and right = m - left (so window length is m in the middle).
    """
    if m <= 1:
        # No smoothing needed or invalid m; return a copy to match "write result to new array"
        return x.copy()

    N = x.shape[0]
    left = m // 2
    right = m - left  # ensures total window length m in the center (works for odd/even m)

    # Cumulative sum along axis 0, padded with an initial zero slice for easy range sums
    cs = np.cumsum(x, axis=0)
    cs_pad = np.concatenate([np.zeros_like(x[:1]), cs], axis=0)  # shape (N+1, B, A, W)

    # Vectorized start/end indices for each n (end is exclusive)
    n_idx = np.arange(N)
    starts = np.maximum(0, n_idx - left)
    ends   = np.minimum(N, n_idx + right)

    # Range sums via prefix sums: sum_{s:e} = cs_pad[e] - cs_pad[s]
    sums = cs_pad[ends] - cs_pad[starts]  # shape (N, B, A, W)

    # Window lengths (may be < m at edges)
    counts = (ends - starts).astype(x.dtype)
    y = sums / counts.reshape(-1, *([1] * (x.ndim - 1)))

    return y