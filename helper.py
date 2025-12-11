import numpy as np
import cvxpy as cp

# Global variables
G = None # grid points
grid_gran = None
# epsilon: max. allowed offset of G@p' from x for solution p'. Depends on granularity of grid.
epsilon = None

def make_grid(positions):
    global G
    global grid_gran
    global epsilon

    # Number of grid points per one dimension
    K = 22
    
    x = np.linspace(np.min(positions[:,0])-0.1, np.max(positions[:,0])+0.1, K)
    y = np.linspace(np.min(positions[:,1])-0.1, np.max(positions[:,1])+0.1, K)
    xv, yv = np.meshgrid(x, y)

    # A 2 by K^2 array of grid points
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