"""
Script to generate a dataset of CSI features from a CAEZ-WIFI dataset.

This script groups CSI data into fixed-size time windows and extracts feature vectors for each window.

@author: Frederik Zumegen
"""

import json
import re
import numpy as np
from pathlib import Path
import pickle
import helper as hp

### Modify this ###
output_filename = "caez-wifi-indoor-lshape.npz"
output_path = "/home/user/"
input_path = "/home/user/caez-wifi-indoor-lshape/"

window_len = 0.1 # Size of window in seconds for grouping the independently measured CSI data

feature_scAmp = True # Always True at the moment; Set to True if we want a dataset of subcarrier amplitude features
feature_delay = False # Set to True if we want a dataset of truncated delay-domain features
apply_csi_feature_smooting = True # Apply a moving-average filter to the frequency CSI features along the time-domain
apply_gt_data_smoothing = False # If the GT (WorldViz) data is jittery, we can apply a moving-average window over the data
###################

# Other presets
W = 52  # Number of subcarriers
A = 4 # Number of antennas
B = 4 # Number of sniffers/APs

# Set data paths - dynamically build from directory structure
def build_sniffer_data_paths(input_path: str) -> dict:
    """
    Build sniffer_data_paths dictionary from directory structure.
    
    Scans data_path for folders named "apX" (where X is an integer) and creates
    a dictionary with integer keys and lists of subfolder paths as values.
    
    Parameters:
    - input_path: Path to the directory containing "apX" folders
    
    Returns:
    - Dictionary with integer keys (AP numbers) and lists of Path objects (subfolders)
    """
    data_path_obj = Path(input_path)
    sniffer_data_paths = {}
    
    if not data_path_obj.exists():
        raise ValueError(f"Input path does not exist: {input_path}")
    
    # Find all folders matching "apX" pattern (case-insensitive)
    ap_pattern = re.compile(r'^ap(\d+)$', re.IGNORECASE)
    
    for item in data_path_obj.iterdir():
        if item.is_dir():
            match = ap_pattern.match(item.name)
            if match:
                ap_number = int(match.group(1))
                # Get all subfolders in this apX folder
                subfolders = [subfolder for subfolder in item.iterdir() if subfolder.is_dir()]
                sniffer_data_paths[ap_number] = subfolders
    
    # Sort by AP number for consistent ordering
    sniffer_data_paths = dict(sorted(sniffer_data_paths.items()))
    
    return sniffer_data_paths

sniffer_data_paths = build_sniffer_data_paths(input_path)

def read_complex_matrix_from_csv(file_path, A):
    # Load the CSV file into a NumPy array, skipping the header
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype='complex64')

    # Return only the rows you need (1 to 4)
    return data[:A, :]  # First four rows (0-based indexing)

def get_gt_pos(timestamp, gt_timestamps, gt_positions):
    """
    Get ground truth position based on the closest timestamp.

    Parameters:
    - timestamp: the timestamp for which to find the closest ground truth position
    - gt_timestamps: numpy array of ground truth timestamps
    - gt_positions: numpy array of ground truth positions (shape: [n, 2] where n is the number of positions)

    Returns:
    - gt_pos: the ground truth position corresponding to the closest timestamp
    """
    # Compute absolute differences between the timestamp and all ground truth timestamps
    abs_differences = np.abs(gt_timestamps - timestamp)
    # Find the index corresponding to the minimum absolute difference
    min_index = np.argmin(abs_differences)
    # Get the corresponding ground truth position
    gt_pos = gt_positions[min_index, :]
    
    return gt_pos

def extract_frame_prefix(json_path: Path) -> str:
    """Get the 'frame' identifier from a metadata file like 'frame9_metadata.json'."""
    return json_path.name.replace("_metadata.json", "")

def get_csv_path(json_path: Path) -> Path:
    """Return corresponding CSV file path from JSON path."""
    frame_prefix = extract_frame_prefix(json_path)
    # return json_path.with_name(f"{frame_prefix}_data_sym_csi.csv") # H_data (CSI from L-SIG and DATA field)
    return json_path.with_name(f"{frame_prefix}_csi.csv") # H_leg (CSI from L-LTF)

def read_timestamp(json_path: Path) -> float:
    with open(json_path) as f:
        meta = json.load(f)
    return meta["timestamp"]

def organize_into_buckets():
    buckets = [[] for _ in range(B)]
    all_timestamps = []
    
    # Flatten the dictionary to iterate over all subfolders
    total_paths = sum(len(subfolders) for subfolders in sniffer_data_paths.values())
    path_counter = 0
    
    for ap_number, subfolders in sorted(sniffer_data_paths.items()):
        for data_dir in subfolders:
            path_counter += 1
            json_paths = list(Path(data_dir).rglob("*.json"))
            nof_samples = len(json_paths)
            for i, json_path in enumerate(json_paths):
                print(f"Unpacking path {path_counter} of {total_paths}: {int(i/nof_samples*100)}%")
                csv_path = get_csv_path(json_path)
                
                if not csv_path.exists():
                    print(f"Warning: Missing CSV for {json_path}")
                    continue

                timestamp = read_timestamp(json_path)
                H = read_complex_matrix_from_csv(csv_path, A)
                
                # Use ap_number - 1 as sniffer index (ap1 -> 0, ap2 -> 1, etc.)
                sniffer_index = ap_number - 1
                if 0 <= sniffer_index < B:  # Ensure index is within valid range
                    buckets[sniffer_index].append((timestamp, H))
                    all_timestamps.append(timestamp)

    return buckets, min(all_timestamps), max(all_timestamps)

def split_by_intervals(bucket, sniffer_idx, start_time, n_intervals, x):
    """
    bucket: list of (timestamp, CSI array)
    start_time: minimum timestamp of interest
    n_intervals: number of x-second intervals
    x: window/interval size in seconds
    """

    bins_freq_feat, bins_delay_feat = [[] for _ in range(n_intervals)], [[] for _ in range(n_intervals)]
    bins_for_t = [[] for _ in range(n_intervals)]
    # Fill the bins with arrays
    print(f"Sorting CSI for sniffer {sniffer_idx+1} ... ")
    for timestamp, csi_array in bucket:
        index = int((timestamp - start_time) // x)
        if 0 <= index < n_intervals:
            # Turn CSI array into a feature already: CSI Magnitude in frequency
            bins_freq_feat[index].append(np.abs(csi_array))
            bins_for_t[index].append(timestamp-start_time)
            if feature_delay:
                # Store another feature variant: delay domain
                bins_delay_feat[index].append(compute_delay_feature(csi_array))

    samples_per_bin = np.array([len(bins_freq_feat[i]) for i in range(n_intervals)])
    bins_avg_freq_feat = compute_mean_feature(bins_freq_feat, n_intervals, num_columns=52)
    if feature_delay:
        bins_avg_delay_feat = compute_mean_feature(bins_delay_feat, n_intervals, num_columns=34)
        return bins_avg_freq_feat, bins_avg_delay_feat, samples_per_bin, bins_for_t
    else:
        return bins_avg_freq_feat, None, samples_per_bin, bins_for_t

def compute_mean_feature(bins, n_intervals, num_columns):
    """
    bins: list of bins with CSI arrays

    Returns: 3-D numpy array with: 1st dim number of intervals, 2nd and 3rd dim CSI array
    """
    result = np.zeros((n_intervals, 4, num_columns))
    for idx, arr_list in enumerate(bins):
        if arr_list:
            stacked = np.stack(arr_list)
            mean_array = np.mean(stacked, axis=0)
        else:
            mean_array = np.zeros((4,num_columns))
        result[idx,:,:] = mean_array
    return result

def compute_delay_feature(array, num_of_taps=34):
    """
    array: original CSI array in subcarrier/frequency domain
    returns absoulte values of a delay domain version of the original CSI
    """
    temp = np.fft.ifft(array)
    return np.abs(temp[:,:num_of_taps])

def main():
    # Step 1: Read GT positions and timestamps
    gt_positions_file = Path(input_path+'gt-positions.csv')
    if gt_positions_file.exists():
        try:
            print('Reading GT file ...\n')
            gt_data = np.genfromtxt(gt_positions_file, delimiter=',', skip_header=1) 
            gt_data = gt_data[:,[0, 1, 3]]  # omit 'y' (height)
        except:
            print("Error reading ground-truth positions.\n")
    else:
        print("[warning]: No GT file available. Setting GT to zeros.\n")
        gt_data = np.zeros((1, 3))

    if apply_gt_data_smoothing:
        print("Applying GT data smoothing ...\n")
        offset = np.array([[100, 100]]) # offset to perform the averaging operation correctly
        gt_data[:, 1:3] = hp.moving_average_over_N(gt_data[:, 1:3]+offset, m=11)-offset
        print(f"Shape of GT positions: {gt_data[:,1:3].shape}\n")
    
    # Step 2: Read and bucket data
    H_buckets, t_min, t_max = organize_into_buckets()

    # Step 3: Compute time intervals for grouping the CSI data
    n_intervals = int(np.ceil((t_max - t_min) / window_len))
    # Compute center timestamps of intervals
    interval_centers = np.array([t_min + (i + 0.5) * window_len for i in range(n_intervals)])

    # Step 4: Create positions array: Find closest position for each interval center
    position_timestamps = gt_data[:, 0]
    interval_mapped_positions = np.zeros((n_intervals,2))
    for i, center_time in enumerate(interval_centers):
        print(f"Creating array of positions for time window centers ...{int(i/n_intervals*100)}%") 
        idx = np.argmin(np.abs(position_timestamps - center_time))
        interval_mapped_positions[i,:] = gt_data[idx, 1:3]  # Assuming x in col 1, y in col 2

    # Step 5: Sort CSI data in time window bins and compute feature
    all_freq_feat, all_delay_feat = [], []
    orig_timestamps_per_AP_and_bin = [None]*B
    samples_per_AP_and_bin = np.zeros((B, n_intervals))
    for b in range(B):
        freq_feat_bins_perSniffer, delay_feat_bins_perSniffer, samples_per_AP_and_bin[b,:], orig_timestamps_per_AP_and_bin[b] = split_by_intervals(H_buckets[b], b, t_min, n_intervals, window_len)
        all_freq_feat.append(freq_feat_bins_perSniffer)
        if feature_delay:
            all_delay_feat.append(delay_feat_bins_perSniffer)

    # Step 6: Convert lists to arrays
    H = np.array(all_freq_feat)
    H = np.transpose(H, (1,0,2,3)) # Put sample index in front, sniffer index second
    if feature_delay:
        H_delay = np.array(all_delay_feat)
        H_delay = np.transpose(H_delay, (1,0,2,3)) # Put sample index in front, sniffer index second
    else:
        H_delay = None

    # Step 7 (optional): Apply moving-average to frequency CSI features along time-domain
    if apply_csi_feature_smooting:
        print("Applying moving-average to CSI features...")
        H = hp.moving_average_over_N(H,m=51)

    # Step 8: Remove empty rows in 'H', i.e., any row of H where all entries across (B, A, W) are zero (i.e., all H[n, :, :, :] == 0)
    """
    Careful: this step removes temporal continuity, as some timesteps are simply dropped.
    """
    print("Removing H-zero-row indices...")
    nonzero_rows = ~(H == 0).all(axis=(1,2,3))
    H = H[nonzero_rows]
    if H_delay is not None:
        H_delay = H_delay[nonzero_rows]
    interval_mapped_positions = interval_mapped_positions[nonzero_rows]
    interval_centers = interval_centers[nonzero_rows]
    samples_per_AP_and_bin = samples_per_AP_and_bin[:, nonzero_rows]

    # Get and add AP positions
    ap_pos = np.load(input_path+'ap-pos.npz')['ap_pos']

    # Save the arrays to a .npz file
    np.savez_compressed(output_path+output_filename,
                        H=H,
                        H_delay=H_delay,
                        pos=interval_mapped_positions,
                        timestamps=interval_centers,
                        orig_nof_samples_per_AP=np.array([len(H_buckets[i]) for i in range(B)]),
                        samples_per_AP_and_bin=samples_per_AP_and_bin,
                        ap_pos=ap_pos)
    
    # with open(output_filename[:-4]+"-orig-timestamps.pkl", "wb") as f:
    #     pickle.dump(orig_timestamps_per_AP_and_bin, f)

    print(f"H.shape: {H.shape}\n")
    print("Dataset build done!\n")

if __name__ == "__main__":
    main()