import numpy as np
import pickle
import scipy as sp

import argparse
import os

from multiprocessing import Pool

import matplotlib.pyplot as plt

import datetime
parser = argparse.ArgumentParser()

parser.add_argument("-data_path", help="path to csi data", type=str)
from tqdm import tqdm
from contextlib import contextmanager

"""
Dataset generation and CSI feature extraction script.

This script processes raw CSI measurements from a 5G testbed and ground-truth UE
position logs to create a preprocessed dataset for neural positioning. The script:

1. Loads CSI samples (pickle files) from a modified NVIDIA PyAerial DataLake PUSCH Multicell notebook (release version 25-2, described below)
2. Extracts and processes CSI features (downsampling, filtering, normalization)
3. Loads WorldViz ground-truth position labels
4. Interpolates WorldViz positions to CSI sample timestamps
5. Applies spatial filtering (bounding box) if configured
6. Saves processed data as .npz files for use in training

Input Data Format:
This script expects CSI samples from a modified NVIDIA PyAerial DataLake PUSCH
Multicell notebook (release version 25-2), available at:
https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/content/notebooks/datalake_pusch_multicell.html

The PyAerial notebook applies the PUSCH receiver for all O-RUs up to the channel
estimator and saves a pickle file for each PUSCH slot containing:
- ch_est: channel estimates of all O-RUs
- noise_var_dB: estimated pre-equalization noise variance of all O-RUs
- cellIds: cell IDs associated to entries of channel estimates list

WorldViz position logs should be in text format with timestamps and positions.

Workflow:
1. Configuration: Set feature extraction parameters and data paths
2. CSI Loading: Load and parse pickle files containing CSI estimates
3. Feature Processing: Apply downsampling, filtering, and normalization
4. Position Loading: Load and parse WorldViz position logs
5. Temporal Alignment: Interpolate positions to CSI timestamps
6. Spatial Filtering: Apply bounding box constraints (optional)
7. Data Saving: Save processed data as .npz files

@author: Reinhard Wiesmayr
"""

@contextmanager
def opened_w_error(filename, mode="r"):
    """Context manager for file operations with error handling."""
    try:
        f = open(filename, mode)
    except IOError as err:
        yield None, err
    else:
        try:
            yield f, None
        finally:
            f.close()


###############################################################################
# Configuration
###############################################################################
# Configure data paths and feature extraction parameters here.
# Switch between INDOOR and OUTDOOR configurations by changing data_path.

# Data path configuration
data_path = "/scratch/rwiesmayr/csi_data/robot_measurement_outdoor_2025_10_11"  # for CAEZ-5G-OUTDOOR
# data_path = "/scratch/rwiesmayr/csi_data/relative_pos_j61_2025_10_04"  # for CAEZ-5G-INDOOR

# Feature extraction mode
single_ap = False  # If True, combine all O-RUs into single AP

# Feature type selection
# Option 1: Downsampled OFDM-domain absolute values (default)
auto_correlation_features = False
time_domain = False
downsampling_factor = 12  # Downsampling factor for subcarrier dimension (-1 to disable)
ftype = 'fir'  # Filter type: 'fir' or 'iir' for anti-aliasing filter
truncation_len = -1  # Truncation length for time-domain features (-1 to disable)
sum_all_orus = False  # Sum features across all O-RUs
no_norm = True  # Skip normalization (True = no normalization)

# Option 2: Delay-domain autocorrelation features (alternative)
# Uncomment to use this feature type instead:
# auto_correlation_features = True
# time_domain = False
# downsampling_factor = -1
# ftype = 'fir'
# truncation_len = 25
# sum_all_orus = False
# no_norm = True

# Feature processing options
absolute_before_downsampling = True  # Compute absolute values before downsampling
mean_absolutes = True  # Average over DMRS symbols (mean)
median_absolutes = False  # Alternative: use median over DMRS symbols

# Advanced feature options
ant_cov_mat_feature = False  # Use antenna covariance matrix features
num_splits = 1  # Number of splits for covariance matrix features

# Data processing options
sorted_samples = True  # Sort samples by timestamp
DEBUG = False  # Debug mode (limits number of samples)

# System parameters
n_prbs = 273  # Number of physical resource blocks
n_orus = 4  # Number of O-RUs (access points)
n_rx_ant_per_oru = 4  # Number of receive antennas per O-RU
n_tx_ant = 1  # Number of transmit antennas (layers)
n_dmrs_symbols = 3  # Number of DMRS symbols per slot

# Processing configuration
num_processing_splits = 10  # Number of splits for batched sequential processing

# Spatial filtering
apply_bounding_box = True  # Apply bounding box filter (True for OUTDOOR, False for INDOOR)
# Bounding box for OUTDOOR: [lower_left_corner, upper_right_corner]
bounding_box = np.array([[-4.67, -5.582],   # lower left corner [x, y]
                         [5.25, 4.5]])      # upper right corner [x, y]

retrieve_boundin_box = True  # Retrieve effective bounding box from remaining samples

###############################################################################
# File Discovery and Initialization
###############################################################################
# Discover all pickle files containing CSI data
data_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and ".pickle" in f]

# Discover O-RU position files
o_ru_pos_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and "position_data_oru" in f]

# WorldViz ground-truth position log filename
pos_file = "position_data.txt"

# Limit number of samples in debug mode
if DEBUG:
    data_files = data_files[:2000]

# Calculate split sizes for batched sequential processing
n_data_samples = len(data_files)
n_data_samples_split = n_data_samples // num_processing_splits
n_data_samples = n_data_samples_split * num_processing_splits  # Round down to multiple of splits
data_files = data_files[:n_data_samples]

###############################################################################
# CSI Data Loading and Processing
###############################################################################
# Initialize arrays for storing processed data
ue_pos_at_timestamp = np.zeros((n_data_samples, 2), dtype=np.float64)
sample_timestamps = np.zeros((n_data_samples), dtype=np.float64)

# Lists for storing split results (will be concatenated later)
sample_timestamps_splits = []
H_splits = []
H_max_sc_splits = []
noise_var_splits = []

# Progress bar for monitoring
t = tqdm(total=n_data_samples)

# Process data in splits for memory efficiency
for i in range(num_processing_splits):
    data_files_split = data_files[i*n_data_samples_split:(i+1)*n_data_samples_split]
    
    H = np.zeros((n_data_samples_split, n_orus, n_rx_ant_per_oru, n_tx_ant, n_prbs*12, n_dmrs_symbols), dtype=np.complex64)
    noise_var = np.zeros((n_data_samples_split,n_orus), dtype=np.float32)

    print(f"Processing CSI estimates for split {i+1}/{num_processing_splits}")
    ###########################################################################
    # Load CSI Data from Pickle Files
    ###########################################################################
    # Each pickle file contains CSI estimates from one PUSCH slot
    for idx, data_file in enumerate(data_files_split):
        t.update()  
        with opened_w_error(os.path.join(data_path, data_file), "rb") as (file, err):
            if err:
                print("File " + data_file + " has IO Error: " + str(err))
            else:
                x = pickle.load(file)
                h_sample = x['ch_est'] # N_ORU x Rx ant x layer x frequency x time
                H[idx, :, :, :, :, :] = np.squeeze(np.array(h_sample), axis=1)[:, :n_rx_ant_per_oru, :, :, :]    # np.array should make a copy, slice assignment should also do the copy on itself
                noise_var[idx, :] = np.squeeze(np.array(x['noise_var_dB']))
                # we also have in x the keys 'start_prb', 'num_prbs' but we always have all 273 PRBs

    ###########################################################################
    # Antenna Covariance Matrix Features (Optional)
    ###########################################################################
    # Compute antenna covariance matrix features if enabled
    if ant_cov_mat_feature:
        # H is n_data_samples, n_orus, n_rx_ant_per_oru, n_tx_ant, n_prbs*12, n_dmrs_symbols
        H = np.reshape(H, newshape=(n_data_samples, n_orus, n_rx_ant_per_oru, n_tx_ant, num_splits, int(n_prbs*12 / num_splits), n_dmrs_symbols))
        H = H.transpose((0, 1, 3, 4, 6, 2, 5))
        # H is n_data_samples, n_orus, n_tx_ant, num_splits, n_dmrs_symbols, n_rx_ant_per_oru, n_prbs*12/num_splits

        F = np.expand_dims(np.fft.fft(np.eye(n_rx_ant_per_oru)), axis=(0,1,2,3,4))
        FH = np.matmul(F,H)
        H = np.abs(np.matmul(FH,FH.conj().transpose(0,1,2,3,4,6,5)))

        H = H.transpose((0,1,2,4,5,6,3))
        H = np.reshape(H, newshape=(n_data_samples, n_orus, n_tx_ant, n_dmrs_symbols, n_rx_ant_per_oru, n_rx_ant_per_oru*num_splits))
        # H is n_data_samples, n_orus, n_tx_ant, n_dmrs_symbols, n_rx_ant_per_oru, n_rx_ant_per_oru*num_splits

        H = H / np.linalg.norm(H, ord="fro", axis=(-2,-1), keepdims=True)
        # H is n_data_samples, n_orus, n_tx_ant, n_dmrs_symbols, n_rx_ant_per_oru, n_rx_ant_per_oru
        # undo transpose
        H = H.transpose((0, 1, 4, 2, 5, 3))
        # H is n_data_samples, n_orus, n_rx_ant_per_oru, n_tx_ant, n_rx_ant_per_oru*num_splits, n_dmrs_symbols

    ###########################################################################
    # Time Domain Transformation (Optional)
    ###########################################################################
    # Transform OFDM-domain CSI to delay domain via IFFT
    # Note: IFFT could be implemented as a DFT matrix-vector product (slower)
    if time_domain:
        # 1st fftshift: H has least subcarrier index at position 0 --> move carrier frequency to position 0
        # ifft pushes time-zero index to position 0
        # 2nd fftshift: move time-zero index to middle of the CIR
        print("Compute IFFT to obtain delay domain CSI samples")
        H = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(H, axes=4), axis=4, norm="ortho"), axes=4) 

    # Compute absolute values before downsampling if enabled
    if absolute_before_downsampling:
        H = np.abs(H).astype(np.float32)

    ###########################################################################
    # Downsampling
    ###########################################################################
    # Downsample subcarrier dimension to reduce feature dimensionality
    # Uses parallel processing for efficiency (sequential downsampling is slow)
    #
    # Implementation via scipy.signal.decimate:
    # 1. Low-pass anti-aliasing filter (FIR/IIR)
    # 2. Zero-phase filtering (compensates for group delay)
    # 3. Subsampling by downsampling_factor

    # Function to apply decimate on a chunk of the array
    def downsample_chunk(chunk_and_params):
        chunk, q, axis = chunk_and_params
        return sp.signal.decimate(chunk, q, axis=axis, zero_phase=True, ftype=ftype)

    # Main function for parallelized downsampling
    def parallel_decimate_with_pool(H, q, axis=4, num_cores=4):
        # Swap the downsampling axis to the last axis for easier chunking
        H = np.moveaxis(H, axis, -1)
        
        # Split the array along the first dimension for parallel processing
        chunks = np.array_split(H, num_cores, axis=0)
        
        # Create a list of arguments for the worker function
        chunk_params = [(chunk, q, -1) for chunk in chunks]
        
        # Use multiprocessing.Pool to process the chunks in parallel
        with Pool(processes=num_cores) as pool:
            downsampled_chunks = pool.map(downsample_chunk, chunk_params)
        
        # Concatenate the processed chunks
        downsampled_H = np.concatenate(downsampled_chunks, axis=0)
        
        # Move the downsampling axis back to its original position
        downsampled_H = np.moveaxis(downsampled_H, -1, axis)
        
        return downsampled_H

    if downsampling_factor > 1:
        print(f"Downsampling by factor {downsampling_factor}")
        # H = sp.signal.decimate(H, q=downsampling_factor, axis=4)    # down-sampling over frequency or time domain
        H_ = parallel_decimate_with_pool(H, q=downsampling_factor, axis=4)
        ## Optional for visualization during debugging:
        # plt.figure()
        # subcarrier_idx = np.arange(273*n_prbs)
        # plt.figure()
        # subcarrier_idx = np.arange(12*n_prbs)
        # plt.plot(subcarrier_idx, np.real(H[1000,1,0,0,:,0]),label='real original')
        # plt.plot(subcarrier_idx, np.imag(H[1000,1,0,0,:,0]),label='imag original')
        # plt.plot(subcarrier_idx[::downsampling_factor], np.real(H_[1000,1,0,0,:,0]),label='real downsampled')
        # plt.plot(subcarrier_idx[::downsampling_factor], np.imag(H_[1000,1,0,0,:,0]),label='imag downsampled')
        # plt.show()
        # plt.legend()
        # plt.savefig('downsampling.pdf', format='pdf')
        H = H_

    ###########################################################################
    # Feature Truncation (Optional)
    ###########################################################################
    # Truncate time-domain features to keep only center samples
    # Useful for delay-domain features where most energy is in center (i.e., around the time-zero index)
    if truncation_len > 0 and not auto_correlation_features:
        print(f"Truncation of center of time domain CSI to length {truncation_len}")
        num_time_samples = int(12*n_prbs/downsampling_factor)
        num_ft_samples = truncation_len
        H__ = H[:,:,:,:,int((num_time_samples-truncation_len)/2):int((num_time_samples+truncation_len)/2),:]
        ## Optional for visualization during debugging:
        # # plt.figure()
        # plt.plot(np.real(H[1000,1,0,0,:,0]),label='real original')
        # plt.plot(np.imag(H[1000,1,0,0,:,0]),label='imag original')
        # plt.plot(np.real(H__[1000,1,0,0,:,0]),label='real downsampled')
        # plt.plot(np.imag(H__[1000,1,0,0,:,0]),label='imag downsampled')
        # plt.show()
        # plt.legend()
        # plt.savefig('truncation.pdf', format='pdf')
        H = H__
    else:
        num_ft_samples = int(12*n_prbs/downsampling_factor)

    ###########################################################################
    # DMRS Symbol Averaging
    ###########################################################################
    # Average CSI features over all DMRS symbols in the slot
    # ARC-OTA testbed typically uses 3 DMRS symbols per slot
    if mean_absolutes or median_absolutes:
        if mean_absolutes:
            print(f"Computing mean absolutes over N_tx={n_tx_ant} and all {n_dmrs_symbols} DMRS symbols")
            H = np.mean(np.abs(H), axis=(3,5))
        elif median_absolutes:
            print(f"Computing median absolutes over N_tx={n_tx_ant} and all {n_dmrs_symbols} DMRS symbols")
            H = np.median(np.abs(H), axis=(3,5))
        # H is n_data_samples, n_orus, n_rx_ant_per_oru, n_prbs*12
        H = H.astype(np.float32)

        if single_ap:
            H = np.reshape(H, (n_data_samples, 1, n_orus*n_rx_ant_per_oru, num_ft_samples))

    # Compute maximum subcarrier power (for analysis/debugging)
    H_max_sc = np.max(np.abs(H), axis=3)

    ###########################################################################
    # Autocorrelation Features (Alternative Feature Type)
    ###########################################################################
    # Compute delay-domain autocorrelation features
    # Method: squared-absolutes of OFDM CSI → IFFT → delay domain autocorrelation
    # Steps:
    # 1. Compute squared absolutes in OFDM domain
    # 2. IFFT to transform to delay domain
    # 3. Truncate to keep center samples (CAEZ-5G setup: typically 25 samples contain most energy)
    # 4. Stack real and imaginary parts
    # 5. Optionally normalize to unit Frobenius norm
    if auto_correlation_features:
        # take squared absolutes
        print("Compute squared absolutes")
        H = np.square(np.abs(H))

        # transform to time domain
        print("Compute IFFT")
        H = np.fft.ifft(np.fft.fftshift(H, axes=3), axis=3, norm="ortho")

        # truncate
        if truncation_len > 0:
            H = H[:,:,:,:truncation_len]
        else:
            H = H[:,:,:,:np.shape(H)[-1]//2]
        
        # stack real and imaginary
        H = np.concatenate((np.real(H), np.imag(H)),axis=-1)

        if sum_all_orus:
            H = np.sum(H, axis=1, keepdims=True)

        # normalize per AP
        if not no_norm:
            print("Normalize per AP")
            H = H / np.linalg.norm(H, ord="fro", axis=(2,3), keepdims=True)

    # Store results for this split
    print(f"Finished split {i+1}/{num_processing_splits}")
    H_splits.append(H)
    noise_var_splits.append(noise_var)
    H_max_sc_splits.append(H_max_sc)

###############################################################################
# Concatenate All Splits
###############################################################################
# Combine all processed splits into single arrays
H = np.concatenate(H_splits, axis=0)
noise_var = np.concatenate(noise_var_splits, axis=0)
H_max_sc = np.concatenate(H_max_sc_splits, axis=0)

###############################################################################
# WorldViz Position Log Processing
###############################################################################
# Load and parse ground-truth UE positions from WorldViz logs
# Positions are stored with timestamps and need to be interpolated to CSI timestamps

print("Processing WorldViz position estimates")
num_processes = 10
def process_chunk(lines):
    """Parse a chunk of WorldViz log lines into timestamps and positions."""
    times = []
    positions = []
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            # Parse timestamp: format "YYYY-MM-DD HH:MM:SS.microseconds"
            time_str = parts[0] + " " + parts[1][:-7] + "." + parts[1][-6:]
            position_str = parts[2].split(";")[0]
            
            # Convert to Unix timestamp
            time = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
            times.append(time.timestamp())
            
            # Parse position coordinates (x, y, z)
            position = tuple(map(float, position_str.split(",")))
            positions.append(position)
    
    return np.array(times, dtype=np.float64), np.array(positions, dtype=np.float32)

def read_file_in_chunks(file_path, chunk_size):
    """Read file in chunks for parallel processing."""
    with open(file_path, "r") as file:
        lines = []
        for line in file:
            lines.append(line)
            if len(lines) >= chunk_size:
                yield lines
                lines = []
        if lines:
            yield lines

def parallel_process_file(file_path, num_processes):
    """Process WorldViz log file in parallel chunks."""
    # Estimate chunk size for balanced parallel processing
    with open(file_path, "r") as file:
        total_lines = sum(1 for _ in file)
    chunk_size = total_lines // num_processes
    
    # Process chunks in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, read_file_in_chunks(file_path, chunk_size))
    
    # Combine results from all processes
    times_list, positions_list = zip(*results)
    times_array = np.concatenate(times_list)
    positions_array = np.vstack(positions_list)
    
    return times_array, positions_array

# Load WorldViz position log
worldwiz_times_array, worldwiz_positions_array = parallel_process_file(os.path.join(data_path, pos_file), num_processes)

###############################################################################
# Temporal Alignment: Interpolate Positions to CSI Timestamps
###############################################################################
# Verify that parallel processing maintained chronological order
is_sorted = lambda a: np.all(a[:-1] <= a[1:])

if is_sorted(worldwiz_times_array):
    print("Parallel processing kept chronological order")
else:
    raise Exception("Parallel processing failed: timestamps not sorted")

# Load O-RU positions
oru_pos_dict = {}
for idx, o_ru_pos_file in enumerate(o_ru_pos_files):
    if idx > 0 and sum_all_orus:
        break
    _, oru_pos = parallel_process_file(os.path.join(data_path, o_ru_pos_file), num_processes=1)
    oru_pos_mean = np.mean(oru_pos, axis=0)  # Average position over time
    oru_pos_dict[o_ru_pos_file.split(".")[0]] = oru_pos_mean

# Extract timestamps from CSI data filenames
print("Extracting timestamps from CSI filenames")
for idx, data_file in enumerate(data_files):
    timestamp_str = data_file.split("_")[0]  # Timestamp is prefix of filename
    timestamp = np.fromstring(timestamp_str, dtype=np.float64, sep='.')
    sample_timestamps[idx] = timestamp[0]

# Filter samples to those within WorldViz time series range
print("Filtering samples to WorldViz time range")
in_worldwiz_timeseries_idx = np.squeeze(np.argwhere(
    np.logical_and(sample_timestamps < np.max(worldwiz_times_array), 
                   sample_timestamps > np.min(worldwiz_times_array))))
sample_timestamps = np.take(sample_timestamps, in_worldwiz_timeseries_idx, axis=0)
H = np.take(H, in_worldwiz_timeseries_idx, axis=0)
noise_var = np.take(noise_var, in_worldwiz_timeseries_idx, axis=0)
H_max_sc = np.take(H_max_sc, in_worldwiz_timeseries_idx, axis=0)
ue_pos_at_timestamp = np.zeros((np.shape(sample_timestamps)[0], 2), dtype=np.float64)

# Interpolate WorldViz positions to CSI sample timestamps
# WorldViz uses (x, y, z) coordinates; we extract (x, z) for 2D positioning
ue_pos_at_timestamp[:, 0] = np.interp(sample_timestamps, worldwiz_times_array, worldwiz_positions_array[:, 0])  # x coordinate
ue_pos_at_timestamp[:, 1] = np.interp(sample_timestamps, worldwiz_times_array, worldwiz_positions_array[:, 2])  # z coordinate (used as y)

###############################################################################
# Spatial Filtering and Final Processing
###############################################################################
# Apply bounding box filter to remove samples outside measurement area
if apply_bounding_box:
    x_in_box = np.logical_and(ue_pos_at_timestamp[:, 0] > bounding_box[0, 0], 
                              ue_pos_at_timestamp[:, 0] < bounding_box[1, 0])
    y_in_box = np.logical_and(ue_pos_at_timestamp[:, 1] > bounding_box[0, 1], 
                              ue_pos_at_timestamp[:, 1] < bounding_box[1, 1])
    sample_idx_in_box = np.squeeze(np.argwhere(np.logical_and(x_in_box, y_in_box)))

    print(f"{len(sample_idx_in_box)} out of {len(ue_pos_at_timestamp)} samples are inside the bounding box")

    # Filter all arrays to keep only samples inside bounding box
    H = np.take(H, sample_idx_in_box, axis=0)
    noise_var = np.take(noise_var, sample_idx_in_box, axis=0)
    ue_pos_at_timestamp = np.take(ue_pos_at_timestamp, sample_idx_in_box, axis=0)
    H_max_sc = np.take(H_max_sc, sample_idx_in_box, axis=0)
    sample_timestamps = np.take(sample_timestamps, sample_idx_in_box, axis=0)

# Sort samples by timestamp (chronological order)
if sorted_samples:
    print("Sorting samples by timestamp")
    time_asc_idx = np.argsort(sample_timestamps)
    H = np.take(H, time_asc_idx, axis=0)
    noise_var = np.take(noise_var, time_asc_idx, axis=0)
    H_max_sc = np.take(H_max_sc, time_asc_idx, axis=0)
    ue_pos_at_timestamp = np.take(ue_pos_at_timestamp, time_asc_idx, axis=0)
    sample_timestamps = np.take(sample_timestamps, time_asc_idx, axis=0)

# Retrieve effective bounding box from remaining samples (optional)
if retrieve_boundin_box:
    until_when = -7400  # Optional: exclude last N samples from bounding box calculation
    bounding_box[0, 0] = np.min(ue_pos_at_timestamp[:until_when, 0])
    bounding_box[1, 0] = np.max(ue_pos_at_timestamp[:until_when, 0])
    bounding_box[0, 1] = np.min(ue_pos_at_timestamp[:until_when, 1])
    bounding_box[1, 1] = np.max(ue_pos_at_timestamp[:until_when, 1])

# Prepare bounding box for saving (one per O-RU or single if single_ap mode)
if single_ap:
    bounding_box = [bounding_box]
else:
    bounding_box = np.repeat([bounding_box], repeats=n_orus, axis=0)

###############################################################################
# Save Processed Dataset
###############################################################################
# Generate filename based on feature extraction parameters
# This ensures unique filenames for different feature configurations

print("Saving processed dataset to .npz file (uncompressed)")

filename = data_path

if ant_cov_mat_feature:
    filename += f'_ant_cov_mat_split{num_splits}'

if time_domain:
    filename += '_time_domain'

if auto_correlation_features:
    filename += '_auto_correlation'
    if no_norm:
        filename += '_no_norm'

if single_ap:
    filename += '_single_ap'

if median_absolutes:
    filename += "_median_absolutes"

if downsampling_factor > 1:
    if absolute_before_downsampling:
        filename += '_abs_before'
    filename += f"_downsampling_{downsampling_factor}"

if truncation_len > 0:
    filename += f"_truncated_{truncation_len}"

if sorted_samples:
    filename += "_sorted"

if sum_all_orus:
    filename += "_sum_oru"

filename += f"_{n_rx_ant_per_oru}_rx_ant"
np.savez(filename + '_wo_last_500.npz', H=H[4100:-8300,:,:,:], noise_var=noise_var[4100:-8300], H_max_sc=H_max_sc[4100:-8300], pos=ue_pos_at_timestamp[4100:-8300,:], oru_pos_dict=oru_pos_dict, timestamps=sample_timestamps[4100:-8300], bounding_box=bounding_box) # CAEZ-5G-OUTDOOR
np.savez(filename + '_only_last_500.npz', H=H[-8300:-7800,:,:,:], noise_var=noise_var[-8300:-7800], H_max_sc=H_max_sc[-8300:-7800], pos=ue_pos_at_timestamp[-8300:-7800,:], oru_pos_dict=oru_pos_dict, timestamps=sample_timestamps[-8300:-7800], bounding_box=bounding_box) # CAEZ-5G-OUTDOOR
# np.savez(filename + '_wo_last_500.npz', H=H[:-1750,:,:,:], noise_var=noise_var[:-1750], H_max_sc=H_max_sc[:-1750], pos=ue_pos_at_timestamp[:-1750,:], oru_pos_dict=oru_pos_dict, timestamps=sample_timestamps[:-1750], bounding_box=bounding_box) # CAEZ-5G-INDOOR
# np.savez(filename + '_only_last_500.npz', H=H[-1750:-1250,:,:,:], noise_var=noise_var[-1750:-1250], H_max_sc=H_max_sc[-1750:-1250], pos=ue_pos_at_timestamp[-1750:-1250,:], oru_pos_dict=oru_pos_dict, timestamps=sample_timestamps[-1750:-1250], bounding_box=bounding_box) # CAEZ-5G-INDOOR
# np.savez(filename + '_wo_last_1000.npz', H=H[:-1000,:,:,:], noise_var=noise_var[:-1000], H_max_sc=H_max_sc[:-1000], pos=ue_pos_at_timestamp[:-1000,:], oru_pos_dict=oru_pos_dict, timestamps=sample_timestamps[:-1000], bounding_box=bounding_box) # CAEZ-5G-BASEMENT

filename += '.npz'

np.savez(filename, H=H, noise_var=noise_var, H_max_sc=H_max_sc, pos=ue_pos_at_timestamp, oru_pos_dict=oru_pos_dict, timestamps=sample_timestamps, bounding_box=bounding_box)
print("Done!")