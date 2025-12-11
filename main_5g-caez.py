"""
Main script for CSI-based neural positioning using probability maps and CAEZ-5G datasets.

This script implements supervised learning for UE positioning based on CSI features
from a 5G testbed. The method uses probability maps as described in [1] to train
a neural network that outputs probability distributions over a spatial grid. These
probability maps are then fused to estimate absolute UE positions.

Workflow:
1. Load preprocessed CSI data and ground-truth UE positions
2. Generate probability maps for ground-truth positions
3. Extract CSI features and create training/test datasets
4. Train neural network(s) for each access point (AP)
5. Evaluate on test sets and compute positioning errors
6. Generate plots and save results

The script supports both CAEZ-5G-INDOOR and CAEZ-5G-OUTDOOR datasets. Configuration
is done via the 'config' dictionary at the top of the file.

References:
[1] E. Goenueltas, E. Lei, J. Langerman, H. Huang, and C. Studer, "CSI-based 
    multi-antenna and multi-point indoor positioning using probability fusion,"
    IEEE Trans. Wireless Commun., vol. 21, no. 4, pp. 2162-2176, 2021.

Created on Fri Feb 23 13:21:40 2024
@author: Frederik Zumegen, Reinhard Wiesmayr

Based on a template provided by Sueda Taner and Victoria Palhares and by an 
online course "Introduction to Neural Networks and PyTorch" available on coursera.org
"""

import matplotlib.pyplot as plt 
import numpy as np
import torch
import os

from NN_models import *
from create_dataset import *
from param_config import *
import helper as hp

import warnings
import argparse

###############################################################################
# Command-line Arguments
###############################################################################
parser = argparse.ArgumentParser(description='CSI-based neural positioning using probability maps and CAEZ-5G datasets')
parser.add_argument('--data_path', type=str, default='~/csi_data/',
                    help='Path to the directory containing CSI data files (default: ~/csi_data/)')
parser.add_argument('--results_path', type=str, default='~/results/',
                    help='Path to the directory containing results files (default: ~/results/)')
args = parser.parse_args()

###############################################################################
# Configuration
###############################################################################
# Configure training parameters and dataset paths here.
# Switch between INDOOR and OUTDOOR configurations by commenting/uncommenting
# the respective config blocks below.

# # For CAEZ-5G-INDOOR
# config = {
#         "learning_rate": 1e-4,  # Initial learning rate for Adam optimizer
#         "architecture": "probability_NN",  # "probability_NN" is the default architecture for this method.
#         "dataset": "relative_pos_j61_2025_10_04_abs_before_downsampling_12_sorted_4_rx_ant_wo_last_500.npz",  # Main training/test dataset
#         "dataset_val": "relative_pos_j61_2025_10_04_abs_before_downsampling_12_sorted_4_rx_ant_only_last_500.npz",  # Additional validation set (last 500 samples, excluded from random split)
#         "epochs": 50,  # Number of training epochs
#         "CSI_type": "H",  # CSI data key in .npz file
#         "temporal_conflation": False,  # Whether to use temporal probability fusion
#         "training_noise_var": 0.0,  # Variance of Gaussian noise added during training (data augmentation)
#         "dropout_rate": 0.  # Dropout rate for regularization (0 = no dropout)
#             }
# results_dir = args.results_path+'relative_pos_j61_2025_10_04_abs_before_downsampling_12_sorted_4_rx_ant_wo_last_500_as1oru/'

# For CAEZ-5G-OUTDOOR
config = {
        "learning_rate": 1e-4,
        "architecture": "probability_NN", # "probability_NN" is the default architecture for this method.
        "dataset": "robot_measurement_outdoor_2025_10_11_abs_before_downsampling_12_sorted_4_rx_ant_wo_last_500.npz",
        "dataset_val": "robot_measurement_outdoor_2025_10_11_abs_before_downsampling_12_sorted_4_rx_ant_only_last_500.npz",  # extra test dataset in addition to random partitioning of "dataset"
        "epochs": 50,
        "CSI_type": "H",
        "temporal_conflation": False,
        "training_noise_var": 0.0,
        "dropout_rate": 0.
            }
results_dir = args.results_path+'robot_measurement_outdoor_2025_10_11_abs_before_downsampling_12_sorted_4_rx_ant_wo_last_500_as1oru/'

# Create results directory if it does not exist
os.makedirs(results_dir, exist_ok=True)

if "training_noise_var" in config:
    print(f"Training noise variance: {config['training_noise_var']}")

if "dropout_rate" in config:
    print(f"Dropout rate: {config['dropout_rate']}")

# Export trained model to ONNX format (for deployment)
export_onnx = False

# Number of access points (APs) / O-RUs
# Determines the number of NNs for this method. For B = 1, typically the CSI features of all O-RUs are stacked to "one" AP.
B = 1

def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    print(f"Warning detected!\nMessage: {message}\nCategory: {category.__name__}\nFile: {filename}\nLine: {lineno}")
    print()
warnings.showwarning = custom_warning_handler

# To use GPU in Pytorch to speed up computation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available(): print('GPU device count:', torch.cuda.device_count())
torch.manual_seed(0)
np.random.seed(0)

par0 = Parameter()

###############################################################################
# Data Loading
###############################################################################
# Load preprocessed CSI dataset and ground-truth UE positions
# The dataset should be generated using gen_dataset.py first

data_path = args.data_path
testbed_data = np.load(data_path+config["dataset"])
H_original = testbed_data['H']
UE_pos = testbed_data['pos']
timestamps = testbed_data['timestamps']

# # Sub-sampling
# H, UE_pos, timestamps = hp.sub_samp_by_rate(H_original,UE_pos,timestamps,rate=5)
# H, UE_pos, timestamps = hp.sub_samp_by_AP(H_original,UE_pos,timestamps,ap=0)
H = H_original.copy()

###############################################################################
# Feature Extraction and Probability Map Generation
###############################################################################
# Set plot axis limits for visualization
par0.set_pos_plot_axis_limits(UE_pos)

# Create spatial grid for probability maps (used for probability fusion)
hp.make_grid(UE_pos)

# Feature extraction parameters
PRELOAD = False  # Set to True to load pre-computed features/probability maps
norm_per_AP = False  # Normalize features per AP (False = normalize over all APs)

# Extract CSI features and generate probability maps for ground-truth positions
# This function:
# - Converts CSI to normalized feature vectors
# - Maps ground-truth positions to probability distributions over the spatial grid
feat_vec, prob_map, prob_map_var = feature_extraction_probability_maps(H,
                                                                       UE_pos,
                                                                       config['dataset'],
                                                                       data_path,
                                                                       prob_maps_preloaded=PRELOAD,
                                                                       feat_vec_preloaded=PRELOAD,
                                                                       norm_per_AP=norm_per_AP)

if config["dataset_val"] is not None:
    testbed_data_ = np.load(data_path+config["dataset_val"])
    H_ = testbed_data_['H']
    UE_pos_ = testbed_data_['pos']
    feat_vec_, prob_map_, prob_map_var_ = feature_extraction_probability_maps(H_.copy(),
                                                                       UE_pos_,
                                                                       config['dataset_val'],
                                                                       data_path,
                                                                       prob_maps_preloaded=PRELOAD,
                                                                       feat_vec_preloaded=PRELOAD,
                                                                       norm_per_AP=norm_per_AP)


print(f"Feature vector shape: {feat_vec.shape}")
per_AP_feat_size = H.shape[2]*H.shape[3]  # Number of features per AP (antennas x subcarriers)

###############################################################################
# Dataset Creation
###############################################################################
# Create training and test datasets with 80:20 random split
# The DataPropMaps class handles the split and prepares data for each AP
dataset = DataPropMaps(feat_vec, prob_map, UE_pos, B, test_to_all_ratio=0.2)

# Load additional validation dataset if specified (e.g., last 500 samples for generalization testing)
if config["dataset_val"] is not None:
    dataset_ = DataPropMaps(feat_vec_, prob_map_, UE_pos_, B, test_to_all_ratio=1.0)

###############################################################################
# Visualization: Test Set Locations
###############################################################################
# Plot the spatial distribution of test samples
par0.set_UE_info(dataset.UE_pos_testing)
ax = par0.plot_scenario(passive=False, dimensions='2d')
if config["dataset_val"] is not None:
    ax.scatter(dataset_.UE_pos_testing[:, 0], dataset_.UE_pos_testing[:, 1], s=5, c='b')
plt.savefig(results_dir+'Test_set.pdf', format='pdf')

###############################################################################
# Neural Network Training
###############################################################################
# Train one neural network per AP for probability map prediction
# The networks are trained independently and their outputs are fused later

losses = []  # Store training and validation losses for each AP

# Initialize estimated probability maps for all APs and test samples
# Initialized to ones (identity element for multiplication in probability conflation)
est_prob_maps = np.ones((B, dataset.test_samples, prob_map.shape[1]))
for ap in range(B):
    model = SupervisedModel(device,dataset.X_size,B,config)
    print(f"Training for AP {ap} ...")
    loss_per_epoch, val_loss_per_epoch = model.train(dataset,ap, noise_var=config["training_noise_var"]/per_AP_feat_size)
    print(loss_per_epoch)
    torch.save(model.network.state_dict(), results_dir + f"nn_weights_ap_{ap}.torch")
    losses.append([loss_per_epoch.cpu().detach().numpy(),val_loss_per_epoch.cpu().detach().numpy()])
    print(f"Testing for AP {ap}")
    Y_test = model.test(dataset,ap)
    est_prob_maps[ap,dataset.test_nonzero_indices[ap],:] = Y_test.cpu().detach().numpy() # take from torch to numpy and take it to the cpu instead of cuda

if config["dataset_val"] is not None:
    est_prob_maps_ = np.ones((B,dataset_.test_samples,prob_map_.shape[1]))
    for ap in range(B):
        Y_test_ = model.test(dataset_,ap)
        est_prob_maps_[ap,dataset_.test_nonzero_indices[ap],:] = Y_test_.cpu().detach().numpy()
    # p_bar_ = hp.prob_conflation(est_prob_maps_)
    x_hat_, var_hat_ = hp.prob_conflation_gaussian(est_prob_maps_, hp.G)

if export_onnx:
    example_inputs = (torch.t(dataset.X_test[0][:,:1].to(device)),)
    onnx_program = torch.onnx.export(model.network, example_inputs, dynamo=True)
    onnx_program.save(results_dir + "pobability_model.onnx")


# Save model weights
# torch.save(model.network.state_dict(), results_dir + "nn_weights.torch")

# # Load weights with :
# model = SupervisedModel(device,dataset.X_size,B,config)
# model.network.load_state_dict(torch.load(results_dir + "nn_weights.torch", weights_only=True))
# model.network.eval()

###############################################################################
# Probability Fusion and Position Estimation
###############################################################################
# Fuse probability maps from all APs using Gaussian probability conflation
# This combines the independent predictions from each AP into a single position estimate
x_hat, var_hat = hp.prob_conflation_gaussian(est_prob_maps, hp.G)

# Compute final position estimates based on fused prob. maps
# if config["temporal_conflation"] == True:
#     p_bar = np.array([p_bar[0:-3, :], p_bar[1:-2, :], p_bar[2:-1, :], p_bar[3:, :]])
#     x_hat, var_hat = hp.prob_conflation_gaussian(p_bar, hp.G)
#     # p_bar = hp.prob_conflation(p_bar)
#     skip_first_n_samples = 3

#     # p_bar = np.array([p_bar[0:-2, :], p_bar[1:-1, :], p_bar[2:, :]])
#     # x_hat = hp.prob_conflation_gaussian(p_bar, hp.G)
#     # # p_bar = hp.prob_conflation(p_bar)
#     # skip_first_n_samples = 2

#     if config["dataset_val"] is not None:
#         p_bar_ = np.array([p_bar_[0:-3, :], p_bar_[1:-2, :], p_bar_[2:-1, :], p_bar_[3:, :]])
#         x_hat_, var_hat_ = hp.prob_conflation_gaussian(p_bar_, hp.G)
#         var_hat_ = np.sum(var_hat_, axis=-1)
# else:
skip_first_n_samples = 0
    # x_hat = hp.G@(p_bar.T) # x_hat: 2 by N
    # if config["dataset_val"] is not None:
        # x_hat_ = hp.G@(p_bar_.T)
        # var_hat_ = np.sum(np.squeeze(np.matmul(np.square(hp.G), np.expand_dims(p_bar_, axis=2)), axis=-1) - np.square(x_hat_.T), axis=-1)

###############################################################################
# Evaluation: Positioning Error Computation
###############################################################################
# Compute 2D Euclidean positioning errors
if config["architecture"] == "single_NN":
    positioning_error = np.linalg.norm(Y_test - dataset.UE_pos_testing[skip_first_n_samples:, :], ord=2, axis=-1)
elif config["architecture"] == "probability_NN":
    positioning_error = np.linalg.norm(x_hat.T - dataset.UE_pos_testing[skip_first_n_samples:, :], ord=2, axis=-1)

# Generate CDF of positioning errors
value = np.linspace(0, 3, 100)
cdfdata = par0.cdfgen(value=value, input=positioning_error)

# Compute error statistics: mean, median, 95th percentile, maximum
print("Error stats:")
error_stats = np.array([np.mean(positioning_error), 
                        np.median(positioning_error), 
                        np.percentile(positioning_error, 95), 
                        np.max(positioning_error)])
print(error_stats)

np.savez(results_dir+'test_set_results.npz',
         pos_err=positioning_error,
         feat_test=dataset.feat_testing,
         pos_test=dataset.UE_pos_testing,
         x_hat=x_hat,
         prob_maps_var_test=prob_map_var[:,dataset.index_testing],
         error_cdfdata=cdfdata)

###############################################################################
# Additional Validation Dataset Analysis
###############################################################################
# Analyze performance on the additional validation set (e.g., last 500 samples)
# This provides insight into generalization capabilities
if config["dataset_val"] is not None:
    if config["architecture"] == "single_NN":
        positioning_error_ = np.linalg.norm(Y_test_ - dataset_.UE_pos_testing[skip_first_n_samples:, :], ord=2, axis=-1)
    elif config["architecture"] == "probability_NN":
        positioning_error_ = np.linalg.norm(x_hat_.T - dataset_.UE_pos_testing[skip_first_n_samples:, :], ord=2, axis=-1)

    value = np.linspace(0, 3, 100)
    cdfdata_ = par0.cdfgen(value=value, input=positioning_error_)
    print("Error stats (dataset_val):")
    error_stats_ = np.array([np.mean(positioning_error_), 
                              np.median(positioning_error_), 
                              np.percentile(positioning_error_, 95), 
                              np.max(positioning_error_)])
    print(error_stats_)

    # Analyze correlation between positioning error and distance to nearest training sample
    # This helps understand generalization performance
    min_dist_to_training_sample = np.min(np.linalg.norm(np.expand_dims(dataset_.UE_pos_testing[skip_first_n_samples:,:], axis=1) - np.expand_dims(dataset.UE_pos_training, axis=0), axis=2), axis=1, keepdims=False)

    min_idx_dist_to_training_sample = np.argmin(np.linalg.norm(np.expand_dims(dataset_.UE_pos_testing[skip_first_n_samples:,:], axis=1) - np.expand_dims(dataset.UE_pos_training, axis=0), axis=2), axis=1, keepdims=False)
    feat_dist_to_next_training_sample = np.linalg.norm(dataset_.X_test[0][:, skip_first_n_samples:] - dataset.X_train[0][:,min_idx_dist_to_training_sample], axis=0)

    # create 2d heatmap
    heatmap, xedges, yedges = np.histogram2d(min_dist_to_training_sample, positioning_error_, bins=100)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.figure()
    plt.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto')
    plt.xlabel("Distance to closest training sample")
    plt.ylabel("Positioning error")
    plt.show()
    plt.savefig(results_dir+'error_vs_dist_to_training_sample.pdf',format='pdf')

    plt.figure()
    plt.scatter(min_dist_to_training_sample, feat_dist_to_next_training_sample)
    plt.xlabel("Distance to closest training sample")
    plt.ylabel("Feature distance to closest training sample")
    plt.show()
    plt.savefig(results_dir+'feature_dist_vs_dist_to_training_sample.pdf',format='pdf')

    plt.figure()
    plt.scatter(feat_dist_to_next_training_sample, positioning_error_)
    plt.xlabel("Feature distance to closest training sample")
    plt.ylabel("Positioning error")
    plt.show()
    plt.savefig(results_dir+'error_vs_feature_dist_to_training_sample.pdf',format='pdf')

    plt.figure()
    plt.scatter(min_dist_to_training_sample, positioning_error_)
    plt.xlabel("Distance to closest training sample")
    plt.ylabel("Positioning error")
    plt.show()
    plt.savefig(results_dir+'error_vs_dist_to_training_sample1.pdf',format='pdf')

    plt.figure()
    plt.scatter(positioning_error_, np.sum(var_hat_, axis=0))
    plt.ylabel("Estimator variance")
    plt.xlabel("Positioning error")
    plt.show()
    plt.savefig(results_dir+'variance_vs_error.pdf',format='pdf')

    np.savez(results_dir+'test_set_val_results.npz',
         pos_err=positioning_error_,
         pos_err_var=var_hat_,
         dist_to_training_sample=min_dist_to_training_sample,
         feat_test=dataset_.feat_testing,
         pos_test=dataset_.UE_pos_testing,
         x_hat=x_hat_,
         prob_maps_var_test=prob_map_var_[:,dataset_.index_testing],
         error_cdfdata=cdfdata_)

###############################################################################
# Visualization: Results and Plots
###############################################################################
# Plot CDF of positioning errors
plt.figure()
plt.plot(value, cdfdata)
if config["dataset_val"] is not None:
    plt.plot(value,cdfdata_, ":")
plt.grid()
plt.xlabel('x[m]')
plt.ylabel(r'Pr[X $\leq$ x]')
plt.show()
plt.savefig(results_dir+'CDF_positioning_error.pdf', format='pdf')

# Plot estimated positions vs. ground truth
par0.set_UE_info(x_hat.T, color_map=par0.color_map[skip_first_n_samples:, :]) # Set the color map to the colormap of the testset
ax = par0.plot_scenario(passive=False, dimensions='2d', error_stats=error_stats)
if config["dataset_val"] is not None:
    ax.scatter(x_hat_.T[:, 0], x_hat_.T[:, 1], s=5, c='b')
    # var_hat_sum = np.sum(var_hat_, axis=0)
    # ax.scatter(x_hat_.T[var_hat_sum<0.1, 0], x_hat_.T[var_hat_sum<0.1, 1], s=5, c='darkblue')
# plt.savefig(results_dir+'Supervised_learning_positioning_plot.pdf',format='pdf')
plt.savefig(results_dir + 'Estimator_output.pdf', format='pdf')

# Plot training and validation loss curves
min_value = np.min(np.array(losses))
max_value = np.max(np.array(losses))
if B>1:
    fig, axs = plt.subplots(2,int(np.ceil(B/2)), figsize=(12, 10))
    for ap in range(B):
        axs[int(np.floor(ap/2)),np.mod(ap,2)].plot(range(model.epochs),losses[ap][0],label='Training loss')
        axs[int(np.floor(ap/2)),np.mod(ap,2)].plot(range(model.epochs),losses[ap][1],label='Validation loss')
        axs[int(np.floor(ap/2)),np.mod(ap,2)].set_ylim(np.floor(min_value*100)/100,np.ceil(max_value*100)/100)
        axs[int(np.floor(ap/2)),np.mod(ap,2)].grid()
        axs[int(np.floor(ap/2)),np.mod(ap,2)].legend()
        axs[int(np.floor(ap/2)),np.mod(ap,2)].set_xlabel('epochs')
        axs[int(np.floor(ap/2)),np.mod(ap,2)].set_ylabel('MSE')
        axs[int(np.floor(ap/2)),np.mod(ap,2)].set_title(f'AP {ap}')
else:
    ap=0
    plt.figure()
    plt.plot(range(model.epochs),losses[ap][0],label='Training loss')
    plt.plot(range(model.epochs),losses[ap][1],label='Validation loss')
    plt.ylim(np.floor(min_value*100)/100,np.ceil(max_value*100)/100)
    plt.grid()
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.title(f'AP 0')
plt.show()
plt.savefig(results_dir+'MSE_vs_epochs.pdf',format='pdf')

