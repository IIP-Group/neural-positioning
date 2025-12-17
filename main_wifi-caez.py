"""
Main script for CSI-based neural positioning using probability maps and CAEZ-WIFI datasets.

This script implements supervised learning for UE positioning based on CSI features
from a Wi-Fi testbed. The method uses probability maps as described in [1] to train
a neural network that outputs probability distributions over a spatial grid. These
probability maps are then fused to estimate absolute UE positions.

Workflow:
1. Load preprocessed CSI data and ground-truth UE positions
2. Generate probability maps for ground-truth positions
3. Extract CSI features and create training/test datasets
4. Train neural network(s) for each access point (AP)
5. Evaluate on test set(s) and compute positioning errors
6. Generate plots and save results

The script supports the CAEZ-WIFI-INDOOR-LSHAPE dataset and future possible CAEZ-WIFI datsets.
Configuration is done via the 'config' dictionary at the top of the file.

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
import datetime
import json

###############################################################################
# Command-line Arguments
###############################################################################
parser = argparse.ArgumentParser(description='CSI-based neural positioning using probability maps and CAEZ-WIFI datasets')
parser.add_argument('--data_path', type=str, default='/home/user/caez-wifi-datasets/',
                    help='Path to the directory containing the generated dataset (default: /home/user/caez-wifi-datasets/)')
parser.add_argument('--results_path', type=str, default='/home/user/caez-wifi-results/',
                    help='Path to the directory containing result files (default: /home/user/caez-wifi-results/)')
args = parser.parse_args()

config = {
        "learning_rate": 1e-4,
        "architecture": "probability_NN", # "probability_NN" is the default architecture for this method.
        "dataset": "caez-wifi-indoor-lshape.npz",
        "epochs": 20,
        "CSI_type": "H",
        "dropout_rate": 0.,
        "conflation_method": "gaussian"
            }
results_dir = args.results_path+'neural_positioning_'+config["dataset"][:-4]+'_'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'/'

# Create results directory if it does not exist and save config
os.makedirs(results_dir, exist_ok=True)
config_path = os.path.join(results_dir, "config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=4)

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
# The dataset should be generated using make_wifi_dataset.py first

data_path = args.data_path
testbed_data = np.load(data_path+config["dataset"])
H_original = testbed_data['H']
UE_pos = testbed_data['pos']
timestamps = testbed_data['timestamps']
if 'ap_pos' in testbed_data:
    ap_pos = testbed_data['ap_pos']
else:
    ap_pos = None

# Sub-sampling
# H, UE_pos, timestamps = hp.sub_samp_by_AP(H_original,UE_pos,timestamps,ap=0)
# H, UE_pos, timestamps = hp.sub_samp_by_rate(H,UE_pos,timestamps,rate=5)
H = H_original.copy()

per_AP_feat_size = H.shape[2]*H.shape[3]
B = H.shape[1] # number of APs
N = H.shape[0] # number of dataset samples

###############################################################################
# Feature Extraction and Probability Map Generation
###############################################################################

# Set plot axis limits for visualization
par0.set_pos_plot_axis_limits(UE_pos)

# Create spatial grid for probability maps (used for probability fusion)
# Set non_square_area to true; fits to caez-wifi-indoor-lshape 
hp.make_grid(UE_pos,non_square_area=True)

# Feature extraction parameters
PRELOAD = True  # Set to True to load pre-computed features/probability maps
norm_per_AP = False  # Normalize features per AP (False = normalize over all APs)
print(f"Feature extraction parameters: PRELOAD={PRELOAD}, norm_per_AP={norm_per_AP}")

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
print(f"Feature vector shape: {feat_vec.shape}")

###############################################################################
# Dataset Creation
###############################################################################
# Create training and test datasets with 80:20 random split
# The DataPropMaps class handles the split and prepares data for each AP
dataset = DataPropMaps(feat_vec,prob_map,UE_pos,B)

# Visualization: plotting only the testing set
par0.set_UE_info(dataset.UE_pos_testing)
par0.plot_scenario(passive=False,dimensions='2d',ap_pos=ap_pos)
plt.savefig(results_dir+'Test_set.png',format='png',dpi=400)

###############################################################################
# Neural Network Training
###############################################################################
# Train one neural network per AP for probability map prediction
# The networks are trained independently and their outputs are fused later

losses = [] # Store training and validation losses for each AP

# For testing: array of estimated prob. maps for each AP and each test sample.
# Initialized to ones (identity element for multiplication in probability conflation)
est_prob_maps = np.ones((B,dataset.test_samples,prob_map.shape[1]))
for ap in range(B):
    model = SupervisedModel(device,dataset.X_size,B,config,prob_map_size=hp.G.shape[1])
    print(f"Training for AP {ap} ...")
    loss_per_epoch, val_loss_per_epoch = model.train(dataset,ap)
    print(loss_per_epoch)
    losses.append([loss_per_epoch.cpu().detach().numpy(),val_loss_per_epoch.cpu().detach().numpy()])
    print(f"Testing for AP {ap}")
    Y_test = model.test(dataset,ap)
    est_prob_maps[ap,dataset.test_nonzero_indices[ap],:] = Y_test.cpu().detach().numpy() # take from torch to numpy and take it to the cpu instead of cuda

# Save model weights
# torch.save(model.network.state_dict(), results_dir + "nn_weights.torch")

# # Load weights with :
# model = SupervisedModel(device,dataset.X_size,B,config)
# model.network.load_state_dict(torch.load(results_dir + "nn_weights.torch", weights_only=True))
# model.network.eval()

###############################################################################
# Probability Fusion and Position Estimation
###############################################################################
# Fuse probability maps from all APs using a conflation method.
# This combines the independent predictions from each AP into a single position estimate

if config["conflation_method"] == "gaussian":
    x_hat, var_hat = hp.prob_conflation_gaussian(est_prob_maps, hp.G)
elif config["conflation_method"] == "probability":
    p_bar = hp.prob_conflation(est_prob_maps).T
    x_hat = hp.G@p_bar # x_hat: 2 by N
else:
    raise ValueError(f"Invalid conflation method: {config['conflation_method']}")

###############################################################################
# Evaluation: Positioning Error Computation
###############################################################################
# Compute 2D Euclidean positioning errors
positioning_error = np.linalg.norm(x_hat.T-dataset.UE_pos_testing,ord=2,axis=-1) # ord=2 for 2 norm and axis=-1 from the last axis

# Generate CDF of positioning errors
value = np.linspace(0,3,100)
cdfdata = par0.cdfgen(value = value,input = positioning_error)
error_stats = np.array([np.mean(positioning_error),np.median(positioning_error),np.percentile(positioning_error,95),np.max(positioning_error)])

# Plot the CDF of positioning error
plt.figure()
plt.plot(value,cdfdata)
plt.grid()
plt.xlabel('x[m]')
plt.ylabel('Pr[X $\leq$ x]')
plt.show()
plt.savefig(results_dir+'CDF_positioning_error.png',format='png',dpi=400)

print(f"Error stats:\n {error_stats}")

# Plot estimates
par0.set_UE_info(x_hat.T,color_map=par0.color_map) # Set the color map to the colormap of the testset
par0.plot_scenario(passive=False,dimensions='2d',error_stats=error_stats)
plt.savefig(results_dir+'Estimator_output.png',format='png',dpi=400)

# Plot the loss
min_value = np.min(np.array(losses))
max_value = np.max(np.array(losses))
fig, axs = plt.subplots(2,2, figsize=(12, 10))
for ap in range(B):
    axs[int(np.floor(ap/2)),np.mod(ap,2)].plot(range(model.epochs),losses[ap][0],label='Training loss')
    axs[int(np.floor(ap/2)),np.mod(ap,2)].plot(range(model.epochs),losses[ap][1],label='Validation loss')
    axs[int(np.floor(ap/2)),np.mod(ap,2)].set_ylim(np.floor(min_value*100)/100,np.ceil(max_value*100)/100)
    axs[int(np.floor(ap/2)),np.mod(ap,2)].grid()
    axs[int(np.floor(ap/2)),np.mod(ap,2)].legend()
    axs[int(np.floor(ap/2)),np.mod(ap,2)].set_xlabel('epochs')
    axs[int(np.floor(ap/2)),np.mod(ap,2)].set_ylabel('MSE')
    axs[int(np.floor(ap/2)),np.mod(ap,2)].set_title(f'AP {ap}')
plt.savefig(results_dir+'MSE_vs_epochs.png',format='png')

np.savez(results_dir+'test_set_results.npz',
         pos_err=positioning_error,
         feat_test=dataset.feat_testing,
         pos_test=dataset.UE_pos_testing,
         x_hat=x_hat,
         prob_maps_var_test=prob_map_var[:,dataset.index_testing],
         error_cdfdata=cdfdata)