import numpy as np
import torch
from torch.utils.data import Dataset

from param_config import *
from feature_extraction import *

class Data(Dataset):    
    # Constructor
    def __init__(self, feat_vec, UE_pos):
        # Create a Dataset object
        # Separate between training and test set
        self.total_samples = feat_vec.shape[1]
        self.test_samples = int(self.total_samples/5)
        self.training_samples = self.total_samples-self.test_samples
        tmp = np.random.choice(self.total_samples,self.test_samples,replace=False)

        self.index_testing = np.sort(tmp)
        self.feat_testing = feat_vec[:,self.index_testing]
        self.UE_pos_testing = UE_pos[self.index_testing,:]
        
        self.index_training = np.array([item for item in range(self.total_samples) if item not in self.index_testing])
        
        self.feat_training = feat_vec[:,self.index_training]
                
        self.UE_pos_training = UE_pos[self.index_training,:]

        self.X_train = torch.Tensor(self.feat_training)
        self.X_size = (self.feat_training).shape[0]
        self.Y_train = torch.Tensor(self.UE_pos_training)
        self.X_test = torch.Tensor(self.feat_testing)
        self.Y_test = torch.Tensor(self.UE_pos_testing)
        
    # Getter
    def __getitem__(self, index):    
        return self.X_train[index], self.Y_train[index],self.X_test[index], self.Y_test[index]
    
    def __len__(self):
        return self.len
    
class DataPropMaps(Dataset):    
    # Constructor
    def __init__(self, feat_vec, prob_maps, UE_pos, B, test_to_all_ratio=1.0/5):
        # Create a Dataset object
        # Separate between training and test set
        self.total_samples = feat_vec.shape[1]
        self.test_samples = int(self.total_samples*test_to_all_ratio)
        self.training_samples = self.total_samples-self.test_samples
        tmp = np.random.choice(self.total_samples,self.test_samples,replace=False)

        self.index_testing = np.sort(tmp)
        self.index_training = np.array([item for item in range(self.total_samples) if item not in self.index_testing])
        if self.index_training.size > 0:
            self.feat_training = feat_vec[:,self.index_training]
            self.prob_maps_training = prob_maps[self.index_training,:]
            self.UE_pos_training = UE_pos[self.index_training, :]
        else:
            self.feat_training = np.array([[]])
            self.prob_maps_training = np.array([[]])
        self.feat_testing = feat_vec[:,self.index_testing]
        self.prob_maps_testing = prob_maps[self.index_testing,:]
        self.UE_pos_testing = UE_pos[self.index_testing,:]
        
        if self.index_training.size > 0:
            self.X_size = (self.feat_training).shape[0]
        else:
            self.X_size = (self.feat_testing).shape[0]
        per_AP_feat_size = int(self.X_size/B)
        # Store X and Y tensors for each AP separately; append to these lists.
        self.X_train, self.X_test, self.Y_train, self.Y_test, self.test_nonzero_indices = [], [], [], [], []
        for ap in range(B):
            # Find zeros in the first feature element of AP "ap"; training first
            if self.index_training.size > 0:
                non_zero_indices = np.nonzero(self.feat_training[ap*per_AP_feat_size,:])[0]
                features = self.feat_training[ap*per_AP_feat_size:(ap+1)*per_AP_feat_size,non_zero_indices]
            else:
                non_zero_indices = np.nonzero(self.feat_training[0,:])[0]
                features = self.feat_training[0:per_AP_feat_size,non_zero_indices]
            self.X_train.append(torch.Tensor(features))
            self.Y_train.append(torch.Tensor(self.prob_maps_training[non_zero_indices,:]))

            # Find non-zero indices for test samples  
            non_zero_indices = np.nonzero(self.feat_testing[ap*per_AP_feat_size,:])[0]
            self.test_nonzero_indices.append(non_zero_indices)
            features = self.feat_testing[ap*per_AP_feat_size:(ap+1)*per_AP_feat_size,non_zero_indices]
            self.X_test.append(torch.Tensor(features))
            self.Y_test.append(torch.Tensor(self.prob_maps_testing[non_zero_indices,:]))
        
    # Getter
    def __getitem__(self, index):    
        return self.X_train[index], self.Y_train[index],self.X_test[index], self.Y_test[index]
    
    def __len__(self):
        return self.len
    
class TestOnlyData(Dataset):
    
    # Constructor
    def __init__(self, feat_vec, UE_pos):

        self.feat_testing = feat_vec[:]
        self.UE_pos_testing = UE_pos[:]

        self.X_size = (self.feat_testing).shape[0]
        self.X_test = torch.Tensor(self.feat_testing)
        
    # Getter
    def __getitem__(self, index):    
        return self.X_test[index], self.Y_test[index]
    
    def __len__(self):
        return self.len