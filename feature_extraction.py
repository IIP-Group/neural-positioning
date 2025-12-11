import numpy as np
from param_config import *
import helper as hp

from tqdm import tqdm

def feature_extraction_simple(H):
        
        print(f"H shape: {H.shape}")        
        if len(H.shape) == 3:
                N,A,W = H.shape
                feature_vector= np.zeros((A*W,N))
                for item in range(N):
                        print(f"Preparing features ... {int((item/N)*100)} %")
                        if len(np.where(np.isnan(H[item,:,:]))[0]) != 0:
                                print("Error: NaN encountered in dataset. Aborting.")
                                exit()
                        feature = np.abs(H[item,:,:]/np.linalg.norm(H[item,:,:],ord="fro"))

                        feature_vector[:,item] = feature.reshape(A*W,)
                non_zero_indices = np.nonzero(feature_vector[0,:])[0] # Find the non-zero entries
                feature_vector = feature_vector[:,non_zero_indices]

        else:
                N,B,A,W = H.shape
                feature_vector= np.zeros((B*A*W,N))
                for item in range(N):
                        print(f"Preparing features ... {int((item/N)*100)} %")
                        if len(np.where(np.isnan(H[item,:,:,:]))[0]) != 0:
                                print("Error: NaN encountered in dataset. Aborting.")
                                exit()
                        feature = np.abs(H[item,:,:,:]/np.linalg.norm(H[item,:,:,:].reshape(B,A*W),ord="fro"))             

                        feature_vector[:,item] = feature.reshape(B*A*W,)

        return feature_vector

def feature_extraction_probability_maps(H,
                                        UE_pos,
                                        dataset_name,
                                        data_path,
                                        prob_maps_preloaded=False,
                                        feat_vec_preloaded=False,
                                        norm_per_AP=False):

        N,B,A,W = H.shape

        if prob_maps_preloaded:
                print("Loading pre-computed prob. maps ...")
                data = np.load(data_path+'features_probMaps_'+dataset_name[8:-4]+'.npz')
                prob_map = data["prob_map"]
                prob_map_var = data["prob_map_var"]
                K = 1 # dummy value
                if feat_vec_preloaded:
                        print("Loading pre-computed CSI features ...")
                        feature_vector = data["feat_vec"]
                        prob_map = np.abs(np.round(prob_map.T,5))
                        return feature_vector, prob_map, prob_map_var
        else:
                K_sq = hp.G.shape[1] # number of grid points
                K = np.sqrt(K_sq).astype(int)
                prob_map = np.zeros((K_sq,N))
                prob_map_var = np.zeros((1,N))

        if norm_per_AP:
                feat_vec_size = B*2*A*W
                max_pwr = np.max(np.linalg.norm(H,ord="fro",axis=(2,3)),axis=(0,1))
        else:
                feat_vec_size = B*A*W
                max_pwr = 1 # dummy value
        feature_vector = np.zeros((feat_vec_size,N))
        t = tqdm(total=N)
        print(f"Preparing features ... ")
        for item in range(N):
                # CSI to feature
                t.update()  
                if len(np.where(np.isnan(H[item,:,:,:]))[0]) != 0:
                        print("Error: NaN encountered in dataset. Aborting.")
                        exit()
                feature_vector[:,item] = make_csi_feature(H[item,:,:,:],max_pwr,norm_per_AP=norm_per_AP)
                
                if not prob_maps_preloaded:
                        # UE_pos to probability vector
                        prob_map[:,item], prob_map_var[:,item] = hp.learn_ref_probability_map(UE_pos[item,:].reshape(2,1),K)

        if not prob_maps_preloaded:
                np.savez(data_path+'features_probMaps_'+dataset_name[8:-4]+'.npz', feat_vec=feature_vector, prob_map=prob_map, prob_map_var=prob_map_var)
                print('features and probability maps saved.')
        
        # transpose prob maps; training needs this
        # round all near-zero values to zero and make positive
        prob_map = np.abs(np.round(prob_map.T,5))

        return feature_vector, prob_map, prob_map_var

def make_csi_feature(H_n,max_pwr,norm_per_AP=False):
        # H_n: one CSI datapoint; 3-D: APs x antenna x subcarrier
        B,A,W = H_n.shape # APs x antenna x subcarrier
        
        if norm_per_AP:
                feature = np.zeros((B*A*W,2))
                power = np.linalg.norm(H_n,ord="fro",axis=(1,2))
                temp = np.abs(H_n/power.reshape(B,1,1))
                feature[:,0] = np.nan_to_num(temp).reshape(B*A*W,) # zero CSI from APs results in NaN entries; replace by zeros.
                # Add average Rx power per AP as feature
                for ap in range(B):
                        if power[ap] != 0:
                                feature[ap*A*W:(ap+1)*A*W,1] = pos_encoding(int(A*W/2),power[ap],max_pwr)
        else: # Normalize over all APs
                feature = np.abs(H_n/np.linalg.norm(H_n.reshape(B,A*W),ord="fro"))
        
        return feature.reshape(-1,1).flatten()

def pos_encoding(L,x_,x_max):
        # Enode a low dimensional input into a higher dimension
        x = x_/x_max # normalize x over all possible x
        temp = np.array([np.sin(2**np.r_[0:L]*np.pi*x),np.cos(2**np.r_[0:L]*np.pi*x)]).T
        return temp.reshape(2*L,)

def feature_extraction_simple_ifft(H):
        N,B,A,W = H.shape
        print(f"H shape: {H.shape}")
        feature_vector= np.zeros((B*A*W,N))
        nan_counter = 0
        for item in range(N):
                if len(np.where(np.isnan(H[item,:,:,:]))[0]) != 0:
                        nan_counter += 1
                        H[item,:,:,:] = np.nan_to_num(H[item,:,:,:], nan=0.0)

                # IFFT where a contribution from an AP is not zero
                feature = np.zeros((B,A*W,1))
                non_zero_indices = np.nonzero(H[item,:,0,0])[0]
                for idx in non_zero_indices:
                        feature_ = np.zeros((4,52),dtype=np.complex64)
                        for j in range(4):
                                feature_[j,:] = np.fft.ifft(H[item,idx,j,:])
                        feature[idx,:] = np.abs(feature_.reshape(-1,1))
                feature_vector[:,item] = feature.reshape(-1,1).flatten()

        print(f"Numer of NaNs: {nan_counter}")
        return feature_vector