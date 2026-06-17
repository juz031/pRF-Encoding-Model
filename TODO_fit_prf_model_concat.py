# TODO: return the the index of the best pRF, use this index to get the parameters of the best prf
# Save weights vector for channels

import sys
import os
import time
import numpy as np
import h5py
import pandas as pd
import scipy
import torch
import matplotlib
import matplotlib.pyplot as plt
import warnings
import pickle
import argparse, gc
from skimage.transform import resize
import random
import time
from tqdm import tqdm
import random
import clip

import torch.multiprocessing as mp
import os
import socket
from torch.cuda.amp import GradScaler
from contextlib import closing
import torch.distributed as dist
# import model
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import math
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

sys.path.insert(0,'/lab_data/hendersonlab/code/utils/')
import stats_utils

# sys.path.insert(0,'/home/junruz/BrainDiVE')
# from encoder_training_CNN import neural_loader

import prf_utils
import model_fitting_utils


def load_nsd_data(data_folder, labels_folder, ss):
    """
    Return:
    - voxel_data: array of shape [num_voxels, num_features] (e.g.. S1 [10000, 19738]), the fmri data for each voxel
    - good_values: array of shape [num_voxels], the indices of the voxels that have valid fmri data
    """

    # load the preprocessed data files, made using code in nsd_preproc/code 
    
    # load info about images on each trial
    info_fn = os.path.join(labels_folder, 'S%d_image_info.csv'%(ss))
    print(info_fn)
    info = pd.read_csv(info_fn)

    image_order = np.array(info['unique_ims'])
    n_reps = np.array(info['n_reps'])

    # load fmri data
    data_filename = os.path.join(data_folder, 'S%d_betas_avg_bigmask.hdf5'%ss)
    print(data_filename)

    t = time.time()
    with h5py.File(data_filename, 'r') as data_set:
        values = np.copy(data_set['/betas'])
        data_set.close() 
    elapsed = time.time() - t
    print('Took %.5f seconds to load file'%elapsed)
    # data is organized as:
    # [images x voxels]

    # Some of these values may be nans, only for some subjects
    # this is for subjects who didn't complete all 40 sessions of NSD experiment.
    # make sure we remove the nans now.
    good_values = ~np.isnan(values[:,0])
    print(values.shape)
    print(np.sum(~good_values))

    # check that nans are exactly where we expect
    # nans happen when n_reps=0
    assert(np.all(good_values[n_reps>0]))
    assert(np.all(~good_values[n_reps==0]))

    voxel_data = values[good_values,:]
    print(voxel_data.shape)
    
    return voxel_data, good_values


def load_nsd_splits(stim_folder, ss, good_values):

    # I computed the data splits ahead of time, so that the random seed is reproducible
    # Always holding out 1000 shared images as val. 
    # Then a random 10% as the "nested held-out" set that is used to choose ridge parameters.
    splits_filename = os.path.join(stim_folder, 'Image_data_partitions.npy')
    splits = np.load(splits_filename, allow_pickle=True).item()

    si = ss-1
    trn_inds = splits['is_trn'][good_values,si]
    val_inds = splits['is_val'][good_values,si]
    nest_inds = splits['is_holdout'][good_values,si]

    return trn_inds, val_inds, nest_inds


def model_fitting(voxel_data, good_values, features_folder, save_fits_folder, args):
    """
    Return:
    - best_prf_idx: list of length of num_voxels, the index of the best pRF for each voxel
    - best_lambda: list of length of num_voxels, the best lambda for each voxel for the best pRF
    - best_r2: list of length of num_voxels, the best r2 for each voxel for the best pRF
    - best_weights: array of length of [num_voxels, num_features + 1], the best weights for each voxel for the best pRF, with the intercept
    """
    num_prfs = len(os.listdir(features_folder))

    for voxel_idx in range(voxel_data.shape[0]): # TODO: no need for a for loop for the voxel 
        for prf_idx in range(num_prfs): # ~1450
            features_path = os.path.join(features_folder, f'features_prf_{prf_idx}.npy')
            features = np.load(features_path)

            n_lambdas = 20
            small_value = 0.0001
            lambdas = np.logspace(np.log(small_value),np.log(10**10+small_value),n_lambdas, \
                                dtype=np.float32, base=np.e) - small_value
            
            for lambda_idx in range(n_lambdas): # TODO: no need for a for loop for the lambda if using solve-ridge
                lambda_ = lambdas[lambda_idx]
                weights = model_fitting_utils.ridge_regression(features, voxel_data[voxel_idx,:], lambda_) # model_fitting_utils.py: solve_ridge
                r2 = model_fitting_utils.compute_r2(voxel_data[voxel_idx,:], weights)
                if r2 > best_r2:
                    best_r2 = r2
                    best_weights = weights
                    best_prf_idx = prf_idx
                    best_lambda = lambda_

        ## TODO: for all layers, use the same prf parameters




def main():
    # Set some paths: where the preprocessed NSD files live
    nsd_path = '/lab_data/hendersonlab/datasets/nsd_preproc'
    data_folder = os.path.join(nsd_path, 'data')
    labels_folder = os.path.join(nsd_path, 'labels')
    stim_folder = os.path.join(nsd_path, 'stimuli')
    rois_folder = os.path.join(nsd_path, 'rois')

    n_pix = 224

    # Create namespace and set attributes
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', nargs='+', default=[1], type=int)
    parser.add_argument('--neural_activity_path', type=str, default=os.path.join(data_folder, 'S{}_betas_avg_bigmask_dict.hdf5'))
    parser.add_argument('--image_path', type=str, default=os.path.join(stim_folder, 'S{}_stimuli_%d_dict.hdf5'%n_pix))
    parser.add_argument('--roi_path', type=str, default=os.path.join(rois_folder, 'S{}_voxel_roi_info.npy'))
    parser.add_argument('--stim_keys_path', type=str, default=os.path.join(stim_folder, "all_keys.pkl"))
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_name', type=str, default="RN50")
    parser.add_argument('--layer_name', type=str, default="avgpool")
    parser.add_argument('--roi', type=str, default="V1")
    parser.add_argument('--save_root', type=str, default="/user_data/junruz/prf_features")
    args = parser.parse_args()

    # where my preprocessed NSD files live
    nsd_path = '/lab_data/hendersonlab/datasets/nsd_preproc'
    data_folder = os.path.join(nsd_path, 'data')
    labels_folder = os.path.join(nsd_path, 'labels')
    stim_folder = os.path.join(nsd_path, 'stimuli')
    rois_folder = os.path.join(nsd_path, 'rois')

    # where the pre-computed features are placed
    # different models are organized in sub-folders in here
    features_folder = '/user_data/junruz/prf_features'
    print(features_folder)

    # where you want to save the model fits.
    save_fits_folder = '/user_data/junruz/prf_models'
    ss = args.subject_id[0]

    voxel_data, good_values = load_nsd_data(data_folder, labels_folder, ss)
    train_ids, val_ids, nest_ids = load_nsd_splits(stim_folder, ss, good_values)

    # split the fmri data, 3 independent sets
    train_data = voxel_data[train_ids,:]
    nest_data = voxel_data[nest_ids,:]
    val_data = voxel_data[val_ids,:]
    pass


if __name__ == "__main__":
    main()