import sys
import os
import time
import numpy as np
import h5py
import pandas as pd
import torch
import argparse
import time
from tqdm import tqdm
import random
import pickle
import os
import numpy as np

import prf_utils
import model_fitting_utils


nsd_path = '/lab_data/hendersonlab/datasets/nsd_preproc'
data_folder = os.path.join(nsd_path, 'data')
labels_folder = os.path.join(nsd_path, 'labels')
stim_folder = os.path.join(nsd_path, 'stimuli')
rois_folder = os.path.join(nsd_path, 'rois')

    # Create namespace and set attributes
parser = argparse.ArgumentParser()
parser.add_argument('--subject_id', nargs='+', default=[1], type=int)
parser.add_argument('--neural_activity_path', type=str, default=os.path.join(data_folder, 'S{}_betas_avg_bigmask_dict.hdf5'))
parser.add_argument('--roi_path', type=str, default=os.path.join(rois_folder, 'S{}_voxel_roi_info.npy'))
parser.add_argument('--stim_keys_path', type=str, default=os.path.join(stim_folder, "all_keys.pkl"))
parser.add_argument('--model_name', type=str, default="RN50")
parser.add_argument('--layer_name', type=str, default="layer2")
args = parser.parse_args()

ss = 1

clip_rn50_layer_size = {
    'relu1': [32, 112, 112],
    'relu2': [32, 112, 112],
    'relu3': [64, 112, 112],
    'avgpool': [64, 56, 56],
    'layer1': [256, 56, 56],
    'layer2': [512, 28, 28],
    'layer3': [1024, 14, 14],
    'layer4': [2048, 7, 7]
}

n_channels = clip_rn50_layer_size[args.layer_name][0]
print(f"Save for {args.layer_name}, Number of channels: {n_channels}")

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
    # print(values.shape)
    # print(np.sum(~good_values))

    # check that nans are exactly where we expect
    # nans happen when n_reps=0
    assert(np.all(good_values[n_reps>0]))
    assert(np.all(~good_values[n_reps==0]))

    voxel_data = values[good_values,:]
    # print(voxel_data.shape)
    
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

# where the pre-computed features are placed
# different models are organized in sub-folders in here

############ for debug ##########################
# features_folder = '/lab_data/hendersonlab/features/NSD_prfmodel/gabor_12ori_8sf_prf_default-log-polar/NSD_S1'
# features_folder = os.path.join(features_root, f"S{ss}", args.model_name, args.layer_name)
##############################

features_root = '/user_data/junruz/prf_features'
features_folder = os.path.join(features_root, f"S{ss}", args.model_name, args.layer_name)
print(f"Loading features from: {features_folder}")

voxel_data, good_values = load_nsd_data(data_folder, labels_folder, ss)
train_ids, val_ids, nest_ids = load_nsd_splits(stim_folder, ss, good_values)

n_prfs = 1450
mean = np.zeros((n_prfs, n_channels))
std = np.zeros((n_prfs, n_channels))
for prf_idx in tqdm(range(n_prfs)): # ~1450
    features_path = os.path.join(features_folder, f'features_prf_{prf_idx}.npy')
    f = np.load(features_path)

    f_trn = f[train_ids,:]
    f_val = f[val_ids,:]
    f_nest = f[nest_ids,:]

    # I'm computing the normalization parameters (mean and std) on my training data only
    # (plus the nested held-out partition), but not the val set.
    # this helps reduce leakage of data between train and val partitions.
    # then apply those same normalization parameters to the val set too.
    f_concat = np.concatenate([f_trn, f_nest], axis=0)
    # f_concat = f_trn
    
    features_m = np.mean(f_concat, axis=0, keepdims=True) #[:trn_size]
    # print(features_m[0,0:10])
    features_s = np.std(f_concat, axis=0, keepdims=True) + 1e-12
    mean[prf_idx, :] = features_m
    std[prf_idx, :] = features_s

np.save(os.path.join(features_folder, 'channel_mean.npy'), mean)
np.save(os.path.join(features_folder, 'channel_std.npy'), std)