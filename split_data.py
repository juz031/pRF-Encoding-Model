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



# sys.path.insert(0,'/home/junruz/BrainDiVE')
# from encoder_training_CNN import neural_loader

import prf_utils
import model_fitting_utils


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


def split_data(trn_inds, val_inds, nest_inds, ratio=0.5):
    # split the data into train, val, and nest sets
    # Split each input array into two according to the specified ratio.
    train_ids = np.argwhere(trn_inds == True).squeeze().tolist()
    val_ids = np.argwhere(val_inds == True).squeeze().tolist()
    nest_ids = np.argwhere(nest_inds == True).squeeze().tolist()

    def split_indices(indices, ratio):
        indices = np.array(indices)
        n_split = int(len(indices) * ratio)
        shuffled = np.random.permutation(indices)
        return shuffled[:n_split].tolist(), shuffled[n_split:].tolist()

    trn1, trn2 = split_indices(train_ids, ratio)
    val1, val2 = split_indices(val_ids, ratio)
    nest1, nest2 = split_indices(nest_ids, ratio)

    return {'ratio': ratio}, {'train': trn1, 'val': val1, 'nest': nest1}, {'train': trn2, 'val': val2, 'nest': nest2}


def main():
    # Set some paths: where the preprocessed NSD files live
    nsd_path = '/lab_data/hendersonlab/datasets/nsd_preproc'
    data_folder = os.path.join(nsd_path, 'data')
    labels_folder = os.path.join(nsd_path, 'labels')
    stim_folder = os.path.join(nsd_path, 'stimuli')
    rois_folder = os.path.join(nsd_path, 'rois')

    # where my preprocessed NSD files live
    nsd_path = '/lab_data/hendersonlab/datasets/nsd_preproc'
    data_folder = os.path.join(nsd_path, 'data')
    labels_folder = os.path.join(nsd_path, 'labels')
    stim_folder = os.path.join(nsd_path, 'stimuli')
    rois_folder = os.path.join(nsd_path, 'rois')

    # Create namespace and set attributes
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', default=1, type=int)
    parser.add_argument('--ratio', default=0.5, type=float)
    parser.add_argument('--output_dir', default='/user_data/junruz/prf_models/split_1', type=str)
    args = parser.parse_args()

    ss = args.subject_id

    voxel_data, good_values = load_nsd_data(data_folder, labels_folder, ss)
    train_ids, val_ids, nest_ids = load_nsd_splits(stim_folder, ss, good_values)

    data_splits = split_data(train_ids, val_ids, nest_ids, ratio=args.ratio)

    save_dir = os.path.join(args.output_dir, f'S{args.subject_id}')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'data_splits_S%d.pkl'%(ss)), "wb") as f:
        pickle.dump(data_splits, f)

if __name__ == '__main__':
    main()