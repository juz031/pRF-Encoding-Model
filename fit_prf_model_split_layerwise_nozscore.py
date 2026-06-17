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
import open_clip


# sys.path.insert(0,'/home/junruz/BrainDiVE')
# from encoder_training_CNN import neural_loader

import prf_utils
import model_fitting_utils


class LayerSize:
    pass

setattr(LayerSize, "CLIP_RN50_layer_size", {
    'relu1': [32, 112, 112],
    'relu2': [32, 112, 112],
    'relu3': [64, 112, 112],
    'avgpool': [64, 56, 56],
    'layer1': [256, 56, 56],
    'layer2': [512, 28, 28],
    'layer3': [1024, 14, 14],
    'layer4': [2048, 7, 7]
})

setattr(LayerSize, "OPEN_CLIP_RN50_layer_size", {
    'act1': [32, 112, 112],
    'act2': [32, 112, 112],
    'act3': [64, 112, 112],
    'avgpool': [64, 56, 56],
    'layer1': [256, 56, 56],
    'layer2': [512, 28, 28],
    'layer3': [1024, 14, 14],
    'layer4': [2048, 7, 7]
})

setattr(LayerSize, "DINO_RN50_layer_size", {
    'relu': [64, 112, 112],
    'maxpool': [64, 56, 56],
    'layer1': [256, 56, 56],
    'layer2': [512, 28, 28],
    'layer3': [1024, 14, 14],
    'layer4': [2048, 7, 7]
})

setattr(LayerSize, "SIMCLR_RN50_layer_size", {
    'relu': [64, 112, 112],
    'maxpool': [64, 56, 56],
    'layer1': [256, 56, 56],
    'layer2': [512, 28, 28],
    'layer3': [1024, 14, 14],
    'layer4': [2048, 7, 7]
})


class Normalize:
    pass

setattr(Normalize, "CLIP_RN50_MEAN", np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.single)[:, None, None])
setattr(Normalize, "CLIP_RN50_STD", np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.single)[:, None, None])

setattr(Normalize, "DINO_RN50_MEAN", np.array((0.485, 0.456, 0.406), dtype=np.single)[:, None, None])
setattr(Normalize, "DINO_RN50_STD", np.array((0.229, 0.224, 0.225), dtype=np.single)[:, None, None])

setattr(Normalize, "SIMCLR_RN50_MEAN", np.array((0.485, 0.456, 0.406), dtype=np.single)[:, None, None])
setattr(Normalize, "SIMCLR_RN50_STD", np.array((0.229, 0.224, 0.225), dtype=np.single)[:, None, None])

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


def model_fitting(voxel_data, train_ids, val_ids, nest_ids, features_folder, args, device):
    """
    Return:
    - best_prf_idx: list of length of num_voxels, the index of the best pRF for each voxel
    - best_lambda: list of length of num_voxels, the best lambda for each voxel for the best pRF
    - best_loss: list of length of num_voxels, the best r2 for each voxel for the best pRF
    - best_weights: array of length of [num_voxels, num_features + 1], the best weights for each voxel for the best pRF, with the intercept
    """
    num_prfs = len(os.listdir(features_folder))

    train_voxel = voxel_data[train_ids,:]
    nest_voxel = voxel_data[nest_ids,:]

    train_voxel = torch.from_numpy(train_voxel).float().to(device)
    nest_voxel = torch.from_numpy(nest_voxel).float().to(device)

    num_features = getattr(LayerSize, args.model_name + '_layer_size')[args.layer_name][0]

    # num_features = 96
    num_voxels = train_voxel.shape[1]

    # best_weights: [num_features + 1, num_voxels] +1 for the intercept
    best_weights_dict = {}
    channel_kept_dict = {}
    best_lambda_array = np.zeros(num_voxels)
    best_nest_loss_array = np.full((num_voxels,), np.inf)
    best_prf_idx_array = np.zeros(num_voxels, dtype=int)
    
    for prf_idx in tqdm(range(num_prfs)): # ~1450
        features_path = os.path.join(features_folder, f'features_prf_{prf_idx}.npy')
        features = np.load(features_path)

        # split train and nest features, then z-score the features across images axis 0 using model_fitting_utils.split_normalize_feats
        # train_features, val_features, nest_features, features_s, features_m = model_fitting_utils.split_normalize_feats(features, train_ids, val_ids, nest_ids)
        # train_features, val_features, nest_features, keep_idx = model_fitting_utils.split_normalize_feats_with_drop(features, train_ids, val_ids, nest_ids)
        train_features, val_features, nest_features, keep_idx = model_fitting_utils.split_feats(features, train_ids, val_ids, nest_ids)

        # add the intercept: a column of ones
        train_features = np.concatenate([train_features, np.ones(shape=(len(train_features), 1), dtype=train_features.dtype)], axis=1)
        nest_features = np.concatenate([nest_features, np.ones(shape=(len(nest_features), 1), dtype=nest_features.dtype)], axis=1)

        train_features = torch.from_numpy(train_features).float().to(device)
        nest_features = torch.from_numpy(nest_features).float().to(device)
        n_lambdas = 20
        small_value = 0.0001
        lambdas = np.logspace(np.log(small_value),np.log(10**10+small_value),n_lambdas, \
                            dtype=np.float32, base=np.e) - small_value
        
        best_weights, best_lambda_idx, best_nest_loss = model_fitting_utils.solve_ridge(train_features, train_voxel, nest_features, nest_voxel, lambdas, eps=1e-4, return_loss=True)
        # best_weights, best_lambda_idx, best_nest_loss = model_fitting_utils.solve_ridge_svd(train_features, train_voxel, nest_features, nest_voxel, lambdas, eps=1e-4, return_loss=True)


        # update the best weights, lambda, loss, and prf_idx if the current best_loss is less than the previous best_loss
        count = 0
        for voxel_idx in range(num_voxels):
            if best_nest_loss[voxel_idx] < best_nest_loss_array[voxel_idx]:
                best_weights_dict[str(voxel_idx)] = best_weights[:,voxel_idx].cpu().numpy().astype(np.float32)
                channel_kept_dict[str(voxel_idx)] = keep_idx
                best_lambda_array[voxel_idx] = lambdas[best_lambda_idx[voxel_idx]]
                best_nest_loss_array[voxel_idx] = best_nest_loss[voxel_idx]
                best_prf_idx_array[voxel_idx] = prf_idx
                count += 1
        # print(f"Number of voxels updated: {count}")
        

    best_lambda_array = best_lambda_array.astype(np.float32)
    best_nest_loss_array = best_nest_loss_array.astype(np.float32)
    best_prf_idx_array = best_prf_idx_array.astype(int)

    return best_weights_dict, channel_kept_dict, best_lambda_array, best_nest_loss_array, best_prf_idx_array





def main():
    # Set some paths: where the preprocessed NSD files live
    nsd_path = '/lab_data/hendersonlab/datasets/nsd_preproc'
    data_folder = os.path.join(nsd_path, 'data')
    labels_folder = os.path.join(nsd_path, 'labels')
    stim_folder = os.path.join(nsd_path, 'stimuli')
    rois_folder = os.path.join(nsd_path, 'rois')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_pix = 224

    # Create namespace and set attributes
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', nargs='+', default=[1], type=int)
    parser.add_argument('--neural_activity_path', type=str, default=os.path.join(data_folder, 'S{}_betas_avg_bigmask_dict.hdf5'))
    parser.add_argument('--image_path', type=str, default=os.path.join(stim_folder, 'S{}_stimuli_%d_dict.hdf5'%n_pix))
    parser.add_argument('--roi_path', type=str, default=os.path.join(rois_folder, 'S{}_voxel_roi_info.npy'))
    parser.add_argument('--stim_keys_path', type=str, default=os.path.join(stim_folder, "all_keys.pkl"))
    parser.add_argument('--model_name', type=str, default="DINO_RN50")
    parser.add_argument('--split_id', type=int, choices=[1,2], default=1)
    parser.add_argument('--layer_name', type=str, default="layer4")
    parser.add_argument('--split_root', type=str, default="/user_data/junruz/prf_models/split_1_zscore")
    args = parser.parse_args()

    # where my preprocessed NSD files live
    nsd_path = '/lab_data/hendersonlab/datasets/nsd_preproc'
    data_folder = os.path.join(nsd_path, 'data')
    labels_folder = os.path.join(nsd_path, 'labels')
    stim_folder = os.path.join(nsd_path, 'stimuli')
    rois_folder = os.path.join(nsd_path, 'rois')

    ss = args.subject_id[0]

    split_dir = os.path.join(args.split_root, f'S{ss}')
    with open(os.path.join(split_dir, f'data_splits_S{ss}.pkl'), 'rb') as f:
        data_splits = pickle.load(f) # 0: ratio, 1:split 1, 2:split 2

    ids = data_splits[args.split_id]
    train_ids, val_ids, nest_ids = ids['train'], ids['val'], ids['nest']

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
    # train_ids, val_ids, nest_ids = load_nsd_splits(stim_folder, ss, good_values)

    best_weights, channel_ignored, best_lambda, best_loss, best_prf_idx = model_fitting(voxel_data, train_ids, val_ids, nest_ids, features_folder, args, device)

    print(f"Max loss: {np.max(best_loss)}, Min loss: {np.min(best_loss)}, Mean loss: {np.mean(best_loss)}")
    # where you want to save the model fits.
    save_fits_roots = args.split_root
    save_fits_folder = os.path.join(save_fits_roots, f"S{ss}", args.model_name+f'_set{args.split_id}', args.layer_name)
    os.makedirs(save_fits_folder, exist_ok=True)
    # save the best weights, lambda, loss, and prf_idx
    with open(os.path.join(save_fits_folder, "best_weights.pkl"), "wb") as f:
        pickle.dump(best_weights, f)
    with open(os.path.join(save_fits_folder, "channel_kept.pkl"), "wb") as f:
        pickle.dump(channel_ignored, f)
    np.save(os.path.join(save_fits_folder, "best_lambda.npy"), best_lambda)
    np.save(os.path.join(save_fits_folder, "best_loss.npy"), best_loss)
    np.save(os.path.join(save_fits_folder, "best_prf_idx.npy"), best_prf_idx)


    pass


if __name__ == "__main__":
    main()