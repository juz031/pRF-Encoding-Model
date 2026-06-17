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
import matplotlib.pyplot as plt
import random

import os
import numpy as np
import pickle




import prf_utils
import model_fitting_utils

import stats_utils


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
    # TODO: Load ROI masks and apply to the data

    # TODO: Load noise ceiling and set threshold to select voxels
    
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
    good_values = ~np.isnan(values[:,0]) # data not missing
    # print(values.shape)
    # print(values.shape)
    # print(np.sum(~good_values))

    # check that nans are exactly where we expect
    # nans happen when n_reps=0
    assert(np.all(good_values[n_reps>0]))
    assert(np.all(~good_values[n_reps==0]))

    voxel_data = values[good_values,:]
    # print(voxel_data.shape)
    
    return voxel_data, good_values


def load_nsd_rois(ss, args):
    roi_path = os.path.join(args.roi_path.format(ss))
    print(f"Loading rois from: {roi_path}")

    roi_info = np.load(roi_path, allow_pickle=True).item()
    noise_ceiling = roi_info['noise_ceiling_avgreps'] / 100.

    big_mask = roi_info['voxel_mask']
    roi_keys = ['roi_labels_retino', 'roi_labels_kastner', 'roi_labels_face', 'roi_labels_place', 'roi_labels_body']
    roi_names = ['ret_prf_roi_names', 'kastner_atlas_roi_names', 'floc_face_roi_names', 'floc_place_roi_names', 'floc_body_roi_names']
    roi_masks = dict()
    for key, name in zip(roi_keys, roi_names):
        roi_labels = roi_info[key][big_mask]
        roi_name = roi_info[name]
        for name in roi_name.keys():
            roi_masks[name] = roi_labels==roi_name[name]
    
    # combine ventral and dorsal retinotopic regions  
    for name in ["V1", "V2", "V3"]:
        roi_masks[name] = roi_masks[name + "v"] | roi_masks[name + "d"]

    # combine FFA-1 and FFA-2
    roi_masks["FFA"] = roi_masks["FFA-1"] | roi_masks["FFA-2"]

    print(f"Loaded {len(roi_masks)} rois: {list(roi_masks.keys())}")
        
    return roi_masks, noise_ceiling


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


def eval_prf_model_layerwise(voxel_data, train_ids, val_ids, nest_ids, features_folder, args, device):
    model_path = os.path.join(args.model_path, args.model_name + f"_set{args.split_id}")
    model_path = os.path.join(model_path, args.layer_name)
    weights_path = os.path.join(model_path, 'best_weights.pkl')
    channel_kept_path = os.path.join(model_path, 'channel_kept.pkl')
    prf_idx_path = os.path.join(model_path, 'best_prf_idx.npy')
    lambda_path = os.path.join(model_path, 'best_lambda.npy')
    features_m_path = os.path.join(model_path, 'best_features_m.npy')
    features_s_path = os.path.join(model_path, 'best_features_s.npy')
    
    with open(weights_path, "rb") as f:
        best_weights = pickle.load(f)
    with open(channel_kept_path, "rb") as f:
        channel_kept = pickle.load(f)
    best_prf_idx = np.load(prf_idx_path)
    best_lambda = np.load(lambda_path)
    best_features_m = np.load(features_m_path)
    best_features_s = np.load(features_s_path)

    # num_layer_feature = clip_rn50_layer_size[args.layer_name][0]

    val_voxel = voxel_data[val_ids,:]
    print(f"number of images in val set: {val_voxel.shape[0]}")
    val_voxel = torch.from_numpy(val_voxel).float()
    num_voxels = val_voxel.shape[1]

    ######### for debug ##########
    # train_voxel = voxel_data[train_ids,:]
    # print(f"number of images in val set: {train_voxel.shape[0]}")
    # train_voxel = torch.from_numpy(train_voxel).float()
    # num_voxels = train_voxel.shape[1]
    ##############################

    assert(num_voxels == len(best_weights.keys()))
    assert(num_voxels == best_prf_idx.shape[0])

    predicted_neural_list = []
    for voxel_idx in tqdm(range(num_voxels)):
    #     if voxel_idx in [619,  3716,  3909,  4450,  4451,  4464,  4494,  5101,  5544, \
    #     5921,  6349,  6399,  6757,  7357,  7364,  7450,  9967, 10451, \
    #    18831]:
    #         pass
        prf_idx = best_prf_idx[voxel_idx]
        lambda_ = best_lambda[voxel_idx]
        model = best_weights[str(voxel_idx)]
        weights = model[:-1]
        intercept = model[-1]

        features_m = best_features_m[voxel_idx]
        features_s = best_features_s[voxel_idx]

        weights = torch.from_numpy(weights).float().to(device)
        intercept = torch.tensor(intercept).to(device)

        features_path = os.path.join(features_folder, f'features_prf_{prf_idx}.npy')
        features = np.load(features_path)

        # delete columns that are all zeros
        # all_zeros_cols = np.all(features == 0, axis=0)
        # zero_cols_idx = np.where(all_zeros_cols)[0]
        # features = np.delete(features, zero_cols_idx, axis=1)

        # z-score the features across images axis 0 using model_fitting_utils.split_normalize_feats
        # train_features, val_features, nest_features, keep_idx = model_fitting_utils.split_normalize_feats_with_drop(features, train_ids, val_ids, nest_ids)
        # train_features, val_features, nest_features, keep_idx = model_fitting_utils.split_feats(features, train_ids, val_ids, nest_ids)
        val_features = features[val_ids,:]
        val_features = (val_features - features_m) / features_s
        val_features = torch.from_numpy(val_features).float().to(device)

        assert(weights.shape[0] == val_features.shape[1])

        ######### for debug ##########
        # train_features = torch.from_numpy(train_features).float().to(device)
        # predicted_neural = train_features @ weights + intercept
        ##############################

        predicted_neural = val_features @ weights + intercept
        predicted_neural = predicted_neural.detach().cpu().numpy()
        predicted_neural = predicted_neural[:, np.newaxis]


        predicted_neural_list.append(predicted_neural)
    
    predicted_neural = np.concatenate(predicted_neural_list, axis=1) # [num_heldout, num_voxels]
    print(f'shape of predicted neural response: {predicted_neural.shape}')
    
    return predicted_neural


def calculate_r2(predicted_neural, true_neural, roi_masks, noise_ceiling, args):
    model_path = os.path.join(args.model_path, args.model_name + f"_set{args.split_id}")
    save_root = os.path.join(model_path, args.layer_name)
    # model_path = os.path.join(model_path, args.layer_name)
    noise_ceiling_threshold = args.noise_ceiling_threshold
    areas = args.roi
    nc_mask = noise_ceiling > noise_ceiling_threshold

    r2 = {}
    for area in areas:
        roi_mask = roi_masks[area]
        mask = roi_mask & nc_mask
        
        num_voxels = np.sum(mask)
        print(f"Selected {num_voxels} voxels in {area}, nc > {noise_ceiling_threshold}")
        r2_voxels = np.zeros((num_voxels, 1))

        predicted_neural_area = predicted_neural[:, mask]
        true_neural_area = true_neural[:, mask]

        ######### for debug ##########
        # predicted_neural_area = np.delete(predicted_neural_area, 765, axis=0)
        # true_neural_area = np.delete(true_neural_area, 765, axis=0)
        ##############################

        print(f"shape of predicted neural response: {predicted_neural_area.shape}")

        for vv in range(num_voxels):    
            r2_voxels[vv] = stats_utils.get_r2(true_neural_area[:,vv], \
                                    predicted_neural_area[:,vv])


        # plot and save the r2 histogram
        plt.figure()
        plt.hist(r2_voxels)
        plt.axvline(np.median(r2_voxels), color='k')
        print("mean r2: ", np.mean(r2_voxels))
        print("median r2: ", np.median(r2_voxels))
        plt.title(f'R2 hist for {area} {args.layer_name}, NC > {noise_ceiling_threshold}')
        # Add a box annotation in the upper right corner to show mean and median R2
        mean_r2 = np.mean(r2_voxels)
        median_r2 = np.median(r2_voxels)
        textstr = f"Mean: {mean_r2:.3f}\nMedian: {median_r2:.3f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.75)
        plt.gca().text(
            0.95, 0.95, textstr,
            transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=props
        )
        plt.xlabel('R2')
        plt.ylabel('Voxel count')
        # save_folder = os.path.join(save_root, args.model_name + f"_set{args.split_id}")
        # plt.xlim(0,1)
        # os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_root, f'hist_{area}_{args.layer_name}_nc_{noise_ceiling_threshold}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

        # plot and save the r2 vs noise ceiling scatter plot
        a, b = np.polyfit(noise_ceiling[mask], r2_voxels, 1)
        plt.figure()
        plt.rcParams.update({'font.size': 18})
        plt.scatter(noise_ceiling[mask], r2_voxels, s=5, )
        plt.plot(noise_ceiling[mask], a*noise_ceiling[mask] + b, color='r', linestyle='--')
        plt.xlabel('Noise Ceiling')
        plt.ylabel('R2 per voxel')
        plt.ylim(-0.2,0.8)
        # plt.xlim(0,1)
        plt.title(f'R2 vs NC for {area}, RN50 {args.layer_name}') #, nc > {noise_ceiling_threshold
        mean_r2 = np.mean(r2_voxels)
        median_r2 = np.median(r2_voxels)
        textstr = f"Mean: {mean_r2:.3f}\nMedian: {median_r2:.3f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.75)
        plt.gca().text(
            0.05, 0.95, textstr,
            transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=props
        )
        plt.axline((0, 0), slope=1, color='k', linestyle='--')
        # save_folder = os.path.join(save_root, 'r2_nc')
        # os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_root, f'scatter_{area}_{args.layer_name}_nc_{noise_ceiling_threshold}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        

        plt.close()

        r2[area] = {
            'r2_voxels': r2_voxels,
            'a': a,
            'b': b,
            'noise_ceiling': noise_ceiling[mask],
        }
        
    return r2


def main():
    # Set some paths: where the preprocessed NSD files live
    nsd_path = '/lab_data/hendersonlab/datasets/nsd_preproc'
    data_folder = os.path.join(nsd_path, 'data')
    labels_folder = os.path.join(nsd_path, 'labels')
    stim_folder = os.path.join(nsd_path, 'stimuli')
    rois_folder = os.path.join(nsd_path, 'rois')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    n_pix = 224

    # Create namespace and set attributes
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', nargs='+', default=[1], type=int)
    parser.add_argument('--neural_activity_path', type=str, default=os.path.join(data_folder, 'S{}_betas_avg_bigmask_dict.hdf5'))
    parser.add_argument('--model_path', type=str, default='/user_data/junruz/prf_models/split_1/S1')
    parser.add_argument('--image_path', type=str, default=os.path.join(stim_folder, 'S{}_stimuli_%d_dict.hdf5'%n_pix))
    parser.add_argument('--features_root', type=str, default='/user_data/junruz/prf_features')
    parser.add_argument('--roi_path', type=str, default=os.path.join(rois_folder, 'S{}_voxel_roi_info.npy'))
    parser.add_argument('--stim_keys_path', type=str, default=os.path.join(stim_folder, "all_keys.pkl"))
    parser.add_argument('--model_name', type=str, default="DINO_RN50")
    parser.add_argument('--layer_name', type=str, default="layer3")
    parser.add_argument('--roi', nargs='+', default=["V1", "V2", "V3", "hV4", "FFA", "PPA"], type=str)
    parser.add_argument('--noise_ceiling_threshold', type=float, default=0.)
    parser.add_argument('--split_root', type=str, default="/user_data/junruz/prf_models/split_1")
    parser.add_argument('--split_id', type=int, choices=[1,2], default=2)
    args = parser.parse_args()

    ss = args.subject_id[0]

    # where the pre-computed features are placed
    # different models are organized in sub-folders in here
    features_folder = os.path.join(args.features_root, f"S{ss}", args.model_name, args.layer_name)

    ############ for debug ##########################
    # features_folder = '/lab_data/hendersonlab/features/NSD_prfmodel/gabor_12ori_8sf_prf_default-log-polar/NSD_S1'
    ##############################

    print(f"Loading features from: {features_folder}")

    # Load rois masks
    roi_masks, noise_ceiling = load_nsd_rois(ss, args)

    
    # Load ground truth voxel response and predict voxel response
    voxel_data, good_values = load_nsd_data(data_folder, labels_folder, ss)

    split_dir = os.path.join(args.split_root, f'S{ss}')
    with open(os.path.join(split_dir, f'data_splits_S{ss}.pkl'), 'rb') as f:
        data_splits = pickle.load(f) # 0: ratio, 1:split 1, 2:split 2

    ids = data_splits[args.split_id]
    train_ids, val_ids, nest_ids = ids['train'], ids['val'], ids['nest']
    val_voxel = voxel_data[val_ids,:]
    # train_voxel = voxel_data[train_ids,:]

    predicted_neural = eval_prf_model_layerwise(voxel_data, train_ids, val_ids, nest_ids, features_folder, args, device)


    r2 = calculate_r2(predicted_neural, val_voxel, roi_masks, noise_ceiling, args)
    # r2 = calculate_r2(predicted_neural, train_voxel, roi_masks, noise_ceiling, args)

    with open(os.path.join(args.model_path, args.model_name + f"_set{args.split_id}", args.layer_name, "r2.pkl"), "wb") as f:
        pickle.dump(r2, f)



if __name__ == "__main__":
    main()