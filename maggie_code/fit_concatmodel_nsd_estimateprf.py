
import sys
import os
import time
import numpy as np
import argparse
import distutils.util
import gc
import h5py
import pandas as pd
import torch
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"
if device=="cuda":
    print('\nUsing GPU device:')
    print(torch.cuda.get_device_name(0))
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# where my preprocessed NSD files live
nsd_path = '/lab_data/hendersonlab/datasets/nsd_preproc'
data_folder = os.path.join(nsd_path, 'data')
labels_folder = os.path.join(nsd_path, 'labels')
stim_folder = os.path.join(nsd_path, 'stimuli')
rois_folder = os.path.join(nsd_path, 'rois')

# where the pre-computed features are placed
# different models are organized in sub-folders in here
features_folder = '/lab_data/hendersonlab/features/NSD_prfmodel/'
print(features_folder)

# where you want to save the model fits.
save_fits_folder = '/lab_data/hendersonlab/projects/gabor_prf/model_fits/'

import model_fitting_utils

sys.path.append('/lab_data/hendersonlab/code/feature_extraction')
import prf_utils

def fit_model(args):

    ########## LOADING THE DATA #############################################################################

    voxel_data, good_values = load_nsd_data(args.subject)

    # by default, fitting all voxels at once. but could sub-select voxels here if needed
    n_voxels = voxel_data.shape[1]
    voxel_inds = np.ones((n_voxels,),dtype=bool)

    voxel_data = voxel_data[:,voxel_inds]

    trn_inds, val_inds, nest_inds = load_nsd_splits(args.subject, good_values)

    # split the fmri data, 3 independent sets
    dat_trn = voxel_data[trn_inds,:]
    dat_nest = voxel_data[nest_inds,:]
    dat_val = voxel_data[val_inds,:]
    
    # load info about rois and voxel mask
    fn = os.path.join(rois_folder, 'S%d_voxel_roi_info.npy'%args.subject)
    rinfo = np.load(fn, allow_pickle=True).item()

    # Params for the spatial aspect of the model (possible pRFs)
    which_prf_grid = 'default-log-polar'
    models, grid_name = prf_utils.get_prf_grid(grid_name = which_prf_grid)   
    n_prfs = models.shape[0]

    ########## FITTING THE MODEL #############################################################################

    # define lambda values
    # lambda is the ridge penalty, bigger = more regularization
    n_lambdas = 20
    small_value = 0.0001
    lambdas = np.logspace(np.log(small_value),np.log(10**10+small_value),n_lambdas, \
                          dtype=np.float32, base=np.e) - small_value
    # lambdas = np.logspace(np.log(0.01),np.log(10**5+0.01),9, dtype=np.float32, base=np.e) - 0.01

    # where the features are located, for this model
    features_subfolder1 = os.path.join(features_folder, '%s_prf_%s'%(args.model_name1, grid_name), 'NSD_S%d'%args.subject)

    if not os.path.exists(features_subfolder1):
        features_subfolder1 = os.path.join(features_folder, 'old', '%s_prf_%s'%(args.model_name1, grid_name), 'NSD_S%d'%args.subject)
        if not os.path.exists(features_subfolder1):
            raise FileNotFoundError('Folder does not exist: %s'%features_subfolder1)

    features_subfolder2 = os.path.join(features_folder, '%s_prf_%s'%(args.model_name2, grid_name), 'NSD_S%d'%args.subject)

    if not os.path.exists(features_subfolder2):
        features_subfolder2 = os.path.join(features_folder, 'old', '%s_prf_%s'%(args.model_name2, grid_name), 'NSD_S%d'%args.subject)
        if not os.path.exists(features_subfolder2):
            raise FileNotFoundError('Folder does not exist: %s'%features_subfolder2)
            
    # load the first set of features, just to get size
    features_filename1 = os.path.join(features_subfolder1, 'features_prf_0.npy')
    f1 = np.load(features_filename1)
    
    features_filename2 = os.path.join(features_subfolder2, 'features_prf_0.npy')
    f2 = np.load(features_filename2)

    f = np.concatenate([f1, f2], axis=1)
    
    print(f1.shape, f2.shape, f.shape)

    
    n_feat = f.shape[1]

    r2_best = np.zeros((n_voxels, ))
    corr_best = np.zeros((n_voxels, ))
    weights_best = np.zeros((n_voxels, n_feat+1)) # n_feat + 1, for the intercept. last weight is intercept.
    featsens_best = np.zeros((n_voxels, n_feat)) # feature sensitivity values 
    best_lambda_inds_best = np.zeros((n_voxels, ))

    # this is needed to choose the best pRF for each voxel, based on nested heldout set
    nest_loss_best = np.ones((n_voxels, )) * 10**8 # initialize w a big value
    nest_loss_all = np.zeros((n_voxels, n_prfs)) 
    
    # convert everything to tensors, send to gpu.
    vtrn = torch.Tensor(dat_trn).to(device).to(torch.float64)
    vnest = torch.Tensor(dat_nest).to(device).to(torch.float64)
    vval = torch.Tensor(dat_val).to(device).to(torch.float64)

    print('Size of data matrices:')
    print(vtrn.shape, vval.shape, vnest.shape)

    # loop over all the prfs in the grid.
    for pi in np.arange(n_prfs):

        st_prep = time.time()

        if args.debug and (pi>1):
            continue

        print('Fitting pRF %d of %d (all vox)'%(pi, n_prfs))

        # load the features corresponding to this specified pRF
        features_filename1 = os.path.join(features_subfolder1, 'features_prf_%d.npy'%pi)
        print('loading features from: %s'%features_filename1)
        sys.stdout.flush()
        f1 = np.load(features_filename1)

        # load the features corresponding to this specified pRF
        features_filename2 = os.path.join(features_subfolder2, 'features_prf_%d.npy'%pi)
        print('loading features from: %s'%features_filename2)
        sys.stdout.flush()
        f2 = np.load(features_filename2)

        f = np.concatenate([f1, f2], axis=1)

        # make sure we only take features that have valid fMRI data
        f = f[good_values,:]
        print(f.shape)

        # divide into three splits
        # z-scoring happens in here as well.
        f_trn, f_val, f_nest = model_fitting_utils.split_normalize_feats(f, trn_inds, val_inds, nest_inds)
    
        # add the intercept: a column of ones
        f_trn = np.concatenate([f_trn, np.ones(shape=(len(f_trn), 1), dtype=f_trn.dtype)], axis=1)
        f_nest = np.concatenate([f_nest, np.ones(shape=(len(f_nest), 1), dtype=f_nest.dtype)], axis=1)
        f_val = np.concatenate([f_val, np.ones(shape=(len(f_val), 1), dtype=f_val.dtype)], axis=1)
        
        # x is our features, send to GPU here
        xtrn = torch.Tensor(f_trn).to(device).to(torch.float64)
        xnest = torch.Tensor(f_nest).to(device).to(torch.float64)
        xval = torch.Tensor(f_val).to(device).to(torch.float64)

        del f, f_trn, f_val, f_nest
        gc.collect()
        
        print('Size of features matrices:')
        print(xtrn.shape, xval.shape, xnest.shape)

        elapsed_prep = time.time() - st_prep
        print('Prep time elapsed: %.5f s'%elapsed_prep)

        
        print('Memory usage just before fitting function')
        model_fitting_utils.print_gpu_memory()  # Check after each epoch
        sys.stdout.flush()
    
        # here is where we actually solve for the weights. 
        st_fit = time.time()
        weights, best_lambda_inds, nest_loss = model_fitting_utils.solve_ridge(xtrn, vtrn, \
                                                                               xnest, vnest, \
                                                                               lambdas, return_loss=True)
        elapsed_fit = time.time() - st_fit
        print('Model fitting time elapsed: %.5f s'%elapsed_fit)
    
        print('Memory usage just after fitting function')
        model_fitting_utils.print_gpu_memory()  # Check after each epoch
        sys.stdout.flush()
    
        # predict voxel response in held-out validation data here.
        st_pred = time.time()
        pred = xval @ weights
        elapsed_pred = time.time() - st_pred
        print('Pred time elapsed: %.5f s'%elapsed_pred)
        sys.stdout.flush()

        st_eval = time.time()

        r2 = model_fitting_utils.get_r2_torch(vval, pred) 
        corr = model_fitting_utils.get_corrcoef_torch(vval, pred)

        featsens = model_fitting_utils.get_featsens_torch(pred, xval[:,0:n_feat])

        # remember to turn these back into numpy, from torch.
        # sometimes tensors will give errors in your subsequent numpy code.
        r2 = r2.cpu().numpy()
        corr = corr.cpu().numpy()
        featsens = featsens.cpu().numpy()
        weights = weights.cpu().numpy()
        
        del xtrn, xnest, xval
        torch.cuda.empty_cache()

        # if the fit for the voxel has improved, then this is the new "best" RF for the voxel
        # we will save all of its parameters
        voxels_update = nest_loss < nest_loss_best
        print('%d voxels to be updated'%np.sum(voxels_update))

        if np.sum(voxels_update)>0:
            
            # for those voxels that improved, we rewrite the arrays with their current params.
            r2_best[voxels_update] = r2[voxels_update] 
            corr_best[voxels_update] = corr[voxels_update] 
            
            weights_best[voxels_update, :] = weights[:,voxels_update].T
            featsens_best[voxels_update, :] = featsens[voxels_update,:]
            
            best_lambda_inds_best[voxels_update] = best_lambda_inds[voxels_update]

            nest_loss_best[voxels_update] = nest_loss[voxels_update]

        nest_loss_all[:,pi] = nest_loss

        del r2, corr, weights, featsens
        gc.collect()

        elapsed_eval = time.time() - st_eval
        print('Eval time elapsed: %.5f s'%elapsed_eval)
        sys.stdout.flush()

    print('Choosing best pRF per voxel:')
    st_choose = time.time()
    
    # Now identifying for each voxel, what was its best pRF
    voxel_prf_grid_inds = np.zeros((n_voxels))
    
    for vi in np.arange(0, n_voxels):

        # the best pRF is the one that gives the best (min) loss on nested heldout set
        best_prf_ind = np.argmin(nest_loss_all[vi,:])
        voxel_prf_grid_inds[vi] = best_prf_ind

        if not args.debug:
            assert(nest_loss_all[vi, best_prf_ind]==nest_loss_best[vi])

        
    elapsed_choose = time.time() - st_eval
    print('Choose time elapsed: %.5f s'%elapsed_choose)
    
    # Now save.
    # Will make a dictionary of things to save.
    dict2save = {'subject': args.subject, \
                 'model1': args.model_name1, \
                 'model2': args.model_name2, \
                 'features_filename1': features_filename1, \
                 'feature2_filename2': features_filename2, \
                 'grid_name': which_prf_grid, \
                 'voxel_prf_grid_inds': voxel_prf_grid_inds, \
                 'prf_grid_params': models, \
                 'lambdas': lambdas, \
                 'voxel_mask': rinfo['voxel_mask'], \
                 'voxel_index': rinfo['voxel_idx'], \
                 'voxel_nc': rinfo['noise_ceiling_avgreps'], \
                 'brain_nii_shape': rinfo['brain_nii_shape'], \
                 'weights_all': weights_best, \
                 'featsens_all': featsens_best, \
                 'r2_all': r2_best, \
                 'corr_all': corr_best, \
                 'best_lambda_inds_all': best_lambda_inds_best, \
                 'nest_loss_all': nest_loss_best, \
                    }

    save_folder = os.path.join(save_fits_folder, '%s_plus_%s_fit_prfs'%(args.model_name1, args.model_name2))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    fn2save = os.path.join(save_folder, 'NSD_S%d_%s_plus_%s_fit_prfs.npy'%(args.subject, args.model_name1, args.model_name2))
    print('saving to %s'%fn2save)
    np.save(fn2save, dict2save, allow_pickle=True)


def load_nsd_data(ss):

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
    

def load_nsd_splits(ss, good_values):

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


# this is the function we go to when main .py script is executed. 
if __name__ == '__main__':

    # this is just a function that helps with argument parsing
    def nice_str2bool(x):
        return bool(distutils.util.strtobool(x))

    # this part handles the arguments 
    parser = argparse.ArgumentParser()

    # we can add as many arguments here as needed, using this same format.
    parser.add_argument("--subject", type=int,default=1,
                    help="number of the subject, 1-8")
    parser.add_argument("--debug",type=nice_str2bool,default=False,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    
    parser.add_argument("--model_name1",type=str,default='',
                    help="which model are the features from?")
    parser.add_argument("--model_name2",type=str,default='',
                    help="which model are the features from?")

    args = parser.parse_args()

    st = time.time()
    
    # then we call the main function. arguments gets passed in.
    fit_model(args)

    elapsed = time.time() - st
    print('fitting took %.5f s total'%elapsed)
    sys.stdout.flush()


    

    
