import argparse
import numpy as np
import sys, os
import torch
import time
import h5py
import pandas as pd

# this has the code we need to do the steerable pyramid decomposition.
# https://pyrtools.readthedocs.io/en/latest/index.html
import pyrtools as pt

# this should be in same folder as this file
import prf_utils

device = "cuda" if torch.cuda.is_available() else "cpu"
if device=="cuda":
    print('\nUsing GPU device:')
    print(torch.cuda.get_device_name(0))
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
  
# where the preproc NSD images / data live
save_preproc_path = '/lab_data/hendersonlab/datasets/nsd_preproc/'

labels_folder = os.path.join(save_preproc_path, 'labels')
stim_folder = os.path.join(save_preproc_path, 'stimuli')

# where i would like to put the feature files when created.
# different models become sub-folders in here
features_folder = '/lab_data/hendersonlab/features/NSD_prfmodel/'

sys.path.append('/lab_data/hendersonlab/code/model_fitting')
import model_fitting_utils

def extract_features(ss, \
                     image_data,\
                     n_ori=12, n_sf=5, \
                     sample_batch_size=100, \
                     which_prf_grid='default-log-polar', \
                     nonlin_fn = True, \
                     elongated = False, \
                     debug=False):

 
    n_pix = image_data.shape[2]
    n_images = image_data.shape[0]
    n_batches = int(np.ceil(n_images/sample_batch_size))

    # Params for the spatial aspect of the model (possible pRFs)
    models, grid_name = prf_utils.get_prf_grid(grid_name = which_prf_grid)   
    n_prfs = models.shape[0]

    pyr_height = n_sf # 5 is max for this, given 224 pix images

    high, center, low = get_pyr_freq_pars(pyr_height = pyr_height)
    sf_cyc_per_pix_rev = center # these go in high-to-low order (how they come out of the pyramid)
    sf_cyc_per_pix = np.flip(sf_cyc_per_pix_rev) # low-to-high order (this will match my other code better, later on).
    
    # pass one image through, this gives us sizes of resulting feature maps.
    first_im = image_data[0,0,:,:]
    pyr_init = pt.pyramids.SteerablePyramidFreq(first_im, is_complex=True, \
                                                   height = pyr_height, order = n_ori-1)
    # coefficients come out in high-to-low order.
    # larger fmaps for higher SF, smaller for lower SF.
    map_sizes = dict([])
    prf_stacks_all = dict([])
    for sfi, sf in enumerate(sf_cyc_per_pix_rev):
        print(sfi, sf)
        n_pix = pyr_init.pyr_coeffs[sfi,0].shape[0]
        map_sizes[sf] = n_pix
        # make the stack of pRFs, at this specified resolution size.
        prf_stacks_all[sf] = prf_utils.get_prf_stack(n_pix = n_pix, \
                                                     which_prf_grid = which_prf_grid, \
                                                     dtype=np.float32)
    
    if nonlin_fn:
        # adding a nonlinearity to the filter activations
        print('\nAdding log(1+sqrt(x)) as nonlinearity fn...')
        nonlin = lambda x: torch.log(1+torch.sqrt(x))
    else:
        nonlin = None

    n_features = n_ori * n_sf
    
    print('number of features total: %d'%n_features)
    
    features_each_prf = np.full((n_images, n_features, n_prfs), np.nan, dtype=np.float32)

    ori_rad = np.linspace(0, np.pi, num=n_ori+1)[:-1]
    ori_deg = ori_rad / np.pi * 180
    
    sf_labs_all = np.repeat(sf_cyc_per_pix, n_ori)
    ori_labs_all = np.tile(ori_deg, n_sf)

    
    with torch.no_grad():
        
        for bb in range(n_batches):

            if debug and bb>1:
                continue

            print('\nstarting batch %d of %d...'%(bb,n_batches))
            sys.stdout.flush()
                  
            st_batch = time.time()
            batch_inds = np.arange(sample_batch_size * bb, np.min([sample_batch_size * (bb+1), n_images]))

            # going to extract steerable pyr maps for images in this batch only.
            
            mag_maps = dict([])
            for sfi, sf in enumerate(sf_cyc_per_pix_rev):
                n_pix = map_sizes[sf]
                mag_maps[sf] = np.zeros((len(batch_inds), n_pix, n_pix, n_ori ), dtype=np.float32)

            for ii, image_ind in enumerate(batch_inds):

                # can only go one image at a time. slow!
                this_im = image_data[image_ind, 0, :, :]

                st_pyr = time.time()
                pyr = pt.pyramids.SteerablePyramidFreq(this_im, is_complex=True, \
                                               height = pyr_height, order = n_ori-1)
                elapsed = time.time() - st_pyr
                if ii==0:
                    print('steerable pyramid decomp took: %.5s sec'%elapsed)
                
                for sfi, sf in enumerate(sf_cyc_per_pix_rev):
                    for oo in range(n_ori):
                        mag_maps[sf][ii, :, :, oo] = np.abs(pyr.pyr_coeffs[sfi, oo])

            print('Memory usage just after extraction')
            model_fitting_utils.print_gpu_memory()  # Check after each epoch
            sys.stdout.flush()

            # now multiply the maps by pRFs, one SF at a time
            # this is the only part for which batching helps...
            st_mult = time.time()
            for sfi, sf in enumerate(sf_cyc_per_pix):

                mag = torch.Tensor(mag_maps[sf]).to(device)

                if nonlin is not None:
                    print('apply nonlinearity')
                    # apply compressive nonlinearity
                    mag = nonlin(mag)

                prf_stack = torch.Tensor(prf_stacks_all[sf]).to(device)

                print('SF %d, %.2f cpp'%(sfi, sf))
                print('mag shape:')
                print(mag.shape)
                print('prf stack shape:')
                print(prf_stack.shape)

                # Multiply features by pRFs
                feats_by_prfs = get_feats_by_rfs(mag, prf_stack)

                assert(not torch.any(torch.isnan(feats_by_prfs)))

                print('feats by prfs shape:')
                print(feats_by_prfs.shape)
                
                print('Memory usage just after mult by prfs')
                model_fitting_utils.print_gpu_memory()  # Check after each epoch
                sys.stdout.flush()

                # find just the indices in features that correspond to this SF. all orients.
                feature_inds = sf_labs_all==sf

                print(feature_inds)
                print(len(feature_inds), np.sum(feature_inds))
                
                print(features_each_prf[batch_inds,:,:][:,feature_inds,:].shape, feats_by_prfs.shape)

                # have to do the indexing in 2 steps, bc multiple boolean indices used
                temp = features_each_prf[batch_inds, :, :]
                temp[:, feature_inds, :] = feats_by_prfs.detach().cpu().numpy()
                features_each_prf[batch_inds, :, :] = temp

                # features_each_prf[batch_inds,:,:][:,feature_inds,:] = feats_by_prfs.detach().cpu().numpy()

            elapsed = time.time() - st_mult
            print('multiplying feat by pRFs took %.4f seconds'%(elapsed))
            sys.stdout.flush()

            
            elapsed = time.time() - st_batch
            print('total batch %d took %.4f seconds'%(bb, elapsed))
            sys.stdout.flush()

    print('Total size of features:')
    print(features_each_prf.shape)

    
    if not debug:
        print(np.sum(np.isnan(features_each_prf)))
        print(np.sum(np.sum(np.isnan(features_each_prf), axis=0), axis=1))
        # no nans allowed
        assert(not np.any(np.isnan(features_each_prf)))
                
    # Now save features. We are making one file for each candidate pRF.
    # All files will be located in the same folder.
    folder_save = os.path.join(features_folder, \
                               'pyramid_%dori_%dsf_prf_%s'%(n_ori, n_sf, which_prf_grid), \
                               'NSD_S%d'%ss)

    if not os.path.exists(folder_save):
        os.makedirs(folder_save)

    for mm in range(n_prfs):

        f = features_each_prf[:,:,mm]
        print('Size of features:')
        print(f.shape)
        
        filename_save = os.path.join(folder_save, 'features_prf_%d.npy'%(mm))
        print('Writing prf %d features to %s\n'%(mm, filename_save))

        np.save(filename_save, f)

def rgb_to_gray(image_array):
        
    # Define the luminance coefficients
    # These are standard coefficients for sRGB luminance calculation
    # We use these to get overall luminance (weighted sum of R,G,B channels)
    R_COEFF = 0.2126
    G_COEFF = 0.7152
    B_COEFF = 0.0722

    assert(image_array.shape[1]==3)

    gray = R_COEFF * image_array[:,0:1,:,:] + \
           G_COEFF * image_array[:,1:2,:,:] + \
           B_COEFF * image_array[:,2:3,:,:]

    return gray


def get_pyr_freq_pars(pyr_height = 5):

    n_levels = pyr_height
    
    high = []; center = []; low = []
    
    start_freq = 0.5 # max freq is nyquist limit (this is upper bound of top band)
    
    freq = start_freq
    
    for level in range(n_levels):
        
        high += [freq]
        low += [freq/4]
        center += [freq/2] # approx center of band 
        
        freq /= 2
        # each band goes down in freq by half

    # each band is 2 octaves wide.
    return high, center, low
      

def get_feats_by_rfs(feat_tensor, rfs_tensor):
    """
    Multiply each feature map by each candidate RF. 
    """
    # feat_tensor = torch.tensor(feat_tensor)
    # rfs_tensor = torch.tensor(rfs_tensor)

    nIms = feat_tensor.shape[0]
    nFeats = feat_tensor.shape[3]
    nRFs = rfs_tensor.shape[2]
    nPixTotal = feat_tensor.shape[1]*feat_tensor.shape[2]

    feats_reshaped = torch.moveaxis(torch.reshape(feat_tensor, [nIms, nPixTotal, nFeats]), 2, 1)
    rfs_reshaped = torch.reshape(rfs_tensor, [nPixTotal, nRFs])

    # this is [nImages x nFeatureMaps x nSpatialRFs]
    feats_by_rfs = torch.matmul(feats_reshaped, rfs_reshaped)

    return feats_by_rfs

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=0,
                    help="number of the subject in NSD, 1-8")
    parser.add_argument("--n_ori", type=int,default=12,
                    help="how many orientation channels?")
    parser.add_argument("--n_sf", type=int,default=8,
                    help="how many frequency channels?")

    parser.add_argument("--sample_batch_size", type=int,default=100,
                    help="batch size to use for feature extraction")

    parser.add_argument("--nonlin_fn", type=int,default=0,
                    help="add compressive nonlinearity to responses? 1 for yes, 0 for no")
 
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    
    args = parser.parse_args()

    args.debug = (args.debug==1)
    args.nonlin_fn = (args.nonlin_fn==1)

    print('\nStarting script...')
    print('device = %s'%(device))
    sys.stdout.flush()
    print('debug = %s'%(args.debug))
    print('subject = %d'%(args.subject))

    ss = args.subject
    
    # Load the images for this subject in NSD
    n_pix = 224
    image_filename = os.path.join(stim_folder, 'S%d_stimuli_%d.h5py'%(ss, n_pix))
    print(image_filename)
    t = time.time()
    with h5py.File(image_filename, 'r') as data_set:
        values = np.copy(data_set['/stimuli'])
        data_set.close() 
    elapsed = time.time() - t
    print('Took %.5f seconds to load file'%elapsed)

    # get images ready for feature extraction
    image_array = values.astype(np.float32) / 255
    # turn to grayscale.
    # the code currently only works for gray, not rgb images
    # would be fairly easy to update...
    image_array = rgb_to_gray(image_array)

    extract_features(ss, image_array, \
                     n_ori=args.n_ori, n_sf=args.n_sf, \
                     sample_batch_size=args.sample_batch_size, \
                     which_prf_grid='default-log-polar', \
                     nonlin_fn = args.nonlin_fn, \
                     debug=args.debug)

    print('\nDone')
    