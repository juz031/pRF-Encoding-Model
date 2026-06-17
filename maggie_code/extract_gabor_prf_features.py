import argparse
import numpy as np
import sys, os
import torch
import time
import h5py
import pandas as pd

# these should both be in same folder as this file
import gabor_model
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
                     n_ori=12, n_sf=8, \
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

    if n_sf==6:
        # min_sf_cyc_per_stim = 6; 
        min_sf_cyc_per_stim = 4; 
        max_sf_cyc_per_stim = 30
        # this is meant to match values in the bent gabor code.
    else:
        # min_sf_cyc_per_stim = 3; 
        min_sf_cyc_per_stim = 4;
        max_sf_cyc_per_stim = 72;
        # otherwise using a larger range.
    sf_cyc_per_stim = np.logspace(np.log10(min_sf_cyc_per_stim), np.log10(max_sf_cyc_per_stim), num = n_sf)
    sf_cyc_per_pix = sf_cyc_per_stim / n_pix

    ori_rad = np.linspace(0, np.pi, num=n_ori+1)[:-1]
    ori_deg = ori_rad / np.pi * 180

    print(sf_cyc_per_pix)
    print(ori_deg)

    # default parameters, rarely want to change these
    padding_mode = 'reflect'
    spat_freq_bw = 1.0
    n_sd_out = 3
    
    # Set up the feature extractor module here
    if nonlin_fn:
        # adding a nonlinearity to the filter activations
        print('\nAdding log(1+sqrt(x)) as nonlinearity fn...')
        nonlin = lambda x: torch.log(1+torch.sqrt(x))
    else:
        nonlin = None

    if elongated:
        spat_aspect_ratio = 1/2
    else:
        spat_aspect_ratio = 1

    print('spat aspect ratio = %.2f'%spat_aspect_ratio)

    bank = gabor_model.filter_bank(orients_deg = ori_deg, \
                                   freqs_cpp = sf_cyc_per_pix, \
                                   image_size = n_pix, \
                                   spat_freq_bw = spat_freq_bw, \
                                   n_sd_out = n_sd_out, \
                                   spat_aspect_ratio = spat_aspect_ratio, \
                                   padding_mode = padding_mode)
  
    extractor = gabor_model.gabor_feature_extractor_spat(filter_bank = bank, \
                                             device = device)

    n_features = n_ori * n_sf
    
    print('number of features total: %d'%n_features)
    
    features_each_prf = np.zeros((n_images, n_features, n_prfs), dtype=np.float32)

    # stack of pRFs: this is [x, y, num_pRFs]
    prf_stack = prf_utils.get_prf_stack(n_pix = n_pix, which_prf_grid = which_prf_grid, dtype=np.float32)
    prf_stack = torch.Tensor(prf_stack).to(device)
    
    with torch.no_grad():
        
        for bb in range(n_batches):

            if debug and bb>1:
                continue

            st = time.time()
            batch_inds = np.arange(sample_batch_size * bb, np.min([sample_batch_size * (bb+1), n_images]))

            print('\nBatch %d: extracting features for images [%d - %d]'%(bb, batch_inds[0], batch_inds[-1]))

            image_batch = torch.from_numpy(image_data[batch_inds,:,:,:]).float().to(device)   

            print('Memory usage just before extraction')
            model_fitting_utils.print_gpu_memory()  # Check after each epoch
            sys.stdout.flush()


            st_extr = time.time()
            # pass into the gabor feature extractor here
            # mag is what we want to use
            mag, phase = extractor(image_batch)
            elapsed = time.time() - st_extr
            print('extraction for batch %d took %.4f seconds'%(bb, elapsed))
            sys.stdout.flush()
            
            if nonlin is not None:
                print('apply nonlinearity')
                # apply compressive nonlinearity
                mag = nonlin(mag)

            print('Memory usage just after extraction')
            model_fitting_utils.print_gpu_memory()  # Check after each epoch
            sys.stdout.flush()

            # Multiply features by pRFs
            feats_by_prfs = gabor_model.get_feats_by_rfs(mag, prf_stack)

            print('Memory usage just after mult by prfs')
            model_fitting_utils.print_gpu_memory()  # Check after each epoch
            sys.stdout.flush()
            
            features_each_prf[batch_inds,:,:] = feats_by_prfs.detach().cpu().numpy()

            elapsed = time.time() - st
            print('batch %d took %.4f seconds'%(bb, elapsed))
            sys.stdout.flush()
                
            
    print('Total size of features:')
    print(features_each_prf.shape)

    # Now save features. We are making one file for each candidate pRF.
    # All files will be located in the same folder.
    folder_save = os.path.join(features_folder, \
                               'gabor_%dori_%dsf_prf_%s'%(n_ori, n_sf, which_prf_grid), \
                               'NSD_S%d'%ss)
    if elongated:
        folder_save = os.path.join(features_folder, \
                               'gabor_%dori_%dsf_elongated_prf_%s'%(n_ori, n_sf, which_prf_grid), \
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

    parser.add_argument("--nonlin_fn", type=int,default=1,
                    help="add compressive nonlinearity to responses? 1 for yes, 0 for no")
    parser.add_argument("--elongated", type=int,default=0,
                    help="use more elongated gaussian (spat aspect ratio)? 1 for yes, 0 for no")
    
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    
    args = parser.parse_args()

    args.debug = (args.debug==1)
    args.nonlin_fn = (args.nonlin_fn==1)
    args.elongated = (args.elongated==1)
    
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
                     elongated = args.elongated, \
                     debug=args.debug)

    print('\nDone')
    