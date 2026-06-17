import sys
import os
import numpy as np
import h5py

nsd_path = '/lab_data/hendersonlab/datasets/nsd_preproc'
data_folder = os.path.join(nsd_path, 'data')
stim_folder = os.path.join(nsd_path, 'stimuli')

image_filename = os.path.join(stim_folder, 'S1_stimuli_224.h5py')

with h5py.File(image_filename, 'r') as data_set:
    images_array = np.copy(data_set['/stimuli'])
    data_set.close()

# Save the first 10 images from images_array
import imageio

save_dir = 'first_10_images'
os.makedirs(save_dir, exist_ok=True)
for i in range(10):
    img = images_array[i]
    # Convert image to uint8 in case it's not
    img_uint8 = np.clip((img * 255) if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
    img_uint8 = np.transpose(img_uint8, (1, 2, 0))
    save_path = os.path.join(save_dir, f'image_{i+1:02d}.png')
    imageio.imwrite(save_path, img_uint8)