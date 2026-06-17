import sys
import os
import numpy as np
import h5py
import torch
import pickle
import argparse
from skimage.transform import resize
import random
from tqdm import tqdm
import random
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model

# open_clip.list_pretrained()

import os
import numpy as np
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

import clip
import open_clip
from torchvision.models import resnet50
from huggingface_hub import hf_hub_download


import prf_utils

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

setattr(LayerSize, "ADV_RN50_layer_size", {
    'relu': [64, 112, 112],
    'maxpool': [64, 56, 56],
    'layer1': [256, 56, 56],
    'layer2': [512, 28, 28],
    'layer3': [1024, 14, 14],
    'layer4': [2048, 7, 7]
})

setattr(LayerSize, "OPEN_CLIP_CONVNEXT_BASE_layer_size", {
    'stem': [128, 56, 56],
    'stage1': [128, 56, 56],
    'stage2': [256, 28, 28],
    'stage3': [512, 14, 14],
    'stage4': [1024, 7, 7]
})

class Normalize:
    pass

setattr(Normalize, "CLIP_RN50_MEAN", np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.single)[:, None, None])
setattr(Normalize, "CLIP_RN50_STD", np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.single)[:, None, None])

setattr(Normalize, "OPEN_CLIP_RN50_MEAN", np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.single)[:, None, None])
setattr(Normalize, "OPEN_CLIP_RN50_STD", np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.single)[:, None, None])

setattr(Normalize, "DINO_RN50_MEAN", np.array((0.485, 0.456, 0.406), dtype=np.single)[:, None, None])
setattr(Normalize, "DINO_RN50_STD", np.array((0.229, 0.224, 0.225), dtype=np.single)[:, None, None])

setattr(Normalize, "SIMCLR_RN50_MEAN", np.array((0.485, 0.456, 0.406), dtype=np.single)[:, None, None])
setattr(Normalize, "SIMCLR_RN50_STD", np.array((0.229, 0.224, 0.225), dtype=np.single)[:, None, None])

setattr(Normalize, "ADV_RN50_MEAN", np.array((0.485, 0.456, 0.406), dtype=np.single)[:, None, None])
setattr(Normalize, "ADV_RN50_STD", np.array((0.229, 0.224, 0.225), dtype=np.single)[:, None, None])

setattr(Normalize, "OPEN_CLIP_CONVNEXT_BASE_MEAN", np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.single)[:, None, None])
setattr(Normalize, "OPEN_CLIP_CONVNEXT_BASE_STD", np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.single)[:, None, None])


def normalize_image(input_ndarray, args):
    # print(input_ndarray.dtype, np.max(input_ndarray), np.min(input_ndarray), input_ndarray.shape)
    # exit()
    input_ndarray = input_ndarray.transpose((1, 2, 0))
    image_resized = resize(input_ndarray, (224, 224), preserve_range=True)
    scaled_image = image_resized.astype(np.single).transpose((2, 0, 1))/(255.0) #*random.uniform(0.95, 1.05)
    # print(scaled_image.shape, OPENAI_CLIP_STD.shape, OPENAI_CLIP_MEAN.shape, "SHAPES")
    return (scaled_image-getattr(Normalize, f"{args.model_name}_MEAN"))/getattr(Normalize, f"{args.model_name}_STD")


class nsd_img_loader(torch.utils.data.Dataset):
    def __init__(self, stim_folder, args):
        self.args = args
        self.ss = args.subject_id[0]
        self.n_pix = 224
        self.image_filename = os.path.join(stim_folder, f'S{self.ss}_stimuli_{self.n_pix}.h5py')
        print(f'Loading images from {self.image_filename}')
        with h5py.File(self.image_filename, 'r') as data_set:
            values = np.copy(data_set['/stimuli'])
            data_set.close()

        # get images ready for feature extraction
        image_array = values.astype(np.float32) / 255
        self.image_array = image_array
        self.transform = normalize_image


    def __len__(self):
        return len(self.image_array)


    def __getitem__(self, idx):
        image_data = self.image_array[idx]
        # image_data = np.expand_dims(image_data, axis=0)
        if self.transform:
            image_data = self.transform(image_data, self.args)
        image_data = torch.from_numpy(image_data)
        return image_data



# Extract intermediate features from a CNN model
def extract_intermediate_features_pRF(data_loader, prf_gaussians, args, num_imgs, device):
    """
    # Input: 
    # - data_loader: neural loader
    # - prf_gaussian: all prf gaussian array

    # Output:
    # prf_features:array in shape of [n_images, n_channels, n_prfs] # TODO: save separately for each prf
    # neural_rsp: array in shape of [n_images, n_neurons] # TODO: don't need to save this, use load_nsd_data function in fit_model_nsd_estimateprf.py to load this
    """
    
    ##########################################################
    # Load the CNN backbone
    ##########################################################
    model_name = args.model_name
    if model_name == "CLIP_RN50":
        visual_encoder, _ = clip.load(model_name, device=device)
        CNN_backbone = visual_encoder.visual.to(device)

        del visual_encoder.transformer
        torch.cuda.empty_cache()
        
        assert not visual_encoder.training

        return_nodes = {
            # 'relu1': 'relu1', #torch.Size([1, 32, 112, 112])
            # 'relu2': 'relu2', #torch.Size([1, 32, 112, 112])
            'relu3': 'relu3', #torch.Size([1, 64, 112, 112])
            # 'avgpool': 'avgpool', #torch.Size([1, 64, 56, 56])
            'layer1': 'layer1', #torch.Size([1, 256, 56, 56])
            'layer2': 'layer2', #torch.Size([1, 512, 28, 28])
            'layer3': 'layer3', #torch.Size([1, 1024, 14, 14])
            'layer4': 'layer4' #torch.Size([1, 2048, 7, 7])
        }
    
    elif model_name == "OPEN_CLIP_RN50":
        model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='yfcc15m')
        CNN_backbone = model.visual
        del model.transformer
        torch.cuda.empty_cache()

        # train_nodes, val_nodes = get_graph_node_names(CNN_backbone)
        
        return_nodes = {
            # 'act1': 'act1', #torch.Size([1, 32, 112, 112])
            # 'act2': 'act2', #torch.Size([1, 32, 112, 112])
            'act3': 'act3', #torch.Size([1, 64, 112, 112])
            # 'avgpool': 'avgpool', #torch.Size([1, 64, 56, 56])
            'layer1': 'layer1', #torch.Size([1, 256, 56, 56])
            'layer2': 'layer2', #torch.Size([1, 512, 28, 28])
            'layer3': 'layer3', #torch.Size([1, 1024, 14, 14])
            'layer4': 'layer4' #torch.Size([1, 2048, 7, 7])
        }
    
    elif model_name == "DINO_RN50":
        CNN_backbone = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        return_nodes = {
            'relu': 'relu', #torch.Size([1, 64, 112, 112])
            # 'maxpool': 'maxpool', #torch.Size([1, 64, 56, 56])
            'layer1': 'layer1', #torch.Size([1, 256, 56, 56])
            'layer2': 'layer2', #torch.Size([1, 512, 28, 28])
            'layer3': 'layer3', #torch.Size([1, 1024, 14, 14])
            'layer4': 'layer4' #torch.Size([1, 2048, 7, 7])
        }
    
    elif model_name == "SIMCLR_RN50":
        repo_id = "lightly-ai/simclrv1-imagenet1k-resnet50-1x"
        filename = "resnet50-1x.pth"

        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
        ckpt = torch.load(ckpt_path, map_location="cpu")

        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        CNN_backbone = resnet50(weights=None)

        CNN_backbone.load_state_dict(state_dict, strict=True)
        return_nodes = {
            'relu': 'relu', #torch.Size([1, 64, 112, 112])
            # 'maxpool': 'maxpool', #torch.Size([1, 64, 56, 56])
            'layer1': 'layer1', #torch.Size([1, 256, 56, 56])
            'layer2': 'layer2', #torch.Size([1, 512, 28, 28])
            'layer3': 'layer3', #torch.Size([1, 1024, 14, 14])
            'layer4': 'layer4' #torch.Size([1, 2048, 7, 7])
        }

    elif model_name == "ADV_RN50":
        dataset = ImageNet("/tmp")  # dummy path

        wrapped_model, _ = make_and_restore_model(
            arch="resnet50",
            dataset=dataset,
            resume_path="/user_data/junruz/prf_features/imagenet_l2_3_0.pt",
            parallel=False,
        )

        # Extract weights from the MadryLab ResNet
        robust_state_dict = {
            key: value.detach().cpu()
            for key, value in wrapped_model.model.state_dict().items()
        }

        # Create a normal torchvision ResNet-50
        CNN_backbone = resnet50(weights=None)

        # The parameter names and shapes should match exactly
        CNN_backbone.load_state_dict(robust_state_dict, strict=True)

        # CNN_backbone = wrapped_model.model
        return_nodes = {
            'relu': 'relu', #torch.Size([1, 64, 112, 112])
            # 'maxpool': 'maxpool', #torch.Size([1, 64, 56, 56])
            'layer1': 'layer1', #torch.Size([1, 256, 56, 56])
            'layer2': 'layer2', #torch.Size([1, 512, 28, 28])
            'layer3': 'layer3', #torch.Size([1, 1024, 14, 14])
            'layer4': 'layer4' #torch.Size([1, 2048, 7, 7])
        }

    elif model_name == "OPEN_CLIP_CONVNEXT_BASE":
        model, _, preprocess = open_clip.create_model_and_transforms('convnext_base', pretrained='laion400m_s13b_b51k')
        CNN_backbone = model.visual
        CNN_backbone.eval()
        # train, eval = get_graph_node_names(CNN_backbone)
        return_nodes = {
            "trunk.stem.1.permute_1": "stem",
            "trunk.stages.0.blocks.2.add": "stage1",
            "trunk.stages.1.blocks.2.add": "stage2",
            "trunk.stages.2.blocks.26.add": "stage3",
            "trunk.stages.3.blocks.2.add": "stage4"
        }


    
    feature_extractor = create_feature_extractor(CNN_backbone, return_nodes=return_nodes)
    for name, param in feature_extractor.named_parameters():
        param.requires_grad = False

    feature_extractor.eval().to(device)


    feature_extractor = create_feature_extractor(CNN_backbone, return_nodes=return_nodes)
    # out = feature_extractor(torch.rand(1, 3, 224, 224).to(device))
    # print([(k, v.shape) for k, v in out.items()])
    print(f"Created {args.model_name} and moved to {device}, feature extractor created")


    ##########################################################
    # Extract intermediate features
    ##########################################################
    features_prf = []
    prf_gaussians = torch.from_numpy(prf_gaussians).to(device)
    print(f"Extracting features for {num_imgs} images")
    for step, img_data in tqdm(enumerate(data_loader)):
            img_data = img_data.to(device) # the 1st dim is zero here
            
            with torch.no_grad():
                feature = feature_extractor(img_data)
                feature = feature[args.layer_name].float() # shape: [B, C, H, W]
                # features = features/(features.norm(dim=-1, keepdim=True)+1e-10)

                # Multiply the features by the pRF gaussian
                # Shape of prf_stack: [H, W, n_prfs]
                feature_prf = torch.einsum('bchw,hwn->bcn', feature, prf_gaussians)
                feature_prf = feature_prf.detach().cpu().numpy()
                features_prf.append(feature_prf)
    
    
    
    features_prf = np.concatenate(features_prf, axis=0)

    return features_prf


# Create a pRF parameter grid and create gaussian based on pRF parameters
def create_prf_gaussian(args, which_prf_grid='default-log-polar'):
    # models, grid_name = prf_utils.get_prf_grid(grid_name = which_prf_grid)
    n_pix = getattr(LayerSize, f"{args.model_name}_layer_size")[args.layer_name][-1]
    prf_gaussian = prf_utils.get_prf_stack(n_pix=n_pix, which_prf_grid = which_prf_grid)

    return prf_gaussian


def save_features_prf(features_prf, args, dtype=np.float32):
    output_folder = os.path.join(args.save_root, f"S{args.subject_id[0]}")
    output_folder = os.path.join(output_folder, args.model_name)
    output_folder = os.path.join(output_folder, args.layer_name)
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving features for {features_prf.shape[-1]} pRFs")
    features_prf = features_prf.astype(dtype)
    for prf_idx in tqdm(range(features_prf.shape[-1])):
        np.save(os.path.join(output_folder, f'features_prf_{prf_idx}.npy'), features_prf[:,:,prf_idx])



def main():
    # Set some paths: where the preprocessed NSD files live
    nsd_path = '/lab_data/hendersonlab/datasets/nsd_preproc'
    data_folder = os.path.join(nsd_path, 'data')
    stim_folder = os.path.join(nsd_path, 'stimuli')

    n_pix = 224

    # Create namespace and set attributes
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', nargs='+', default=[1], type=int)
    parser.add_argument('--neural_activity_path', type=str, default=os.path.join(data_folder, 'S{}_betas_avg_bigmask_dict.hdf5'))
    parser.add_argument('--image_path', type=str, default=os.path.join(stim_folder, 'S{}_stimuli_%d_dict.hdf5'%n_pix))
    parser.add_argument('--stim_keys_path', type=str, default=os.path.join(stim_folder, "all_keys.pkl"))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_name', type=str, default="OPEN_CLIP_CONVNEXT_BASE")
    parser.add_argument('--layer_name', type=str, default="stage4")
    parser.add_argument('--save_root', type=str, default="/user_data/junruz/prf_features")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ss = args.subject_id[0]

    ##########################################################
    # Instantiate the data loader
    ##########################################################
    loader = nsd_img_loader(stim_folder, args)
    num_images = len(loader)
    neural_train_loader = torch.utils.data.DataLoader(loader, \
                                                batch_size=args.batch_size, \
                                                shuffle=False, \
                                                num_workers=0, \
                                                drop_last=False)
    prf_gaussian = create_prf_gaussian(args, which_prf_grid='default-log-polar')
    print(f"Using {args.layer_name} of {args.model_name}, pRF Gaussian mask shape: {prf_gaussian.shape}")
    features_prf = extract_intermediate_features_pRF(neural_train_loader, prf_gaussian, args, num_images, device)
    print(f"Features shape: {features_prf.shape}")
    save_features_prf(features_prf, args)



if __name__ == "__main__":
    main()
