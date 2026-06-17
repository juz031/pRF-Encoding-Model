import torch
import torch.nn as nn
from torchvision.models import resnet50
from huggingface_hub import hf_hub_download
import open_clip
open_clip.list_pretrained()
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

model, _, preprocess = open_clip.create_model_and_transforms('convnext_base', pretrained='laion400m_s13b_b51k')
CNN_model = model.visual
CNN_model.eval()
train, eval = get_graph_node_names(CNN_model)
return_nodes = {
    "trunk.stem.1.permute_1": "stem",
    "trunk.stages.0.blocks.2.add": "stage1",
    "trunk.stages.1.blocks.2.add": "stage2",
    "trunk.stages.2.blocks.26.add": "stage3",
    "trunk.stages.3.blocks.2.add": "stage4"
}
# print(node_names)
extractor = create_feature_extractor(CNN_model, return_nodes=return_nodes)
# print(extractor)
pass


# repo_id = "lightly-ai/simclrv1-imagenet1k-resnet50-1x"
# filename = "resnet50-1x.pth"

# # 1. Download checkpoint from Hugging Face
# ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)

# # 2. Load checkpoint
# ckpt = torch.load(ckpt_path, map_location="cpu")

# # Some checkpoints are raw state_dict; some wrap it under "state_dict"
# state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

# # 3. Clean common prefixes, in case they exist
# # clean_state_dict = {}
# # for k, v in state_dict.items():
# #     k = k.replace("module.", "")
# #     k = k.replace("backbone.", "")
# #     k = k.replace("encoder.", "")
# #     k = k.replace("encoder_q.", "")
# #     clean_state_dict[k] = v

# # 4. Build matching architecture
# model = resnet50(weights=None)

# # 5. Load weights
# missing, unexpected = model.load_state_dict(state_dict, strict=True)

# print("Missing keys:", missing)
# print("Unexpected keys:", unexpected)