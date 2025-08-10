from .Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
import os
import torch

MODEL = {}
SUPPORTED_MODEL = ["depth_anything_v2"]

def get_model(model_name):
    if model_name in MODEL:
        return MODEL[model_name]

    if model_name not in SUPPORTED_MODEL:
        return None    

    if model_name == "depth_anything_v2":
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = 'vitl'
        model = DepthAnythingV2(**model_configs[encoder])
        model_ckpt_path = os.path.join(os.path.dirname(__file__), "Depth_Anything_V2/ckpt")
        state_dict = torch.load(f'{model_ckpt_path}/depth_anything_v2_{encoder}.pth', map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to(DEVICE).eval()
        result = {
            "forward" : lambda x : model.infer_image(x, input_size=518),
            "inv_depth" : True
        }
        MODEL[model_name] = result
        return result


