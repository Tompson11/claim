from typing import NamedTuple
import numpy as np
import torch

class FrameData(NamedTuple):
    frame_id : str
    image: np.array
    pointcloud: torch.Tensor
    mono_depth: torch.Tensor
    K: torch.Tensor
    T_gt : torch.Tensor = None
    intensity: torch.Tensor = None
    image_gray : torch.tensor = None
    is_mono_inv_depth : bool = True