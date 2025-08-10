import torch
import utils.transform_utils as transform_utils
from scipy.spatial.transform import Rotation as R

def print_transform_in_euler_and_translation(T):
    if isinstance(T, torch.Tensor):
        eulers = torch.rad2deg(transform_utils.matrix_to_euler_angles(T[:3, :3], "XYZ"))
    else:
        eulers = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    
    print("rot: ", eulers)
    print("pos: ", T[:3, 3:4].transpose(1, 0).squeeze())