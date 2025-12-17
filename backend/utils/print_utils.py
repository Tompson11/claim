import torch
import utils.transform_utils as transform_utils
from scipy.spatial.transform import Rotation as R
import numpy as np

def print_transform_in_euler_and_translation(T, outline=None, digit=4):
    if isinstance(T, torch.Tensor):
        eulers = torch.rad2deg(
            transform_utils.matrix_to_euler_angles(T[:3, :3], "XYZ"))
    else:
        eulers = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)

    trans = T[:3, 3]

    if outline == "initial":
        print("------------- ⛳ Initial ⛳ -------------")
    elif outline == "final":
        print("------------- 🎯  Final  🎯 -------------")
    else:
        pass

    print(f"rot: [{eulers[0]:.{digit}f}, {eulers[1]:.{digit}f}, {eulers[2]:.{digit}f}]")
    print(f"pos: [{trans[0]:.{digit}f}, {trans[1]:.{digit}f}, {trans[2]:.{digit}f}]")        

    if outline is not None:
        print("-----------------------------------------")
