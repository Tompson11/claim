import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))
from lidar_image_align import generate_lidar_depth, generate_lidar_intensity
# from lidar_image_align_ceres import optimize_3d2d
from utils.loss_utils import batch_project_points_to_image, get_gradient
import utils.transform_utils as transform_utils
from dataset import FrameData
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from typing import List, Tuple, Union, Callable, NamedTuple
from tqdm import tqdm

class FrameData:
    pass

def _get_grid_values(ranges : torch.Tensor, resolutions : torch.Tensor) -> torch.Tensor:
    ranges = torch.tensor(ranges)
    assert len(ranges.shape) > 0
    if len(ranges.shape) == 1:
        ranges = ranges[None, ...]
    assert ranges.shape[1] == 2

    resolutions = torch.tensor(resolutions)
    if len(resolutions.shape) > 1:
        resolutions = resolutions.squeeze()
    assert len(resolutions.shape) == 1 and resolutions.shape[0] == ranges.shape[0]

    def _get_samples(start, end, resolution):
        sample_nums = int((end - start) / resolution + 1 + 0.5)
        return torch.linspace(start, end, sample_nums)
    
    N_dim = ranges.shape[0]
    samples = [_get_samples(ranges[i, 0], ranges[i, 1], resolutions[i]) for i in range(N_dim)]
    return samples

def perturb_transform(
    T_init : torch.Tensor,
    rotation_ranges : torch.Tensor = None, # (1, 2) or (3, 2), "xyz order"
    translation_ranges : torch.Tensor = None, # (1, 2) or (3, 2), "xyz order"
    seed : int = None
) -> torch.Tensor:
    if rotation_ranges is None and translation_ranges is None:
        return T_init
    
    if seed is None:
        rng = None
    else:
        rng = torch.Generator(device=T_init.device)
        rng.manual_seed(seed)

    T_out = T_init.clone()
    batch_size = T_init.shape[0] if len(T_init.shape) == 3 else 1
    device = T_init.device

    if rotation_ranges is not None:
        rotation_ranges = torch.deg2rad(torch.tensor(rotation_ranges, device = device))
        if len(rotation_ranges.shape) == 1:
            rotation_ranges = rotation_ranges[None, ...]
        assert rotation_ranges.shape[0] == 1 or rotation_ranges.shape[0] == 3
        if rotation_ranges.shape[1] == 1: # perturb specific value:
            if rotation_ranges.shape[0] == 1:
                euler_angles = rotation_ranges.repeat(1, 3)
            else:
                euler_angles = rotation_ranges

            R_pert = transform_utils.euler_angles_to_matrix(euler_angles, "XYZ")
            T_out[..., :3, :3] = torch.matmul(R_pert, T_out[..., :3, :3])
        else: # perturb random value:
            euler_angles = torch.rand((batch_size, 3), generator = rng, device = device) * (rotation_ranges[:, 1] - rotation_ranges[:, 0]) + rotation_ranges[:, 0]
            R_pert = transform_utils.euler_angles_to_matrix(euler_angles, "XYZ")
            T_out[..., :3, :3] = torch.matmul(R_pert, T_out[..., :3, :3])

    if translation_ranges is not None:
        translation_ranges = torch.tensor(translation_ranges, device = device)
        if len(translation_ranges.shape) == 1:
            translation_ranges = translation_ranges[None, ...]
        assert translation_ranges.shape[0] == 1 or translation_ranges.shape[0] == 3
        if translation_ranges.shape[1] == 1: # perturb specific value:
            if translation_ranges.shape[0] == 1:
                t_pert = translation_ranges.repeat(1, 3)
            else:
                t_pert = translation_ranges
            T_out[..., :3, 3] = T_out[..., :3, 3] + t_pert
        else: # perturb random value:
            t_pert = torch.rand((batch_size, 3), generator = rng, device = device) * (translation_ranges[:, 1] - translation_ranges[:, 0]) + translation_ranges[:, 0]
            T_out[..., :3, 3] = T_out[..., :3, 3] + t_pert
    
    return T_out

def grid_search_transform(
    T_init : torch.Tensor,
    loss_func : Tuple[Callable, ...],
    frame_data : Tuple[FrameData, ...],
    rotation_ranges : torch.Tensor = None, # (1, 2) or (3, 2), "xyz order"
    rotation_resolutions : torch.Tensor = None, # (1, ) or (3, )
    translation_ranges : torch.Tensor = None, # (1, 2) or (3, 2), "xyz order"
    translation_resolutions : torch.Tensor = None, # (1, ) or (3, )
    batch_size : int = 200,
    show_progress : bool = True,
    loss_func_intensity : Tuple[Callable, ...] = None,
    callback : Callable = None
) -> torch.Tensor :
    if rotation_ranges is None and translation_ranges is None:
        return T_init
    if rotation_ranges is not None and rotation_resolutions is None:
        return T_init
    if translation_ranges is not None and translation_resolutions is None:
        return T_init

    desc = None
    if rotation_ranges is not None:
        angle_values = _get_grid_values(rotation_ranges, rotation_resolutions)
        assert len(angle_values) == 1 or len(angle_values) == 3
        if len(angle_values) == 1:
            angle_values += [angle_values[-1], angle_values[-1]]
        desc = "rotation"
    else:
        angle_values = [torch.zeros((1, ))] * 3
    
    if translation_ranges is not None:
        trans_values = _get_grid_values(translation_ranges, translation_resolutions)
        assert len(trans_values) == 1 or len(trans_values) == 3
        if len(trans_values) == 1:
            trans_values += [trans_values[-1], trans_values[-1]]
        desc = "translation" if desc is None else "both"
    else:
        trans_values = [torch.zeros((1, ))] * 3
    

    device = T_init.device
    angle_values = list(map(lambda x : torch.deg2rad(x).to(device), angle_values))
    trans_values = list(map(lambda x : x.to(device), trans_values))

    R_init = T_init[:3, :3]
    t_init = T_init[:3, 3]

    grid_list = torch.meshgrid(*angle_values, *trans_values, indexing='ij')
    grid_list = list(map(lambda x : x.reshape(-1, 1), grid_list))
    grid_list = torch.cat(grid_list, dim = -1)
    
    init_angles = transform_utils.matrix_to_euler_angles(R_init, "XYZ")

    N = grid_list.shape[0]
    T_list = torch.zeros((N, 4, 4), device=device)
    T_list[:, 3, 3] = 1.0
    T_list[:, :3, 3] = grid_list[:, 3:] + t_init
    T_list[:, :3, :3] = transform_utils.euler_angles_to_matrix(grid_list[:, :3] + init_angles, "XYZ")

    best_T = T_init.clone()
    best_score = torch.inf
    batch_num = (N + batch_size - 1) // batch_size
    for i in tqdm(range(batch_num), desc=f"grid searching ({desc})", disable=(not show_progress)):
        start = i * batch_size
        end = min(N, start + batch_size)
        B = end - start

        total_loss = 0.0
        for data in frame_data:
            # unzip frame data
            image = data.image
            points_torch = data.pointcloud
            mono_depth_torch = data.mono_depth
            K_torch = data.K
            inv_depth = data.is_mono_inv_depth

            # project points to image
            H, W, _ = image.shape
            P = points_torch.shape[0]
            uvd, _ = batch_project_points_to_image(
                points_torch, image, T_list[start:end], K_torch, return_mask=False)
            torch.cuda.synchronize()

            lidar_depths = generate_lidar_depth(uvd, P, B, H, W, inv_depth)
            torch.cuda.synchronize()

            # calculate loss
            for func in loss_func:
                total_loss += func(lidar_depths, mono_depth_torch)

            if loss_func_intensity is not None and data.intensity is not None:
                intensity_torch = data.intensity
                image_gray_torch = data.image_gray
                lidar_intensity = generate_lidar_intensity(uvd, intensity_torch, P, B, H, W)
                torch.cuda.synchronize()
                for func in loss_func_intensity:
                    int_loss = func(lidar_intensity, image_gray_torch)
                    total_loss += int_loss
            
            torch.cuda.synchronize()

        min_index = torch.argmin(total_loss)
        min_score = total_loss[min_index]
        if min_score < best_score:
            best_score = min_score
            best_index = min_index + start
            best_T = T_list[best_index]

        if callback is not None:
            callback(i, batch_num, best_T)
        
    return best_T

def random_search_transform(
    T_init : torch.Tensor,
    loss_func : Tuple[Callable, ...],
    frame_data : Tuple[FrameData, ...],
    iterations : int,
    mode : int, # 0: coarse, 1: fine
    translation_ranges : torch.Tensor = None, # (1, 2) or (3, 2), "xyz order"
    translation_perturb_at_opt : bool = True,
    show_progress : bool = True,
    seed : int = None,
    loss_func_intensity : Tuple[Callable, ...] = None,
    callback : Callable = None
) -> torch.Tensor:    
    if seed is None:
        rng = None
    else:
        rng = torch.Generator(device=T_init.device)
        rng.manual_seed(seed)
    
    device = T_init.device
    desc = None

    if translation_ranges is not None:
        translation_ranges = torch.tensor(translation_ranges, device = device)
        if len(translation_ranges.shape) == 1:
            translation_ranges = translation_ranges[None, ...]
        assert translation_ranges.shape[0] == 1 or translation_ranges.shape[0] == 3
        assert translation_ranges.shape[1] == 2
        desc = "translation" if desc is None else "both"

    
    R_init = T_init[:3, :3]
    t_init = T_init[:3, 3]

    best_T = T_init.clone()
    best_score = np.inf
    batch_size = 6 ** 3
    B = batch_size
    T_list = torch.zeros((B, 4, 4), device=device)
    T_list[:, 3, 3] = 1.0
    T_list[:, :3, :3] = R_init
    T_list[:, :3, 3] = t_init

    ang_list = torch.deg2rad(torch.tensor([-0.5, -0.2, -0.1, 0.5, 0.2, 0.1])).float().cuda()
    if mode == 1:
        ang_list /= 5.0
    grid_list = torch.meshgrid(ang_list, ang_list, ang_list, indexing='ij')
    grid_list = list(map(lambda x : x.reshape(-1, 1), grid_list))
    grid_list = torch.cat(grid_list, dim = -1)

    for iter in tqdm(range(iterations), desc=f"random searching ({desc})", disable=(not show_progress)):
        cur_angle = transform_utils.matrix_to_euler_angles(best_T[:3, :3], "XYZ")
        T_list[:, :3, :3] = transform_utils.euler_angles_to_matrix(cur_angle + grid_list, "XYZ")
        
        if translation_ranges is not None:
            trans_values = torch.rand((batch_size // 2, 3), generator=rng, device = device) * (translation_ranges[:, 1] - translation_ranges[:, 0]) + translation_ranges[:, 0]
            if translation_perturb_at_opt:
                T_list[:batch_size // 2, :3, 3] = trans_values + best_T[:3, 3]
                T_list[batch_size // 2:, :3, 3] = trans_values + best_T[:3, 3]
            else:
                T_list[:batch_size // 2, :3, 3] = trans_values + t_init
                T_list[batch_size // 2:, :3, 3] = trans_values + t_init

        
        total_loss = 0.0
        for data in frame_data:
            # unzip frame data
            image = data.image
            points_torch = data.pointcloud
            mono_depth_torch = data.mono_depth
            K_torch = data.K
            inv_depth = data.is_mono_inv_depth

            # project points to image
            H, W, _ = image.shape
            P = points_torch.shape[0]
            uvd, _ = batch_project_points_to_image(
                points_torch, image, T_list, K_torch, return_mask=False)
            torch.cuda.synchronize()

            lidar_depths = generate_lidar_depth(uvd, P, B, H, W, inv_depth)
            torch.cuda.synchronize()

            # calculate loss
            for func in loss_func:
                total_loss += func(lidar_depths, mono_depth_torch)
            
            if loss_func_intensity is not None and data.intensity is not None:
                intensity_torch = data.intensity
                image_gray_torch = data.image_gray
                lidar_intensity = generate_lidar_intensity(uvd, intensity_torch, P, B, H, W)
                torch.cuda.synchronize()

                for func in loss_func_intensity:
                    int_loss = func(lidar_intensity, image_gray_torch)
                    total_loss += int_loss

            torch.cuda.synchronize()
            
        min_index = torch.argmin(total_loss)
        min_score = total_loss[min_index]
        if min_score < best_score:
            best_score = min_score
            best_T[...] = T_list[min_index] # do not touch

        if callback is not None:
            callback(iter, iterations, best_T)

    return best_T


def random_search_transform2(
    T_init : torch.Tensor,
    loss_func : Tuple[Callable, ...],
    frame_data : Tuple[FrameData, ...],
    iterations : int,
    rotation_ranges : torch.Tensor = None, # (1, 2) or (3, 2), "xyz order"
    rotation_perturb_at_opt : bool = True,
    translation_ranges : torch.Tensor = None, # (1, 2) or (3, 2), "xyz order"
    translation_perturb_at_opt : bool = True,
    rotation_candidate_num : int = 3,
    show_progress : bool = True,
    seed : int = None,
    loss_func_intensity : Tuple[Callable, ...] = None,
    callback : Callable = None
) -> torch.Tensor:
    if rotation_ranges is None and translation_ranges is None:
        return T_init
    
    if seed is None:
        rng = None
    else:
        rng = torch.Generator(device=T_init.device)
        rng.manual_seed(seed)
    
    device = T_init.device
    desc = None
    if rotation_ranges is not None:
        rotation_ranges = torch.deg2rad(torch.tensor(rotation_ranges, device = device))
        if len(rotation_ranges.shape) == 1:
            rotation_ranges = rotation_ranges[None, ...]
        assert rotation_ranges.shape[0] == 1 or rotation_ranges.shape[0] == 3
        assert rotation_ranges.shape[1] == 2
        assert torch.all(rotation_ranges[:, 0] == -rotation_ranges[:, -1])
        desc = "rotation"

    if translation_ranges is not None:
        translation_ranges = torch.tensor(translation_ranges, device = device)
        if len(translation_ranges.shape) == 1:
            translation_ranges = translation_ranges[None, ...]
        assert translation_ranges.shape[0] == 1 or translation_ranges.shape[0] == 3
        assert translation_ranges.shape[1] == 2
        desc = "translation" if desc is None else "both"

    
    R_init = T_init[:3, :3]
    t_init = T_init[:3, 3]

    best_T = T_init.clone()
    best_score = np.inf
    batch_size = (rotation_candidate_num * 2) ** 3
    B = batch_size
    T_list = torch.zeros((B, 4, 4), device=device)
    T_list[:, 3, 3] = 1.0
    T_list[:, :3, :3] = R_init
    T_list[:, :3, 3] = t_init
    
    for iter in tqdm(range(iterations), desc=f"random searching ({desc})", disable=(not show_progress)):        
        if rotation_ranges is not None:
            ang_list = torch.rand((3, rotation_candidate_num), generator=rng, device = device) * rotation_ranges[:, 1]
            ang_list = torch.concat([-ang_list, ang_list], dim=-1)
            ang_values = torch.meshgrid(ang_list[0], ang_list[1], ang_list[2], indexing='ij')
            ang_values = list(map(lambda x : x.reshape(-1, 1), ang_values))
            ang_values = torch.cat(ang_values, dim = -1)
            
            if rotation_perturb_at_opt:
                cur_angle = transform_utils.matrix_to_euler_angles(best_T[:3, :3], "XYZ")
            else:
                cur_angle = transform_utils.matrix_to_euler_angles(R_init, "XYZ")
                
            T_list[:, :3, :3] = transform_utils.euler_angles_to_matrix(cur_angle + ang_values, "XYZ")

        if translation_ranges is not None:
            trans_values = torch.rand((batch_size // 2, 3), generator=rng, device = device) * (translation_ranges[:, 1] - translation_ranges[:, 0]) + translation_ranges[:, 0]
            if translation_perturb_at_opt:
                T_list[:batch_size // 2, :3, 3] = trans_values + best_T[:3, 3]
                T_list[batch_size // 2:, :3, 3] = trans_values + best_T[:3, 3]
            else:
                T_list[:batch_size // 2, :3, 3] = trans_values + t_init
                T_list[batch_size // 2:, :3, 3] = trans_values + t_init
        
        total_loss = 0.0
        for data in frame_data:
            # unzip frame data
            image = data.image
            points_torch = data.pointcloud
            mono_depth_torch = data.mono_depth
            K_torch = data.K
            inv_depth = data.is_mono_inv_depth

            # project points to image
            H, W, _ = image.shape
            P = points_torch.shape[0]
            uvd, _ = batch_project_points_to_image(
                points_torch, image, T_list, K_torch, return_mask=False)
            torch.cuda.synchronize()
            
            lidar_depths = generate_lidar_depth(uvd, P, B, H, W, inv_depth)
            torch.cuda.synchronize()

            # calculate loss
            for func in loss_func:
                total_loss += func(lidar_depths, mono_depth_torch)

            if loss_func_intensity is not None and data.intensity is not None:
                intensity_torch = data.intensity
                image_gray_torch = data.image_gray
                lidar_intensity = generate_lidar_intensity(uvd, intensity_torch, P, B, H, W)
                torch.cuda.synchronize()

                for func in loss_func_intensity:
                    int_loss = func(lidar_intensity, image_gray_torch)
                    total_loss += int_loss
            
            torch.cuda.synchronize()

        min_index = torch.argmin(total_loss)
        min_score = total_loss[min_index]
        if min_score < best_score:
            best_score = min_score
            best_T[...] = T_list[min_index] # do not touch

        if callback is not None:
            callback(iter, iterations, best_T)
        
    # print(best_score)
    return best_T

