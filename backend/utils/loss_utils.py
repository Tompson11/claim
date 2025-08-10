import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))

from lidar_image_align import depth_sim_loss, depth_sim_loss2, generate_lidar_depth, generate_histogram
from typing import NamedTuple
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import matplotlib.pyplot as plt
import json
import torch
import cv2
import open3d as o3d
import os

def batch_project_points_to_image(points: torch.Tensor, image: np.array, T_l2c: torch.Tensor, K: torch.Tensor, return_mask: bool = True):
    points_cam = torch.matmul(points, T_l2c[:, :3, :3].transpose(
        2, 1)) + T_l2c[:, :3, 3:4].transpose(2, 1)
    uvs = torch.matmul(points_cam, K.transpose(1, 0)) / points_cam[:, :, -1:]
    uvs[:, :, -1] = points_cam[:, :, -1]

    if return_mask:
        h, w, _ = image.shape
        mask_d = points_cam[:, :, -1] > 0.02
        mask_u = torch.logical_and(uvs[:, :, 0] >= 0, uvs[:, :, 0] < w)
        mask_v = torch.logical_and(uvs[:, :, 1] >= 0, uvs[:, :, 1] < h)
        mask_uv = torch.logical_and(mask_u, mask_v)
        mask = torch.logical_and(mask_d, mask_uv)
    else:
        mask = None

    return uvs, mask

def get_gradient(mono_depth_torch, ksize=3, x_grad=True, y_grad=True, normalize = False):
    depth = mono_depth_torch.contiguous().cpu().numpy()

    if x_grad:
        sobelx = np.abs(cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=ksize))
        # sobelx = cv2.convertScaleAbs(sobelx)
    else:
        sobelx = np.zeros(depth.shape, dtype=np.float64)
    
    if y_grad:
        sobely = np.abs(cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=ksize))
        # sobely = cv2.convertScaleAbs(sobely)
    else:
        sobely = np.zeros(depth.shape, dtype=np.float64)

    sobelxy2 = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

    if normalize:
        v_max = np.max(sobelxy2)
        v_min = np.min(sobelxy2)
        sobelxy2 = (sobelxy2 - v_min) / (v_max - v_min)

    depth_torch = torch.from_numpy(sobelxy2).to(mono_depth_torch.device)
    return depth_torch

# def get_block_weight(mono_depth_torch, box_p, shift=0, ksize=3):
#     depth_torch = get_gradient(mono_depth_torch, ksize=ksize, normalize=True)

#     H, W = depth_torch.shape
#     nums_box_h = (H - shift) // box_p
#     nums_box_w = (W - shift) // box_p
#     blocks = torch.nn.functional.unfold(depth_torch[None, None, shift: shift + nums_box_h * box_p,
#                                         shift: shift + nums_box_w * box_p], (box_p, box_p), stride=box_p)[0].transpose(1, 0)
#     good_mask = torch.max(blocks, dim=-1).values > 0.5
#     weight = torch.zeros_like(good_mask).float()
#     weight[good_mask] = 1.0
#     weight[~good_mask] = 0.1
#     return weight.reshape((nums_box_h, nums_box_w))

def get_block_weight(mono_depth_torch, box_p, shift=0, ksize=3):
    depth_torch = get_gradient(mono_depth_torch, ksize=ksize, x_grad=True, y_grad=True, normalize=True)
    H, W = depth_torch.shape
    nums_box_h = (H - shift) // box_p
    nums_box_w = (W - shift) // box_p
    blocks = torch.nn.functional.unfold(depth_torch[None, None, shift: shift + nums_box_h * box_p,
                                        shift: shift + nums_box_w * box_p], (box_p, box_p), stride=box_p)[0].transpose(1, 0)
    
    good_mask = torch.max(blocks, dim=-1).values > 0.3
    weight = torch.zeros_like(good_mask).float()
    weight[good_mask] = 1.0
    weight[~good_mask] = 0.1
    return weight.reshape((nums_box_h, nums_box_w))

    # mean_dep = torch.mean(blocks, dim=-1)
    # min_dep = mean_dep.min()
    # max_dep = mean_dep.max()
    # weight = (1.0 - (mean_dep - min_dep) / (max_dep - min_dep)) * 10.0
    # return weight.reshape((nums_box_h, nums_box_w))
    
def get_block_weight2(mono_depth_torch, box_p, shift=0, ksize=3):
    depth_torch = get_gradient(mono_depth_torch, ksize=ksize, x_grad=False, y_grad=True, normalize=True)
    H, W = depth_torch.shape
    nums_box_h = (H - shift) // box_p
    nums_box_w = (W - shift) // box_p
    blocks = torch.nn.functional.unfold(depth_torch[None, None, shift: shift + nums_box_h * box_p,
                                        shift: shift + nums_box_w * box_p], (box_p, box_p), stride=box_p)[0].transpose(1, 0)
    # return 1.0 - torch.max(blocks, dim=-1).values.reshape((nums_box_h, nums_box_w))
    good_mask = torch.max(blocks, dim=-1).values > 0.5
    weight = torch.zeros_like(good_mask).float()
    weight[good_mask] = 0.1
    weight[~good_mask] = 1.0
    return weight.reshape((nums_box_h, nums_box_w))

# def get_block_weight2(mono_depth_torch, box_p, shift=0, ksize=3):
#     depth_grad_y = get_gradient(mono_depth_torch, ksize=ksize, x_grad=False, y_grad=True, normalize=True)
#     depth_grad_x = get_gradient(mono_depth_torch, ksize=ksize, x_grad=True, y_grad=False, normalize=True)
#     H, W = depth_grad_y.shape
#     nums_box_h = (H - shift) // box_p
#     nums_box_w = (W - shift) // box_p
#     blocks_y = torch.nn.functional.unfold(depth_grad_y[None, None, shift: shift + nums_box_h * box_p,
#                                         shift: shift + nums_box_w * box_p], (box_p, box_p), stride=box_p)[0].transpose(1, 0)
#     blocks_x = torch.nn.functional.unfold(depth_grad_x[None, None, shift: shift + nums_box_h * box_p,
#                                         shift: shift + nums_box_w * box_p], (box_p, box_p), stride=box_p)[0].transpose(1, 0)

#     # return 1.0 - torch.max(blocks, dim=-1).values.reshape((nums_box_h, nums_box_w))
#     good_mask = torch.logical_and(torch.max(blocks_y, dim=-1).values > 0.6, torch.max(blocks_x, dim=-1).values < 0.6)
#     weight = torch.zeros_like(good_mask).float()
#     weight[good_mask] = 0.1
#     weight[~good_mask] = 1.0
#     return weight.reshape((nums_box_h, nums_box_w))

# def get_block_weight2(mono_depth_torch, box_p, shift=0, ksize=3):
#     depth_grad_y = get_gradient(mono_depth_torch, ksize=ksize, x_grad=False, y_grad=True, normalize=True)
#     depth_grad_x = get_gradient(mono_depth_torch, ksize=ksize, x_grad=True, y_grad=False, normalize=True)
#     H, W = depth_grad_y.shape
#     nums_box_h = (H - shift) // box_p
#     nums_box_w = (W - shift) // box_p
#     blocks_y = torch.nn.functional.unfold(depth_grad_y[None, None, shift: shift + nums_box_h * box_p,
#                                         shift: shift + nums_box_w * box_p], (box_p, box_p), stride=box_p)[0].transpose(1, 0)
#     blocks_x = torch.nn.functional.unfold(depth_grad_x[None, None, shift: shift + nums_box_h * box_p,
#                                         shift: shift + nums_box_w * box_p], (box_p, box_p), stride=box_p)[0].transpose(1, 0)

#     # return 1.0 - torch.max(blocks, dim=-1).values.reshape((nums_box_h, nums_box_w))
#     good_mask = torch.max(blocks_x, dim=-1).values > 0.5
#     weight = torch.zeros_like(good_mask).float()
#     weight[good_mask] = 1.0
#     weight[~good_mask] = 0.1
#     return weight.reshape((nums_box_h, nums_box_w))

def get_sky_block_mask(mono_depth_torch, box_p, shift=0):
    H, W = mono_depth_torch.shape
    nums_box_h = (H - shift) // box_p
    nums_box_w = (W - shift) // box_p
    blocks = torch.nn.functional.unfold(mono_depth_torch[None, None, shift: shift + nums_box_h * box_p,
                                        shift: shift + nums_box_w * box_p], (box_p, box_p), stride=box_p)[0].transpose(1, 0)
    sky_mask = ~torch.any(blocks, dim = -1)
    return sky_mask.reshape((nums_box_h, nums_box_w))

def pearson_loss(lidar_depths, mono_depth, box_p, shift = 0, weight = None, set_zero_to = None, min_hit_ratio = -1.0):
    B, H, W = lidar_depths.shape
    losses = depth_sim_loss(lidar_depths, mono_depth, B, H, W, shift, box_p)
    torch.cuda.synchronize()
    if min_hit_ratio > 0 or set_zero_to is not None:
        miss_mask = (losses == 0.0)
    if set_zero_to is not None:
        sky_block_mask = get_sky_block_mask(mono_depth, box_p, shift=shift).transpose(1, 0).flatten()
        miss_mask = torch.logical_and(miss_mask, (~sky_block_mask).repeat(B, 1))
        losses[miss_mask] = set_zero_to
    if weight is not None:
        losses = losses * weight
    
    if set_zero_to is not None and set_zero_to < 0:
        good_mask = ~miss_mask
        # losses[losses > 0.4] *= 0.4
        final_loss = torch.sum(losses * good_mask, dim = -1) / torch.sum(good_mask, dim = -1)
    else:
        final_loss = torch.mean(losses, dim = -1)

    if min_hit_ratio > 0:
        block_nums = losses.shape[1]
        punish_mask = torch.sum(miss_mask, dim = -1) > (1.0 - min_hit_ratio) * block_nums
        final_loss[punish_mask] = 1.0
    return final_loss
    
def order_loss(lidar_depths, mono_depth, box_p, shift = 0, weight = None, set_zero_to = None, min_hit_ratio = -1.0):
    B, H, W = lidar_depths.shape
    losses = depth_sim_loss2(lidar_depths, mono_depth, B, H, W, shift, box_p)
    torch.cuda.synchronize()
    if min_hit_ratio > 0 or set_zero_to is not None:
        miss_mask = (losses == 0.0)
    if set_zero_to is not None:
        sky_block_mask = get_sky_block_mask(mono_depth, box_p, shift=shift).transpose(1, 0).flatten()
        miss_mask = torch.logical_and(miss_mask, (~sky_block_mask).repeat(B, 1))
        losses[miss_mask] = set_zero_to
    if weight is not None:
        losses = losses * weight
    
    if set_zero_to is not None and set_zero_to < 0:
        good_mask = ~miss_mask
        final_loss = torch.sum(losses * good_mask, dim = -1) / torch.sum(good_mask, dim = -1)
    else:
        final_loss = torch.mean(losses, dim = -1)
    
    if min_hit_ratio > 0:
        block_nums = losses.shape[1]
        punish_mask = torch.sum(miss_mask, dim = -1) > (1.0 - min_hit_ratio) * block_nums
        final_loss[punish_mask] = 1.0
    return final_loss

def NID_loss(lidar_intensity, image_intensity, bin_nums = 16):
    B, H, W = lidar_intensity.shape
    hists_tmp = generate_histogram(
        lidar_intensity,
        image_intensity,
        B, H, W, bin_nums,
    )
    torch.cuda.synchronize()
    
    hists = [torch.sum(hist, dim = 1) for hist in hists_tmp]
    P = torch.sum(hists[0], dim = -1, keepdim=True)
    hists = [hist / P for hist in hists]
    entropys = [-torch.sum(hist * (hist + 1e-6).log(), dim = -1) for hist in hists]
    mult_info = entropys[0] + entropys[1] - entropys[2]
    loss = (entropys[2] - mult_info) / entropys[2]
    return loss

def NID_loss2(lidar_intensity, image_intensity, bin_nums = 16):    
    B, H, W = lidar_intensity.shape
    d_min = torch.min(lidar_intensity.reshape(B, -1), dim = -1).values[..., None, None]
    d_max = torch.max(lidar_intensity.reshape(B, -1), dim = -1).values[..., None, None]
    lidar_intensity_new = (lidar_intensity - d_min) / (d_max - d_min)
    image_intensity_new = (image_intensity) / 255

    hists_tmp = generate_histogram(
        lidar_intensity_new,
        image_intensity_new,
        B, H, W, bin_nums,
    )

    hists = [torch.sum(hist, dim = 1) for hist in hists_tmp]
    P = torch.sum(hists[0], dim = -1, keepdim=True)
    hists = [hist / P for hist in hists]
    entropys = [-torch.sum(hist * (hist + 1e-6).log(), dim = -1) for hist in hists]
    mult_info = entropys[0] + entropys[1] - entropys[2]
    loss = (entropys[2] - mult_info) / entropys[2]
    return loss