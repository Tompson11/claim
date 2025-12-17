import os
from torch.utils.data import Dataset
from dataset import FrameData
import open3d as o3d
import json
import torch
import cv2
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from model import get_model, SUPPORTED_MODEL
from typing import Optional, Union
import random

class PrivateDataset(Dataset):
    def __init__(self,
                 base_dir: str,
                 K: Union[list, np.array],
                 D: Union[list, np.array],
                 mono_depth_model: str = "depth_anything_v2",
                 half_resolution: bool = False,
                 points_down_sample_step: int = 1,
                 intensity_equalization: bool = True,
                 gray_image_equalization: bool = True,
                 shuffle=False):
        assert mono_depth_model in SUPPORTED_MODEL, f"The given mono depth model [{mono_depth_model}] must be in {SUPPORTED_MODEL}"

        if isinstance(K, list):
            K = np.array(K)
        if K.shape[0] == 4:
            K = np.array([
                [K[0], 0, K[2]],
                [0, K[1], K[3]],
                [0, 0, K[2]],
            ])
        assert K.shape == (
            3, 3), f"The shape of K must be (3, 3)!, while the given shape is {K.shape}"

        if isinstance(D, np.ndarray):
            D = D.squeeze().tolist()
        assert len(D) in [
            4, 5], f"The length of D must be 4 or 5!, while the given length is {len(D)}"

        self.image_path = os.path.join(base_dir, f"img")
        self.lidar_path = os.path.join(base_dir, f"pcd")

        self.half_resolution = half_resolution
        self.points_down_sample_step = points_down_sample_step
        self.intensity_equalization = intensity_equalization
        self.gray_image_equalization = gray_image_equalization

        self.image_lists = os.listdir(self.image_path)
        self.image_lists.sort()
        if shuffle:
            random.shuffle(self.image_lists)

        self.K = K
        self.D = D

        self.mono_depth_model = get_model(mono_depth_model)

    def __len__(self):
        return len(self.image_lists)

    def __getitem__(self, frame_id):
        # load image
        image_name = self.image_lists[frame_id]
        image_file = os.path.join(self.image_path, image_name)
        image = cv2.imread(image_file)

        # undistortion
        h, w, _ = image.shape
        K = self.K
        D = self.D
        if np.any(D):
            if len(D) == 5:
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, D, None, K, (w, h), 5)
                image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            else:
                mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
                    K, D, None, K, (w, h), 5)
                image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.gray_image_equalization:
            image_gray = cv2.equalizeHist(image_gray)

        # load lidar
        image_postfix = image_name.split(".")[-1]
        frame_name = image_name[:-len(image_postfix)-1]
        points = None
        for lidar_postfix in ["pcd", "ply"]:
            lidar_name = frame_name + f".{lidar_postfix}"
            lidar_file = os.path.join(self.lidar_path, lidar_name)
            if os.path.exists(lidar_file):
                o3d_pcd = o3d.t.io.read_point_cloud(lidar_file)
                points = o3d_pcd.point["positions"].numpy()
                if "intensity" in o3d_pcd.point:
                    intensity = o3d_pcd.point["intensity"].numpy()
                else:
                    intensity = None
                break

        if points is None:
            raise ValueError(
                f"Cannot find pointcloud w.r.t the image {image_name} in the directory {self.lidar_path}")

        if self.points_down_sample_step > 1:
            points = points[::int(self.points_down_sample_step)]
            if intensity is not None:
                intensity = intensity[::int(self.points_down_sample_step)]

        if intensity is not None:
            if self.intensity_equalization:
                indices = [i for i in range(intensity.shape[0])]
                indices.sort(key=lambda x: intensity[x])
                bins = 256
                for cnt, ori_index in enumerate(indices):
                    value = int(cnt / len(indices) * bins) / bins
                    intensity[ori_index] = value
            else:
                i_min = intensity.min()
                i_max = intensity.max()
                intensity = (intensity - i_min) / (i_max - i_min)
            intensity[intensity == 0.0] = 1e-3

        # load mono depth
        mono_depth = self.mono_depth_model["forward"](image)

        if not self.half_resolution:
            return FrameData(
                frame_id=frame_name,
                image=image,
                pointcloud=torch.from_numpy(points).float().cuda(),
                mono_depth=torch.from_numpy(mono_depth).float().cuda(),
                K=torch.from_numpy(K).float().cuda(),
                intensity=torch.from_numpy(intensity).float().cuda() if intensity is not None else None,
                image_gray=torch.from_numpy(image_gray).float().cuda() / 255.0,
                is_mono_inv_depth=self.mono_depth_model["inv_depth"]
            )
        else:
            K_half = np.array(K)
            K_half[0, :] /= 2
            K_half[1, :] /= 2
            return FrameData(
                frame_id=frame_name,
                image=image[::2, ::2, :],
                pointcloud=torch.from_numpy(points).float().cuda(),
                mono_depth=torch.from_numpy(
                    mono_depth[::2, ::2]).float().cuda(),
                K=torch.from_numpy(K_half).float().cuda(),
                intensity=torch.from_numpy(intensity).float().cuda() if intensity is not None else None,
                image_gray=torch.from_numpy(
                    image_gray[::2, ::2]).float().cuda() / 255.0,
                is_mono_inv_depth=self.mono_depth_model["inv_depth"]
            )
