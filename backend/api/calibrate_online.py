import torch
import torch.nn.functional as F
from typing import Optional, Union
from scipy.spatial.transform import Rotation as R
import numpy as np
import open3d as o3d
import io
import cv2
import tempfile
from dataset import FrameData
from model import get_model
from pipeline import get_pipeline

def calibrate_online(packed_data, socket = None):
    # step 0: unpack params
    params = packed_data["param"]

    extrinsics = params["extrinsics"]
    T_init = np.eye(4)
    T_init[:3, :3] = R.from_quat(extrinsics["rotation"], scalar_first=True).as_matrix()
    T_init[:3, -1] = extrinsics["translation"]

    intrinsics = params["intrinsics"]
    D = intrinsics["D"]
    K = np.eye(3)
    K[0, 0] = intrinsics["K"][0]
    K[1, 1] = intrinsics["K"][1]
    K[0, -1] = intrinsics["K"][2]
    K[1, -1] = intrinsics["K"][3]

    pipelines = params["pipelines"]
    mode = pipelines["mode"]

    # step 1: build dataset
    frame_nums = len(packed_data["image"])
    frames = []
    for i in range(frame_nums):
        # process image
        img_data = np.frombuffer(packed_data["image"][i]["data"], dtype=np.uint8)
        image = cv2.imdecode(img_data, 1)
        h, w, c = image.shape

        if np.any(D):
            if len(D) == 5:
                mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), 5)
                image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            else:
                mapx, mapy = cv2.fisheye.initUndistortRectifyMap(K, D, None, K, (w, h), 5)
                image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).squeeze()
        image_gray = cv2.equalizeHist(image_gray)
        # cv2.imwrite("/data/claim/test.jpg", image_gray)

        # process pcd
        with tempfile.NamedTemporaryFile(mode="wb", delete=True) as tmp:
            tmp.write(packed_data["pointcloud"][i]["data"])
            fmt_int = int(packed_data["pointcloud"][i]["format"]) 
            fmt = ["pcd", "ply", "xyz"][fmt_int]        
            pcd = o3d.t.io.read_point_cloud(tmp.name, format = fmt)
            points = pcd.point["positions"].numpy()
            print(points.shape)
        
        try:
            intensity = pcd.point["intensity"].numpy()
            indices = [i for i in range(intensity.shape[0])]
            indices.sort(key=lambda x: intensity[x])
            bins = 256
            for cnt, ori_index in enumerate(indices):
                value = int(cnt / len(indices) * bins) / bins
                intensity[ori_index] = value
        except:
            intensity = None

        # monodepth
        model = get_model("depth_anything_v2")
        mono_depth = model["forward"](image)
        dep_min, dep_max = mono_depth.min(), mono_depth.max()
        mono_depth = (mono_depth - dep_min) / (dep_max - dep_min)

        # build frame
        frame = FrameData(
            frame_id = f"{i}",
            image = image,
            pointcloud = torch.from_numpy(points).float().cuda(),
            mono_depth = torch.from_numpy(mono_depth).float().cuda(),
            K = torch.from_numpy(K).float().cuda(),
            intensity = torch.from_numpy(intensity).float().cuda() if intensity is not None else None,
            image_gray = torch.from_numpy(image_gray).float().cuda() / 255.0,
            is_mono_inv_depth = model["inv_depth"]
        )
        frames.append(frame)

    pipeline = get_pipeline(pipelines)
    result = pipeline(torch.from_numpy(T_init).float().cuda(), frames, socket=socket)
    return result
    