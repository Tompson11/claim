import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))
from lidar_image_align import depth_sim_loss, depth_sim_loss2, generate_lidar_depth, generate_lidar_intensity, generate_histogram
import cv2
import torch
import json
import matplotlib.pyplot as plt
from utils.loss_utils import batch_project_points_to_image, NID_loss
from utils.transform_utils import matrix_to_euler_angles
import numpy as np

def get_color(value, minmax=[0, 1], cmap="bwr"):
    assert minmax[0] < minmax[1]
    cm = plt.get_cmap(cmap)
    value = np.clip(value, minmax[0], minmax[1])
    return cm(value / (minmax[1] - minmax[0]))


def draw_project_results(points_torch, image, T_l2c, K_torch, intensity_torch=None, draw_loss=False, mono_depth_torch=None, box_p = 80, shift = 0):
    uvd_torch, mask = batch_project_points_to_image(
        points_torch, image, T_l2c[None, ...], K_torch, return_mask=True)
    uvd = uvd_torch.cpu().numpy()
    mask = mask.cpu().numpy()
    uvd = uvd[mask]
    min_d = np.min(uvd[:, -1])
    max_d = np.max(uvd[:, -1])
    uvd[:, -1] = (uvd[:, -1] - min_d) / (max_d - min_d)
    H, W, C = image.shape
    if C == 1:
        image = np.stack([image.copy(), image.copy(), image.copy()], axis=-1).squeeze()

    if intensity_torch is not None:
        intensity = intensity_torch.cpu().numpy()[mask.squeeze()]
        min_i = intensity.min()
        max_i = intensity.max()
        intensity = (intensity - min_i) / (max_i - min_i)
    
    color = get_color(uvd[:, -1], cmap="Spectral") if intensity_torch is None else get_color(intensity[:], cmap="bwr").squeeze()
    color = color[:, [2, 1, 0]]
    uvd = uvd.astype(np.int32)
    image[uvd[:, 1], uvd[:, 0], :] = color * 255
    image[np.minimum(uvd[:, 1] + 1, H - 1), uvd[:, 0], :] = color * 255
    image[np.maximum(uvd[:, 1] - 1, 0), uvd[:, 0], :] = color * 255
    image[uvd[:, 1], np.minimum(uvd[:, 0] + 1, W - 1), :] = color * 255
    image[uvd[:, 1], np.maximum(uvd[:, 0] - 1, 0), :] = color * 255
    
    if draw_loss and mono_depth_torch is not None:
        P = points_torch.shape[0]
        lidar_depths = generate_lidar_depth(uvd_torch, P, 1, H, W)
        losses = depth_sim_loss(
            lidar_depths, mono_depth_torch, 1, H, W, shift, box_p)
        # losses = depth_sim_loss2(
        #     lidar_depths, mono_depth_torch, 1, H, W, shift, box_p)
        losses[losses == 0.0] = -1.0
        # lidar_depths = generate_lidar_intensity(uvd_torch, intensity_torch, P, 1, H, W)
        # losses = depth_sim_loss(
        #     lidar_depths, torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).squeeze()).float().cuda(), 1, H, W, shift, box_p)

        print(torch.mean(losses, dim = -1))
        nums_box_h = (H - shift) // box_p
        nums_box_w = (W - shift) // box_p
        for patch_id_x in range(nums_box_w):
            for patch_id_y in range(nums_box_h):
                u0 = patch_id_x * box_p + shift
                v0 = patch_id_y * box_p + shift
                u1 = u0 + box_p
                v1 = v0 + box_p
                patch_id = patch_id_x * nums_box_h + patch_id_y
                cv2.putText(image, "{:.3f}".format(float(
                    losses[0, patch_id])), (u0 + 10, v0 + box_p // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * box_p / 80, (0, 0, 255), 2)

        for patch_id_x in range(nums_box_w):
            cv2.line(image, (patch_id_x * box_p + shift, 0), (patch_id_x * box_p + shift, H - 1), (255, 255, 255), 1)
        for patch_id_y in range(nums_box_h):
            cv2.line(image, (0, patch_id_y * box_p + shift), (W - 1, patch_id_y * box_p + shift), (255, 255, 255), 1)
    
    return image

def draw_depth_and_gradient(mono_depth, ksize = 5, draw_gradient = True):
    if isinstance(mono_depth, torch.Tensor):
        depth = mono_depth.contiguous().cpu().numpy()
    else:
        depth = mono_depth
    
    depth_color = get_color(depth, minmax=[0, 255], cmap="Spectral")[..., :-1]

    if draw_gradient:
        def normalize(data):
            v_max = np.max(data)
            v_min = np.min(data)
            return (data - v_min) / (v_max - v_min)
        
        sobelx = np.abs(cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=ksize))
        sobelx_normalized = normalize(sobelx)

        sobely = np.abs(cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=ksize))
        sobely_normalized = normalize(sobely)

        sobelxy2 = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
        sobelxy_normalized = normalize(sobelxy2)
        
        row0 = np.concatenate([depth_color, sobelxy_normalized[..., None].repeat(3, axis=-1)], axis=1)
        row1 = np.concatenate([sobelx_normalized[..., None].repeat(3, axis=-1), sobely_normalized[..., None].repeat(3, axis=-1)], axis=1)
        image = (np.concatenate([row0, row1], axis=0) * 255).astype(np.uint8)
    else:
        image = (depth_color * 255).astype(np.uint8)
    
    return image 

def draw_histogram(points_torch, image_torch, intensity_torch, T_l2c, K_torch, bin_nums = 16):
    uvd_torch, _ = batch_project_points_to_image(
        points_torch, image_torch, T_l2c[None, ...], K_torch, return_mask=False)
    
    P = points_torch.shape[0]
    H, W = image_torch.shape
    lidar_intensity = generate_lidar_intensity(uvd_torch, intensity_torch, P, 1, H, W)

    hists_tmp = generate_histogram(
        lidar_intensity,
        image_torch,
        1, H, W, bin_nums,
    )
    NID_loss(lidar_intensity, image_torch)
    hists = [torch.sum(hist, dim = 1) for hist in hists_tmp]
    P = torch.sum(hists[0], dim = -1, keepdim=True)
    hists = [hist / P for hist in hists]
    entropys = [-torch.sum(hist * (hist + 1e-6).log(), dim = -1) for hist in hists]
    mult_info = entropys[0] + entropys[1] - entropys[2]
    loss = (entropys[2] - mult_info) / entropys[2]

    img_00 = (image_torch.contiguous().cpu().numpy()[..., None].repeat(3, axis = -1) * 255).astype(np.uint8)
    img_01 = (lidar_intensity[0].contiguous().cpu().numpy()[..., None].repeat(3, axis = -1) * 255).astype(np.uint8)

    def plot_hist(fig_id, hist, color):
        fig = plt.figure(fig_id, (W // 100, H // 100))
        plt.bar(np.linspace(0, bin_nums - 1, bin_nums), hist, color = color)
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)[:, :, ::-1]
        return image
    
    img_10 = plot_hist(0, hists[1].cpu().numpy().squeeze(), color='r')
    img_10 = cv2.resize(img_10, (W, H))
    cv2.putText(img_10, "NID: {:.4f}".format(float(loss)), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 128), 2)
    img_11 = plot_hist(1, hists[0].cpu().numpy().squeeze(), color='b')
    img_11 = cv2.resize(img_11, (W, H))

    row0 = np.concatenate([img_00, img_01], axis=1)
    row1 = np.concatenate([img_10, img_11], axis=1)
    image = np.concatenate([row0, row1], axis=0)
    return image

def draw_batch_results(
    frames : list, 
    T_init : torch.Tensor,
    T_est : torch.Tensor
):  
    assert len(frames) > 0

    img_rows = []
    img_ids = []
    for frame_data in frames:
        image_show = frame_data.image.copy()
        image_init = draw_project_results(frame_data.pointcloud, image_show,
                                        T_init,
                                        frame_data.K,
                                        draw_loss=False, mono_depth_torch=frame_data.mono_depth)
        
        image_show = frame_data.image.copy()
        image_est = draw_project_results(frame_data.pointcloud, image_show,
                                        T_est,
                                        frame_data.K,
                                        draw_loss=False, mono_depth_torch=frame_data.mono_depth)
        
        image_depth_proj = np.hstack([image_init, image_est])
        
        if frame_data.intensity is not None:
            image_show = frame_data.image.copy()
            image_init = draw_project_results(frame_data.pointcloud, image_show,
                                            T_init,
                                            frame_data.K,
                                            intensity_torch=frame_data.intensity,
                                            draw_loss=False, mono_depth_torch=frame_data.mono_depth)
            
            image_show = frame_data.image.copy()
            image_est = draw_project_results(frame_data.pointcloud, image_show,
                                            T_est,
                                            frame_data.K,
                                            intensity_torch=frame_data.intensity,
                                            draw_loss=False, mono_depth_torch=frame_data.mono_depth)
            
            image_int_proj = np.hstack([image_init, image_est])
            image_final = np.vstack([image_depth_proj, image_int_proj])
        else:
            image_final = image_depth_proj

        img_rows.append(image_final)
        img_ids.append(frame_data.frame_id)
    
    # add header
    def _draw_header(h, w, T, label):
        # calculate text width
        (text_w0, text_h0), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 4, 4)
        
        eulers = torch.rad2deg(matrix_to_euler_angles(T[:3, :3], "XYZ"))
        pos = T[:3, -1]

        rot_text = f"Rot: {eulers[0]:.2f}, {eulers[1]:.2f}, {eulers[2]:.2f}"
        pos_text = f"Pos: {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}"
        ref_text = rot_text if len(rot_text) > len(pos_text) else pos_text
        (text_w, text_h), baseline = cv2.getTextSize(ref_text, cv2.FONT_HERSHEY_COMPLEX, 2, 2)
        
        # base param
        margin = 20
        _w = 1525
        _h = max(margin * 2 + text_h0, margin * 3 + 2 * text_h)
        header = np.zeros((_h, _w, 3), np.uint8)

        # add text
        cv2.putText(header, label, (margin, margin + text_h0), cv2.FONT_HERSHEY_COMPLEX, 4, (0, 0, 255), 4)
        cv2.putText(header, rot_text, (_w - text_w - margin, margin + text_h), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
        cv2.putText(header, pos_text, (_w - text_w - margin, 2 * (margin + text_h)), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
        
        header = cv2.resize(header, (w, h))
        return header
    
    h, w, _ = img_rows[0].shape
    header = np.hstack([_draw_header(180, w // 2, T_init, "Initial"), _draw_header(180, w // 2, T_est, "Final")])
    
    # add column
    def _draw_column(col_w):
        # calculate text width
        max_len = 0
        for text in img_ids:
            if len(text) > max_len:
                ref_text = text
                max_len = len(text)        
        (text_w, text_h), baseline = cv2.getTextSize(ref_text, cv2.FONT_HERSHEY_COMPLEX, 2, 2)

        margin = 30
        h, w, _ = img_rows[0].shape
        N = len(img_rows)
        column = np.zeros((N * h, 2 * margin + text_w, 3), dtype=np.uint8)

        for i in range(N):
            if i % 2 == 0:
                bg_color = (200, 100, 0)
            else:
                bg_color = (200, 200, 0)
            
            column[i * h: (i + 1) * h, :, ...] = bg_color
            cv2.putText(column, str(img_ids[i]), (margin, margin + int((i + 0.5) * h)), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
        
        column = cv2.resize(column, (col_w, N * h))
        return column
    
    column = _draw_column(180)

    # final output
    img_00 = np.zeros((header.shape[0], column.shape[1], 3), dtype=np.uint8)
    img_01 = header
    img_10 = column
    img_11 = np.vstack(img_rows)
    img_final = np.vstack([np.hstack([img_00, img_01]), np.hstack([img_10, img_11])])
    return img_final

