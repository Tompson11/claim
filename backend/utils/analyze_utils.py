import argparse
import json
import os
import io
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def tensor_mode(Q, precision=3):
    if isinstance(Q, torch.Tensor):
        Q = Q.clone().detach()
    else:
        Q = torch.tensor(Q)
    rounded_quats = torch.tensor(np.around(Q.numpy(), precision))
    values, counts = torch.unique(rounded_quats, dim=0, return_counts=True)
    return values[torch.argmax(counts)]

def bin_mode(Q, margin, step):
    if isinstance(Q, torch.Tensor):
        Q = Q.clone().detach().numpy()
    
    centers = np.mean(Q, axis=0)
    N = len(centers)

    modes = np.zeros_like(centers)
    for i in range(N):
        # draw histogram
        center = centers[i]
        bins = np.arange(center - margin, center + margin, step=step)
        hist, bin_edges = np.histogram(Q[:, i], bins=bins)
        index = np.argmax(hist)
        modes[i] = (bin_edges[index] + bin_edges[index + 1]) / 2
    
    return modes


def _strucured_result(euler, trans, seq = "XYZ", degrees=True):
    result = {}
    result["translation"] = {
        "x" : float(trans[0]),
        "y" : float(trans[1]),
        "z" : float(trans[2]),
    }
    result["euler"] = {
        "order" : seq,
        "degree" : degrees,
        "x" : float(euler[0]),
        "y" : float(euler[1]),
        "z" : float(euler[2])
    }

    rot = R.from_euler(seq, euler, degrees=degrees)
    
    quat = rot.as_quat(scalar_first=True)
    result["quaternion"] = {
        "x" : float(quat[1]),
        "y" : float(quat[2]),
        "z" : float(quat[3]),
        "w" : float(quat[0]),
    }

    result["rotationMatrix"] = rot.as_matrix().tolist()
    
    so3 = rot.as_rotvec(degrees = degrees)
    angle = np.linalg.norm(so3)
    axis = so3 / angle
    result["axisAngle"] = {
        "degree" : degrees,
        "x" : float(axis[0]),
        "y" : float(axis[1]),
        "z" : float(axis[2]),
        "angle" : float(angle)
    }

    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, -1] = trans
    result["transformMatrix"] = T.tolist()

    return result


def analyze_results(
    jdata : list,
    quantile_points : list = [0.25, 0.75],
    vis : bool = False
):
    frame_ids = []
    T_inits = []
    T_ests = []
    for frame_ret in jdata:
        frame_ids.append(frame_ret["frame_id"])
        T_inits.append(np.array(frame_ret["T_init"]))
        T_ests.append(np.array(frame_ret["T_est"]))
    
    N = len(frame_ids)
    T_inits = np.stack(T_inits) # (N, 4, 4)
    T_ests = np.stack(T_ests) # (N, 4, 4)
    results = {}

    # delta
    # T_rels = T_ests @ np.linalg.inv(T_inits)
    T_rels = T_ests
    delta_eulers = R.from_matrix(T_rels[:, :3, :3]).as_euler("XYZ", degrees=True)
    delta_trans = T_rels[:, :3, 3] 

    results["num_frames"] = N

    # mean
    results["mean"] = _strucured_result(np.mean(delta_eulers, axis=0), np.mean(delta_trans, axis=0))
    
    # mode
    # results["mode"] = _strucured_result(tensor_mode(delta_eulers, precision=2), tensor_mode(delta_trans, precision=2))
    results["mode"] = _strucured_result(bin_mode(delta_eulers, margin=0.5, step=0.02), bin_mode(delta_trans, margin=0.2, step=0.01))

    # median
    results["median"] = _strucured_result(np.median(delta_eulers, axis=0), np.median(delta_trans, axis=0))

    # quantile
    results["quantile"] = []
    
    for quan in quantile_points:
        euler_quantiles = []
        trans_quantiles = []

        index = min(int(N * quan), N - 1)
        for i in range(3):
            euler_indexes = list(range(N))
            euler_indexes.sort(key=lambda x : delta_eulers[x, i])
            euler_quantiles.append(float(delta_eulers[euler_indexes[index], i]))

            trans_indexes = list(range(N))
            trans_indexes.sort(key=lambda x : delta_trans[x, i])
            trans_quantiles.append(float(delta_trans[trans_indexes[index], i]))

        ret = {"quantile_point" : quan}
        ret.update(_strucured_result(euler_quantiles, trans_quantiles))
        results["quantile"].append(ret)

    # visualization
    if vis:
        def draw_val_vlines(val, text_height_ratio, linestyle='--', color='red', zorder=200):
            line = plt.axvline(val, linestyle = linestyle, color = color, zorder = zorder)
            ymin, ymax = plt.ylim()
            plt.text(val, ymin + (ymax - ymin) * text_height_ratio, f"{val:.2f}", color = color, zorder = zorder)
            return line
        
        plt.figure(0, figsize=(12, 8))
        plt.suptitle(f"Total Frames: {N}")

        titles = ["roll(deg)", "pitch(deg)", "yaw(deg)"]
        labels = ["x", "y", "z"]
        for i in range(3):
            # draw histogram
            center = results["mean"]["euler"][labels[i]]
            bins = np.arange(center - 0.5, center + 0.5, step=0.02)

            plt.subplot(2, 3, i + 1)
            hist, bin_edges = np.histogram(delta_eulers[:, i], bins=bins)
            plt.bar(bins[:-1], hist / hist.sum(), width=0.02, align='edge', edgecolor="black", zorder=100)
            plt.title(titles[i])
            plt.grid(True, "both")

            # draw statistics
            mean_line = draw_val_vlines(results["mean"]["euler"][labels[i]], 0.8, color='red', linestyle='--', zorder=200)
            mode_line = draw_val_vlines(results["mode"]["euler"][labels[i]], 0.7, color='orange', linestyle='--', zorder=200)
            median_line = draw_val_vlines(results["median"]["euler"][labels[i]], 0.6, color='green', linestyle='--', zorder=200)
            plt.legend(handles=[mean_line, mode_line, median_line], labels = ["mean", "mode", "median"])
        
        titles = ["x(m)", "y(m)", "z(m)"]
        labels = ["x", "y", "z"]
        for i in range(3):
            center = results["mean"]["translation"][labels[i]]
            bins = np.arange(center - 0.2, center + 0.2, step=0.01)

            plt.subplot(2, 3, 4 + i)
            hist, bin_edges = np.histogram(delta_trans[:, i], bins=bins)
            plt.bar(bins[:-1], hist / hist.sum(), width=0.01, align='edge', edgecolor="black", zorder=100)
            plt.title(titles[i])
            plt.grid(True, "both")

            # draw statistics
            mean_line = draw_val_vlines(results["mean"]["translation"][labels[i]], 0.8, color='red', linestyle='--', zorder=200)
            mode_line = draw_val_vlines(results["mode"]["translation"][labels[i]], 0.7, color='orange', linestyle='--', zorder=200)
            median_line = draw_val_vlines(results["median"]["translation"][labels[i]], 0.6, color='green', linestyle='--', zorder=200)
            plt.legend(handles=[mean_line, mode_line, median_line], labels = ["mean", "mode", "median"])
        
        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png')
        results["fig"] = buffer
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='visual recon pipline')
    parser.add_argument('--result_file', type=str, help='results json file', default="/data/claim/example_dataset/Waymo/results/results.json")
    parser.add_argument('--result_path', type=str, help='results json file', default="/data/")
    parser.add_argument('--debug_vis', type=int, help='debug', default=1)
    args = parser.parse_args()

    if os.path.exists(args.result_file):
        jdata = json.load(open(args.result_file))
        results = analyze_results(jdata, vis=args.debug_vis)
        with open(os.path.join(args.result_path, "euler_debug.png"), 'wb') as file:
            file.write(results["fig"].getvalue())
        results.pop("fig")
        with open(os.path.join(args.result_path, "euler_debug.json"), 'w') as file:
            json.dump(results, file)
    else:
        print(f"No such file [{args.result_file}]!")
    
    
