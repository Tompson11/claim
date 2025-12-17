import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2
import torch
import argparse
import json
import shutil
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from pipeline import get_pipeline
from dataset.private_dataset import PrivateDataset
from utils.analyze_utils import analyze_results
from utils.vis_utils import draw_batch_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calibrate with private dataset')
    parser.add_argument('--config', type=str, help='path of config file', required=True)
    parser.add_argument('--seed', type=int, help='seed for reproduction. Set a negetive number means using a random seed', default=525)
    parser.add_argument('--result_name', type=str, help='name of output result directory', default="results")
    parser.add_argument('--vis_proj', action='store_true', help='whether to visualize projection results')
    args = parser.parse_args()

    # load config
    assert os.path.exists(args.config) and args.config.endswith(".json")
    config = json.load(open(args.config))
    
    # build dataset
    dataset = PrivateDataset(
        base_dir = config["base_dir"],
        K = np.array(config["intrinsics"]["K"]),
        D = config["intrinsics"]["D"],
        **config["data_params"]
    )

    # get pipeline
    pipe = get_pipeline(config["pipeline_params"])

    # get initial guess
    T_init = np.eye(4)
    T_init[:3, :3] = R.from_quat(config["extrinsics"]["rotation"], scalar_first=True).as_matrix()
    T_init[:3, -1] = config["extrinsics"]["translation"]
    T_init = torch.from_numpy(T_init).float().cuda()
    
    # calib
    assert config["overlap_nums_between_batch"] <= config["frame_nums_per_batch"]
    
    # make result path
    result_path = os.path.join(config["base_dir"], args.result_name)
    if os.path.exists(result_path):
        shutil.rmtree(result_path, ignore_errors=True)
    os.makedirs(result_path, exist_ok=True)
    if args.vis_proj:
        proj_path = os.path.join(result_path, "proj")
        os.makedirs(proj_path, exist_ok=True)

    # set seed
    if args.seed > 0:
        torch.manual_seed(args.seed)

    results = []
    for i in range(0, len(dataset), config["frame_nums_per_batch"] - config["overlap_nums_between_batch"]):
    # for i in range(0, 5, config["frame_nums_per_batch"] - config["overlap_nums_between_batch"]):
        frames = []
        for k in range(i, min(i + config["frame_nums_per_batch"], len(dataset))):
            frames.append(dataset[k])
        
        # print frame infos
        print("*" * 50, "CLAIM", "*" * 50)
        for frame in frames:
            text = f"  🖼️   [Frame] {frame.frame_id}"
            print(f"\033[1;30;42m{text}{' '*(107 - len(text))}\033[0m")
        print("*" * 107)

        # calibration!
        T_est = pipe(T_init, frames)
        
        info = {
            "frame_id" : [frame.frame_id for frame in frames],
            "T_init" : T_init.cpu().numpy().tolist(),
            "T_est" : T_est.cpu().numpy().tolist()
        }
        results.append(info)

        # visualize projection
        if args.vis_proj:
            img_final = draw_batch_results(frames, T_init, T_est)
            img_name = '_'.join(info['frame_id'])
            cv2.imwrite(f"{proj_path}/{img_name}.jpg", img_final)
        
        
    # output result
    print(f"🖨️ Saving results to {result_path}")
    with open(os.path.join(result_path, "results.json"), "w") as f:
        json.dump(results, f)
    
    # output analyzed results
    analyzed_results = analyze_results(results, vis=True)
    with open(os.path.join(result_path, "analyzed_results.png"), 'wb') as file:
        file.write(analyzed_results["fig"].getvalue())
    analyzed_results.pop("fig")
    with open(os.path.join(result_path, "analyzed_results.json"), "w") as f:
        json.dump(analyzed_results, f)

    








    