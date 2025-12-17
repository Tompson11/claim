import os
import sys

CLAIM_BASE_PATH = os.path.join(os.path.dirname(__file__), "../../..")
CLAIM_BACKEND_PATH = os.path.join(CLAIM_BASE_PATH, "backend")
sys.path.append(CLAIM_BACKEND_PATH)

from django.shortcuts import render
from channels.generic.websocket import WebsocketConsumer
from django.http import JsonResponse
import json
import struct
from api.calibrate_online import calibrate_online

def index_page(request):
    return render(request, "index.html")

def get_example_info(request):
    results = []

    # waymo
    waymo_path = os.path.join(CLAIM_BASE_PATH, "example_dataset", "Waymo", "img")
    img_lists = os.listdir(waymo_path)
    postfix = img_lists[0].split(".")[-1] if len(img_lists) else None
    for img_name in img_lists:
        info = {
            "name" : f"waymo_{img_name[:-len(postfix)-1]}",
            "img_path" : f"/static/Waymo/img/{img_name}",
            "pcd_path" : f"/static/Waymo/pcd/{img_name[:-len(postfix)] + 'pcd'}",
            "gt_path" : f"/static/Waymo/gt.json"
        }
        results.append(info)
    
    # kitti
    kitti_path = os.path.join(CLAIM_BASE_PATH, "example_dataset", "KITTI", "img")
    img_lists = os.listdir(kitti_path)
    postfix = img_lists[0].split(".")[-1] if len(img_lists) else None
    for img_name in img_lists:
        info = {
            "name" : f"KITTI_{img_name[:-len(postfix)-1]}",
            "img_path" : f"/static/KITTI/img/{img_name}",
            "pcd_path" : f"/static/KITTI/pcd/{img_name[:-len(postfix)] + 'pcd'}",
            "gt_path" : f"/static/KITTI/gt.json"
        }
        results.append(info)

    return JsonResponse(results, safe=False)
    

class MyConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        self.reset_data(None)

    def disconnect(self, close_code):
        pass

    def receive(self, text_data=None, bytes_data=None):
        if text_data is not None:
            params = json.loads(text_data)
            self.reset_data(params)
        elif bytes_data is not None:
            header = struct.unpack("<BBBB", bytes_data[:4])
            if header != (5, 25, 5, 25):
                return

            is_image = struct.unpack("<B", bytes_data[4:5])[0]
            index = struct.unpack("<B", bytes_data[5:6])[0]
            format = struct.unpack("<B", bytes_data[6:7])[0]
            if index < len(self.data["image"]):
                self.data["image" if is_image else "pointcloud"][index] = {
                    "format": format,
                    "data": bytes_data[7:]
                }

                img_flags = []
                pcd_flags = []

                for img, pcd in zip(self.data["image"], self.data["pointcloud"]):
                    img_flags.append(img is not None)
                    pcd_flags.append(pcd is not None)

                response = {
                    "type" : 0,
                    "img_flags" : img_flags,
                    "pcd_flags" : pcd_flags
                }
                self.send(text_data=json.dumps(response))

                if all(img_flags) and all(pcd_flags):
                    print("calib!")
                    try:
                        T_est = calibrate_online(self.data, socket=self)
                        response = {
                            "type" : 2,
                            "status" : 1,
                            "result" : T_est.cpu().numpy().tolist()
                        }
                    except:
                        response = {
                            "type" : 2,
                            "status" : 0
                        }
                    finally:
                        self.send(text_data=json.dumps(response))

        else:
            pass

    def reset_data(self, params=None):
        self.data = {
            "image": [None for _ in range(params["pair_nums"])] if params is not None else [],
            "pointcloud": [None for _ in range(params["pair_nums"])] if params is not None else [],
            "param": params
        }
