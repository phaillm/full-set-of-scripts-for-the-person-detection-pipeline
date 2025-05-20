import os
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression

os.chdir('./Dataset prep/DATASET_SFE/Images_set')

dataset_path = "dataset_yolo"
yaml_path = os.path.join(dataset_path, "dataset.yaml")

if not os.path.exists(yaml_path):
    raise FileNotFoundError(f"Can't find {yaml_path}. Make sure dataset.yaml is in {dataset_path}")

model = YOLO("yolov8n.pt")

model.train(data=yaml_path, epochs=1000, imgsz=960, batch=16, patience=15)



