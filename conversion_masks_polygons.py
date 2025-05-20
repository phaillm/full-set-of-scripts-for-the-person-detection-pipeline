import os
import cv2
import numpy as np
from tqdm import tqdm

# Dataset paths
DATASET_PATH = "./segmentation_dataset/dataset_yolo"
IMAGE_DIR = os.path.join(DATASET_PATH, "images")
MASK_DIR = os.path.join(DATASET_PATH, "masks")
LABEL_DIR = os.path.join(DATASET_PATH, "labels")

# Create label folders if they don't exist
for split in ["train", "val"]:
    os.makedirs(os.path.join(LABEL_DIR, split), exist_ok=True)

def convert_mask_to_yolo(mask_path, img_width, img_height):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f" Could not read mask: {mask_path}")
        return []

    # Binarize
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # Find ALL contours (chain none = max points)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return []

    annotations = []
    for contour in contours:
        if len(contour) < 6:
            continue

        polygon = []
        for point in contour:
            x, y = point[0]
            x_norm = x / img_width
            y_norm = y / img_height
            polygon.extend([x_norm, y_norm])

        annotations.append("0 " + " ".join(f"{p:.6f}" for p in polygon))

    return annotations

print(" Starting conversion to YOLOv8-Seg with max detail...")
total_saved = 0

for split in ["train", "val"]:
    img_dir = os.path.join(IMAGE_DIR, split)
    mask_dir = os.path.join(MASK_DIR, split)
    label_dir = os.path.join(LABEL_DIR, split)

    files = sorted(os.listdir(img_dir))
    for filename in tqdm(files, desc=f"Processing {split} set"):
        if filename.endswith((".jpg", ".png")):
            img_path = os.path.join(img_dir, filename)
            mask_filename = filename.replace(".jpg", ".png").replace(".jpeg", ".png")
            mask_path = os.path.join(mask_dir, mask_filename)

            if not os.path.exists(mask_path):
                print(f"️ Mask not found for: {filename}")
                continue

            image = cv2.imread(img_path)
            if image is None:
                print(f" Could not read image: {img_path}")
                continue

            h, w = image.shape[:2]
            annotations = convert_mask_to_yolo(mask_path, w, h)

            if annotations:
                label_file = os.path.join(label_dir, filename.rsplit(".", 1)[0] + ".txt")
                with open(label_file, "w") as f:
                    f.write("\n".join(annotations))
                total_saved += 1
            else:
                print(f" No valid annotation in: {mask_path}")

print(f"{total_saved} label files created in total under {LABEL_DIR}")
