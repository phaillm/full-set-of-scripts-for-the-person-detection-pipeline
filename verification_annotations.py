import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Dataset path
dataset_path = "./segmentation_dataset/dataset_yolo"
image_folder = os.path.join(dataset_path, "images/train")
label_folder = os.path.join(dataset_path, "labels/train")

# Image selection
image_name = "child-childrens-baby-children-s.jpg"
image_path = os.path.join(image_folder, image_name)
label_path = os.path.join(label_folder, image_name.replace(".jpg", ".txt").replace(".png", ".txt"))

# Existence of the picture
if not os.path.exists(image_path):
    raise FileNotFoundError(f" Image introuvable : {image_path}")
if not os.path.exists(label_path):
    raise FileNotFoundError(f" Annotations introuvables : {label_path}")

# Load image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Conversion in RGB for Matplotlib display

# Read YOLO annotations
with open(label_path, "r") as file:
    annotations = file.readlines()

# Display annotations
for annotation in annotations:
    data = annotation.strip().split()  # Read every line and separate values
    class_id = int(data[0])  # First value : class person
    points = np.array(data[1:], dtype=np.float32).reshape(-1, 2)  # Normalization of the coordinates
    points[:, 0] *= image.shape[1]  # X converted in pixels
    points[:, 1] *= image.shape[0]  # Y converted in pixels
    points = points.astype(int)

    # Draw the polygon on the picture
    cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)  # Inside of the ontour in blue
    cv2.fillPoly(image, [points], color=(0, 0, 255, 100))  # Contour in red

# Display the picture with the labels
plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.axis("off")
plt.title(f"Annotations YOLO pour {image_name}")
plt.show()
