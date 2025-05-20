import os
import json
import cv2
import numpy as np

os.chdir('./Dataset prep/DATASET_SFE/Images_set') # Working directory

list1 = os.listdir()
list_images = [x for x in list1 if '.jpg' in x]

odgt_file = './annotation_train.odgt'
output_dir = "./dataset_yolo"
log_file = open(f"{output_dir}/log.txt", "w")  # Create a text file to write down every failure in the process


os.makedirs(f"{output_dir}/images", exist_ok=True) # Create output directory
os.makedirs(f"{output_dir}/labels", exist_ok=True)

with open(odgt_file, "r") as f:
    data = [json.loads(line) for line in f]  # Read the annotation file

for entry in data:
    image_id = entry["ID"]
    gtboxes = entry["gtboxes"]

    image_name = f"{image_id}.jpg"
    image_path = os.path.join(image_name)
    label_path = os.path.join(f"{output_dir}/labels", f"{image_id}.txt")

    if image_name not in list_images:
        print(f"Image not found : {image_name}") # Make sure the picture exists

    img = cv2.imread(image_path)
    if img is None:
        print(f"Impossible to load the picture : {image_path}")
        continue

    height, width = img.shape[:2]
    has_annotations = False  # Make sure valid annotations exist

    with open(label_path, "w") as label_file:
        for box in gtboxes:
            if box["tag"] != "person":
                continue

            x_min, y_min, box_width, box_height = box["fbox"]

            x_center = (x_min + box_width / 2) / width
            y_center = (y_min + box_height / 2) / height
            w = box_width / width
            h = box_height / height

            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                log_file.write(f"Out-of-bounds annotations in {image_name}: {x_center}, {y_center}, {w}, {h}\n")
                continue

            label_file.write(f"0 {x_center} {y_center} {w} {h}\n")
            has_annotations = True

    if has_annotations:
        cv2.imwrite(os.path.join(f"{output_dir}/images", image_name), img)
    else:
        os.remove(label_path)  # Delete every empty text file
        image_output_path = os.path.join(f"{output_dir}/images", image_name)
        if os.path.exists(image_output_path):
            os.remove(image_output_path)  # Delete every picture with no annotation
        print(f"No annotation found for {image_name}, image deleted from dataset_yolo")

log_file.close()
print("Conversion completed.")




