import os
import cv2
import numpy as np

os.chdir('/Users/pierrehauss/Downloads/Visdrone')
print(os.getcwd())

source_images_train = os.listdir('Visdrone_images/train')
source_images_val = os.listdir('Visdrone_images/val')
source_labels_train = os.listdir('Visdrone_labels/train')
source_labels_val = os.listdir('Visdrone_labels/val')

output_dir = "/Dataset prep/DATASET_SFE/Images_set/dataset_yolo"
os.makedirs(f"{output_dir}/images/train", exist_ok=True)
os.makedirs(f"{output_dir}/images/val", exist_ok=True)
os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
os.makedirs(f"{output_dir}/labels/val", exist_ok=True)

log_file = open(f"{output_dir}/log_visdrone.txt", "w") # Follow conversion failures

file_index = 373275 # New base index for the fusion with the old dataset

valid_classes = {"0", "1"}  #pedestrian (0) et people (1)
new_class_id = "0"  # New unique class "person" (0)

def convert_annotations(source_images, source_labels, images_dir, labels_dir, output_img_dir, output_label_dir):
    global file_index

    for label_file in source_labels:
        if not label_file.endswith(".txt"):
            continue

        image_file = label_file.replace(".txt", ".jpg")
        if image_file not in source_images:
            log_file.write(f"Image not found for annotation : {label_file}\n")
            continue

        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, label_file)

        img = cv2.imread(image_path)
        if img is None:
            log_file.write(f"Impossible to load the image : {image_path}\n")
            continue

        height, width = img.shape[:2]
        new_name = f"{file_index},{hash(image_file) & 0xfffff}"
        new_image_path = os.path.join(output_img_dir, f"{new_name}.jpg")
        new_label_path = os.path.join(output_label_dir, f"{new_name}.txt")

        with open(label_path, "r") as file:
            lines = file.readlines()

        yolo_annotations = []
        for line in lines:
            values = line.strip().split(",")
            if len(values) < 6:
                continue

            x_min, y_min, box_width, box_height = map(int, values[:4])
            class_id = values[5]  # Object class in column 5

            if class_id in valid_classes:  # Sorting in object classes
                x_center = (x_min + box_width / 2) / width
                y_center = (y_min + box_height / 2) / height
                w = box_width / width
                h = box_height / height

                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    log_file.write(f"Out-of-bounds annotations for {label_file}: {x_center}, {y_center}, {w}, {h}\n")
                    continue

                yolo_annotations.append(f"{new_class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}") # Merge under the 'person' class

        if yolo_annotations: # Save valid annotations
            with open(new_label_path, "w") as file:
                file.write("\n".join(yolo_annotations))

            cv2.imwrite(new_image_path, img)

            print(f"Processed file : {new_label_path}")

        file_index += 1

convert_annotations(source_images_train, source_labels_train, "Visdrone_images/train", "Visdrone_labels/train",
                    f"{output_dir}/images/train", f"{output_dir}/labels/train")

convert_annotations(source_images_val, source_labels_val, "Visdrone_images/val", "Visdrone_labels/val",
                    f"{output_dir}/images/val", f"{output_dir}/labels/val")

log_file.close()
print("Annotation conversion completed.")
