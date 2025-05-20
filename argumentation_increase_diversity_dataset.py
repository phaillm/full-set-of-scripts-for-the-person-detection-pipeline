import os
import cv2
import numpy as np
import albumentations as A

# Working directory
os.chdir('Dataset prep/DATASET_SFE/Images_set')

# Datasets path
dataset_path = "dataset_yolo"
output_path = "dataset_yolo_augmented"

# New directories
for split in ["train", "val"]:
    os.makedirs(f"{output_path}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_path}/labels/{split}", exist_ok=True)

# Basic transformations applied to every image
transform_flip = A.OneOf([
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0)
], p=1.0)

transform_blur = A.OneOf([
    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
    A.MotionBlur(blur_limit=5, p=1.0)
], p=1.0)

transform_brightness = A.OneOf([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    A.ToGray(p=1.0)
], p=1.0)

# Additional transformations
optional_transforms = A.Compose([
    A.MedianBlur(blur_limit=3, p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def clip_bbox(bbox):
    """ Make sure the box is staying in [0,1] """
    x_center, y_center, w, h = bbox
    x_min = max(0, x_center - w / 2)
    y_min = max(0, y_center - h / 2)
    x_max = min(1, x_center + w / 2)
    y_max = min(1, y_center + h / 2)

    new_w = x_max - x_min
    new_h = y_max - y_min

    if new_w > 0 and new_h > 0:
        return [ (x_min + x_max) / 2, (y_min + y_max) / 2, new_w, new_h ]
    else:
        return None

def augment_and_save(image_path, label_path, output_img_dir, output_label_dir, augment_factor=3):
    """ The augmentations are applied and the results are saved """

    image = cv2.imread(image_path)
    if image is None:
        print(f" Impossible to load the picture : {image_path}")
        return

    img_height, img_width = image.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    original_bboxes = []
    class_labels = []

    for line in lines:
        values = line.strip().split()
        class_id = int(values[0])
        yolo_bbox = list(map(float, values[1:]))

        fixed_bbox = clip_bbox(yolo_bbox)
        if fixed_bbox:
            original_bboxes.append(fixed_bbox)
            class_labels.append(class_id)

    if len(original_bboxes) == 0:
        print(f" No valid bounding box for : {image_path}, image ignored.")
        return

    # Original picture saved
    new_image_name = os.path.basename(image_path)
    cv2.imwrite(os.path.join(output_img_dir, new_image_name), image)
    with open(os.path.join(output_label_dir, os.path.basename(label_path)), "w") as f:
        f.writelines(lines)

    for i in range(augment_factor):
        # **Correction : Correct conversion in np.array for Albumentations**
        original_bboxes_np = np.array(original_bboxes, dtype=np.float32)

        # Transformations applied
        augmented = transform_flip(image=image, bboxes=original_bboxes_np.tolist(), class_labels=class_labels)
        augmented = transform_blur(image=augmented["image"], bboxes=augmented["bboxes"], class_labels=class_labels)
        augmented = transform_brightness(image=augmented["image"], bboxes=augmented["bboxes"], class_labels=class_labels)

        # Additional transformations applied
        augmented = optional_transforms(image=augmented["image"], bboxes=augmented["bboxes"], class_labels=class_labels)

        aug_image = augmented["image"]
        aug_bboxes = [clip_bbox(bbox) for bbox in augmented["bboxes"]]
        aug_bboxes = [bbox for bbox in aug_bboxes if bbox]

        if len(aug_bboxes) == 0:
            print(f"Image ignored because no valid bbox after transformation : {image_path}")
            continue

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        new_image_name = f"{base_name}_aug{i}.jpg"
        new_label_name = f"{base_name}_aug{i}.txt"

        cv2.imwrite(os.path.join(output_img_dir, new_image_name), aug_image)

        with open(os.path.join(output_label_dir, new_label_name), "w") as f:
            for bbox, cls in zip(aug_bboxes, class_labels):
                f.write(f"{cls} {' '.join(map(str, bbox))}\n")

for split in ["train", "val"]:
    image_dir = f"{dataset_path}/images/{split}"
    label_dir = f"{dataset_path}/labels/{split}"

    output_img_dir = f"{output_path}/images/{split}"
    output_label_dir = f"{output_path}/labels/{split}"

    for image_file in os.listdir(image_dir):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, image_file.replace(".jpg", ".txt"))

            if os.path.exists(label_path):
                augment_and_save(image_path, label_path, output_img_dir, output_label_dir, augment_factor=3)

print("Artificial augmentation of the dataset completed.")
