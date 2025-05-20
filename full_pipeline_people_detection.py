import os
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from ultralytics import YOLO
import clip
import torchvision.transforms as T
from pathlib import Path


def load_models():
    """Load YOLOv8, YOLOv8-seg and CLIP models"""
    print("Loading models...")
    # Load YOLOv8 for person detection
    yolo_model = YOLO('dataset_segmentation/segmentation_dataset/best_trained_model.pt')

    # Load YOLOv8-seg for person segmentation
    yolo_seg_model = YOLO('dataset_segmentation/segmentation_dataset/yolov8x-seg.pt')

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    return yolo_model, yolo_seg_model, clip_model, clip_preprocess, device


def detect_people(image_path, yolo_model):
    """Detect people in the image and return their bounding boxes"""
    print("Detecting people...")
    results = yolo_model(image_path)

    # Extract bounding boxes for the 'person' class
    person_boxes = []
    for result in results:
        boxes = result.boxes
        for box_idx, cls in enumerate(boxes.cls):
            if int(cls) == 0:  # Class 0 is 'person'
                box = boxes.xyxy[box_idx].cpu().numpy()
                person_boxes.append(box)

    print(f"Found {len(person_boxes)} people in the image")
    return person_boxes


def segment_and_encode_people(image_path, person_boxes, yolo_seg_model, clip_model, clip_preprocess, device):
    """Segment each person, create transparent PNGs, and encode with CLIP"""
    print("Segmenting and encoding people...")
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    segmented_images = []
    clip_features = []
    valid_indices = []  # Track which indices we're keeping

    for i, box in enumerate(person_boxes):
        try:
            # Convert to integers for cropping
            x1, y1, x2, y2 = map(int, box)

            # Crop the person from the original image
            person_crop = original_image[y1:y2, x1:x2]

            # Apply YOLOv8-seg to the crop
            seg_results = yolo_seg_model(person_crop)

            # Check for masks
            if not hasattr(seg_results[0], 'masks') or seg_results[0].masks is None or len(seg_results[0].masks) == 0:
                print(f"No masks found for person crop {i}")
                continue

            # Get segmentation mask for the first detected person
            mask = seg_results[0].masks[0].data[0].cpu().numpy()

            # Resize mask to match crop size
            mask = cv2.resize(mask, (person_crop.shape[1], person_crop.shape[0]))

            # Create a 4-channel RGBA image (with transparency)
            mask_3d = np.stack([mask] * 3, axis=-1)
            person_crop_float = person_crop.astype(np.float32) / 255.0
            masked_person = person_crop_float * mask_3d


            # Add alpha channel (transparency)
            alpha_channel = mask * 255
            rgba_image = np.dstack((person_crop, alpha_channel.astype(np.uint8)))

            # Convert to PIL Image for CLIP
            pil_image = Image.fromarray(rgba_image)

            # Encode with CLIP
            with torch.no_grad():
                image_tensor = clip_preprocess(pil_image).unsqueeze(0).to(device)
                image_features = clip_model.encode_image(image_tensor)
                normalized_features = image_features / image_features.norm(dim=-1, keepdim=True)

            segmented_images.append(pil_image)
            clip_features.append(normalized_features)
            valid_indices.append(i)  # Track which original index this corresponds to

        except Exception as e:
            print(f"Error processing person {i}: {e}")
            continue

    # If we filtered some detections, we need to update person_boxes to match
    if len(valid_indices) < len(person_boxes):
        print(f"Filtered {len(person_boxes) - len(valid_indices)} invalid detections")
        # Create a new list of boxes that only includes those we processed successfully
        filtered_boxes = [person_boxes[i] for i in valid_indices]
        person_boxes[:] = filtered_boxes  # Update the original list in-place

    # Stack all image features if we have any
    if clip_features:
        clip_features = torch.cat(clip_features, dim=0)
    else:
        clip_features = torch.zeros((0, clip_model.visual.output_dim), device=device)

    return segmented_images, clip_features


def encode_text(text_description, clip_model, device):
    """Encode the text description with CLIP"""
    print(f"Encoding text description: '{text_description}'")
    with torch.no_grad():
        text_tokens = clip.tokenize([text_description]).to(device)
        text_features = clip_model.encode_text(text_tokens)
        normalized_text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return normalized_text_features


def compute_similarities(image_features, text_features):
    """Compute cosine similarities between image and text features"""
    print("Computing similarities...")
    if len(image_features) == 0:
        return np.array([])

    # Calculate cosine similarity
    similarities = (image_features @ text_features.T).squeeze(1).cpu().numpy()

    return similarities


def save_results(segmented_images, similarities, output_dir, original_image, person_boxes):
    """Save top 10 results with similarity scores and context-aware segmented images"""
    if len(similarities) == 0:
        print("No results to save (no people were detected or segmented).")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save segmented images for top 10
    top10_idx = np.argsort(similarities)[-10:][::-1] if len(similarities) >= 10 else np.argsort(similarities)[::-1]

    # Save results to text file
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        for i, idx in enumerate(top10_idx):
            f.write(f"Person {idx} - Similarity: {similarities[idx]:.4f}\n")

    # Save top 10 segmented images with context
    segmented_dir = os.path.join(output_dir, "results_top10")
    os.makedirs(segmented_dir, exist_ok=True)

    # Convert original image for processing if needed
    if isinstance(original_image, np.ndarray):
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) if original_image.shape[2] == 3 and len(
            original_image.shape) == 3 else original_image
    else:
        original_rgb = np.array(original_image)

    for i, idx in enumerate(top10_idx):
        try:
            # Get the bounding box for this person
            if idx < len(person_boxes):
                box = person_boxes[idx]
                x1, y1, x2, y2 = map(int, box)

                # Get full crop from original image
                full_crop = original_rgb[y1:y2, x1:x2].copy()

                # Convert segmented image to numpy array with alpha
                seg_img_array = np.array(segmented_images[idx])

                # Resize segmented image to match full_crop dimensions
                resized_seg = cv2.resize(seg_img_array, (full_crop.shape[1], full_crop.shape[0]))

                # Create a new image with the same dimensions as full_crop
                result_image = np.zeros((full_crop.shape[0], full_crop.shape[1], 4), dtype=np.uint8)

                # Set background to 70% opacity
                background_opacity = 0.7
                result_image[:, :, :3] = full_crop * background_opacity

                # Get alpha mask from resized segmented image
                alpha_mask = resized_seg[:, :, 3:] / 255.0 if resized_seg.shape[2] == 4 else np.ones(
                    (resized_seg.shape[0], resized_seg.shape[1], 1))

                # Apply segmented person at 100% opacity where alpha > 0
                foreground = resized_seg[:, :, :3] if resized_seg.shape[2] == 4 else resized_seg
                result_image[:, :, :3] = result_image[:, :, :3] * (1 - alpha_mask) + foreground * alpha_mask

                # Set alpha channel (100% where person is, 40% elsewhere)
                if resized_seg.shape[2] == 4:
                    result_image[:, :, 3] = np.where(
                        resized_seg[:, :, 3] > 0,
                        255,  # 100% opacity for person
                        int(255 * background_opacity)  # 70% opacity for background
                    )
                else:
                    result_image[:, :, 3] = 255  # Full opacity if no alpha channel

                # Save as PNG
                img_path = os.path.join(segmented_dir, f"person_{idx}_similarity_{similarities[idx]:.4f}.png")
                Image.fromarray(result_image).save(img_path)
            else:
                # Fallback if there's an index mismatch
                img_path = os.path.join(segmented_dir, f"person_{idx}_similarity_{similarities[idx]:.4f}.png")
                segmented_images[idx].save(img_path)

        except Exception as e:
            print(f"Error saving result for person {idx}: {e}")
            # Fallback to saving just the segmented image
            try:
                img_path = os.path.join(segmented_dir, f"person_{idx}_similarity_{similarities[idx]:.4f}_fallback.png")
                segmented_images[idx].save(img_path)
            except:
                print(f"Could not save even the fallback image for person {idx}")

    print(f"Results saved to {output_dir}")


def display_top5(similarities, segmented_images, original_image, person_boxes):
    """Display top 5 results in console and show them visually"""
    if len(similarities) == 0:
        print("No results to display (no people were detected or segmented).")
        return

    print("\n--- Top 5 Results ---")
    top5_idx = np.argsort(similarities)[-5:][::-1] if len(similarities) >= 5 else np.argsort(similarities)[::-1]

    for i, idx in enumerate(top5_idx):
        print(f"Person {idx} - Similarity: {similarities[idx]:.4f}")
    print("--------------------\n")

    # Visual display using matplotlib
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        # Create a figure to display the results
        fig = plt.figure(figsize=(15, 10))

        # First, show the original image with bounding boxes
        ax1 = plt.subplot(2, 3, 1)

        # Ensure we're displaying the original image correctly
        # Create a proper copy of the image to avoid modifying the original
        if isinstance(original_image, np.ndarray):
            # Make sure we have RGB order for matplotlib
            if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                plt_image = original_image.copy()  # Make a copy to be safe
            else:
                plt_image = original_image
        else:
            plt_image = np.array(original_image)

        ax1.imshow(plt_image)
        ax1.set_title("Original Image with Detections")

        # Add bounding boxes for top 5 persons - without modifying the image content
        colors = ['r', 'g', 'b', 'y', 'c']
        for i, idx in enumerate(top5_idx):
            if idx < len(person_boxes):
                box = person_boxes[idx]
                x1, y1, x2, y2 = map(int, box)
                width, height = x2 - x1, y2 - y1
                color = colors[i % len(colors)]

                # Just draw rectangle borders, don't modify pixel values
                rect = Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color, facecolor='none')
                ax1.add_patch(rect)
                ax1.text(x1, y1 - 5, f"#{i + 1}: {similarities[idx]:.3f}", color=color, fontsize=10,
                         bbox=dict(facecolor='white', alpha=0.7))

        # Then show the top 5 person crops with their similarity scores
        for i, idx in enumerate(top5_idx):
            if i < 5:  # Just in case we have fewer than 5
                ax = plt.subplot(2, 3, i + 2)

                # Display the segmented person
                seg_img = np.array(segmented_images[idx])
                ax.imshow(seg_img)

                # Set title with similarity score
                ax.set_title(f"#{i + 1}: Person {idx}\nSimilarity: {similarities[idx]:.4f}")
                ax.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Could not display visual results: {e}")
        print("Continuing with console output only.")


def main():
    # Get image and text description from user
    image_number = input("Enter the image number : ").strip()
    image_path = f"dataset_segmentation/segmentation_dataset/Test2/Image_test{image_number}.jpg"
    text_description = input("Enter the text description (e.g., 'a woman with a red jacket'): ").strip()

    # Check if the image exists
    if not os.path.isfile(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    # Create output directory
    output_dir = "detection_results"

    # Load models
    yolo_model, yolo_seg_model, clip_model, clip_preprocess, device = load_models()

    # Detect people
    person_boxes = detect_people(image_path, yolo_model)

    if len(person_boxes) == 0:
        print("No people detected in the image. Pipeline stopped.")
        return

    # Load original image for context-aware saving
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Person_boxes might be modified inside segment_and_encode_people
    # to filter out unsuccessful segmentations
    segmented_images, image_features = segment_and_encode_people(
        image_path, person_boxes, yolo_seg_model, clip_model, clip_preprocess, device
    )

    if len(segmented_images) == 0:
        print("No people were successfully segmented. Pipeline stopped.")
        return

    # Encode text description
    text_features = encode_text(text_description, clip_model, device)

    # Compute similarities
    similarities = compute_similarities(image_features, text_features)

    # Display top 5 results with visual output
    display_top5(similarities, segmented_images, original_image, person_boxes)

    # Save top 10 results with context
    save_results(segmented_images, similarities, output_dir, original_image, person_boxes)


if __name__ == "__main__":
    main()