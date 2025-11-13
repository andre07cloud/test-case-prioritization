
import os
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.metrics import bbox_iou
from .load_config_file import load_config_file
#config_file = "/home/andre/My-PHD/Poly/test-case-prioritization/config.yaml"




def load_gt_boxes(label_path, img_shape):

    print(f"Loading ground truth boxes from: {label_path} with image shape: {img_shape}")
    h,w = img_shape
    boxes = []

    if not os.path.exists(label_path):
        return torch.empty((0, 4))
    
    with open(label_path, "r") as f:

        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            coords = list(map(float, parts[1:]))

            # Extract x and y coordinates
            xs = coords[0::2]
            ys = coords[1::2]

            # Convert normalized coordinates to absolute pixel values
            xs = [x * w for x in xs]
            ys = [y * h for y in ys]

            # Bounding box englobante
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            boxes.append([x1, y1, x2, y2])

    return torch.tensor(boxes)

def compute_ciou_matrix(pred_boxes, gt_boxes):
    """
    Retourne la matrice CIoU (n_pred, n_gt) entre tous les couples de boxes.
    """
    n_pred = pred_boxes.shape[0]
    n_gt = gt_boxes.shape[0]
    ciou_matrix = torch.zeros((n_pred, n_gt))
    for i in range(n_pred):
        for j in range(n_gt):
            ciou_matrix[i, j] = bbox_iou(pred_boxes[i].unsqueeze(0), gt_boxes[j].unsqueeze(0), CIoU=True)
    return ciou_matrix

#+++++++++++++++++++++
# EVALUATE PERCEPTION DIFFICULTY
#+++++++++++++++++++++

def difficulty_evaluation(image_path, label_path, config_file):
    """
    Compute perception difficulty score for an image based on (1 -mean CIoU)
    """
    # Load configuration
    config = load_config_file(config_file)


    MODEL_PATH = config['best_model']
    print(f"Loading model from: {MODEL_PATH}")
    TEST_IMAGES = config['data_images']
    TEST_LABELS = config['data_labels']
    img_path = os.path.join(TEST_IMAGES, image_path)
    label_file = os.path.join(TEST_LABELS, label_path)
    print(f"Evaluating difficulty for image: {img_path}")
    # Load model
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")
    print(model)

    print(f"Using model path: {MODEL_PATH}")
    print(f"Using test images path: {TEST_IMAGES}")
    print(f"Using test labels path: {TEST_LABELS}")

    print(f"Processing image: {img_path} with label: {label_file}")
    # Prediction
    results = model(img_path)
    print(f"Model prediction completed: {results}")
    pred_boxes = results[0].obb.xyxy.cpu()  # x1, y1, x2, y2
    print(f"Predicted boxes: {pred_boxes}")
    image_shape = results[0].orig_shape  # (height, width)
    gt_boxes = load_gt_boxes(label_file, image_shape)
    print(f"Ground truth boxes: {gt_boxes}")

    # Case 1: No GT and no prediction -> Easy difficulty
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0.0  # No objects in the image, considered easy
    
    #Case 2: False positive (no GT but predictions) -> Hard difficulty
    if len(gt_boxes) == 0 and len(pred_boxes) > 0:
        return 1.0  # No ground truth but predictions exist, considered hard
    # Case 3: False negative (GT but no predictions) -> Hard difficulty
    if len(gt_boxes) > 0 and len(pred_boxes) == 0:
        return 1.0  # Ground truth exists but no predictions, considered hard
    
    # Case 4: Both GT and predictions exist
    print(f"Number of GT boxes: {len(gt_boxes)}, Number of predicted boxes: {len(pred_boxes)}")
    ciou_matrix = compute_ciou_matrix(pred_boxes, gt_boxes)  # Compute CIoU matrix between predicted and ground truth boxes
    print(f"***************CIoU Matrix:{ciou_matrix}")
    print(f"{ciou_matrix}")
    mean_ciou = ciou_matrix.mean().item()
    print(f"########### Mean CIoU: {mean_ciou}")

    difficulty_score = 1 - mean_ciou  # Higher CIoU means easier, so we take 1 - CIoU
    print(f"Difficulty score for image {img_path}: {difficulty_score}")
    return difficulty_score