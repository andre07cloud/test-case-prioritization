from src.common.split_range_data import split_dataset
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou
from src.common.split_range_data import YAMEL_FILE
import sys


if __name__ == "__main__":
    

# Load a model
    #model = YOLO("yolov8n.pt")  # load a pretrained model AABB(Axis-Aligned Bounding Box) standard YOLOv8n
    model = YOLO("yolov8n-obb.pt") # load a pretrained model OBB(Oriented Bounding Box) custom YOLOv8n
    # Train the model
    data = f"data/{sys.argv[1]}_{YAMEL_FILE}"
    print(f"*********** Training with data config: {data}")
    model.train(data=data, epochs=50, imgsz=640, batch=16, name=f"train_results_{sys.argv[1]}", project=f"train_results/{sys.argv[1]}")  # train the model
    print("Model training completed.")

    #iou = box_iou([0, 0, 1, 1], [0, 0, 1, 1])  # IoU between boxes

    #ciou = complete_box_iou([0, 0, 1, 1], [0, 0, 1, 1])  # CIoU between boxes