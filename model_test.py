from ultralytics import YOLO
from src.common.perception_difficulty import difficulty_evaluation
import os

if __name__ == "__main__":

    model = YOLO("train_results/uc1/train_results_uc15/weights/best.pt")  # load a custom model
    # Test the model
    image_path = "collected_data/all_stats_05_nov_uc1_2woods/images/0.png"
    results = model(image_path, show=True, save= True)  # predict on an image
    print(f"list of results: {results[0].obb} and type: {type(results)}")
    print("Model testing completed.")
    # #model.export(format="onnx")      # ONNX
    # model.export(format="torchscript")
    # model.export(format="engine")    # TensorRT (rapide sur GPU NVIDIA)
    # model.export(format="coreml")    # iOS
    # model.export(format="pb")         # TensorFlow SavedModel
    # model.export(format="tfjs")       # TensorFlow.js
    # model.export(format="openvino")  # OpenVINO
    # model.export(format="edgetpu")   # Edge TPU
    # model.export(format="paddle")    # PaddlePaddle

 
    #diff = difficulty_evaluation(image_path)
    #print(f"Perception difficulty score for the image {image_path}: {diff}")