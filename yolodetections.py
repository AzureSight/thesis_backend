import os
import cv2
import time
from PIL import Image
from ultralytics import YOLO


results = []

def yolo_detect(cropped_folder,image_path,device,model_path,confidence_threshold):
    
    # Ensure output directories exist
    os.makedirs(cropped_folder, exist_ok=True)
    
    # Load YOLOv3u model
    model = YOLO(model_path)
    
    # Read input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return
    H, W = image.shape[:2]
    
    # Run inference
    yolotime_start = time.time()
    results = model.predict(image_path, device=device, conf=confidence_threshold, imgsz=416)
    yolotime_end = time.time()

    elapsed_time = yolotime_end - yolotime_start

    return results, elapsed_time


