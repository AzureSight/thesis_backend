from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "preds"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def detect_persons(image_path: str, model_path: str = "yolov3u.pt") -> dict:
    # Load the YOLO model
    model = YOLO(model_path)

    # Run the model on the given image
    results = model(image_path)

    person_detections = []

    # Filter the detections to only include persons (class ID 0)
    for result in results:
        person_boxes = [box for box in result.boxes if result.names[int(box.cls)] == "person"]

        for box in person_boxes:
            cls = int(box.cls)  # Class index
            label = result.names[cls]  # Class label
            confidence = box.conf.item()  # Confidence score
            bbox = box.xyxy.tolist()  # Bounding box coordinates
            person_detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": bbox
            })

        result.boxes = person_boxes

    # Generate the output image
    output_image = results[0].plot()

    # Save the output image with the detections
    custom_filename = os.path.join(OUTPUT_FOLDER, os.path.basename(image_path))
    cv2.imwrite(custom_filename, output_image)

    return {
        "data": {
            "num_persons_detected": len(person_detections),
            "detections": person_detections,
            "output_image": custom_filename
        }
    }