from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # Import CORS
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "preds"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def detect_persons(image_path: str, model_path: str = "yolov3u.pt") -> dict:
    model = YOLO(model_path)
    results = model(image_path)
    person_detections = []

    for result in results:
        person_boxes = [box for box in result.boxes if result.names[int(box.cls)] == "person"]

        for box in person_boxes:
            cls = int(box.cls)
            label = result.names[cls]
            confidence = box.conf.item()
            bbox = box.xyxy.tolist()
            person_detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": bbox
            })

        result.boxes = person_boxes

    output_image = results[0].plot()
    custom_filename = os.path.join(OUTPUT_FOLDER, os.path.basename(image_path))
    cv2.imwrite(custom_filename, output_image)

    return {
        "data": {
            "num_persons_detected": len(person_detections),
            "detections": person_detections,
            "output_image": f"/preds/{os.path.basename(custom_filename)}" 
        }
    }

@app.route("/api/yolo", methods=["POST"])
def handle_upload():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        results = detect_persons(file_path)
        return jsonify(results)
   
# Serve files from the preds folder
@app.route("/preds/<path:filename>")
def serve_predictions(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True, port=8080) 