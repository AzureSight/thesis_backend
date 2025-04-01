from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # Import CORS
from ultralytics import YOLO
from backend.Proposed_old import proposed_get_yolo_preds 
from backend.Baseline import baseline_get_yolo_preds
import os
import cv2
import uuid
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = "uploads_from_web"
OUTPUT_FOLDER = "Whole Image Output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/api/analyze", methods=["POST"])
def handle_upload():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Generate a unique filename using UUID and the original file extension
    extension = os.path.splitext(file.filename)[1]  # Get the file extension
    unique_filename = f"{uuid.uuid4()}{extension}"  # Create a unique filename

    # print(f"Saving file as: {unique_filename}") 
    # print(f"./{UPLOAD_FOLDER}/{unique_filename}")

    image = f"./{UPLOAD_FOLDER}/{unique_filename}"

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
 
        baseline_predictions =  baseline_get_yolo_preds(image)
        # predictions_proposed =
        proposed_predictions = proposed_get_yolo_preds(image)
        # baseline_predictions = float(baseline_predictions) if isinstance(baseline_predictions, np.float32) else baseline_predictions
        # proposed_predictions = float(proposed_predictions) if isinstance(proposed_predictions, np.float32) else proposed_predictions

        return jsonify({"message": "File uploaded",  "baseline_predictions": baseline_predictions, "proposed_predictions": proposed_predictions, "filename": unique_filename}), 200
    

@app.route("/api/proposed")
def baseline_output():
    return send_from_directory(OUTPUT_FOLDER, "Poutput.jpg")

@app.route("/api/baseline")
def proposed_output():
    return send_from_directory(OUTPUT_FOLDER, "Boutput.jpg")

if __name__ == "__main__":
    app.run(debug=True, port=8080) 
    