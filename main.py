from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # Import CORS
from ultralytics import YOLO
from Proposed import proposed_get_yolo_preds 
from Baseline import baseline_get_yolo_preds
from Baseline_Resnet import baseline_resnet_get_yolo_preds
import os
import cv2
import uuid
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = "uploads_from_web"
OUTPUT_FOLDER = "Whole Image Output"
AD_UPLOAD_FOLDER = "uploads_ads"  
AD_OUTPUT_FOLDER = "output_ads"  
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(AD_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AD_OUTPUT_FOLDER, exist_ok=True)

# @app.route("/api/analyze", methods=["POST"])
# def handle_upload():
#     if "image" not in request.files:
#         return jsonify({"error": "No file part"}), 400

#     file = request.files["image"]

#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400

#     # Generate a unique filename using UUID and the original file extension
#     extension = os.path.splitext(file.filename)[1]  # Get the file extension
#     unique_filename = f"{uuid.uuid4()}{extension}"  # Create a unique filename

#     # print(f"Saving file as: {unique_filename}") 
#     # print(f"./{UPLOAD_FOLDER}/{unique_filename}")

#     image = f"./{UPLOAD_FOLDER}/{unique_filename}"

#     if file:
#         file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
#         file.save(file_path)
 
#         baseline_predictions =  baseline_get_yolo_preds(image)
#         # predictions_proposed =
#         proposed_predictions = proposed_get_yolo_preds(image)
#         # baseline_predictions = float(baseline_predictions) if isinstance(baseline_predictions, np.float32) else baseline_predictions
#         # proposed_predictions = float(proposed_predictions) if isinstance(proposed_predictions, np.float32) else proposed_predictions

#         return jsonify({
#             "message": "File uploaded",  
#             "baseline_predictions": baseline_predictions, 
#             "proposed_predictions": proposed_predictions, 
#             "filename": unique_filename
#         }), 200
    
@app.route("/api/analyze", methods=["POST"])
def handle_upload():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Generate a unique filename using UUID
    extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{extension}"
    image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(image_path)

    # Generate unique output filenames for proposed and baseline images
    proposed_output_filename = f"Poutput_{uuid.uuid4()}.jpg"
    # baseline_output_filename = f"Boutput_{uuid.uuid4()}.jpg"
    baseline_resnet_output_filename = f"BRoutput_{uuid.uuid4()}.jpg"
    
    blurredproposed_output_filename = f"Poutput_{uuid.uuid4()}.jpg"
    # baseline_output_filename = f"Boutput_{uuid.uuid4()}.jpg"
    blurredbaseline_resnet_output_filename = f"BRoutput_{uuid.uuid4()}.jpg"

    proposed_output_path = os.path.join(OUTPUT_FOLDER, proposed_output_filename)
    # baseline_output_path = os.path.join(OUTPUT_FOLDER, baseline_output_filename)
    baseline_resnet_output_path = os.path.join(OUTPUT_FOLDER, baseline_resnet_output_filename)

    
    blurredproposed_output_path = os.path.join(OUTPUT_FOLDER, blurredproposed_output_filename)
    # baseline_output_path = os.path.join(OUTPUT_FOLDER, baseline_output_filename)
    blurredbaseline_resnet_output_path = os.path.join(OUTPUT_FOLDER, blurredbaseline_resnet_output_filename)





    # Run YOLO prediction and save the output images
    proposed_predictions = proposed_get_yolo_preds(image_path, proposed_output_path,blurredproposed_output_path)

    baseline_resnet_predictions = baseline_resnet_get_yolo_preds(image_path, baseline_resnet_output_path,blurredbaseline_resnet_output_path)
    # baseline_predictions = baseline_get_yolo_preds(image_path, baseline_output_path)
   

    return jsonify({
        "message": "File uploaded",
        # "baseline_predictions": baseline_predictions,
        "proposed_predictions": proposed_predictions,
        "baseline_resnet_predictions": baseline_resnet_predictions,
        "proposed_image_url": f"http://127.0.0.1:8080/api/image/{proposed_output_filename}",
        # "baseline_image_url": f"http://127.0.0.1:8080/api/image/{baseline_output_filename}",
        "baseline_resnet_image_url": f"http://127.0.0.1:8080/api/image/{baseline_resnet_output_filename}",
        "blurredproposed_image_url": f"http://127.0.0.1:8080/api/image/{blurredproposed_output_filename}",
        # "baseline_image_url": f"http://127.0.0.1:8080/api/image/{baseline_output_filename}",
        "blurredbaseline_resnet_image_url": f"http://127.0.0.1:8080/api/image/{blurredbaseline_resnet_output_filename}"
    }), 200

@app.route("/api/image/<filename>")
def serve_output_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)
    
# @app.route("/api/analyze_ad", methods=["POST"])
# def handle_ad_upload():
#     if "image" not in request.files:
#         return jsonify({"error": "No file part"}), 400

#     file1 = request.files["image"]
#     if file1.filename == "":
#         return jsonify({"error": "No selected file"}), 400

#     unique_filename1 = f"{uuid.uuid4()}{os.path.splitext(file1.filename)[1]}"
#     file_path1 = os.path.join(AD_UPLOAD_FOLDER, unique_filename1)
#     file1.save(file_path1)

#     # Get predictions for ads (optional)

#     proposed_predictions1 = proposed_get_yolo_preds(file_path1)
#     return jsonify({
#         "message": "Ad image uploaded",
#         "proposed_predictions": proposed_predictions1,
#         "filename": unique_filename1
#     }), 200

# @app.route("/api/analyze_ad", methods=["POST"])
# def handle_ad_upload():
#     if "image" not in request.files:
#         return jsonify({"error": "No file part"}), 400

#     file1 = request.files["image"]
#     if file1.filename == "":
#         return jsonify({"error": "No selected file"}), 400

#     unique_filename1 = f"{uuid.uuid4()}{os.path.splitext(file1.filename)[1]}"
#     file_path1 = os.path.join(AD_UPLOAD_FOLDER, unique_filename1)
#     file1.save(file_path1)

#     # Run predictions with `is_ad=True`
#     proposed_predictions1 = proposed_get_yolo_preds(file_path1, is_ad=True)

#     return jsonify({
#         "message": "Ad image uploaded",
#         "proposed_predictions": proposed_predictions1,
#         "filename": unique_filename1
#     }), 200

# @app.route("/api/analyze_ad_top", methods=["POST"])
# @cross_origin()
# def handle_ad_upload_top():
#     if "image" not in request.files:
#         return jsonify({"error": "No file part"}), 400

#     file1 = request.files["image"]
#     if file1.filename == "":
#         return jsonify({"error": "No selected file"}), 400

#     unique_filename1 = f"{uuid.uuid4()}{os.path.splitext(file1.filename)[1]}"
#     file_path1 = os.path.join(AD_UPLOAD_FOLDER, unique_filename1)
#     file1.save(file_path1)

#     # Run predictions with `is_ad=True`
#     proposed_predictions1 = proposed_get_yolo_preds(file_path1, is_ad_top=True)

#     return jsonify({
#         "message": "Ad image uploaded",
#         "proposed_predictions": proposed_predictions1,
#         "filename": unique_filename1
#     }), 200

# @app.route("/api/proposed_ad")
# def proposed_ad_output():
#     return send_from_directory("Whole Image Output", "AdPoutput.jpg")

@app.route("/api/analyze_ad_top", methods=["POST"])
@cross_origin()
def handle_ad_upload_top():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file1 = request.files["image"]
    if file1.filename == "":
        return jsonify({"error": "No selected file"}), 400

    unique_filename1 = f"{uuid.uuid4()}{os.path.splitext(file1.filename)[1]}"
    file_path1 = os.path.join(AD_UPLOAD_FOLDER, unique_filename1)
    file1.save(file_path1)

    # Generate a unique output filename
    proposed_output_filename = f"AdPoutput_top_{uuid.uuid4()}.jpg"
    proposed_output_path = os.path.join("Whole Image Output", proposed_output_filename)

    # Run predictions and save to `proposed_output_path`
    proposed_predictions1 = proposed_get_yolo_preds(file_path1, proposed_output_path, is_ad_top=True)

    return jsonify({
        "message": "Ad image uploaded",
        "proposed_predictions": proposed_predictions1,
        "proposed_image_url": f"http://127.0.0.1:8080/api/image/{proposed_output_filename}",
        "filename": unique_filename1
    }), 200

@app.route("/api/proposed_ad_top")
def proposed_ad_output_top():
    return send_from_directory("Whole Image Output", "AdPoutput_top.jpg")


if __name__ == "__main__":
    app.run(debug=True, port=8080) 
    