
import cv2
import numpy as np
import tensorflow as tf
import os
import time
import joblib
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet

def compute_image_confidence(predictions):
    # Separate person-level predictions
    person_predictions = [pred for pred in predictions if 'person' in pred]
    
    # Count the occurrences of each label
    inappropriate_count = sum(1 for pred in person_predictions if pred['label'] == 'inappropriate')
    appropriate_count = sum(1 for pred in person_predictions if pred['label'] == 'appropriate')
    total_count = len(person_predictions)
    
    # Handle edge case: no predictions
    if total_count == 0:
        
        return {"image_label": "undetermined", "total_confidence": .87}

    # Compute confidence for inappropriate and appropriate
    inappropriate_confidence = inappropriate_count / total_count
    appropriate_confidence = appropriate_count / total_count
    
    # Determine the image label based on majority vote
    if inappropriate_count > appropriate_count:
        # image_label = "inappropriate"
        confidence = inappropriate_confidence
    else:
        # image_label = "appropriate"
        confidence = appropriate_confidence

    return {"total_confidence": confidence}

class_names = ['appropriate', 'inappropriate']  
# Load the ResNet feature extractor
load_extractor = tf.keras.models.load_model('Saved models/svm/resnetextractor.keras')

# Load the SVM model
svm_model_path = os.path.join('Saved models/svm', 'svm_model.pkl')
load_svm = joblib.load(svm_model_path)


def predict_image(img_path):
    

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224), color_mode='rgb')
    img_array = preprocess_resnet(np.expand_dims(tf.keras.preprocessing.image.img_to_array(img), axis=0))
   
   
    feature_maps = load_extractor.predict(img_array)  
    flattened_features = feature_maps.reshape(1, -1)  

 
    prediction = load_svm.predict(flattened_features)  
  
    predicted_class = class_names[prediction[0]]  

    return predicted_class


def baseline_get_yolo_preds(image_path: str):
    start_time = time.time()  
    """
    Run YOLO predictions, save cropped persons, classify them, and label the entire image.
    """
    labels_path = "./DARKNET/coco.txt"
    yolo_cfg = "./DARKNET/model_data/yolov3.cfg"
    yolo_weights = "./DARKNET/model_data/yolov3.weights"
    input_img_path = image_path
    output_img_path = "./Whole Image Output/Boutput.jpg"
    confidence_threshold = 0.5
    overlapping_threshold = 0.5

    # Load COCO labels
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = f.read().strip().split("\n")
    except FileNotFoundError:
        print(f"Error: {labels_path} not found. Please check the file path.")
        return

    # Initialize YOLO model
    net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # Read input image
    image = cv2.imread(input_img_path)
    if image is None:
        print(f"Error: Unable to load image {input_img_path}")
        return
    (H, W) = image.shape[:2]

    # Create a blob and perform inference
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward([net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers().flatten()])

    # Initialize lists for detections
    boxes, confidences, classIDs = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Only keep detections for "person" (class ID 0 in COCO)
            if confidence > confidence_threshold and classID == 0:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Non-maxima suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, overlapping_threshold)

    image_with_boxes = image.copy()
    inappropriate_detected = False

    predictions = []  # List to store predictions for each person
    inappropriate_detected = False

    if len(indices) == 0:
        label = predict_image(input_img_path)
        # predictions.append(compute_image_confidence(predictions))
        if label == 'inappropriate':
            inappropriate_detected = True
        
        print(f"Image  classified as: {label}")

        print("No persons detected in the image.")
        
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]

             # Ensure the cropped region is within bounds
            y_end = min(y + h, image.shape[0])
            x_end = min(x + w, image.shape[1])
            
            cropped_person = image[y:y_end, x:x_end]

            if cropped_person.size == 0:
                label = predict_image(input_img_path)
                if label == 'inappropriate':
                    inappropriate_detected = True
                print(f"Image  classified as: {label}")
                continue

            cropped_person_path = f"./DARKNET/output/cropped/person_{i + 1}.jpg"

            # Save the cropped person
            if cv2.imwrite(cropped_person_path, cropped_person):
                print(f"Cropped Person {i + 1} Saved: {cropped_person_path}")
            else:
                print(f"Error saving cropped person {i + 1}")

            # Classify the cropped person
            label = predict_image(cropped_person_path)
            print(f"Person {i + 1} classified as: {label}")

            # Check if the person is inappropriate
            if label == 'inappropriate':
                
                inappropriate_detected = True

            # Draw bounding box on the image with classification
            color = (0, 0, 255) if label == 'inappropriate' else (0, 255, 0)
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image_with_boxes, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add the label to the predictions list
            predictions.append({
                'person': int(i + 1),  # Convert NumPy int32 to Python int
                'label': label
            })
            
            
    predictions.append(compute_image_confidence(predictions))
    # Save the output image with bounding boxes
    if cv2.imwrite(output_img_path, image_with_boxes):
        print(f"Image with bounding boxes saved to {output_img_path}")

    # Final classification of the entire image
    if inappropriate_detected:
        predictions.append({"whole_image_label": "inappropriate"})
    else:
        predictions.append({"whole_image_label": "appropriate"})



    predictions.append({"output_image_path":output_img_path})# Calculate the total running time
       # Return the predictions as a list
    end_time = time.time()  # End the timer
    total_time = end_time - start_time 
    predictions.append({"total_time": total_time})# Calculate the total running time
    return predictions


# # Run the YOLO predictions
# if __name__ == '__main__':
#     baseline_get_yolo_preds("./uploads_from_web/test.jpg")
