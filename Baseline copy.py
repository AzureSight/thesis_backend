
import cv2
import numpy as np
import tensorflow as tf
import os
import time
import joblib
import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

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
load_ex = torch.load('Saved models/svm/fullresnetextractor.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_ex.to(device)

svm_model_path = os.path.join('Saved models/svm', 'fulltorch_svm.pkl')
load_svm = joblib.load(svm_model_path)

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet normalization
])

def preprocess_image(img_path):
    img_resnet = Image.open(img_path).convert('RGB')
    img_tensor_resnet = resnet_transform(img_resnet).unsqueeze(0).to(device)

 
    
    return img_tensor_resnet


def predict_image(img_path):
    

    img = Image.open(img_path).convert("RGB")
    img_tensor = resnet_transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device
    load_ex.eval()  # Ensure the model is in evaluation mode
    
    with torch.no_grad():
        feature_map = load_ex(img_tensor)  
        flattened_features = feature_map.squeeze().cpu().numpy().reshape(1, -1)  
    prediction = load_svm.predict(flattened_features)  # Predict using the trained SVM classifier
    return class_names[prediction[0]]



def baseline_get_yolo_preds(image_path: str):
    start_time = time.time()
    """
    Run YOLOv3u predictions, save cropped persons, classify them, and label the entire image.
    """
    model_path = "yolov3u.pt"  # Ensure this path is correct
    output_img_path = "./Whole Image Output/Boutput.jpg"
    cropped_folder = "./DARKNET/output/cropped/"
    confidence_threshold = 0.4
    
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
    results = model.predict(image_path, device="cuda", conf=confidence_threshold)
    
    # Initialize lists for detections
    boxes, confidences = [], []
    image_with_boxes = image.copy()
    inappropriate_detected = False
    predictions = []
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())  # Get class ID
            confidence = box.conf[0].item()  # Confidence score
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            
            if confidence > confidence_threshold and class_id == 0:  # Class ID 0 is "person"
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(confidence)
    
    # Process detected persons
    if not boxes:
        label = predict_image(image_path)
        if label == 'inappropriate':
            inappropriate_detected = True
        print(f"Image classified as: {label}")
        print("No persons detected in the image.")
    
    for i, (x, y, w, h) in enumerate(boxes):
        cropped_person = image[y:y + h, x:x + w]
        
        if cropped_person.size == 0:
            label = predict_image(image_path)
            if label == 'inappropriate':
                inappropriate_detected = True
            print(f"Image classified as: {label}")
            continue
        
        cropped_person_path = os.path.join(cropped_folder, f"person_{i + 1}.jpg")
      
        if cv2.imwrite(cropped_person_path, cropped_person):
           print(f"Cropped Person {i + 1} Saved: {cropped_person_path} with confidence {confidences[i]:.2f}")
        else:
            print(f"Error saving cropped person {i + 1}")
        # Classify the cropped person
        label = predict_image(cropped_person_path)
        print(f"Person {i + 1} classified as: {label}")
        
        # Check if the person is inappropriate
        if label == 'inappropriate':
            inappropriate_detected = True
        
        # Draw bounding box on the image
        color = (0, 0, 255) if label == 'inappropriate' else (0, 255, 0)
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image_with_boxes, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        predictions.append({
            'person': int(i + 1),
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
    
    predictions.append({"output_image_path": output_img_path})
    end_time = time.time()
    predictions.append({"total_time": end_time - start_time})
    
    return predictions



# # # Run the YOLO predictions
# if __name__ == '__main__':
#     baseline_get_yolo_preds("./DARKNET/test.jpg")

