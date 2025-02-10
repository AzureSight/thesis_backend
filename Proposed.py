import os
import cv2
import numpy as np
import time
from torchvision import transforms
from PIL import Image
import torch
import tensorflow as tf
from ultralytics import YOLO


print(torch.cuda.is_available())

import warnings
class_names = ['appropriate', 'inappropriate']

# Suppress specific UserWarning for InceptionV3
warnings.filterwarnings("ignore", message="Scripted Inception3 always returns Inception3 Tuple")

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fuse = torch.jit.load("Saved models/resnet_inception_fullmodel_croppedv2.pt")
fuse.to(device)
fuse.eval()


resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet normalization
])

inception_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Inception normalization
])

def preprocess_image(img_path):
    img_resnet = Image.open(img_path).convert('RGB')
    img_inception = Image.open(img_path).convert('RGB')

    # Apply transformations for ResNet and Inception
    img_tensor_resnet = resnet_transform(img_resnet).unsqueeze(0)  # Add batch dimension
    img_tensor_inception = inception_transform(img_inception).unsqueeze(0)
    
    return img_tensor_resnet, img_tensor_inception

def predict_image(img_path):
    """
    Predict the class of an image using the combined ResNet50 and InceptionV3 models.
    """
    img_tensor_resnet, img_tensor_inception = preprocess_image(img_path)
    img_tensor_resnet, img_tensor_inception = img_tensor_resnet.to(device), img_tensor_inception.to(device)

    # Forward pass to get predictions
    with torch.no_grad():
        outputs = fuse(img_tensor_resnet, img_tensor_inception)
        predicted_value = outputs.item()  # Extract scalar prediction

    # Compute confidence (confidence is the output of the model for a class)
    confidence = outputs[0][0] if predicted_value > 0.5 else 1 - outputs[0][0]
    print(confidence)
    # Return the predicted class and confidence level
    predicted_class = class_names[int(predicted_value > 0.5)]
    return {
        'predicted_class': predicted_class,
        'confidence': confidence
    }


def proposed_get_yolo_preds(image_path: str):
    start_time = time.time()
    """
    Run YOLOv3u predictions, save cropped persons, classify them, and label the entire image.
    """
    model_path = "yolov3u.pt"  # Ensure this path is correct
    output_img_path = "./Whole Image Output/Poutput.jpg"
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
    boxes, confidences, predictions = [], [], []
    image_with_boxes = image.copy()
    inappropriate_detected = False
    individual_confidences = []
    
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
        individual_confidences.append(float(label['confidence']))
        print(f"Image classified as: {label['predicted_class']} with confidence: {label['confidence']}%")
        
        if label['predicted_class'] == 'inappropriate':
            inappropriate_detected = True
        print("No persons detected in the image.")
    
    for i, (x, y, w, h) in enumerate(boxes):
        cropped_person = image[y:y + h, x:x + w]
        
        if cropped_person.size == 0:
            label = predict_image(image_path)
            individual_confidences.append(float(label['confidence']))
            print(f"Image classified as: {label['predicted_class']} with confidence: {label['confidence']}%")
            if label['predicted_class'] == 'inappropriate':
                inappropriate_detected = True
            continue
        
        cropped_person_path = os.path.join(cropped_folder, f"person_{i + 1}.jpg")
        if cv2.imwrite(cropped_person_path, cropped_person):
            print(f"Cropped Person {i + 1} Saved: {cropped_person_path}")
        else:
            print(f"Error saving cropped person {i + 1}")

        # Classify the cropped person
        label = predict_image(cropped_person_path)
        print(f"Person {i + 1} classified as: {label['predicted_class']} with confidence: {label['confidence']}%")
        
        if label['predicted_class'] == 'inappropriate':
            inappropriate_detected = True
        
        color = (0, 0, 255) if label['predicted_class'] == 'inappropriate' else (0, 255, 0)
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image_with_boxes, label['predicted_class'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        predictions.append({
            'person': int(i + 1),
            'label': label['predicted_class'],
            'confidence': float(label['confidence'])
        })
        individual_confidences.append(float(label['confidence']))
    
    # Compute average confidence
    average_confidence = np.mean(individual_confidences) if individual_confidences else 0.0
    
    # Save the output image
    if cv2.imwrite(output_img_path, image_with_boxes):
        print(f"Image with bounding boxes saved to {output_img_path}")

    if inappropriate_detected:
        predictions.append({"whole_image_label": "inappropriate", "total_confidence": average_confidence})
    else:
        predictions.append({"whole_image_label": "appropriate", "total_confidence": average_confidence})
    
    # Total running time
    end_time = time.time()
    predictions.append({"output_image_path": output_img_path})
    predictions.append({"total_time": end_time - start_time}) 
    
    return predictions




# # # Run the YOLO predictions
# if __name__ == '__main__':
#     proposed_get_yolo_preds("./DARKNET/test.jpg")

