import os
import cv2
import numpy as np
import time
from torchvision import transforms
from PIL import Image
import torch
from ultralytics import YOLO
from torch import sigmoid
from yolodetections import yolo_detect
import json
import pickle

print(torch.cuda.is_available())

import warnings
class_names = ['appropriate', 'inappropriate']

# Suppress specific UserWarning for InceptionV3
warnings.filterwarnings("ignore", message="Scripted Inception3 always returns Inception3 Tuple")

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fuse = torch.jit.load("Saved models/fixedresnet_inception_fullmodel_cropped15epochs_32.pt")
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
    _warmup_performed = False
    img_tensor_resnet, img_tensor_inception = preprocess_image(img_path)
    img_tensor_resnet, img_tensor_inception = img_tensor_resnet.to(device), img_tensor_inception.to(device)
    if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if not _warmup_performed:
        with torch.no_grad():
            _ = fuse(img_tensor_resnet, img_tensor_inception)

        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _warmup_performed = True
        print("Warmup completed")
        
    with torch.no_grad():
        
        start_time = time.time()
        outputs = fuse(img_tensor_resnet, img_tensor_inception)
        end_time = time.time()
        elapsed_time = (end_time - start_time)
        predicted_value = outputs.item()  
  
    # Compute confidence (confidence is the output of the model for a class)
    confidence = outputs[0][0] if predicted_value > 0.5 else 1 - outputs[0][0]
    print(confidence)
    # Return the predicted class and confidence level
    predicted_class = class_names[int(predicted_value > 0.5)]
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'time': elapsed_time
    }


def get_yolo_results(cropped_folder,image_path,device,model_path,confidence_threshold):

    results, yolo_elapsed_time = yolo_detect(cropped_folder,image_path,device,model_path,confidence_threshold)
    
    # Save the output to a file (JSON for readability)
    # output_data = {"results": results, "elapsed_time": yolo_elapsed_time}
    # with open("yolo_output.json", "w") as f:
    #     json.dump(output_data, f)
    
    with open("yolo_output.pkl", "wb") as f:
        pickle.dump((results, yolo_elapsed_time), f)

    return results, yolo_elapsed_time

def proposed_get_yolo_preds(image_path: str, output_img_path: str, blurredoutput_img_path: str, is_ad_top=False):

    
    """
    Run YOLOv3u predictions, save cropped persons, classify them, and label the entire image.
    """
    # if is_ad:
    #     output_img_path = "./Whole Image Output/AdPoutput.jpg"  # Ads processed image

    if is_ad_top and not output_img_path:
        output_img_path = "./Whole Image Output/AdPoutputop.jpg"

    else:
       bluroutput_img_path = blurredoutput_img_path
       output_img_path = output_img_path
    cropped_folder = "./DARKNET/output/cropped/"
    model_path = "yolov3u.pt"  # Ensure this path is correct
    confidence_threshold = 0.4
    
    # Ensure output directories exist
    os.makedirs(cropped_folder, exist_ok=True)
    
    # Load YOLOv3u model
    # model = YOLO(model_path)
    
    # Read input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return
    H, W = image.shape[:2]
    
    # # Run inference
    # yolotime = time.time()
    # results = model.predict(image_path, device=device, conf=confidence_threshold, imgsz=416)
    # yolotime_end = time.time()

    results, yolo_elapsed_time = get_yolo_results(cropped_folder,image_path,device,model_path,confidence_threshold)
    
    # Initialize lists for detections
    boxes, confidences, predictions = [], [], []
    image_with_boxes = image.copy()
    blurimage_with_boxes = image.copy()
    inappropriate_detected = False
    I_individual_confidences = []
    A_individual_confidences = []
    totaltime=[]
    
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
        print(f"Image classified as: {label['predicted_class']} with confidence: {label['confidence']}% time: {label['time']} ")
        
        if label['predicted_class'] == 'inappropriate':
            I_individual_confidences.append(float(label['confidence']))
            inappropriate_detected = True
        else:
            A_individual_confidences.append(float(label['confidence']))
        
        totaltime.append(float(label['time']))  
        predictions.append({
            'label': label['predicted_class'],
            'confidence': float(label['confidence']),
            'time': label['time']
            })
        blurredimage=cv2.GaussianBlur(blurimage_with_boxes, (249, 249), 0) 
        
        if cv2.imwrite(output_img_path, image_with_boxes):
            print(f"Image with bounding boxes saved to {output_img_path}")
        if cv2.imwrite(bluroutput_img_path, blurredimage):
            print(f"Image with bounding boxes saved to {bluroutput_img_path}")

        print("No persons detected in the image.")
    
    for i, (x, y, w, h) in enumerate(boxes):
        cropped_person = image[y:y + h, x:x + w]
        person = int(i + 1)
        if cropped_person.size == 0:
            label = predict_image(image_path)           
            print(f"Image classified as: {label['predicted_class']} with confidence: {label['confidence']}% time: {label['time']} ")
            if label['predicted_class'] == 'inappropriate':
                I_individual_confidences.append(float(label['confidence']))
                inappropriate_detected = True
            else:
                A_individual_confidences.append(float(label['confidence']))
            
            totaltime.append(float(label['time']))  
            predictions.append({
            'person': person,
            'label': label['predicted_class'],
            'confidence': float(label['confidence']),
            'time': label['time']
            })

            color = (0, 0, 255) if label['predicted_class'] == 'inappropriate' else (0, 255, 0)
        
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image_with_boxes, f"(person {person}) {label['predicted_class']} ({label['confidence']:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Only blur if the detection is inappropriate (red color)
            if color == (0, 0, 255) or (0, 255, 0):
            # if color == (0, 0, 255): 
                roi = blurimage_with_boxes[y:y+h, x:x+w]
                blurred_roi = cv2.GaussianBlur(roi, (249, 249), 0)
                blurimage_with_boxes[y:y+h, x:x+w] = blurred_roi

            cv2.rectangle(blurimage_with_boxes, (x, y), (x + w, y + h), color, 2)
            cv2.putText(blurimage_with_boxes, f"(person {person}) {label['predicted_class']} ({label['confidence']:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if cv2.imwrite(output_img_path, image_with_boxes):
                print(f"Image with bounding boxes saved to {output_img_path}")
            if cv2.imwrite(bluroutput_img_path, blurimage_with_boxes):
                print(f"Image with bounding boxes saved to {bluroutput_img_path}")

            continue
        
        cropped_person_path = os.path.join(cropped_folder, f"person_{person}.jpg")
        if cv2.imwrite(cropped_person_path, cropped_person):
            print(f"Cropped Person {person} Saved: {cropped_person_path}")
        else:
            print(f"Error saving cropped person {person}")

        # Classify the cropped person
       
        label = predict_image(cropped_person_path)
     
        print(f"Person {person} classified as: {label['predicted_class']} with confidence: {label['confidence']}% time: {label['time']} ")
        
        if label['predicted_class'] == 'inappropriate':
            I_individual_confidences.append(float(label['confidence']))
            inappropriate_detected = True
        else:
            A_individual_confidences.append(float(label['confidence']))

        
        color = (0, 0, 255) if label['predicted_class'] == 'inappropriate' else (0, 255, 0)
        
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image_with_boxes, f"(person {person}) {label['predicted_class']} ({label['confidence']:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Only blur if the detection is inappropriate (red color)
        if color == (0, 0, 255): 
        # if color == (0, 0, 255) or (0, 255, 0):  
            roi = blurimage_with_boxes[y:y+h, x:x+w]
            blurred_roi = cv2.GaussianBlur(roi, (249, 249), 0)
            blurimage_with_boxes[y:y+h, x:x+w] = blurred_roi

        cv2.rectangle(blurimage_with_boxes, (x, y), (x + w, y + h), color, 2)
        cv2.putText(blurimage_with_boxes, f"(person {person}) {label['predicted_class']} ({label['confidence']:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        totaltime.append(float(label['time']))
        predictions.append({
            'person': person,
            'label': label['predicted_class'],
            'confidence': float(label['confidence']),
            'time': label['time']
        })
 
        # Save the output image
        if cv2.imwrite(output_img_path, image_with_boxes):
            print(f"Image with bounding boxes saved to {output_img_path}")
        if cv2.imwrite(bluroutput_img_path, blurimage_with_boxes):
            print(f"Image with bounding boxes saved to {bluroutput_img_path}")

        

    if inappropriate_detected:
        average_confidence = np.mean(I_individual_confidences) if I_individual_confidences else 0.0
        Ap_average_confidence = np.mean(A_individual_confidences) if A_individual_confidences else 0.0
        predictions.append({"whole_image_label": "inappropriate", "total_confidence": average_confidence, "average_opposite_confidence": Ap_average_confidence})
    else:
        average_confidence = np.mean(A_individual_confidences) if A_individual_confidences else 0.0
        In_average_confidence = np.mean(I_individual_confidences) if I_individual_confidences else 0.0
        predictions.append({"whole_image_label": "appropriate", "total_confidence": average_confidence, "average_opposite_confidence": In_average_confidence})
        
   
    # Total running time
    # end_time = time.time()
    totaltime = sum(totaltime)
    
    predictions.append({"output_image_path": output_img_path})
    predictions.append({"blurredoutput_image_path": bluroutput_img_path})
    # predictions.append({"total_time": end_time - start_time}) 
    predictions.append({"total_time": totaltime})
    predictions.append({"yolo_time": yolo_elapsed_time})
    predictions.append({"total_time_with_yolo": yolo_elapsed_time+totaltime})


    # print(predictions)
    return predictions

import uuid
if __name__ == '__main__':
    OUTPUT_FOLDER = "Whole Image Output"
    proposed_output_filename = f"Poutput_{uuid.uuid4()}.jpg"
    blurredproposed_output_filename = f"blurredPoutput_{uuid.uuid4()}.jpg"
    proposed_output_path = os.path.join(OUTPUT_FOLDER, proposed_output_filename)
    blurredproposed_output_path = os.path.join(OUTPUT_FOLDER, blurredproposed_output_filename)
    proposed_get_yolo_preds("./DARKNET/test1.jpg", proposed_output_path,blurredproposed_output_path)


