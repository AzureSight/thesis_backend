import os
import cv2
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception

# Load the classifier model
model_save_path = 'Saved models/resnetxinceptionVVV2.keras'
# model_save_path = 'Saved models/resnetxinceptionV5_TRAINED_ON_CROPPED.keras'
loaded_proposed = tf.keras.models.load_model(model_save_path)
class_names = ['appropriate', 'inappropriate']  # Adjust these based on your model's classes


def predict_image(img_path):
    """
    Predict the class of an image using the combined ResNet50 and InceptionV3 models.
    """
    # Load the image for ResNet50 and InceptionV3
    img_resnet = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224), color_mode='rgb')
    img_inception = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299), color_mode='rgb')

    # Preprocess the images
    img_array_resnet = preprocess_resnet(np.expand_dims(tf.keras.preprocessing.image.img_to_array(img_resnet), axis=0))
    img_array_inception = preprocess_inception(np.expand_dims(tf.keras.preprocessing.image.img_to_array(img_inception), axis=0))

    # Predict using the loaded model
    prediction = loaded_proposed.predict([img_array_resnet, img_array_inception])
    predicted_value = prediction[0].item()

    # Compute confidence (confidence is the output of the model for a class)
    confidence = prediction[0][0] if predicted_value > 0.5 else 1 - prediction[0][0]
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
    Run YOLO predictions, save cropped persons, classify them, and label the entire image.
    """
    labels_path = "./DARKNET/coco.txt"
    yolo_cfg = "./DARKNET/model_data/yolov3.cfg"
    yolo_weights = "./DARKNET/model_data/yolov3.weights"
    input_img_path = image_path
    output_img_path = "./Whole Image Output/Poutput.jpg"
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
    
    predictions = []  # List to store predictions for each person
    inappropriate_detected = False
    individual_confidences = []  # List to store individual confidence val
    
 
    if len(indices) == 0:
        label = predict_image(input_img_path)
        individual_confidences.append(float(label['confidence']))
        print(f"Image  classified as: {label['predicted_class']} with confidence: {label['confidence']}%")
        
        if label['predicted_class'] == 'inappropriate':
                inappropriate_detected = True
                
        print("No persons detected in the image.")
   
           
    if len(indices) > 0:
        print("Number of Personsiin:", indices)
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]

             # Ensure the cropped region is within bounds
            y_end = min(y + h, image.shape[0])
            x_end = min(x + w, image.shape[1])
            
            cropped_person = image[y:y_end, x:x_end]

            if cropped_person.size == 0:
                label = predict_image(input_img_path)
                individual_confidences.append(float(label['confidence']))
                print(f"Image  classified as: {label['predicted_class']} with confidence: {label['confidence']}%")
                if label['predicted_class'] == 'inappropriate':
                        inappropriate_detected = True                    
                continue

            cropped_person_path = f"./DARKNET/output/cropped/person_{i + 1}.jpg"

            # Save the cropped person
            if cv2.imwrite(cropped_person_path, cropped_person):
                print(f"Cropped Person {i + 1} Saved: {cropped_person_path}")
            else:
                print(f"Error saving cropped person {i + 1}")

            # Classify the cropped person
            label = predict_image(cropped_person_path)
            print(f"Person {i + 1} classified as: {label['predicted_class']} with confidence: {label['confidence']}%")

            # Check if the person is inappropriate
            if label['predicted_class'] == 'inappropriate':
                inappropriate_detected = True

            # Draw bounding box on the image with classification
            color = (0, 0, 255) if label['predicted_class'] == 'inappropriate' else (0, 255, 0)
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image_with_boxes, label['predicted_class'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add the label to the predictions list
            predictions.append({
                'person': int(i + 1),  # Convert NumPy int32 to Python int
                'label': label['predicted_class'],
                'confidence': float(label['confidence'])
            })
            individual_confidences.append(float(label['confidence']))
            
    
    # If we have any predictions, calculate the average confidence for the entire image
    if individual_confidences:
        average_confidence = np.mean(individual_confidences)
    else:
        average_confidence = 0.0  # If no persons detected, set it to 0
        
    # Save the output image with bounding boxes
    if cv2.imwrite(output_img_path, image_with_boxes):
        print(f"Image with bounding boxes saved to {output_img_path}")

     # Final classification of the entire image based on majority vote of person labels
    if inappropriate_detected:
        predictions.append({"whole_image_label": "inappropriate", "total_confidence": average_confidence})
    else:
        predictions.append({"whole_image_label": "appropriate", "total_confidence": average_confidence})

    
    
    # Total running time
    end_time = time.time()
    total_time = end_time - start_time
    predictions.append({"output_image_path":output_img_path})# Calculate the total running time
    predictions.append({"total_time": total_time}) # Calculate the total running time

    return predictions



# # Run the YOLO predictions
# if __name__ == '__main__':
#     proposed_get_yolo_preds("./uploads_from_web/test.jpg")
