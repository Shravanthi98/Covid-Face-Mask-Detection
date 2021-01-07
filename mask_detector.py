# import the necessary packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2

# Input arguments
ap = argparse.ArgumentParser()
ap.add_argument("--i", "--image", required=True, help="Path to the input image")
ap.add_argument("--m", "--model", required=True, help="Path to the saved model")
ap.add_argument("--p", "--prototxt", required=True, help="Path to the deploy.prototxt file")
ap.add_argument("--w", "--weights", required=True, help="Path to the model weights for face detection")
ap.add_argument("--c", "--confidence", default=0.5)
args = vars(ap.parse_args())

# Loading the image and extracting its shape
image = cv2.imread(args["i"])
orig_image = image.copy()
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
    (300, 300), (104.0, 177.0, 123.0))

# Loading the model
model = load_model(args["m"]+'covid_mask_detector.model')

# Perform Face Detection
# Load the corresponding model architecture and weights
print("Loading the model")
net = cv2.dnn.readNetFromCaffe(args["p"], args["w"])

# Pass the blob through the network and obtain the detections and predictions
print("Computing face detections")
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
    # Extract the confidence (i.e., probability) associated with the prediction
    confidence = detections[0, 0, i, 2]
    # Filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence > args["c"]:
        # Compute the (x, y)-coordinates of the bounding box for the face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # Ensure the bounding boxes fall within the dimensions of the frame
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        # Extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
        face = image[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # Feed the face to the model
        (mask, withoutMask) = model.predict(face)[0]

        # Determine the class label and color to draw the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Include the probability/confidence in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # Display the label and bounding box rectangle on the image
        cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# Show the output image
cv2.imshow("Output_image", image)
cv2.waitKey(0)







