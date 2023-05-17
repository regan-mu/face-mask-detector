import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Facemask Detector",
    page_icon="random",
    initial_sidebar_state="expanded",
    layout="wide"
)


def load_face_detector_model():
    config_path = os.path.sep.join(
        ["face_detector", "deploy.prototxt.txt"])
    model_path = os.path.sep.join(
        ["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(config_path, model_path)
    return net


def prediction(image):
    mask_prediction = None
    image = cv2.imdecode(np.frombuffer(image.read(), dtype='uint8'), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    network = load_face_detector_model()
    model = load_model("../Models/model.h5")
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    network.setInput(blob)
    detection = network.forward()
    for i in range(0, detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.5:
            # Get the X and Y coordinates
            box = detection[0, 0, i, 3:7]*np.array([h, w, h, w])
            (startx, starty, endx, endy) = box.astype('int')

            # Ensure the bounding boxes fall within the dimensions of the frame
            startx, starty = (max(0, startx), max(0, starty))
            endx, endy = (min(w-1, endx), min(h-1, endy))

            # Extract the ROI of the image, convert it to RGB and resize it to 350x350 and preprocess it
            face = image[starty:endy, startx:endx]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (350, 350))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            with_mask, without_mask = model.predict(face)[0]

            # determine the bounding box and label color
            mask_prediction = 'Mask' if with_mask > without_mask else 'No mask'
            color = (0, 255, 0) if mask_prediction == 'Mask' else (255, 0, 0)

            # include probability in the label
            label = '{} : {:.1f}%'.format(mask_prediction, max(with_mask, without_mask)*100)

            # Add the label and bounding boxes
            cv2.putText(image, label, (startx, starty-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startx, starty), (endx, endy), color, 2)
    return image, mask_prediction


# Sidebar
st.sidebar.header("Facemask Detector".upper())
capture_method = st.sidebar.radio("Face Capture Method", ("Face Upload", "Camera Capture"))

face_image = None
if capture_method == "Face Upload":
    face_image = st.sidebar.file_uploader("Upload Face Image", type=["png", "jpg", "svg", "jpeg"])
elif capture_method == "Camera Capture":
    face_image = st.sidebar.camera_input("Capture Face")

st.sidebar.subheader("Source code [Github](https://github.com/regan-mu/face-mask-detector)")
st.sidebar.write("Predict whether the face detected is wearing a mask or not.")

# Main Page
uploaded, predicted = st.columns(2)
if face_image:
    with uploaded:
        st.image(face_image)
        st.text("Original Image")
    with predicted:
        image_pred, label = prediction(face_image)
        st.image(image_pred)
        st.text(f"Prediction: {label}")
