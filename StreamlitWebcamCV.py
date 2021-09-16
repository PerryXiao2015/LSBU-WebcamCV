#Example 5.10a
#pip install --upgrade protobuf
#pip install streamlit

import streamlit as st
import cv2 
import numpy as np
import pandas as pd
import time

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions

# import the models for further classification experiments=====================
from tensorflow.keras.applications import (
        vgg16,
        resnet50,
        mobilenet,
        inception_v3
    )

import matplotlib.pyplot as plt

# imports for reproducibility
import tensorflow as tf
import random
import os
from keras import backend as K

# Import libraries for YOLO 3/4 ========================================
from yolo_utils import infer_image, show_image
from types import SimpleNamespace

d = {'confidence':0.5,
     'threshold':0.3,
     'weights':'./yolov3-coco/yolov4-tiny.weights',
     'config':'./yolov3-coco/yolov4-tiny.cfg',
     'show_time':True}
FLAGS = SimpleNamespace(**d)
cocolabels='./yolov3-coco/coco-labels'
labels = open(cocolabels).read().strip().split('\n')
print(FLAGS)


colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
net = cv2.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Face Detection Haarcascade code ======================================
cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Webcam ===============================================================
@st.cache(allow_output_mutation=True)
def get_cap():
    return cv2.VideoCapture(0)

# VGG 16 ===============================================================
@st.cache(allow_output_mutation=True)
def vgg16_predict(cam_frame, image_size):
    frame= cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = vgg16.preprocess_input(image_batch.copy())
    predictions = model.predict(processed_image)
    label_vgg = decode_predictions(predictions)
    cv2.putText(cam_frame, "VGG16: {}, {:.2f}".format(label_vgg[0][0][1], label_vgg[0][0][2]) , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
    return cam_frame

@st.cache(allow_output_mutation=True)
def resnet50_predict(cam_frame, image_size):
    frame= cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = resnet50.preprocess_input(image_batch.copy())
    predictions = model.predict(processed_image)
    label_resnet = decode_predictions(predictions, top=3)
    cv2.putText(cam_frame, "ResNet50: {}, {:.2f}".format(label_resnet[0][0][1], label_resnet[0][0][2]) , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
    return cam_frame    

@st.cache(allow_output_mutation=True)
def mobilenet_predict(cam_frame, image_size):
    frame= cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = mobilenet.preprocess_input(image_batch.copy())
    predictions = model.predict(processed_image)
    label_mobilenet = decode_predictions(predictions)
    cv2.putText(cam_frame, "MobileNet: {}, {:.2f}".format(label_mobilenet[0][0][1], label_mobilenet[0][0][2]) , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
    return cam_frame    
    
@st.cache(allow_output_mutation=True)
def inception_v3_predict(cam_frame, image_size):
    frame= cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = inception_v3.preprocess_input(image_batch.copy())
    predictions = model.predict(processed_image)
    label_inception = decode_predictions(predictions)
    cv2.putText(cam_frame, "Inception: {}, {:.2f}".format(label_inception[0][0][1], label_inception[0][0][2]) , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
    return cam_frame    

# Face Detection ====================================================================
@st.cache(allow_output_mutation=True)
def face_detection(cam_frame):
    gray_img=cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
    faces= cascade_classifier.detectMultiScale(gray_img, minNeighbors=5)

    for (x,y,w,h) in faces:
        cv2.rectangle(cam_frame,(x,y),(x+w,y+h),(255,255,0),2)
        cv2.putText(cam_frame,'face', (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
    return cam_frame    

# Find Shapes =======================================================================
@st.cache(allow_output_mutation=True)
def find_shapes(cam_frame):
    gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY) 
    edged = cv2.Canny(gray, 30, 200) 
    _, contours, _ = cv2.findContours(edged,  
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
      
    print("Number of Contours found = " + str(len(contours))) 
    for c in contours:  
        # -1 signifies drawing all contours  3: thickness
        cv2.drawContours(cam_frame, c, -1, (0, 255, 0), 3)
    return cam_frame    
# YOLO v3 and v4 ===================================================================
#@st.cache(allow_output_mutation=True)
def yolov3(cam_frame):
    height, width = cam_frame.shape[:2]
    cam_frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
			height, width, cam_frame, colors, labels, FLAGS)

    return cam_frame    

# Open Webcam ========================================================================
cap = get_cap()

mode = 1
frameST = st.empty()
st.sidebar.markdown("# Image Functions")

# Function Checkbox ==================================================================
function = st.sidebar.selectbox(
     'Select a Function:',
     ["Original","Face Detection","Image Classification","Object Detection", "Shapes"], index=0)
#st.sidebar.write('You selected:', option)

if function == "Original":
    K.clear_session()
    st.title("Webcam Demo")
    mode = 0
elif function == "Face Detection":
    K.clear_session()
    st.title("Face Detection")
    mode = 1
elif function == "Image Classification":
    K.clear_session()
    option = st.sidebar.selectbox(
     'Select a Deep Learning Model:',
     ["VGG16","RESNET50","MOBILENET","INCEPTION_V3"], index=0)
    #st.sidebar.write('You selected:', option)
    if option == "VGG16":
        K.clear_session()
        model = vgg16.VGG16(weights='imagenet')
        image_size = 224
        st.title("Image Classification - VGG16")
        mode = 11
    elif option == "RESNET50":
        K.clear_session()
        model = resnet50.ResNet50(weights='imagenet')
        image_size = 224
        st.title("Image Classification - RESNET50")
        mode = 12
    elif option == "MOBILENET":
        K.clear_session()
        model = mobilenet.MobileNet(weights='imagenet')
        image_size = 224
        st.title("Image Classification - MOBILENET")
        mode = 13
        
    elif option == "INCEPTION_V3":
        K.clear_session()
        model = inception_v3.InceptionV3(weights='imagenet')
        image_size = 299
        st.title("Image Classification - INCEPTION_V3")
        mode = 14

elif function == "Object Detection":
    K.clear_session()
    st.title("Object Detection")
    mode = 3
elif function == "Shapes":
    K.clear_session()
    st.title("Find Shapes")
    mode = 4
    


while True:
    ret, frame = cap.read()

    if mode == 1:
        frame = face_detection(frame)
    elif mode == 3:
        frame = yolov3(frame)
    elif mode == 4:
        frame = find_shapes(frame)
    elif mode == 11:
        frame = vgg16_predict(frame, image_size)
    elif mode == 12:
        frame = resnet50_predict(frame, image_size)
    elif mode == 13:
        frame = mobilenet_predict(frame, image_size)
    elif mode == 14:
        frame = inception_v3_predict(frame, image_size)
       
    # Stop the program if reached end of video
    if not ret:
        cv2.waitKey(3000)
        # Release device
        cap.release()
        break

    frameST.image(frame, channels="BGR")

