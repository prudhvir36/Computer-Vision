######## Webcam Classification Using Tensorflow-trained Classifier #########
#
# Date: 11/9/20
# Description: 
# This script uses a TensorFlow Lite image classification model to classify
# frames from a USB camera in real time. It draws the inferenced label on each frame.
#
# This code is based off my TensorFlow Lite object detection example at:
# https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_webcam.py

# Import packages
import os
import cv2
import numpy as np
import sys
import importlib.util

# User-defined settings

GRAPH_NAME = 'mobilenet_v1_1.0_224_quant.tflite'   # Name of TFLite model file
LABELMAP_NAME = 'labels.txt'   # Name of labelmap file inside model folder
resolution = [1280, 720]   # Desired camera resolution (width by height)

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the Tensorflow Lite model.
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
#output_details = interpreter.get_output_details()[0]
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize camera
cap = cv2.VideoCapture(1)
cap.set(3, resolution[0])
cap.set(4, resolution[1])

# Initialize framerate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Loop over every image and perform detection
while True:

    t1 = cv2.getTickCount()

    # Get frame from camera and resize to expected shape [1xHxWx3]
    hasFrame, image = cap.read()
    if not hasFrame:
        print('Unable to get frames from camera!')
        break
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape 
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    output_details = interpreter.get_output_details()[0]
    results = interpreter.get_tensor(output_details['index'])[0]

    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        results = scale * (results - zero_point)

    # Get the top result
    top_result = np.argmax(results)
    label = labels[top_result]
    score = results[top_result]

    # Draw results on the image and display it
    cv2.putText(image,"FPS: %.2f" % frame_rate_calc,(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
    cv2.putText(image,'%s: %.3f' % (label, score),(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),5,cv2.LINE_AA)
    cv2.putText(image,'%s: %.3f' % (label, score),(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
    cv2.imshow('Classifier', image)

    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1

    # Press any key to continue to next image, or press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
cap.release()
