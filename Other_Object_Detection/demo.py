from __future__ import division, print_function, absolute_import
import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
warnings.filterwarnings('ignore')


def main(yolo):
    video_capture = cv2.VideoCapture("5.mp4")

    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output_5.avi', fourcc, 15, (w, h))
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs, predicted_classes = yolo.detect_image(image)
        for bbox, class_name in zip(boxs, predicted_classes):
            cv2.putText(frame, str(class_name), (int(bbox[0]), int(bbox[1])-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(
                bbox[2]+bbox[0]), int(bbox[3]+bbox[1])), (255, 255, 0), 2)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())
