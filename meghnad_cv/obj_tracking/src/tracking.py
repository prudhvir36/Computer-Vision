import os
import sys
import cv2
import json
import torch
from tqdm import tqdm
from typing import List

from utils.log import Log
from utils.ret_values import *
from utils.common_defs import class_header, method_header

from meghnad.core.cv.obj_tracking.src.algorithm.object_detector import YOLO
from meghnad.core.cv.obj_tracking.src.tracker_utils.detections import draw
from meghnad.core.cv.obj_tracking.cfg import ObjTrackingConfig
from meghnad.cfg.config import MeghnadConfig

log = Log()

@class_header(
    description="""Tracking Class for Video & Webcam""")
class VideoTracking:

    def __init__(self, 
                weights_path: str, 
                class_yaml_path: str,
                conf_thres: float = 0.25,
                iou_thres:float = 0.45,
                img_size: int = 640, 
                device: str = "cpu"):
        
        self.weights_path = weights_path
        self.class_yaml_path = class_yaml_path
        self.device = device
        self.config = ObjTrackingConfig(MeghnadConfig())

        if self.config.get_meghnad_configs('DEVICE') != 'cpu' and self.config.get_meghnad_configs('DEVICE') != None:
            if torch.cuda.is_available():
                self.device = self.config.get_meghnad_configs('DEVICE')
        
        self.setting = {
            'conf_thres':conf_thres,
            'iou_thres':iou_thres,
            'img_size':img_size,
        }

        self.yolo = YOLO(self.setting['conf_thres'],
                         self.setting["iou_thres"],
                         self.setting['img_size'])
        
        self.load = self.yolo.load(self.weights_path,
                                     classes=self.class_yaml_path,
                                     device=self.device)

    @method_header(
        description="""
        Tracking Videos in a directory""",
        arguments="""
        video_path:str = Input directory for video files,
        detect_class:List[str] = List of Specific Classes to be Tracked
         """)
    def track_video(self,
                    video_path: str,
                    output_dir: str,
                    detect_class: List[str] = None):

        video_file_paths = []
        for root, dirs, files in os.walk(video_path):
            for name in files:
                if name.endswith((".mp4", ".mkv")):
                    video_file_paths.append((os.path.join(root, name), name))

        for idx in video_file_paths:
            input_video_path, name = idx
            video = cv2.VideoCapture(input_video_path)
            width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video.get(cv2.CAP_PROP_FPS))
            frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            output = cv2.VideoWriter(os.path.join(output_dir, f"tracked_{name}"),
                                     fourcc, fps, (width, height))

            if video.isOpened() == False:
                log.ERROR(sys._getframe().f_lineno, __file__, __name__,
                          "[!] Error Opening Video")

            log.STATUS(sys._getframe().f_lineno, __file__, __name__,"[-->] Tracking video")
            pbar = tqdm(total=frames_count, unit=' frames', dynamic_ncols=True, position=0, leave=True)

            try:
                while video.isOpened():
                    ret, frame = video.read()
                    if ret == True:
                        detections = self.yolo.detect(frame, track=True)
                        if detect_class is not None:
                            classes_detections = [det for det in detections if det["class"] in detect_class]
                            detected_frame = draw(frame, classes_detections)
                            output.write(detected_frame)
                            pbar.update(1)
                        else:
                            detected_frame = draw(frame, detections)
                            output.write(detected_frame)
                            pbar.update(1)
                    else:
                        break
            except KeyboardInterrupt:
                pass

            pbar.close()
            video.release()
            output.release()
            self.yolo.unload()


    @method_header(
        description="""Tracking Webcam""",
        arguments="""
        use_webcam:int = Which Webcam to Use""")
    def track_webcam(self, use_webcam:int = 0) -> None:

        webcam = cv2.VideoCapture(use_webcam)
        if webcam.isOpened() == False:
            log.ERROR(sys._getframe().f_lineno, __file__, __name__,
                      "[!] Error Opening Webcam")
        try:
            while webcam.isOpened():
                ret, frame = webcam.read()
                if ret == True:
                    detections = self.yolo.detect(frame, track=True)
                    detected_frame = draw(frame, detections)
                    print(json.dumps(detections, indent=4))
                    cv2.imshow('webcam', detected_frame)
                    cv2.waitKey(1)
                else:
                    break
        except KeyboardInterrupt:
            pass

        webcam.release()
        log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                   "[-] Webcam Closed")
        self.yolo.unload()