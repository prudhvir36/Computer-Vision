import os
import yaml
import torch
import random
import numpy as np
from typing import List

from utils.log import Log
from utils.ret_values import *
from utils.common_defs import class_header, method_header

from meghnad.core.cv.obj_tracking.src.byte_tracker import BYTETracker
from meghnad.repo.obj_det.yolov5.utils.torch_utils import select_device
from meghnad.repo.obj_det.yolov5.models.experimental import attempt_load
from meghnad.core.cv.obj_tracking.src.tracker_utils.datasets import letterbox
from meghnad.core.cv.obj_tracking.src.tracker_utils.general import scale_coords
from meghnad.core.cv.obj_tracking.src.tracker_utils.detections import Detections
from meghnad.repo.obj_det.yolov5.utils.general import check_img_size, non_max_suppression

log = Log()

@class_header(description="""
YOLO class for loading & detecting
""")
class YOLO:
    def __init__(self, 
                 conf_thres: float = 0.25, 
                 iou_thres: float = 0.45, 
                 img_size: int = 640):
        
        self.settings = {
            'conf_thres':conf_thres,
            'iou_thres':iou_thres,
            'img_size':img_size,
        }
        self.tracker = BYTETracker()

    @method_header(description = """
    loading model weights
    we are generating hex color codes for object bounding box""",
    arguments = """
    weights:str = best weights to load the model,
    classes:List[str]  = classes to be detected,
    device:str = device to be used - cpu/gpu
    """)
    def load(self, weights_path: str, classes: str, device:str ='cpu'):                    
        with torch.no_grad():
            self.device = select_device(self.device)
            self.model = attempt_load(weights_path, device= self.device)        

            if device != 'cpu':
                self.model.half()
                self.model.to(self.device).eval()
                
            stride = int(self.model.stride.max())
            self.imgsz = check_img_size(self.settings['img_size'], s=stride)
            class_names = []
            class_yaml = {}

            with open(classes, 'r') as inputs:
            
                input_content = yaml.safe_load(inputs)
                no_of_classes = len(input_content["names"])
                
                r = lambda: random.randint(0,255)
                hex_colors_lst = ['#%02X%02X%02X' % (r(),r(),r()) for i in range(no_of_classes)]
                
                for i, j in zip(hex_colors_lst, input_content["names"]):
                    color_class_dct = {'name': j, 'color': i}
                    color_class_dct["color"] = i
                    class_names.append(color_class_dct)
                class_yaml["classes"] = class_names
                self.classes = class_yaml["classes"]
            self.classes = self.classes

    @method_header(description="""
    Releases and Empties Cuda Cache""")
    def unload(self):
        if self.device.type != 'cpu':
            torch.cuda.empty_cache()


    @method_header(description="""
    Preprocess an input image: Resize, Scale, Pad etc..""",
    arguments="""
    img: Input of an Image""")
    def __parse_image(self, img):
        im0 = img.copy()
        img = letterbox(im0, self.imgsz, auto=self.imgsz != 1280)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.device.type != 'cpu' else img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return im0, img


    @method_header(description="""
    Detecting the input image""",
    arguments="""
    img = Input Image to be detected""")
    def detect(self, img, track=True):
        with torch.no_grad():
            im0, img = self.__parse_image(img)
            pred = self.model(img)[0]
            pred = non_max_suppression(pred, self.settings['conf_thres'], self.settings['iou_thres'])
            raw_detection = np.empty((0,6), float)

            for det in pred:
                if len(det) > 0:
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        raw_detection = np.concatenate((raw_detection, [[int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), round(float(conf), 2), int(cls)]]))

            if track:
                raw_detection = self.tracker.update(raw_detection)
            
            detections = Detections(raw_detection, self.classes, tracking=track).to_dict()

            return detections