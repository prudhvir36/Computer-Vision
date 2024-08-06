import os
import sys
from typing import Tuple, Any

from utils.log import Log
from utils import ret_values
from utils.common_defs import class_header, method_header

from meghnad.core.cv.obj_tracking.src.tracking import VideoTracking

log = Log()

@class_header(
    description='''
    Wrapper class for Object detection predictions''')
class Predictor:
    def __init__(self):
        pass

    @method_header(
        description="""Instantiates Video Tracking Class with required parameters
        """,
        arguments="""
            weights_path: Weights Path to Video Tracking
            class_yaml_path: data.yaml
            conf_thres: Minimum confidence required for the object to be shown as detection
            iou_thres: Intersection over Union Threshold
            img_size: image size
            device: device to run the tracking on
            """,
        returns="""Prints out the time taken for actual inferencing, and then the post processing steps""")
    def pred(self,
            weights_path:str,
            class_yaml_path:str,
            conf_thres: float = 0.25,
            iou_thres:float = 0.45,
            img_size: int = 640, 
            device:str = "cpu"):
        
        return VideoTracking(class_yaml_path, conf_thres, iou_thres, img_size, device, weights_path)