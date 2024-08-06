import os
import cv2
import json
from datetime import datetime
from typing import Optional, Tuple, Any, List

import torch.cuda

from utils.log import Log
from utils.common_defs import class_header, method_header

from detectron2.config import get_cfg
from detectron2.data.catalog import Metadata
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from meghnad.core.cv.obj_det.cfg import ObjDetConfig

setup_logger()

__all__ = ['DT2ObjDetPred']

log = Log()
_config = ObjDetConfig()
_defaults = _config.get('default_constants')['pred']


@class_header(
    description='''
    Class for Detectron2 Object detection predictions''')
class DT2ObjDetPred:
    def __init__(self,
                 weights: str,
                 output_dir: Optional[str] = './results') -> None:

        self.weights = weights
        self.output_dir = output_dir
  
        cls_names_file_path = os.path.join(output_dir)

        with open(os.path.join(cls_names_file_path, "cls_name_categories.json"), "r") as read_file:
            dct_cls_names = json.load(read_file)
            cls_names = list(dct_cls_names.values())

        my_metadata = Metadata()
        self.my_metadata = my_metadata.set(thing_classes = cls_names)

        metadata_path = os.path.join(f"runs", f"dt2",
                                        f"metadata")

        
        #setting the required config for the model and loading the model
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(metadata_path, "config.yaml"))
        cfg.MODEL.WEIGHTS = self.weights
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'

        self.predictor = DefaultPredictor(cfg)

    @method_header(
    description="""Curates directories, runs inference, performs post processing and processes detections
    """,
    arguments="""
        input: image
        conf_thres: Minimum confidence required for the object to be shown as detection
        iou_thres: Intersection over Union Threshold
    """,
    returns="""Prints out the time taken for actual inferencing, and then the post processing steps""")
    def pred(self,
                input: Any,
                conf_thres: float = _defaults['conf_thres'],
                iou_thres: float = _defaults['iou_thres'],
                ) -> Tuple:

        cfg = get_cfg()
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thres
        cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = iou_thres
        cfg.OUTPUT_DIR = self.output_dir
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        for root, dirs, files in os.walk(input):
            for name in files:
                if name.endswith((".jpg", ".png")):
                    image_file = name
                    im = cv2.imread(os.path.join(input, image_file))
                    outputs = self.predictor(im)  
                    v = Visualizer(im[:, :, ::-1],
                                    metadata= self.my_metadata, 
                                    scale=0.5, 
                                    instance_mode=ColorMode.IMAGE
                                )
                    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                    current_time = datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")
                    cv2.imwrite(os.path.join(self.output_dir, f"results_image_{current_time}.jpg"), out.get_image()[:, :, ::-1])