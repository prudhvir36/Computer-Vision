import sys
from typing import Tuple, Any

import torch

from utils import ret_values
from utils.log import Log
from utils.common_defs import class_header, method_header
from meghnad.core.cv.obj_det.cfg import ObjDetConfig


__all__ = ['PTObjDetPred']

log = Log()
_config = ObjDetConfig()
_defaults = _config.get('default_constants')['pred']

@class_header(
    description='''
    Class for Object detection predictions''')
class PTObjDetPred:
    def __init__(self,
                 weights: str) -> None:
        self.weights = weights

        # get arch
        arch = torch.load(self.weights, map_location='cpu').get('arch')
        if arch in ('yolov5', 'yolov7'):
            from meghnad.core.cv.obj_det.src.pt.inference.infer_utils.infer_utils_yolo import detect
            self.predict_fn = detect
        else:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, f"Not supported arch {self.predict_fn}")
            #raise ValueError(f'Not supported arch {self.predict_fn}')

    @method_header(
        description="""Curates directories, runs inference, performs post processing and processes detections
        """,
        arguments="""
            input: image
            conf_thres: Minimum confidence required for the object to be shown as detection
            iou_thres: Intersection over Union Threshold
            max_predictions: Maximum nuber of predictions.
            save_img: Whether to save the output images or not.
            result_dir: Output directory.
        """,
        returns="""Prints out the time taken for actual inferencing, and then the post processing steps""")
    def pred(self,
             input: Any,
             conf_thres: float = _defaults['conf_thres'],
             iou_thres: float = _defaults['iou_thres'],
             max_predictions: int = _defaults['max_predictions'],
             save_img: bool = _defaults['save_img'],
             result_dir: str = _defaults['result_dir']) -> Tuple:
        self.predict_fn(
            input,
            self.weights,
            result_dir,
            conf_thres,
            iou_thres,
            max_predictions,
            save_img)