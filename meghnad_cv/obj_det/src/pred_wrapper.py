import os
import sys
from typing import List, Tuple, Any

from utils.log import Log
from utils import ret_values
from utils.common_defs import class_header, method_header
from meghnad.core.cv.obj_det.src.pt.inference.pred import PTObjDetPred
from meghnad.core.cv.obj_det.src.tf.inference.pred import TFObjDetPred
from meghnad.core.cv.obj_det.src.dt.inference.pred import DT2ObjDetPred
from meghnad.core.cv.obj_det.src.trf.inference.pred import TRFObjDetPred
from meghnad.core.cv.obj_det.cfg import ObjDetConfig


log = Log()
_config = ObjDetConfig()
_defaults = _config.get('default_constants')['pred']


@class_header(
    description='''
    Wrapper class for Object detection predictions''')
class Predictor:
    def __init__(self, ckpt_path: str):
        self.ckpt_path = ckpt_path
        self.predictor = self._load(ckpt_path)

    @method_header(
        description="""Method that applies Pytorch or Tensorflow pred classes based on the backend selected
            """,
        arguments="""
                path: path to weights file
                """,
        returns="""PT or TF predictor class""")
    def _load(self, path: str):
        if path is None:
            log.ERROR(sys._getframe().f_lineno,
                        __file__, __name__,
                        f'Unable to load model from {path}')
            return ret_values.IXO_RET_INVALID_INPUTS
        # TODO: We may need our own weights format in the future
        if os.path.isfile(path):
            # pytorch weights
            # TODO:
            if True:
                return PTObjDetPred(path)
            else:
                return DT2ObjDetPred(path)
        elif os.path.isdir(path):
            # tensorflow saved model
            return TFObjDetPred(path)
        else:
            # Transformers model id
            return TRFObjDetPred(path)


    @method_header(
        description="""Curates directories, runs inference, performs post processing and processes detections
        """,
        arguments="""
            input: image
            candidate_labels: A list of candidate classes. Its only available for Zero-Shot models
            conf_thres: Minimum confidence required for the object to be shown as detection
            iou_thres: Intersection over Union Threshold
            max_predictions: Maximum nuber of predictions.
            save_img: Whether to save the output images or not.
            result_dir: Output directory.
        """,
        returns="""Prints out the time taken for actual inferencing, and then the post processing steps""")
    def pred(self,
             input: Any,
             candidate_labels: List[str] = _defaults['candidate_labels'],
             conf_thres: float = _defaults['conf_thres'],
             iou_thres: float = _defaults['iou_thres'],
             max_predictions: int = _defaults['max_predictions'],
             save_img: bool = _defaults['save_img'],
             result_dir: str = _defaults['result_dir']) -> Tuple:
        if isinstance(self.predictor, TRFObjDetPred):
            return  self.predictor.predict(
                input, candidate_labels, conf_thres, iou_thres, max_predictions, save_img, result_dir)
        else:
            return self.predictor.pred(
                input, conf_thres, iou_thres, max_predictions, save_img, result_dir)
