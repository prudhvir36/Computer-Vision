import os
import glob
import sys
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf

from meghnad.core.cv.obj_det.src.tf.model_loader.utils import decode, compute_nms
from utils import ret_values
from utils.log import Log
from utils.common_defs import class_header, method_header
from meghnad.core.cv.obj_det.cfg import ObjDetConfig

__all__ = ['TFObjDetPred']

log = Log()
_config = ObjDetConfig()
_defaults = _config.get('default_constants')['pred']

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']


@class_header(
    description='''
    Class for Object detection predictions''')
class TFObjDetPred:
    def __init__(self,
                 model_dir: str) -> None:
        self.saved_dir = model_dir
        model_params_path = os.path.join(model_dir, 'metadata.npz')
        metadata = np.load(model_params_path)
        self.default_boxes = metadata['default_boxes']
        self.input_shape = metadata['input_shape']

        self.model = tf.saved_model.load(model_dir)
        log.STATUS(sys._getframe().f_lineno,
                   __file__, __name__,
                   f'Loaded model from {model_dir}')

    @method_header(
        description='''
                Function to preprocess images''',
        arguments='''
                image : image : Pass image to the function
                ''',
        returns='''
                returns image in form of a numpy array and a tuple having height and width''')
    def _preprocess(self, image: np.ndarray) -> Tuple:
        if image.ndim != 3:
            raise ValueError('Input image should be a 3-D tensor.')
        input_height, input_width = self.input_shape[:2]
        h, w = image.shape[:2]
        if h != input_height or w != input_width:
            image = cv2.resize(image, (input_width, input_height))
        image = image.astype(np.float32)
        image /= 255.
        return image, (h, w)

    @method_header(
        description='''
                Predict the input''',
        arguments='''
                input: input is image to the function
                conf_thres: Score threshold value.
                iou_thres: NMS threshold value.
                max_predictions: Maximum number of predictions per image.''',
        returns='''
                bboxes: Predicted bounding boxes (N, 4)
                classes: Predicted classes (N, num_classes)
                scores: Predicted scores (N)
        ''')
    def pred(self,
             input: Union[str, np.ndarray],
             conf_thres: float = _defaults['conf_thres'],
             iou_thres: float = _defaults['iou_thres'],
             max_predictions: int = _defaults['max_predictions']) -> Tuple:
        if not self.model:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, "Model is null")
            return ret_values.IXO_RET_INVALID_INPUTS

        if isinstance(input, np.ndarray):
            if input.ndim == 3:
                input, shape = self._preprocess(input)
                inputs = np.array([input])
                shapes = np.array([shape])
            elif input.ndim == 4:
                inputs = input
                shape = input.shape[1:3]
                shapes = np.tile([shape], [input.shape[0], 1])
            else:
                log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, f"Invalid input shape.")
                return ret_values.IXO_RET_INVALID_INPUTS, (None, None, None)
                
        elif isinstance(input, str):
            if os.path.isdir(input):
                image_paths = []
                for ext in IMAGE_EXTENSIONS:
                    image_paths += list(glob.glob(os.path.join(input,
                                                               f'*{ext}')))

                inputs = []
                shapes = []
                for image_path in image_paths:
                    image = cv2.imread(image_path)
                    image = image[:, :, ::-1]
                    image, shape = self._preprocess(image)
                    inputs.append(image)
                    shapes.append(shape)
                inputs = np.array(inputs)
                shapes = np.array(shapes)
            elif os.path.isfile(input):
                image = cv2.imread(input)
                image = image[:, :, ::-1]
                image, shape = self._preprocess(image)
                inputs = [image]
                shapes = [shape]
            else:
                log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, f"Invalid input type: {input} has type of {type(input)}")
                return ret_values.IXO_RET_INVALID_INPUTS, (None, None, None)
        else:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, "Not supoorted input type")
            return ret_values.IXO_RET_NOT_SUPPORTED, (None, None, None)

        # (H, W, C) -> (N, H, W, C)
        batch_confs, batch_locs = self.model(inputs, training=False)

        batch_boxes = []
        batch_classes = []
        batch_scores = []
        for confs, locs, shape in zip(batch_confs, batch_locs, shapes):
            image_height, image_width = shape

            confs = tf.math.softmax(confs, axis=-1)

            boxes = decode(self.default_boxes, locs)

            out_boxes = []
            out_labels = []
            out_scores = []

            num_classes = confs.shape[1]
            for c in range(1, num_classes):
                cls_scores = confs[:, c]

                score_idx = cls_scores > conf_thres
                cls_boxes = boxes[score_idx]
                cls_scores = cls_scores[score_idx]

                nms_idx = compute_nms(cls_boxes, cls_scores,
                                      iou_thres, max_predictions)
                cls_boxes = tf.gather(cls_boxes, nms_idx)
                cls_scores = tf.gather(cls_scores, nms_idx)
                cls_labels = [c] * cls_boxes.shape[0]

                out_boxes.append(cls_boxes)
                out_labels.extend(cls_labels)
                out_scores.append(cls_scores)

            out_boxes = tf.concat(out_boxes, axis=0)
            out_scores = tf.concat(out_scores, axis=0)

            boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
            boxes = boxes * np.array([[image_width, image_height,
                                       image_width, image_height]])
            classes = np.array(out_labels)
            scores = out_scores.numpy()

            batch_boxes.append(boxes)
            batch_classes.append(classes)
            batch_scores.append(scores)

        return ret_values.IXO_RET_SUCCESS, (batch_boxes, batch_classes, batch_scores)
