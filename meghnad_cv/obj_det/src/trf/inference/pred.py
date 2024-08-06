import sys
import os
import glob
from pathlib import Path
from typing import Optional, List, Tuple, Any

from transformers import pipeline
import cv2
from PIL import Image
import numpy as np

from utils.log import Log
from utils.common_defs import class_header, method_header
from meghnad.core.cv.obj_det.cfg import ObjDetConfig


__all__ = ['TRFObjDetPred']

log = Log()
_config = ObjDetConfig()
_defaults = _config.get('default_constants')['pred']


@class_header(
    description='''
    Class for Object detection predictions''')
class TRFObjDetPred:
    def __init__(self,
                 weights: str) -> None:
        self.weights = weights

        try:
            self.detector = pipeline(model=weights, task="zero-shot-object-detection")
        except Exception as e:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      f'An error has occurred while initializing detector')

    def _unpack_predictions(slef, predictions: List) -> List:
        boxes = []
        scores = []
        labels = []
        for prediction in predictions:
            boxes.append(list(prediction['box'].values()))
            scores.append(prediction['score'])
            labels.append(prediction['label'])
        return boxes, scores, labels

    def _draw_img(self, img: np.ndarray, boxes: List, scores: List, labels: List) -> np.ndarray:
        for box, score, label in zip(boxes, scores, labels):
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            cv2.putText(img, f'{label}: {round(score, 2)}', (xmin, ymin),
                        cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 255, 255), 1, cv2.LINE_AA)
        return img

    @method_header(
        description="""Curates directories, runs inference, performs post processing and processes detections
        """,
        arguments="""
            input: image
            candidate_labels: A list of candiate labels, e.g ['dog', 'cat', 'handbag']
            conf_thres: Minimum confidence required for the object to be shown as detection
            iou_thres: Intersection over Union Threshold
        """,
        returns="""Prints out the time taken for actual inferencing, and then the post processing steps""")
    def predict(self,
             input: Any,
             candidate_labels: List[str] = _defaults['candidate_labels'],
             conf_thres: float = _defaults['conf_thres'],
             iou_thres: float = _defaults['iou_thres'],
             max_predictions: int = _defaults['max_predictions'],
             save_img: bool = _defaults['save_img'],
             result_dir: str = _defaults['result_dir']) -> Tuple:
        os.makedirs(result_dir, exist_ok=True)
        if os.path.isfile(input):
            log.STATUS(sys._getframe().f_lineno,
                     __file__, __name__,
                     f'Running inference on {input}')
            i = input.rfind('.')
            ext = input[i:]
            if ext.lower() in _config.get('default_constants')['trf']['inference']['pred']['supported_image_exts']:
                image = Image.open(input).convert('RGB')
                predictions = self.detector(image, candidate_labels=candidate_labels)
                boxes, scores, labels = self._unpack_predictions(predictions)
                if save_img:
                    outpath = os.path.join(result_dir, Path(input).stem + '.jpg')
                    image = np.array(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = self._draw_img(image, boxes, scores, labels)
                    cv2.imwrite(outpath, image)
                return [[boxes, scores, labels]]
            elif ext.lower() in _config.get('default_constants')['trf']['inference']['pred']['supported_video_exts']:
                cap = cv2.VideoCapture(input)

                outpath = os.path.join(result_dir, Path(input).stem + '.avi')
                writer = None
                results = []
                while True:
                    success, im0 = cap.read()
                    if not success:
                        break
                    image = Image.fromarray(im0[:, :, ::-1])  # BGR -> RGB
                    predictions = self.detector(image, candidate_labels=candidate_labels)

                    if save_img:
                        if writer is None:
                            fps = int(cap.get(cv2.CAP_PROP_FPS))
                            frame_h, frame_w = im0.shape[:2]
                            writer = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_w, frame_h))

                        log.STATUS(sys._getframe().f_lineno,
                                  __file__, __name__,
                                  f'Saving output')

                        boxes, scores, labels = self._unpack_predictions(predictions)
                        im0 = self._draw_img(im0, boxes, scores, labels)

                        results.append([boxes, scores, labels])

                        if writer:
                            writer.write(im0)

                if writer:
                    writer.release()
                return results
            else:
                log.ERROR(sys._getframe().f_lineno,
                          __file__, __name__,
                          f'Not supported file: {ext}')
        elif os.path.isdir(input):
            exts = ['.jpg', '.jpeg', '.png']
            image_paths = []
            for ext in exts:
                image_paths += list(glob.glob(os.path.join(input, f'*{ext}')))

            # Inference
            results = []
            for image_path in image_paths:
                image = Image.open(image_path).convert('RGB')
                predictions = self.detector(image, candidate_labels=candidate_labels)
                boxes, scores, labels = self._unpack_predictions(predictions)
                if save_img:
                    outpath = os.path.join(result_dir, Path(image_path).stem + '.jpg')
                    image = np.array(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = self._draw_img(image, boxes, scores, labels)
                    cv2.imwrite(outpath, image)
                results.append([boxes, scores, labels])
            return results