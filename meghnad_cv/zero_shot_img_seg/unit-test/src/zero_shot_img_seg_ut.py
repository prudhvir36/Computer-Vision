import gc
import sys
import unittest

from utils.log import Log
from utils.ret_values import *

from meghnad.core.cv.zero_shot_img_seg.src.zero_shot_img_seg import ZeroShotImageSegmentation

log = Log()

def _cleanup():
    gc.collect()

def test_case1():
    settings = ['default']
    predictor=ZeroShotImageSegmentation(model_type= settings, device= "cpu")

    data_path = r"input_files"
    bbox_json = r"image_bboxes.json"
    output_path = r"output_path"
    segmentation_lst = predictor.pred(input_path = data_path,
                                      bbox_json= bbox_json,
                                  output_path = output_path, iou_thresh = 0.7)

def _perform_tests():
    test_case1()


if __name__ == '__main__':
    _perform_tests()

    unittest.main()
    _cleanup()