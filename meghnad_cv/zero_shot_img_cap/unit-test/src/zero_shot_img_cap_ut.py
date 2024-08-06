import gc
import sys
import unittest

from utils.log import Log
from utils.ret_values import *

from meghnad.core.cv.zero_shot_img_cap.src.zero_shot_img_cap import ZeroShotImageCaptioning

log = Log()

def _cleanup():
    gc.collect()

def test_case1():
    settings = ['light']
    predictor=ZeroShotImageCaptioning(model_type= settings, device= "cpu")

    data_path = r"C:\Users\lokes\OneDrive\Pictures\img_cap_pic"
    captions_lst = predictor.pred(input_path = data_path, 
                                  output_path = r"meghnad\core\cv\zero_shot_img_cap\unit-test\results")

def _perform_tests():
    test_case1()


if __name__ == '__main__':
    _perform_tests()

    unittest.main()
    _cleanup()

