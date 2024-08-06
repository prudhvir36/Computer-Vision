#######################################################################################################################
# Image OCR Module
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Sreyasha Sengupta
#######################################################################################################################

from utils.ret_values import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.cv.img_ocr.src.image_ocr import TextExtraction

import json
import gc

import unittest


def _cleanup():
    gc.collect()


def _write_results(result, results_path):
    with open(results_path, 'w') as file:
        file.write(json.dumps(result))


def _tc_1(text_extractor, testcases_path, results_path):
    image = testcases_path + 'z2.jpg'
    file_name = results_path + 'results_tc_2.txt'

    result = text_extractor.pred(image)

    _write_results(result, file_name)


def _perform_tests():
    ut_path = MeghnadConfig().get_meghnad_configs(
        'BASE_DIR') + "core/cv/img_sim/unit-test/"
    testcases_path = ut_path + "testcases/"
    results_path = ut_path + "results/"
    text_extractor = TextExtraction()
    _tc_1(text_extractor, testcases_path, results_path)


if __name__ == '__main__':
    _perform_tests()

    unittest.main()

    _cleanup()
