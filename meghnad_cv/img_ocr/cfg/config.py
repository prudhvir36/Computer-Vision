#######################################################################################################################
# Image Similarity Configuration 
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Sreyasha Sengupta,Chethan & Haristh
#######################################################################################################################
from utils.common_defs import *

_img_ocr_cfg=\
{
    'lang': 'en'
}

class ImageOCRConfig():
    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_img_ocr_configs(self):
        return _img_ocr_cfg.copy()

if __name__ == '__main__':
    pass

