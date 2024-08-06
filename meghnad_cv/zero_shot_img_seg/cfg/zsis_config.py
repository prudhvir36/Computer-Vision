#######################################################################################################################
# Configurations for Zero Shot Image Segmentation.
#
# Copyright: All rights reserved. Inxite Out. 2023.
#
# Author: Lokesh Kankalapati
#######################################################################################################################
from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from meghnad.cfg.config import MeghnadConfig

import sys

__all__ = ['ZeroShotImageSegmentationConfig']


log = Log()

_zsis_cfg =\
{
    'models': {
        'default': {
                'model_type': "vit_b",
                'model_url': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            },
        'light': {
                'model_type': "vit_b",
                'model_url': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            },
        'large': {
                'model_type': "vit_l",
                'model_url': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            },
        'heavy': {
            'model_type': "vit_h",
            'model_url': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        }
        },
}


class ZeroShotImageSegmentationConfig(MeghnadConfig):
    def __init__(self, *args):
        super().__init__()

    def get_model_cfg(self, model_type: str) -> dict:
        try:
            return _zsis_cfg['models'].copy()[model_type]
        except:
            return _zsis_cfg['models'][model_type]
