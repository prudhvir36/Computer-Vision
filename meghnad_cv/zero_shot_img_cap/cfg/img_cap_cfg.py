#######################################################################################################################
# Configurations for Zero Shot Image Captioning. 
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

__all__ = ['ZeroShotImageCaptioningConfig']

log = Log()

_zero_shot_img_cap_cfg =\
{
    'models': {
        'default': {
                'model_name': "blip-image-captioning-base",
                'repo_id': "Salesforce/blip-image-captioning-base",
            },
        'light': {
                'model_name': "blip-image-captioning-base",
                'repo_id': "Salesforce/blip-image-captioning-base",
            },
        'large': {
                'model_name': "blip-image-captioning-large",
                'repo_id': "Salesforce/blip-image-captioning-large",
            }
        },
}


class ZeroShotImageCaptioningConfig(MeghnadConfig):
    def __init__(self, *args):
        super().__init__()

    def get_model_cfg(self, model_type: str) -> dict:
        try:
            return _zero_shot_img_cap_cfg['models'].copy()[model_type]
        except:
            return _zero_shot_img_cap_cfg['models'][model_type]



