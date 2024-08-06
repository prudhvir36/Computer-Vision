#######################################################################################################################
# Configurations for Object Detection & Tracking - Video.
#
# Copyright: All rights reserved. Inxite Out. 2023.
#
# Author: Lokesh Kankalapati
#######################################################################################################################
from typing import List
from dataclasses import dataclass
from meghnad.cfg.config import MeghnadConfig

__all__ = ['ObjTrackingConfig']

@dataclass
class BACKENDS:
    PYTORCH: str = 'pytorch'

_obj_det_video_cfg = {
    'models':
    {
        'YOLOv5',
        'YOLOv7',
    },
    'model_cfg':
    {
        'YOLOv5': {
            'config_name': 'YOLOv5',
            'arch': 'yolov5',
            'backend': 'pytorch',
            'weights': 'yolov5s.pt',
            'imgsz': 640,
            'cfg': '',
            'hyp': 'data/hyps/hyp.scratch-med.yaml',
            'rect': False,
            'resume': False,
            'nosave': False,
            'noval': False,
            'noautoanchor': False,
            'noplots': False,
            'evolve': False,
            'cache': '',
            'image_weights': False,
            'multi_scale': False,
            'single_cls': False,
            'optimizer': 'SGD',
            'sync_bn': False,
            'workers': 8,
            'project': 'runs/train',
            'name': 'yolov5s',
            'exist_ok': False,
            'quad': False,
            'cos_lr': False,
            'label_smoothing': 0.0,
            'patience': 100,
            'freeze': [0],
            'save_period': -1,
            'seed': 0,
        },
        'YOLOv7Light': {
            'config_name': 'YOLOv7Light',
            'arch': 'yolov7',
            'backend': 'pytorch',
            'weights': 'yolov7.pt',
            'img_size': [640],
            'cfg': '',
            'hyp': 'data/hyp.scratch.p5.yaml',
            'rect': False,
            'resume': False,
            'nosave': False,
            'noautoanchor': False,
            'notest': False,
            'evolve': False,
            'bucket': '',
            'cache_images': False,
            'image_weights': False,
            'multi_scale': False,
            'single_cls': False,
            'adam': False,
            'sync_bn': False,
            'workers': 4,
            'project': 'runs/train',
            'name': 'yolov7_light',
            'exist_ok': False,
            'quad': False,
            'linear_lr': False,
            'label_smoothing': 0.0,
            'freeze': [0],
            'save_period': -1,
            'local_rank': -1,
            'v5_metric': False
        },
        'YOLOv7Large': {
            'config_name': 'YOLOv7Large',
            'arch': 'yolov7',
            'backend': 'pytorch',
            'weights': 'yolov7.pt',
            'img_size': [1280],
            'cfg': '',
            'hyp': 'data/hyp.scratch.p5.yaml',
            'rect': False,
            'resume': False,
            'nosave': False,
            'noautoanchor': False,
            'notest': False,
            'evolve': False,
            'bucket': '',
            'cache_images': False,
            'image_weights': False,
            'multi_scale': False,
            'single_cls': False,
            'adam': False,
            'sync_bn': False,
            'workers': 4,
            'project': 'runs/train',
            'name': 'yolov7_large',
            'exist_ok': False,
            'quad': False,
            'linear_lr': False,
            'label_smoothing': 0.0,
            'freeze': [0],
            'save_period': -1,
            'local_rank': -1,
            'v5_metric': False
        }, 

    },
    'model_settings':
    {
        'default_models': ['YOLOv5'],
        'light_models': ['YOLOv5', 'YOLOv7Light'],
        'large_models': ['YOLOv7Large']
    }
}

class ObjTrackingConfig(MeghnadConfig):
    def __init__(self, *args):
        super().__init__()

    def get_model_cfg(self, model_name: str) -> dict:
        try:
            return _obj_det_video_cfg['model_cfg'].copy()[model_name]
        except:
            return _obj_det_video_cfg['model_cfg'][model_name]

    def get_data_cfg(self) -> dict:
        try:
            return _obj_det_video_cfg['data_cfg'].copy()
        except:
            return _obj_det_video_cfg['data_cfg']

    def get_model_settings(self, setting_name: str = None) -> dict:
        if setting_name and setting_name in _obj_det_video_cfg['model_settings']:
            try:
                return _obj_det_video_cfg['model_settings'][setting_name].copy()
            except:
                return _obj_det_video_cfg['model_settings'][setting_name]

    def set_user_cfg(self, user_cfg):
        for key in user_cfg:
            self.user_cfg[key] = user_cfg[key]

    def get_user_cfg(self) -> dict:
        return self.user_cfg

    def get_models_by_names(self) -> List[str]:
        models = []
        for key in _obj_det_video_cfg['model_archs']:
            models.append(key)
        return models
