#######################################################################################################################
# Configurations for Image Classification.
#
# Copyright: All rights reserved. Inxite Out. 2023.
#
# Author: Chethan Ningappa
#######################################################################################################################
from typing import List
from meghnad.cfg.config import MeghnadConfig

__all__ = ['ImgClfConfig']

_img_clf_cfg = {
    'models':
    {
        'resnet50',
        'resnet101',
        'googlenet',
        'vit_l_16',
        'efficientnet_b7'
    },
    'data_cfg':
    {
        'train_test_val_split': (0.7, 0.2, 0.1),
    },
    'model_cfg':
    {
        'resnet50':
        {
            'arch': 'resnet50',
            'pretrained': None,
            'input_shape': [28,28,3],
            'augmentations': {
                'train':
                {
                    'resize': {'width': 28, 'height': 28},
                    'rotation': {'p': (-10, 10)},
                    'resized': {'p': 28},
                    'centercrop': {'p': 28}
                },
            },
            'hyp_params':
            {
                'batch_size': 32,
                'optimizer': 'SGD',
                'learning_rate': 0.0001,
                'momentum': 0.9
            },},
        'resnet152':
        {
            'arch': 'resnet152',
            'pretrained': None,
            'input_shape': [28,28,3],
            'augmentations': {
                'train':
                {
                    'resize': {'width': 28, 'height': 28},
                    'rotation': {'p': (-10, 10)},
                    'resized': {'p': 28},
                    'centercrop': {'p': 28}
                },},
            'hyp_params':
            {
                'batch_size': 32,
                'optimizer': 'SGD',
                'learning_rate': 0.0001,
                'momentum': 0.9
            },
        },
        'googlenet':
        {
            'arch': 'googlenet',
            'pretrained': None,
            'input_shape': [28,28,3],
            'augmentations': {
                'train':
                {
                    'resize': {'width': 28, 'height': 28},
                    'rotation': {'p': (-10, 10)},
                    'resized': {'p': 28},
                    'centercrop': {'p': 28},
                },
            },
            'hyp_params':
            {
                'batch_size': 32,
                'optimizer': 'SGD',
                'learning_rate': 0.0001,
                'momentum': 0.9
            },

        },
        'vit_l_16':
        {
                'arch': 'vit_l_16',
                'pretrained': None,
                'input_shape': [224, 224, 3],
                'augmentations': {
                    'train':
                        {
                            'resize': {'width': 224, 'height': 224},
                            'rotation': {'p': (-10, 10)},
                            'resized': {'p': 224},
                            'centercrop': {'p': 224},
                        },
                },
                'hyp_params':
                    {
                        'batch_size': 32,
                        'optimizer': 'SGD',
                        'learning_rate': 0.0001,
                        'momentum': 0.9
                    },},
        'efficientnet_b3':
        {
                'arch': 'efficientnet_b3',
                'pretrained': None,
                'input_shape': [28, 28, 3],
                'augmentations': {
                    'train':
                        {
                            'resize': {'width': 28, 'height': 28},
                            'rotation': {'p': (-10, 10)},
                            'resized': {'p': 28},
                            'centercrop': {'p': 28},
                        },
                },
                'hyp_params':
                    {
                        'batch_size': 32,
                        'optimizer': 'SGD',
                        'learning_rate': 0.0001,
                        'momentum': 0.9
                    },

            },
        'efficientnet_b7':
        {
                'arch': 'efficientnet_b7',
                'pretrained': None,
                'input_shape': [28, 28, 3],
                'augmentations': {
                    'train':
                        {
                            'resize': {'width': 28, 'height': 28},
                            'rotation': {'p': (-10, 10)},
                            'resized': {'p': 28},
                            'centercrop': {'p': 28},
                        },
                },
                'hyp_params':
                    {
                        'batch_size': 32,
                        'optimizer': 'SGD',
                        'learning_rate': 0.0001,
                        'momentum': 0.9
                    },
            },
        'mobilenet_v2':
        {
                'arch': 'mobilenet_v2',
                'pretrained': None,
                'input_shape': [28, 28, 3],
                'augmentations': {
                    'train':
                        {
                            'resize': {'width': 28, 'height': 28},
                            'rotation': {'p': (-10, 10)},
                            'resized': {'p': 28},
                            'centercrop': {'p': 28},
                        },
                },
                'hyp_params':
                    {
                        'batch_size': 32,
                        'optimizer': 'SGD',
                        'learning_rate': 0.0001,
                        'momentum': 0.9
                    },

            },
        'mobilenet_v3_small':
        {
                'arch': 'mobilenet_v3_small',
                'pretrained': None,
                'input_shape': [28, 28, 3],
                'augmentations': {
                    'train':
                        {
                            'resize': {'width': 28, 'height': 28},
                            'rotation': {'p': (-10, 10)},
                            'resized': {'p': 28},
                            'centercrop': {'p': 28},
                        },
                },
                'hyp_params':
                    {
                        'batch_size': 32,
                        'optimizer': 'SGD',
                        'learning_rate': 0.0001,
                        'momentum': 0.9
                    },
            },
        'resnet34':
        {
                'arch': 'resnet34',
                'pretrained': None,
                'input_shape': [28, 28, 3],
                'augmentations': {
                    'train':
                        {
                            'resize': {'width': 28, 'height': 28},
                            'rotation': {'p': (-10, 10)},
                            'resized': {'p': 28},
                            'centercrop': {'p': 28}
                        },
                },
                'hyp_params':
                    {
                        'batch_size': 32,
                        'optimizer': 'SGD',
                        'learning_rate': 0.0001,
                        'momentum': 0.9
                    },},
    },
    'model_settings':{
        'default_models': ['googlenet','resnet50'],
        'light_models': ['resnet34', 'mobilenet_v2','mobilenet_v3_small','efficientnet_b3'],
        'large_models': ['resnet152','vit_l_16','efficientnet_b7'],
    }
}


class ImgClfConfig(MeghnadConfig):
    def __init__(self, *args):
        super().__init__()

    def get_model_cfg(self, model_name: str) -> dict:
        try:
            return _img_clf_cfg['model_cfg'].copy()[model_name]
        except:
            return _img_clf_cfg['model_cfg'][model_name]

    def get_data_cfg(self) -> dict:
        try:
            return _img_clf_cfg['data_cfg'].copy()
        except:
            return _img_clf_cfg['data_cfg']

    def get_model_settings(self, setting_name: str = None) -> dict:
        if setting_name in _img_clf_cfg['model_settings']:
            try:
                return _img_clf_cfg['model_settings'][setting_name].copy()
            except:
                return _img_clf_cfg['model_settings'][setting_name]