#######################################################################################################################
# Image Similarity Configuration 
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Harshit Bardana and Chethan Ningappa
#######################################################################################################################

from utils.common_defs import *

cosine_img_sim_cfg =\
{
    'weights': 'IMAGENET1K_V1',
    'model': 'resnet50',
    'stride_number': 3,
    'list_of_stride_combinations': [[1, 2, 4, 5], [2, 3, 5, 6], [4, 5, 7, 8], [5, 6, 8, 9]],
}

_annoy_img_sim_cfg = \
{
    'weights': 'IMAGENET1K_V1',
    'model': 'resnet50',
    'metric': 'angular',
    'neighbours': 5,
    'n_trees': 20,
}

_faiss_img_sim_cfg = \
{
    'weights': 'IMAGENET1K_V1',
    'model': 'resnet50',
    'neighbours': 5,
}


class ImageSimConfig:
    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_cosine_img_sim_configs(self):
        return cosine_img_sim_cfg.copy()

    def get_typeA_img_sim_configs(self):
        return _annoy_img_sim_cfg.copy()

    def get_typeF_img_sim_configs(self):
        return _faiss_img_sim_cfg.copy()


if __name__ == '__main__':
    pass

