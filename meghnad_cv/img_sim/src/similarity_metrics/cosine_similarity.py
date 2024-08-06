#######################################################################################################################
# Image Similarity Module
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Harshit Bardana
#######################################################################################################################

import torch
import json
from PIL import Image
from torchvision import transforms
from torchvision.models import get_model
import torch.nn as nn

from utils.common_defs import *
from utils.log import Log
from meghnad.core.cv.img_sim.cfg.config import ImageSimConfig

log = Log()

__all__ = ['SimilarityMetricCosine']


@class_header(
description='''
Image Similarity between two images.''')
class SimilarityMetricCosine:
    def __init__(self, *args, **kwargs):
        self.configs = ImageSimConfig().get_cosine_img_sim_configs()

    @method_header(
        description='''
       Connecting to s3 buckets'''
    )
    def _config_connectors(self, data_path: str = None, *args, **kwargs):
        self.connector_pred = {}
        self.data_path = data_path

    @method_header(
        description='''
        Extracting feature vector''',
        arguments='''
        image_path: Location of test image,
        stride_number: Number of strides,
        stride: Boolean variable signifying if the image has to be divided into strides or not''',
        returns='''
        A tensor of feature vectors and it's dimensions. ''')
    def _feature_vector_extractor(self, image_path: str, stride_number: int = None, stride: bool = False) -> (object, int):

        image = Image.open(image_path)
        use_model = self.configs['model']
        weights_version = self.configs['weights']
        model = get_model(name=use_model, weights=weights_version)
        dimensions = list(model.children())[-1].in_features
        model.fc = nn.Identity()
        model.eval()

        if stride:
            feature_dict = {}
            count = 0
            for i in range(stride_number):
                for j in range(stride_number):
                    count += 1
                    name = (count)
                    start_x = int(i) * int(image.shape[1] / stride_number)
                    start_y = int(j) * int(image.shape[0] / stride_number)
                    end_x = int(image.shape[1] / stride_number) * (i + 1)
                    end_y = int(image.shape[0] / stride_number) * (j + 1)
                    crop = image[start_y:end_y, start_x: end_x]

                    transform = transforms.Compose([transforms.ToTensor()])
                    input_tensor = transform(image).unsqueeze(0)
                    feature_tensor = model(input_tensor)
                    feature_dict[name] = feature_tensor.reshape((-1))
            return feature_dict, dimensions

        else:
            transform = transforms.Compose([transforms.ToTensor()])
            input_tensor = transform(image).unsqueeze(0)
            feature_tensor = model(input_tensor)

            return feature_tensor, dimensions

    @method_header(
        description='''
            This function calculates the cosine similarity between 2 sets of features of images''',
        arguments='''
            features1: Tensor representing features of image 1,
            features2: Tensor representing features of image 2,
            full_image: Boolean variable signifying if the feature is of full image or of strides''',
        returns='''
            A dictionary containing similarity scores''')
    def _cosine_similarity_scores(self, features1: object, features2: object, full_image: bool = True) -> dict:
        similarity_scores = {}

        if not full_image:
            config = self.configs
            list_of_stride_combinations = config['list_of_stride_combinations']
            stride_number = config['stride_number']
            count = 0
            for comparison_list in list_of_stride_combinations:
                stride_list = []
                count += 1
                for i in comparison_list:
                    for j in comparison_list:
                        cos = torch.nn.CosineSimilarity(dim=0)
                        output = cos(features1[i], features2[j])
                        stride_list.append(output)
                stride_name = "stride_sim_score_" + str(count)
                similarity_scores[stride_name] = max(stride_list).numpy().tolist()
        else:
            cos = torch.nn.CosineSimilarity(dim=0)
            similarity_scores['full_sim_score'] = cos(features1, features2)
            similarity_scores['full_sim_score'] = similarity_scores['full_sim_score'].detach().numpy().tolist()

        return similarity_scores

    @method_header(
        description='''
        gives resultant json with indices of nearest neighbours''',
        arguments='''
        result: the dictionary with distances and nearest nbrs,
        results_path: path to save the json file''',
        returns='''
        saves the results as json at output_path''')
    def _write_results_json(self, result: dict, results_path: str):
        with open(results_path, 'w') as file:
            file.write(json.dumps(result))
