#######################################################################################################################
# Image Similarity Module
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Harshit Bardana, Chethan Ningappa
#######################################################################################################################

import json
import annoy
import torch
import os
import sys
import faiss
from torch import nn
from PIL import Image, ImageDraw
from torchvision import models, transforms

from utils.common_defs import *
from utils.log import Log
from utils.ret_values import *
from meghnad.core.cv.img_sim.cfg.config import ImageSimConfig

__all__ = ['TypeFaissSimilarImages']

log = Log()

@class_header(
    description='''Finds the Similar Images from Given Database for a query image'''
)
class TypeFaissSimilarImages:
    def __init__(self, *args, **kwargs):
        self.configs = ImageSimConfig().get_typeF_img_sim_configs()

    @method_header(
        description='''Connecting to s3 buckets'''
    )
    def _config_connectors(self, data_path: str = None, *args, **kwargs):
        self.connector_pred = {}
        self.data_path = data_path

    @method_header(
        description='''
        Extracting feature vector using''',
        arguments='''
        image_path: Location of test image
        ''',
        returns='''
        A tensor of feature vectors and it's dimensions. ''')
    def _feature_vector_extractor(self, image_path: str) -> (torch.Tensor, int):

        use_model = self.configs['model']
        weights_version = self.configs['weights']
        model = models.get_model(name=use_model, weights=weights_version)
        dimensions = list(model.children())[-1].in_features
        model.fc = nn.Identity()
        model.eval()
        transform = transforms.Compose([transforms.ToTensor()])
        image = Image.open(image_path)
        input_tensor = transform(image).unsqueeze(0)
        output_tensor = model(input_tensor)

        return output_tensor, dimensions

    @method_header(
        description='''
    Creates Index which can be stored and read-write''',
        arguments='''
    dimensions: vector dimensions''',
        returns='''
    a new index''')
    def _create_index(self, dimensions: int = 2048) -> faiss.IndexFlatL2:
        index = faiss.IndexFlatL2(dimensions)
        return index

    @method_header(
        description='''
    adds feature tensor to existing index''',
        arguments='''
    index: created index,
    in_tensor = tensor to add''',
        returns='''\
    index with given item added''')
    def _add_item(self, index: faiss.IndexFlatL2, in_tensor: torch.Tensor) -> faiss.IndexFlatL2:
        return index.add(in_tensor.detach().numpy())

    @method_header(
        description='''
    saves index''',
        arguments='''
    index: created index,
    path: path to save index''',
        returns='''saves the index file''')
    def _save_index(self, index: faiss.IndexFlatL2, path: str):
        return faiss.write_index(index, path)

    @method_header(
        description='''
    loads index''',
        arguments='''
    index: created index,
    path: path to load index from''',
        returns='''\
        bool value.''')
    def _load_index(self, path: str):
        try:
            return faiss.read_index(path)
        except:
            log.ERROR(sys._getframe().f_lineno, __file__, __name__, "Index size is not a multiple of vector size. "
                                                                    " OR 'large_data' must be same for "
                                                                    "train and inference")
            return IXO_RET_INVALID_INPUTS


    @method_header(
        description='''
    finds nearest neighours''',
        arguments='''
    index: created index,
    in_tensor: query tensor
    ''',
        returns='''
    nnlist: list of index for nearest nbrs,
    distance: list of distances for nearest nbrs''')
    def _get_nearest_nbrs(self, index: faiss.IndexFlatL2, in_tensor: torch.Tensor) -> dict:
        neighbours = self.configs['neighbours']
        distance, nnlist = index.search(in_tensor.detach().numpy(), neighbours)
        result = {'index': [int(x) for x in list(nnlist[0])], 'distance': [float(x) for x in list(distance[0])]}
        return result

    @method_header(
        description='''
        gets images from database for given dict''',
        arguments='''
        database_path: path to database,
        nndict: dict with indices and distances for nearest nbrs''',
        returns='''
        list of images''')
    def _get_images_from_nn_dict(self, database_path: str, nndict: dict) -> list:
        database = os.listdir(database_path)
        images = []
        nnlist = nndict['index']
        for i in range(len(nnlist)):
            image = Image.open(os.path.join(
                database_path, database[nnlist[i]]))
            images.append(image)
        return images

    @method_header(
        description='''
        creates resultant image with five nearest neighbours''',
        arguments='''
        base_image_path: path of query image,
        result_list: list of nearest neighbour images,
        output_path: path for resultant jpg file.(user needs to include '.jpg' or '.jpeg'),
        start: starting index of result_list,
        end: end index of result_list ''',
        returns='''
        saves the resultant image at output_path''')
    def _write_results(self, base_image_path: str, result_list: list, output_path: str, start: int = None,
                       end: int = None) -> int:
        image_grid = Image.new('RGB', (400, 600))
        base_image = Image.open(base_image_path)
        base_image = base_image.resize((200, 200))
        image_draw = ImageDraw.Draw(base_image)
        image_draw.rectangle(((0, 0), (200, 200)), outline='green', width=8)
        image_grid.paste(base_image, (0, 0))

        if (start is None) & (end is None):
            for j in range(len(result_list[:5])):
                image_grid.paste(
                    result_list[j], (200 * ((j + 1) % 2), 200 * ((j + 1) // 2)))

        elif (type(start) is int) & (type(end) is int):
            if end > len(result_list):
                log.ERROR(sys._getframe().f_lineno, __file__, __name__,
                          " 'end > neighbours' not enough neighbours to show")
                return IXO_RET_INVALID_INPUTS
            if start <= -1:
                log.ERROR(sys._getframe().f_lineno, __file__, __name__, "start must be zero or a positive number")
                return IXO_RET_INVALID_INPUTS
            if end <= start:
                log.ERROR(sys._getframe().f_lineno, __file__, __name__, "start must be greater than end")
                return IXO_RET_INVALID_INPUTS
            if end - start > 5:
                log.ERROR(sys._getframe().f_lineno, __file__, __name__, "cannot show more than 5 images.")
                return IXO_RET_INVALID_INPUTS
            else:
                for j in range(len(result_list[start:end])):
                    image_grid.paste(
                        result_list[j], (200 * ((j + 1) % 2), 200 * ((j + 1) // 2)))

        elif start is not None:
            if start <= -1:
                log.ERROR(sys._getframe().f_lineno, __file__, __name__, "start must be zero or a positive number")
                return IXO_RET_INVALID_INPUTS
            if start + 5 <= len(result_list):
                for j in range(len(result_list[start:start + 5])):
                    image_grid.paste(
                        result_list[j], (200 * ((j + 1) % 2), 200 * ((j + 1) // 2)))
            else:
                log.ERROR(sys._getframe().f_lineno, __file__, __name__, "resultant image requires five neighbours")
                return IXO_RET_NOT_SUPPORTED

        elif end is not None:
            if end <= 0:
                log.ERROR(sys._getframe().f_lineno, __file__, __name__, "end must be a positive number")
                return IXO_RET_INVALID_INPUTS
            if end - 5 >= 0:
                for j in range(len(result_list[end - 5:end])):
                    image_grid.paste(
                        result_list[j], (200 * ((j + 1) % 2), 200 * ((j + 1) // 2)))
            else:
                log.ERROR(sys._getframe().f_lineno, __file__, __name__, "resultant image requires five neighbours")
                return IXO_RET_NOT_SUPPORTED

        image_grid.save(output_path)

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
