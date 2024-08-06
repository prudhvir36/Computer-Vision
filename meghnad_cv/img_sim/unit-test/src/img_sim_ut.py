#######################################################################################################################
# Image Similarity Module
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Harshit Bardana, Chethan Ningappa
#######################################################################################################################

import gc
import os
import sys

import unittest

from meghnad.core.cv.img_sim.src.indexing.faiss import TypeFaissSimilarImages
from meghnad.core.cv.img_sim.src.indexing.annoy import TypeAnnoySimilarImages
from meghnad.core.cv.img_sim.src.similarity_metrics.cosine_similarity import SimilarityMetricCosine
from utils.log import Log
from utils.ret_values import *

log = Log(subsystem='apps')


def _testcase_1(mode, img_path, img_2_path=None, index_path=None, database_path=None, image_results_path=None, json_results_path=None, large_data=False,
                similarity_metric='cosine', start=None, end=None):

    if mode == 'indexing':
        if large_data:
            img_sim = TypeFaissSimilarImages()
            database = os.listdir(database_path)

            base_image_features, dimensions = img_sim._feature_vector_extractor(img_path)
            meghnad_index = img_sim._create_index(dimensions=dimensions)
            count = 0
            log.STATUS(sys._getframe().f_lineno, __file__, __name__, f'no. of images in database: {len(database)}')

            for img in database:
                features, dimensions = img_sim._feature_vector_extractor(
                    image_path=os.path.join(database_path, img))
                img_sim._add_item(meghnad_index, features)
                count = count + 1
                if count % 100 == 0:
                    log.STATUS(sys._getframe().f_lineno, __file__, __name__, f'completed images: {count}')

            img_sim._save_index(meghnad_index, index_path)

            nn_dict = img_sim._get_nearest_nbrs(meghnad_index, base_image_features)
            result_images = img_sim._get_images_from_nn_dict(database_path, nn_dict)
            img_sim._write_results(img_path, result_images, image_results_path, start=start, end=end)
            img_sim._write_results_json(nn_dict, json_results_path)

        elif large_data is False:
            img_sim = TypeAnnoySimilarImages()
            database = os.listdir(database_path)

            base_image_features, dimensions = img_sim._feature_vector_extractor(img_path)
            meghnad_index = img_sim._create_index(dimensions=dimensions)
            print('status will run')
            log.STATUS(sys._getframe().f_lineno, __file__, __name__, f'no. of images in database: {len(database)}')
            print('status is run')
            count = 0

            for img in database:
                features, dimensions = img_sim._feature_vector_extractor(
                    image_path=os.path.join(database_path, img))
                img_sim._add_item(meghnad_index, count, features)
                count = count + 1
                if count % 100 == 0:
                    log.STATUS(sys._getframe().f_lineno, __file__, __name__, f'completed images: {count}')

            img_sim._build_trees(meghnad_index)
            img_sim._save_index(meghnad_index, index_path)

            nn_dict = img_sim._get_nearest_nbrs(
                meghnad_index, base_image_features)
            result_images = img_sim._get_images_from_nn_dict(database_path, nn_dict)
            img_sim._write_results(img_path, result_images, image_results_path, start=start, end=end)
            img_sim._write_results_json(nn_dict, json_results_path)

        else:
            log.ERROR(sys._getframe().f_lineno, __file__, __name__, "'large_data' accepts only a boolean value")
            return IXO_RET_INVALID_INPUTS

    elif mode == 'similarity_metric':

        if similarity_metric == 'cosine':
            img_sim = SimilarityMetricCosine()
            try:
                img_path_1 = img_path
                img_path_2 = img_2_path

                image_features_1, dimensions1 = img_sim._feature_vector_extractor(img_path_1)
                image_features_2, dimensions2 = img_sim._feature_vector_extractor(img_path_2)

                final_similarity_scores = img_sim._cosine_similarity_scores(image_features_1, image_features_2)

                img_sim._write_results_json(final_similarity_scores, json_results_path)
                return IXO_RET_SUCCESS, final_similarity_scores

            except Exception as e:
                log.ERROR(sys._getframe().f_lineno,
                          __file__, __name__,
                          e)
                return IXO_RET_GENERIC_FAILURE, None
        else:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      f'No metric found named {similarity_metric} ')
            return IXO_RET_NOT_SUPPORTED, None


def _testcase_2(img_path, index_path, database_path, image_results_path, json_results_path, large_data=False,
                start=None, end=None):
    if large_data is False:
        img_sim = TypeAnnoySimilarImages()
        base_image_features, dimensions = img_sim._feature_vector_extractor(img_path)
        meghnad_index = img_sim._create_index(dimensions=dimensions)

        img_sim._load_index(meghnad_index, index_path)

        nn_dict = img_sim._get_nearest_nbrs(
            meghnad_index, base_image_features)
        result_images = img_sim._get_images_from_nn_dict(database_path, nn_dict)

        img_sim._write_results(img_path, result_images, image_results_path, start=start, end=end)
        img_sim._write_results_json(nn_dict, json_results_path)

    elif large_data:
        img_sim = TypeFaissSimilarImages()
        base_image_features, dimensions = img_sim._feature_vector_extractor(img_path)
        meghnad_index = img_sim._create_index(dimensions=dimensions)
        meghnad_index = img_sim._load_index(index_path)

        nn_dict = img_sim._get_nearest_nbrs(
            meghnad_index, base_image_features)
        result_images = img_sim._get_images_from_nn_dict(database_path, nn_dict)

        img_sim._write_results(img_path, result_images, image_results_path, start=start, end=end)
        img_sim._write_results_json(nn_dict, json_results_path)

    else:
        log.ERROR(sys._getframe().f_lineno, __file__, __name__, "large_data accepts only a boolean value")
        return IXO_RET_INVALID_INPUTS


def _perform_tests():

    common_path = r'C:\Users\Harsh\PycharmProjects\ixolerator\meghnad\core\cv\img_sim\unit-test\results'
    base_image_path = r"C:\Users\Harsh\Documents\Meghnad\ANNOY\freiburg_groceries_dataset\test_images\CAKE0016.png"
    database_path = r'C:\Users\Harsh\Documents\Meghnad\ANNOY\freiburg_groceries_dataset\images\merge_folder'
    base_image_2_path = r'C:\Users\Harsh\Documents\Meghnad\ANNOY\freiburg_groceries_dataset\test_images\CAKE0133.png'
    large_data = False

    image_results_path_tc1 = os.path.join(common_path, r'indexing\result_image_tc1.JPG')
    json_results_path_tc1 = os.path.join(common_path, r"indexing\result_json_tc1.json")

    image_results_path_tc2 = os.path.join(common_path, r"indexing\result_image_tc2.JPG")
    json_results_path_tc2 = os.path.join(common_path, r"indexing\result_json_tc2.json")

    index_path = os.path.join(common_path, r'indexing\search_index.index')

    # Indexing
    _testcase_1(mode='indexing', img_path=base_image_path, index_path=index_path, database_path=database_path,
                image_results_path=image_results_path_tc1, json_results_path=json_results_path_tc1, large_data=False)

    _testcase_2(img_path=base_image_path, index_path=index_path, database_path=database_path,
                image_results_path=image_results_path_tc2, json_results_path=json_results_path_tc2, large_data=False)

    # Similarity Metric
    similarity_metric_json_results_path_tc1 = os.path.join(common_path, r"similarity_metric\cosine_full_score.json")
    _testcase_1(mode='similarity_metric', img_path=base_image_path, img_2_path=base_image_2_path,
                json_results_path=similarity_metric_json_results_path_tc1, similarity_metric='cosine')

def _cleanup():
    gc.collect()


if __name__ == '__main__':
    _perform_tests()

    unittest.main()

    _cleanup()
