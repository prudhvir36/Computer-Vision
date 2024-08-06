import os, sys
import unittest

from meghnad.core.cv.obj_tracking.src.trn_wrapper import Trainer
from meghnad.core.cv.obj_tracking.src.pred_wrapper import Predictor

from utils.log import Log

import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
num_of_gpus = torch.cuda.device_count()
torch.cuda.set_device(0)

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
   tf.config.set_visible_devices(gpus, 'GPU')

log = Log()

#Object Detection Training Pipeline + Video Tracking
def test_case1():
    settings = ['light']
    trainer = Trainer(settings)
    data_path = r"D:\object_detection_image\datasets\face_mask_yolov5\dataset.yaml"
    augmentations = {
        'train':
        {
            'resize': {'width': 300, 'height': 300},
            'random_fliplr': {'p': 0.5},
            'random_brightness': {'p': 0.2},
            'normalize': {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}
        },
        'test':
        {
            'resize': {'width': 300, 'height': 300},
            'normalize': {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}
        }
    }
    hyp = {'fliplr': 0.4, 'lr0': 0.02,
           'lrf': 0.2, 'weight_decay': 0.0003,
           'translate': 0.2, 'scale': 0.8,
           'optimizer': 'Adam'}

    trainer.config_connectors(data_path, augmentations=augmentations)
    _, result = trainer.trn(
        batch_size=2,
        epochs=3,
        workers=4,
        device='cpu',
        hyp=hyp
    )

    log.VERBOSE(sys._getframe().f_lineno,
               __file__, __name__,
               f'Best mAP: {result.best_metric}')
    
    log.VERBOSE(sys._getframe().f_lineno,
               __file__, __name__,
               f'Best path: {result.best_model_path}')

    print('Best mAP:', result.best_metric)
    print('Best path:', result.best_model_path)
    video_path = r"D:\object_tracking\testing_video"

    tracker = Predictor()
    tracker = tracker.pred(weights_path = result.best_model_path,
                    class_yaml_path = data_path,
                    device="cpu",
                    conf_thres = 0.25)
    
    tracker.track_video(video_path, 
                      output_dir=r"D:\object_tracking\codes\vdo_obj_det_v06\ixolerator\meghnad\core\cv\obj_tracking\unit-test\results")


# Tracking Video
def test_case2():
    tracker = Predictor()
    tracker = tracker.pred(weights_path = r"D:\object_tracking\working_input_files\best.pt",
                    class_yaml_path = r"D:\object_tracking\working_input_files\data.yaml",
                    device="cpu",
                    conf_thres = 0.25)

    tracker.track_video(r"D:\object_tracking\testing_video", 
                        output_dir=r"D:\object_tracking\codes\vdo_obj_det_v06\ixolerator\meghnad\core\cv\obj_tracking\unit-test\results")


def _perform_tests():
    test_case1()


if __name__ == '__main__':
    _perform_tests()

    unittest.main()
