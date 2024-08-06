import os,sys
import unittest
import torch
from torchvision import datasets, transforms, models
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from meghnad.core.cv.img_clf.src.pt.data_loader.data_loader import PTImgClfDataLoader

from utils.log import Log
from meghnad.core.cv.img_clf.src.pt.trn.trn import PTImgClfTrn
from meghnad.core.cv.img_clf.src.pt.inference.pred import PTImgClfPred


log = Log()


def test_case1():
    model_cfgs='default_models'
    data_path=r'D:\books\archive\LabelledRice\Labelled'
    output_path = r'D:\Meghanad\ixolerator_imgocr\meghnad\core\cv\img_clf\unit-test\results'
    augmentations= {
        'train':
            {
                'resize': {'width': 28, 'height': 28},
                'rotation': {'p': (-10, 10)},
                'resized': {'p': 28},
                'centercrop': {'p': 28}},}
    hyp_params= {
        'batch_size': 32,
        'learning_rate': 0.0001,
        'momentum': 0.9
    }
    classes = ['LeafBlast', 'BrownSpot', 'Healthy', 'Hispa']
    clftrain=PTImgClfTrn(model_cfgs='default_models')
    best_model, best_accuracy =clftrain.train(epochs=1,
                                              data_path=data_path,
                                              augmentations=augmentations,
                                              save_path=output_path,
                                              classes=classes,
                                              device='cpu')
    test_path=r'D:\books\archive\LabelledRice\test\BrownSpot'
    clfpred = PTImgClfPred(classes=classes,
                           test_path=test_path,
                           output_path=output_path,
                           device='cpu')
    results=clfpred.pred()


def _perform_tests():
    test_case1()


if __name__ == '__main__':
    _perform_tests()

    unittest.main()