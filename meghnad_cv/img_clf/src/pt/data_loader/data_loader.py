import json
import os

import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from utils.log import Log
from utils.common_defs import class_header, method_header

from meghnad.core.cv.img_clf.cfg.img_clf_config import ImgClfConfig
from meghnad.core.cv.img_clf.src.pt.data_loader.loader_utils import mean_std

__all__ = ['PTImgClfDataLoader']
log = Log()


@class_header(
    description='''Data loader for Image classification.''')
class PTImgClfDataLoader:
    def __init__(self,
                 model_cfg: str,
                 augmentations:dict,
                 data_path: str,
                 classes:list,
                 save_path:str) -> None:
        self.model_name=model_cfg['arch']
        self.model_cfg= model_cfg
        self.data_path = data_path
        self.batch_size = self.model_cfg['hyp_params']['batch_size']
        self.input_shape = self.model_cfg['input_shape']
        self.num_classes = len(classes)
        self.augmenations=augmentations
        self.rotation = augmentations['train']['rotation']
        self.resizecrop = augmentations['train']['resize']
        self.centercrop = augmentations['train']['resized']
        self.train_size = ImgClfConfig().get_data_cfg()['train_test_val_split'][0]
        self.val_size = ImgClfConfig().get_data_cfg()['train_test_val_split'][1]
        self.test_size = ImgClfConfig().get_data_cfg()['train_test_val_split'][2]
        self.save_path=save_path
        self.mean,self.std=mean_std(images_directory=self.data_path,
                            image_size=(self.input_shape['width'],self.input_shape['height']),
                            batch_size=self.batch_size)


    @method_header(description='''Saving augmentation for test''',
                   returns='''Returns augmentation as dictionary''')
    def save_ann(self) -> dict:
        augmentations = {
            'train':
                {
                    'resize': {'width': self.input_shape[0], 'height': self.input_shape[1]},
                    'normalize': {'mean': self.mean, 'std': self.std}
                },}
        with open(os.path.join(self.save_path, 'augmenations.json'), 'w') as f:
            json.dump(augmentations, f)
        return augmentations


    @method_header(
        description='''
        data augmentation using pytorch module this include rotation, resize and cropping
        ''',
        returns='''We are getting augmented data and saving the augmentation data''')
    def datatransform(self) ->any:
        save_annnotation=self.save_ann()
        transform = transforms.Compose(
            [transforms.RandomRotation(self.rotation['p']),
             transforms.RandomResizedCrop((self.resizecrop['width'],self.resizecrop['height'])),
             transforms.Resize((self.resizecrop['width'],self.resizecrop['height'])),
             transforms.RandomHorizontalFlip(),
             transforms.CenterCrop(self.resizecrop['width']),
             transforms.ToTensor(),
             transforms.Normalize(self.mean,self.std)])
        return transform,save_annnotation


    @method_header(
        description='''
        split the data into train, test and validation, 80:20 80 for training 10 and 10 for test and validation
        ''',
        returns='''
        train_idx: Number images for training
        test_idx: Number images for testing
        valid_idx: Number images for validation
        train_data: contains images for train
        valid_data: contains images for validation
        test_data: contains images for test
                ''')
    def split_train_test_val(self) ->any:
        data_transforms = {"train_transforms": self.datatransform(),
                           "valid_transforms": self.datatransform(),
                           "test_transforms": self.datatransform()}
        train_data = datasets.ImageFolder(self.data_path, transform=data_transforms["train_transforms"])
        valid_data = datasets.ImageFolder(self.data_path, transform=data_transforms["valid_transforms"])
        test_data = datasets.ImageFolder(self.data_path, transform=data_transforms["test_transforms"])
        num_train = len(train_data)
        indices = list(range(num_train))
        train_count = int(self.train_size * num_train)
        valid_count = int(self.val_size * num_train)
        test_count = num_train - train_count - valid_count
        train_idx = indices[:train_count]
        valid_idx = indices[train_count:train_count + valid_count]
        test_idx = indices[train_count + valid_count:]
        train_idx, valid_idx, test_idx = len(train_idx), len(valid_idx), len(test_idx)
        return train_idx, valid_idx, test_idx, train_data, valid_data, test_data


    @method_header(
        description='''
        data should pass train, test and validation
        ''',
        returns='''
        data loading function for train,validation and test
        ''')
    def dataloader(self, data: str, batch_size: int, shuffle=False):
        return torch.utils.data.DataLoader(data, batch_size=batch_size,shuffle=shuffle)

