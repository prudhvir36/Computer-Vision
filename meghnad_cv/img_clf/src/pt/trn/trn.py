import os
import sys
import numpy as np
import torch
from torchvision.models import get_model

from torch import nn, optim
from collections import OrderedDict
from tqdm import tqdm
from utils import ret_values
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models, datasets, transforms

from utils.log import Log
from utils.common_defs import class_header, method_header

from meghnad.cfg.config import MeghnadConfig
from meghnad.core.cv.img_clf.src.pt.trn.metric import AverageMeter, accuracy
from meghnad.core.cv.img_clf.cfg.img_clf_config import ImgClfConfig
from meghnad.core.cv.img_clf.src.pt.data_loader.data_loader import PTImgClfDataLoader
from meghnad.core.cv.img_clf.src.pt.utils.general import get_sync_dir


log = Log()


@class_header(description='''Class for Image classification training,
              arguments=model_cfgs: the dictionary that contains all the necessary parameter''')
class PTImgClfTrn:
    def __init__(self, model_cfgs) -> None:
        self.model_cfgs = model_cfgs
        self.best_model_path = None
        self.configs=ImgClfConfig(MeghnadConfig())


    @method_header(
        description='''
                Helper for configuring data connectors.''',
        arguments='''
                data_path: location of the training data, each class is divided by folder,
                augmentation: Refer to the various augmentation has been used for images like rotate,resize
                classes:
                ''')
    def config_connectors(self, data_path: str, augmentation: dict,classes:list,save_path:str) -> None:
        sync_dir = get_sync_dir(data_path)
        data_path = os.path.join(sync_dir, data_path)
        self.data_loaders = PTImgClfDataLoader(self.model_cfg, augmentation,data_path,classes,save_path)


    @method_header(
        description='''
                This will help us to loop over multiple model and start training''',
        arguments='''
                n_epochs: set epochs for the training by default it is 10
                trainloader: We have loaded that dataset into the train loader
                validloader: We have loaded that dataset into the valid loader
                model_transfer: model name which we are about to train
                Class: list of class name
                learning_rate: an argument to specify when the function should print or after how many epochs
                momentum:Momentum is a accelerate convergence and overcome local minima
                save_path: directory from where the checkpoints should be loaded
                ''',
        returns='''model_transfer: Writes the model name
                   train_loss: Losses for the training data
                   valid_loss:Losses for the validation data 
                   log_val_acc: accuracy for validation data 
                   log_train_acc: accuracy for train data
                ''')
    def seq_train(self, n_epochs: int,
                  trainloader: str,
                  validloader: str,
                  model_transfer: str,
                  classes: list,
                  learning_rate: float,
                  momentum: float,
                  model_name: str,
                  device='cpu'):

        if n_epochs <= 0:
            log.ERROR(sys._getframe().f_lineno,
                          __file__, __name__, "Epochs value must be a positive integer")
            return ret_values.IXO_RET_INVALID_INPUTS

        scores_train = AverageMeter()
        scores_val = AverageMeter()
        clfconfig = ImgClfConfig()
        if model_transfer == clfconfig.get_model_settings('large_models')[1]:
            model_transfer = get_model(model_transfer, weights="DEFAULT")
            model_transfer.heads = nn.Linear(model_transfer.heads.head.in_features, len(classes))
        elif model_transfer == clfconfig.get_model_settings('large_models')[2] or \
                model_transfer == clfconfig.get_model_settings('light_models')[1] or \
                model_transfer == clfconfig.get_model_settings('light_models')[3]:
            model_transfer = get_model(model_transfer, weights="DEFAULT")
            model_transfer.heads = nn.Linear(model_transfer.classifier[1].in_features, len(classes))
        elif model_transfer == clfconfig.get_model_settings('light_models')[1] \
                or model_transfer == clfconfig.get_model_settings('light_models')[2] :
            model_transfer = get_model(model_transfer, weights="DEFAULT")
            model_transfer.heads = nn.Linear(model_transfer.classifier[3].in_features, len(classes))
        else:
            model_transfer = get_model(model_transfer, weights="DEFAULT")
            model_transfer.fc = nn.Linear(model_transfer.fc.in_features, len(classes))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model_transfer.parameters(), lr=learning_rate, momentum=momentum)

        train_loss = []
        valid_loss = []
        for epoch in tqdm(range(1, n_epochs + 1)):
            log.STATUS(sys._getframe().f_lineno,
                       __file__, __name__,
                       f'training started {epoch}')

            train_loss = 0.0
            valid_loss = 0.0
            model_transfer.train()
            for batch_idx, (data, target) in enumerate(trainloader):
                if self.configs.get_meghnad_configs('DEVICE') != 'cpu'\
                        and self.configs.get_meghnad_configs('DEVICE') != None:
                    if torch.cuda.is_available():
                        device = self.configs.get_meghnad_configs('DEVICE')
                        data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = model_transfer(data)
                if torch.is_tensor(output) == False:
                    output = output[0]
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
                acc_train = accuracy(output, target)[0]
                scores_train.update(acc_train.item(), data.size(0))

            log_train_acc = OrderedDict([('acc', scores_train.avg)])
            model_transfer.eval()
            for batch_idx, (data, target) in enumerate(validloader):
                log.STATUS(sys._getframe().f_lineno,
                           __file__, __name__,
                           f'Validation...')
                if self.configs.get_meghnad_configs('DEVICE') != 'cpu'\
                        and self.configs.get_meghnad_configs('DEVICE') != None:
                    if torch.cuda.is_available():
                        device = self.configs.get_meghnad_configs('DEVICE')
                        data, target = data.cuda(), target.cuda()
                output = model_transfer(data)
                loss = criterion(output, target)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
                acc_train = accuracy(output, target)[0]
                scores_train.update(acc_train.item(), data.size(0))
                acc_val = accuracy(output, target)[0]
                scores_val.update(acc_val.item(), data.size(0))
            log_val_acc = OrderedDict([('acc', scores_val.avg)])
        return model_transfer, train_loss, valid_loss, log_val_acc, log_train_acc

    @method_header(
        description='''iterate over the list of model in setting file and give the best model details''',
        arguments='''
            epochs: set epochs for the training by default it is 10
            data_path: The path/checkpoint from where the training should be resumed
            save_path: directory from where the checkpoints should be loaded
            augmentations: ''',

        returns='''model: best model name,
                   best_accuracy: Best model accuracy'''
                   )
    def train(self,epochs:int,data_path:str,augmentations: dict,save_path:str,classes:list,device:str):
        clfconfig = ImgClfConfig()
        models_list = clfconfig.get_model_settings(self.model_cfgs)
        dict_result = {}
        best_loss = float('inf')
        best_model = None
        for models_name in tqdm(models_list):
            model_cfg = clfconfig.get_model_cfg(models_name)
            classes = classes
            data_path = data_path
            clfdataloader = PTImgClfDataLoader(model_cfg,augmentations,data_path,classes,save_path)
            train_idx, valid_idx, test_idx, train_data, valid_data, test_data = clfdataloader.split_train_test_val()
            batch_size = clfconfig.get_model_cfg(model_cfg['arch'])['hyp_params']['batch_size']
            trainloader = clfdataloader.dataloader(train_data, batch_size, True)
            validloader = clfdataloader.dataloader(valid_data, batch_size, True)
            clftrain = PTImgClfTrn(model_cfg)
            result_path = save_path
            model_transfer = model_cfg['arch']
            learning_rate = model_cfg['hyp_params']['learning_rate']
            momentum = model_cfg['hyp_params']['momentum']

            mt, tl, valid_loss, log_val_acc, log_train_acc = clftrain.seq_train(epochs,
                                                                                trainloader,
                                                                                validloader,
                                                                                model_transfer,
                                                                                classes,
                                                                                learning_rate,
                                                                                momentum,
                                                                                models_name,
                                                                                device='cpu')


            if valid_loss <= best_loss:
                best_model = mt
                best_loss = valid_loss

            dict_result.update({f'{models_name}': log_val_acc['acc']})

        model_scripted = torch.jit.script(best_model)  # Export to TorchScript
        model_scripted.save(os.path.join(save_path, 'metadata.pt'))
        log.STATUS(sys._getframe().f_lineno,
                   __file__, __name__,
                   f'Saved the best model as {save_path}/metadata.pt')

        with open(os.path.join(result_path, "classfication.txt"), 'w') as write_results:
            count = 0
            for key, value in zip(dict_result.values(), dict_result.keys()):
                count = count+1
                write_results.write(f'model_{count}: {key} ')

        best_accuracy = max(zip(dict_result.values(), dict_result.keys()))[0]
        model = max(zip(dict_result.values(), dict_result.keys()))[1]
        return model, best_accuracy
