import os
import json
import torch.multiprocessing
from typing import Dict, List
torch.multiprocessing.set_sharing_strategy('file_system')

import detectron2.utils.comm as comm
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.engine.hooks import BestCheckpointer
from utils.common_defs import class_header, method_header
from detectron2.data import DatasetMapper, build_detection_test_loader, build_detection_train_loader
from detectron2.data import transforms as T

from meghnad.core.cv.obj_det.src.dt.trn.trn_utils.eval import LossEvalHook

@class_header(
    description='''
    Defining Hooks for Augmentation, Evaluation, and Best Checkpointer.
    ''')
class DT2ObjDetTrainer(DefaultTrainer):

    def build_transforms(cfg: Dict) -> List[str]:

      transforms_map = {
        'resize': T.Resize,
        'random_crop': T.RandomCrop,
        'random_flip': T.RandomFlip,
        'random_brightness': T.RandomBrightness,
        'random_crop': T.RandomCrop,
        'random_contrast':T.RandomContrast
          }

      if cfg is not None:
          transforms_list = []
          for name, values in cfg.items():
              if name in transforms_map.keys():
                  if name == "resize":
                      resize_value = tuple(values.values())
                      resize_value = T.Resize(shape = resize_value)
                      transforms_list.append(resize_value)
                  elif name == "random_fliplr":
                      random_fliplr_value = list(values.values())[0]
                      random_fliplr_value = T.RandomFlip(prob=random_fliplr_value, horizontal=True)
                      transforms_list.append(random_fliplr_value)
                  elif name == "random_brightness":
                      random_brightness = list(values.values())[0]
                      random_brightness = T.RandomBrightness(intensity_min = 0.0, intensity_max = random_brightness)
                      transforms_list.append(random_brightness)
          return transforms_list
      elif cfg is None:
          return None


    @classmethod
    def _build_train_loader(cls, cfg: Dict) -> object:
        try:
            train_augs_path = os.path.join(f"runs", f"dt",
                                        "metadata", )
            with open(os.path.join(train_augs_path, "train_augmentations.json"), "r") as f:
                content = json.load(f)
            train_augmentations = DT2ObjDetTrainer.build_transforms(content)
        
            if train_augmentations is not None:
                return build_detection_train_loader(cfg,
                                mapper=DatasetMapper(cfg, is_train=True, augmentations=train_augmentations))
        except:
            return build_detection_train_loader(cfg,
                    mapper=DatasetMapper(cfg, is_train=True))

    @classmethod
    def _build_evaluator(cls, cfg: Dict, dataset_name: str, output_folder: str = None) -> COCOEvaluator:
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @method_header(
        description='''
        Build custom hooks
        ''',
        returns='''
            A list of hooks (callbacks)
        '''
    )
    def _build_hooks(self) -> List:
        hooks = super()._build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))

        if self.cfg.SOLVER.BEST_CHECKPOINTER and comm.is_main_process():
            hooks.append(BestCheckpointer(
                self.cfg.TEST.EVAL_PERIOD,
                self.checkpointer,
                self.cfg.SOLVER.BEST_CHECKPOINTER.METRIC,
                mode=self.cfg.SOLVER.BEST_CHECKPOINTER.MODE
                ))

        hooks = hooks[:-2] + hooks[-2:][::-1]
        return hooks