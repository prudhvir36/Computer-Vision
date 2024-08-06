import os
import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass

from utils import ret_values
from utils.log import Log
from meghnad.core.cv.obj_tracking.cfg.obj_det_video_config import ObjTrackingConfig, BACKENDS
from meghnad.core.cv.obj_det.src.pt.trn import PTObjDetTrn
from meghnad.core.cv.obj_det.src.metric import Metric
from utils.common_defs import class_header, method_header

log = Log()

@method_header(
    description="""Returns configs from given settings

    Parameters
    ----------
    settings : List[str]
        A list of string represents settings.

    Returns
    -------
    [model_cfgs]
        A list of string represents corresponding model configs
    """)
def load_config_from_settings(settings: List[str]) -> List:
    settings = [f'{setting}_models' for setting in settings]
    cfg_obj = ObjTrackingConfig()

    model_cfgs = []
    for setting in settings:
        model_names = cfg_obj.get_model_settings(setting)
        for model_name in model_names:
            model_cfg = cfg_obj.get_model_cfg(model_name)
            model_cfgs.append(model_cfg)
    return model_cfgs

@method_header(
        description='''
                Dataclass that contains only data
                ''')
@dataclass
class TrainingResult:
    best_metric: Metric = None
    best_model_path: str = None

@class_header(
    description='''
    Wrapper class for Object detection training''')
class Trainer:
    def __init__(self, settings: List[str]) -> None:
        self.settings = settings
        self.model_cfgs = load_config_from_settings(settings)
        self.data_path = None

        pt_model_cfgs = []
        for model_cfg in self.model_cfgs:
            backend = model_cfg['backend']
            if backend == BACKENDS.PYTORCH:
                pt_model_cfgs.append(model_cfg)

            else:
                log.ERROR(sys._getframe().f_lineno,
                          __file__, __name__, f'Unsupported backend: {backend}')
                return ret_values.IXO_RET_NOT_SUPPORTED
            raise ValueError(f'Unsupported backend: {backend}')
        self.pt_trainer = PTObjDetTrn(pt_model_cfgs)


    @method_header(
        description='''
                Helper for configuring data connectors.''',
        arguments='''
                data_path: location of the training data (should point to the file in case of a single file, should point to
                the directory in case data exists in multiple files in a directory structure)
                ''')
    def config_connectors(self, data_path: str, augmentations: Dict = None) -> None:
        self.data_path = data_path
        self.augmentations = augmentations

    @method_header(
        description='''
                Function to set training configurations and start training.''',
        arguments='''
                batch_size: batch size.
                epochs: set epochs for the training by default it is 10
                output_dir: directory from where the checkpoints should be loaded
                ''')
    def trn(self,
            batch_size: int = 32,
            epochs: int = 5,
            workers: int = 4,
            output_dir: str = r'meghnad\core\cv\obj_det\unit-test\results',
            device: str = 'cuda',
            **kwargs) -> Tuple:

        os.makedirs(output_dir, exist_ok=True)

        pt_output_dir = os.path.join(output_dir, 'pt')
        self.pt_trainer.config_connectors(self.data_path)
        success, pt_best_metric, pt_best_path = self.pt_trainer.trn(
            batch_size=batch_size,
            epochs=epochs,
            workers=workers,
            device=device,
            output_dir=pt_output_dir,
            hyp=kwargs.get('hyp', dict())
        )
        del self.pt_trainer

        best_metric = pt_best_metric
        best_model_path = pt_best_path

        result = TrainingResult(
            best_metric=best_metric,
            best_model_path=best_model_path
        )

        return ret_values.IXO_RET_SUCCESS, result

