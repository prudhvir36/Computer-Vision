from typing import List, Tuple, Dict
import numpy as np

from utils import ret_values
from utils.log import Log
from utils.common_defs import class_header, method_header
from meghnad.core.cv.obj_det.src.pt.trn.trn_utils import get_train_pipeline, get_train_opt
from meghnad.core.cv.obj_det.src.metric import Metric
from meghnad.core.cv.obj_det.cfg import ObjDetConfig


__all__ = ['PTObjDetTrn']


log = Log()
_config = ObjDetConfig()
_defaults = _config.get('default_constants')['pt']['trn']


@method_header(
    description="""Combines precision, recall, mAP@0.5, and mAP@0.5:0.95 to final metric.
    """,
    arguments="""
        x: A 2D-array of metrics.
    """,
    returns="""
        metric: (float) weighted combination of metrics
    """)
def fitness(x: np.ndarray) -> float:
    # Model fitness as a weighted combination of metrics
    w = _defaults['metric_weights']  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


@class_header(
    description='''
        Class for object detection model training''')
class PTObjDetTrn:
    def __init__(self, model_cfgs: List[Dict]) -> None:
        self.model_cfgs = model_cfgs
        self.data_path = None
        self.best_model_path = None

    @method_header(
        description='''
                Helper for configuring data connectors.''',
        arguments='''
                data_path: location of the training data (should point to the file in case of a single file, should point to
                the directory in case data exists in multiple files in a directory structure)
                ''')
    def config_connectors(self, data_path: str) -> None:
        self.data_path = data_path

    @method_header(
        description='''
                Function to set training configurations and start training.''',
        arguments='''
                epochs: number of complete passes through the training dataset. Set epochs for the training by default as 10
                imgsz: image size
                batch_size: number of samples processed before the model is updated
                workers: the number of sub process that ingest data
                hyp: tuple that accepts the hyper parameters from data/hyps/
                ''',
        returns='''
                metric: (Metric) training metrics
                best_path: (str) best model path for prediction
            '''
    )
    def trn(self,
            batch_size: int = _defaults['trn']['batch_size'],
            epochs: int = _defaults['trn']['epochs'],
            imgsz: int = _defaults['trn']['imgsz'],
            device: str = _defaults['trn']['device'],
            workers: int = _defaults['trn']['workers'],
            output_dir: str = _defaults['trn']['output_dir'],
            hyp: Dict = _defaults['trn']['hyp'],
            **kwargs) -> Tuple:
        best_fitness = 0.0
        best_path = None
        for model_cfg in self.model_cfgs:
            train_pipeline = get_train_pipeline(model_cfg['arch'])
            opt = get_train_opt(
                model_cfg,
                data=self.data_path,
                epochs=epochs,
                batch_size=batch_size,
                imgsz=imgsz,
                device=device,
                workers=workers,
                output_dir=output_dir,
                hyp=hyp
            )

            results, best = train_pipeline(opt)
            fi = fitness(np.array(results).reshape(1, -1))
            if fi > best_fitness:
                best_fitness = fi
                best_path = best

        best_metric = Metric(map=best_fitness)
        return ret_values.IXO_RET_SUCCESS, best_metric, best_path
