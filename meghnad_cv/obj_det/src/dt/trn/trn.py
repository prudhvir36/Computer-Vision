import os
import sys
import json
from typing import List, Tuple, Dict

from utils import ret_values
from utils.log import Log
from utils.common_defs import class_header, method_header

from meghnad.core.cv.obj_det.src.dt.data_loader import DT2ObjDetDataLoader
from meghnad.core.cv.obj_det.src.dt.trn.trn_utils.trainer import DT2ObjDetTrainer
from meghnad.core.cv.obj_det.src.metric import Metric
from meghnad.core.cv.obj_det.cfg import ObjDetConfig


from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog 


__all__ = ['DT2ObjDetTrn']

log = Log()
_config = ObjDetConfig()
_defaults = _config.get('default_constants')['dt']['trn']['trn']['trn_params']


@class_header(
    description='''
    Class for training Detectron2 Object Detection Model.''')
class DT2ObjDetTrn:
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
    def config_connectors(self, data_path: str, augmentations: Dict = None) -> None:
        self.dataloader = [DT2ObjDetDataLoader(data_path, model_cfg, augmentations)
                             for model_cfg in self.model_cfgs]
        
        
    @method_header(
    description='''
            Function to set training configurations and start training.''',
    arguments='''
            epochs: number of complete passes through the training dataset. Set epochs for the training by default as 10
            imgsz: image size
            batch_size: number of samples processed before the model is updated
            workers: the number of sub process that ingest data
            hyp: tuple that accepts the hyper parameters from data/hyps/
            ''')
    def trn(self,
            batch_size: int = _defaults['batch_size'],
            epochs: int = _defaults['epochs'],
            device: str = _defaults['device'],
            workers: int = _defaults['workers'],
            output_dir: str = _defaults['output_dir'],
            hyp: Dict = _defaults['hyp']
            ) -> Tuple:

        best_metric = Metric(map=0.)
        best_model_path = None
        if device == 'cpu':
            log.ERROR(sys._getframe().f_lineno, 
                        __file__, __name__, "device should be set to cuda")
            return ret_values.IXO_RET_INVALID_INPUTS, best_metric, best_model_path
        
        elif epochs <= 0:
            log.ERROR(sys._getframe().f_lineno,
                        __file__, __name__, "epochs value must be a positive integer")
            return ret_values.IXO_RET_INVALID_INPUTS, best_metric, best_model_path

        for model_cfg in self.model_cfgs:
            model_name = model_cfg.get("model_name")
            dt_model_cfg = model_cfg.get("dt_model_cfg")
            learning_rate = model_cfg.get("hyp_params").get("learning_rate")

            cfg = get_cfg()
            cfg.set_new_allowed(True)
            cfg.merge_from_file(model_zoo.get_config_file(dt_model_cfg))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(dt_model_cfg)
            cfg.DATASETS.TRAIN = ("dataset_train", )
            cfg.DATASETS.TEST = ("dataset_test", )
            cfg.DATALOADER.NUM_WORKERS = 2
            cfg.MODEL.DEVICE = device
            cfg.SOLVER.IMS_PER_BATCH = batch_size
            cfg.SOLVER.BASE_LR = hyp.get("lr0")
            cfg.TEST.EVAL_PERIOD = 200
            cfg.SOLVER.MAX_ITER = epochs
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get("dataset_train").thing_classes)
            os.makedirs(output_dir, exist_ok=True)
            dt_output_dir = os.path.join(output_dir, 'dt')
            cfg.OUTPUT_DIR = dt_output_dir  + os.sep + f"best_model_path"
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            cfg.SOLVER.BEST_CHECKPOINTER = CfgNode({"ENABLED": False})
            cfg.SOLVER.BEST_CHECKPOINTER.METRIC = "bbox/AP50"
            cfg.SOLVER.BEST_CHECKPOINTER.MODE = "max"
            trainer = DT2ObjDetTrainer(cfg)
            trainer.resume_or_load(resume=False)
            trainer.train()

            best_model_path = os.path.join(cfg.OUTPUT_DIR, "model_best.pt")

            metadata_path = os.path.join(dt_output_dir, f"metadata")
            os.makedirs(metadata_path, exist_ok=True)
            
            with open(os.path.join(metadata_path, "config.yaml"), "w") as f:
                f.write(cfg.dump())

            metrics_path = cfg.OUTPUT_DIR
            best_metric = []
            try: 
                with open(os.path.join(metrics_path,"metrics.json")) as f:
                    content = f.readlines()
                    for i in content:
                        load_json = json.loads(i)
                        try:
                            value = load_json["bbox/AP50"]
                            best_metric.append(value)     
                        except:
                            pass
                best_metric = max(best_metric)
            except:
                continue
            else:
                best_metric = 0.00
            
            best_metric = Metric(map=best_metric)

        return ret_values.IXO_RET_SUCCESS, best_metric, best_model_path
        