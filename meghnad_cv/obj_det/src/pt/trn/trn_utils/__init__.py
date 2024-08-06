from typing import Dict, Callable
import yaml

from utils.common_defs import class_header, method_header
from meghnad.core.cv.obj_det.src.utils.general import get_meghnad_repo_dir


@method_header(
    description='''
        Get training function from given arch name.''',
    arguments='''
        arch: Model arch.''',
    returns='''
        A training function.''')
def get_train_pipeline(arch: str) -> Callable:
    if arch == 'yolov5':
        from meghnad.core.cv.obj_det.src.pt.trn.trn_utils.trn_utils_v5 import train as trn_yolov5
        return trn_yolov5
    elif arch == 'yolov7':
        from meghnad.core.cv.obj_det.src.pt.trn.trn_utils.trn_utils_v7 import train as trn_yolov7
        return trn_yolov7


@class_header(description='''Config object.''')
class Opt:
    def __str__(self):
        repr_dict = dict()
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                repr_dict[k] = v
        return str(repr_dict)


@method_header(
    description='''
    Get training configs.''',
    arguments='''
        model_cfg: Config dict.
        **kwargs: User-defined arguments.''',
    returns='''
        Config object.'''
)
def get_train_opt(model_cfg: Dict, **kwargs) -> object:
    opt = Opt()
    for k, v in model_cfg.items():
        if k == 'hyp' and isinstance(v, str):
            hyp_filepath = get_meghnad_repo_dir() / model_cfg['arch'] / v
            with open(hyp_filepath, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
                setattr(opt, 'hyp', hyp)
        else:
            setattr(opt, k, v)

    for k, v in kwargs.items():
        if k == 'hyp':
            if v:
                if getattr(opt, 'hyp'):
                    opt.hyp.update(v)
                else:
                    opt.hyp = v
        else:
            setattr(opt, k, v)
    return opt
