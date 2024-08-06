import time
import os
import math
import sys
from typing import List, Tuple, Union, Callable, Dict

import numpy as np
import tensorflow as tf

from meghnad.core.cv.obj_det.src.tf.data_loader import TFObjDetDataLoader
from meghnad.core.cv.obj_det.src.tf.model_loader.losses import Loss
from meghnad.core.cv.obj_det.src.metric import Metric

from utils import ret_values
from utils.log import Log
from utils.common_defs import class_header, method_header

from meghnad.core.cv.obj_det.src.tf.trn.select_model import TFObjDetSelectModel
from meghnad.core.cv.obj_det.src.tf.trn.eval import TFObjDetEval
from meghnad.core.cv.obj_det.src.tf.trn.trn_utils import get_optimizer
from meghnad.core.cv.obj_det.src.utils.general import get_sync_dir
from meghnad.core.cv.obj_det.cfg import ObjDetConfig


__all__ = ['TFObjDetTrn']

log = Log()
_config = ObjDetConfig()
_defaults = _config.get('default_constants')['tf']['trn']['trn']['trn_params']


@method_header(
    description="""Process a training step.

    Parameters
    ----------
    imgs : tf.Tensor
        Images for training. A tensor has shape of [N, H, W, C]
    gt_confs : tf.Tensor
        Classification targets. A tensor has shape of [B, num_default]
    gt_locs : tf.Tensor
        Regression targets. A tensor has shape of [B, num_default, 4]
    model : tf.keras.Model
        An instance of tf.keras.Model
    criterion : function
        Loss function
    optimizer : Optimizer class
        Optimizer for updating weights
    weight_decay : float
        Weights decay

    Returns
    -------
    [loss, conf_loss, loc_loss, l2_loss]
        Returns a list of losses.
    """)
@tf.function
def _train_step(
        imgs: tf.Tensor,
        gt_confs: tf.Tensor,
        gt_locs: tf.Tensor,
        model: tf.keras.Model,
        criterion: Union[tf.keras.losses.Loss, Callable],
        optimizer: str,
        weight_decay: float = 1e-5) -> Tuple[float]:
    with tf.GradientTape() as tape:
        confs, locs = model(imgs)

        conf_loss, loc_loss = criterion(
            confs, locs, gt_confs, gt_locs)

        loss = conf_loss + loc_loss
        l2_loss = [tf.nn.l2_loss(t) for t in model.trainable_variables]
        l2_loss = weight_decay * tf.math.reduce_sum(l2_loss)
        loss += l2_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    # Normalize gradients
    gradients = [tf.clip_by_norm(grad, 0.2) for grad in gradients]
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, conf_loss, loc_loss, l2_loss


@class_header(
    description='''
        Class for object detection model training''')
class TFObjDetTrn:
    def __init__(self, model_cfgs: List[Dict]) -> None:
        self.model_cfgs = model_cfgs
        self.model_selection = None
        self.data_loaders = []
        self.best_model_path = None

    @method_header(
        description='''
                Helper for configuring data connectors.''',
        arguments='''
                data_path: location of the training data (should point to the file in case of a single file, should point to
                the directory in case data exists in multiple files in a directory structure)
                ''')
    def config_connectors(self, data_path: str, augmentations: Dict = None) -> None:
        sync_dir = get_sync_dir()
        data_path = os.path.join(sync_dir, data_path)
        self.data_loaders = [TFObjDetDataLoader(data_path, model_cfg, augmentations)
                             for model_cfg in self.model_cfgs]

    @method_header(
        description='''
                Function to set training configurations and start training.''',
        arguments='''
                epochs: set epochs for the training by default it is 10
                output_dir: directory from where the checkpoints should be loaded
                resume_path: The path/checkpoint from where the training should be resumed
                print_every: an argument to specify when the function should print or after how many epochs
                ''')
    def trn(self,
            epochs: int = _defaults['epochs'],
            output_dir: str = _defaults['output_dir'],
            resume_path: str = _defaults['resume_path'],
            print_every: int = _defaults['print_every'],
            hyp: Dict = _defaults['hyp']) -> Tuple:
        try:
            epochs = int(epochs)
            if epochs <= 0:
                log.ERROR(sys._getframe().f_lineno,
                          __file__, __name__, "Epochs value must be a positive integer")
                return ret_values.IXO_RET_INVALID_INPUTS

        except ValueError:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, "Epochs value must be a positive integer")
            return ret_values.IXO_RET_INVALID_INPUTS

        # Layzy loading
        if self.model_selection is None:
            self.model_selection = TFObjDetSelectModel(self.model_cfgs)

        sync_dir = get_sync_dir()
        output_dir = os.path.join(sync_dir, output_dir)
        os.makedirs(output_dir, exist_ok=True)

        best_metric = Metric()
        for i, model in enumerate(self.model_selection.models):
            data_loader = self.data_loaders[i]
            model_cfg = self.model_cfgs[i]
            opt = hyp.get('optimizer', 'Adam')
            weight_decay = hyp.get('weight_decay', 1e-5)

            optimizer = get_optimizer(opt)
            criterion = Loss(
                model_cfg['neg_ratio'], model_cfg['num_classes'])
            evaluator = TFObjDetEval(model)

            model_name = model_cfg['arch']
            model_savedir = os.path.join(output_dir, model_name)

            # Found a checkpoint
            ckpt = tf.train.Checkpoint(
                model=model, optimizer=optimizer, start_epoch=tf.Variable(0))
            if resume_path:
                ckpt.read(resume_path)
                log.STATUS(sys._getframe().f_lineno,
                           __file__, __name__,
                           f'Resuming training from {resume_path}')
            else:
                if model_savedir and os.path.isdir(model_savedir):
                    ckpt_filenames = os.listdir(model_savedir)
                    prefix = f'{model_name}_last.ckpt'
                    for filename in ckpt_filenames:
                        if filename.startswith(prefix):
                            model_savedir = os.path.join(
                                model_savedir, prefix)
                            ckpt.read(model_savedir)
                            log.STATUS(sys._getframe().f_lineno,
                                       __file__, __name__,
                                       f'Resuming training from {model_savedir}')
                            break

            # Setup summary writers
            if os.path.isfile(model_savedir):
                logdir = os.path.join(os.path.dirname(model_savedir), 'logs')
            else:
                logdir = os.path.join(model_savedir, 'logs')
            train_logdir = os.path.join(logdir, 'logs/train')
            val_logdir = os.path.join(logdir, 'logs/val')
            os.makedirs(train_logdir, exist_ok=True)
            os.makedirs(val_logdir, exist_ok=True)
            train_summary_writer = tf.summary.create_file_writer(train_logdir)
            val_summary_writer = tf.summary.create_file_writer(val_logdir)

            # Setup learning rate scheduler
            base_lr = float(optimizer.learning_rate.numpy())
            warmup_learning_rate = base_lr / 6
            warmup_steps = 2000
            optimizer.learning_rate.assign(warmup_learning_rate)
            steps_per_epoch = max(
                1, data_loader.train_size // data_loader.batch_size)
            total_steps = epochs * steps_per_epoch
            global_step = 0

            log.VERBOSE(sys._getframe().f_lineno,
                        __file__, __name__,
                        f'Steps per epoch {steps_per_epoch}')

            log.VERBOSE(sys._getframe().f_lineno,
                        __file__, __name__,
                        f'Total steps {total_steps}')

            # Start training
            start_epoch = ckpt.start_epoch.numpy()
            for epoch in range(start_epoch, epochs):
                print('Epoch', epoch + 1)
                avg_loss = 0.0
                avg_conf_loss = 0.0
                avg_loc_loss = 0.0
                avg_l2_loss = 0.0
                start = time.time()
                for i, (imgs, gt_confs, gt_locs) in enumerate(data_loader.train_dataset):
                    # Forward + Backward
                    loss, conf_loss, loc_loss, l2_loss = _train_step(
                        imgs, gt_confs, gt_locs,
                        model, criterion, optimizer, weight_decay
                    )

                    # Compute average losses
                    avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)
                    avg_conf_loss = (avg_conf_loss * i +
                                     conf_loss.numpy()) / (i + 1)
                    avg_loc_loss = (avg_loc_loss * i +
                                    loc_loss.numpy()) / (i + 1)
                    avg_l2_loss = (avg_l2_loss * i + l2_loss.numpy()) / (i + 1)
                    if (i + 1) % print_every == 0:
                        log.VERBOSE(sys._getframe().f_lineno,
                                    __file__, __name__,
                                    'Epoch: {} Batch {} Time: {:.2}s | Loss: {:.4f} Conf: {:.4f} Loc: {:.4f} L2 Loss '
                                    '{:.4f}'.format(
                                        epoch + 1, i + 1, time.time() - start, avg_loss, avg_conf_loss, avg_loc_loss,
                                        avg_l2_loss))

                    # Learning rate scheduler
                    global_step = epoch * steps_per_epoch + i + 1
                    if global_step <= warmup_steps:
                        slope = (base_lr - warmup_learning_rate) / warmup_steps
                        new_lr = warmup_learning_rate + slope * \
                            tf.cast(global_step, tf.float32)
                        optimizer.learning_rate.assign(new_lr)
                    else:
                        new_lr = 0.5 * base_lr * (1 + tf.cos(
                            math.pi *
                            (tf.cast(i + 1, tf.float32) - warmup_steps
                             ) / float(total_steps - warmup_steps)))
                        optimizer.learning_rate.assign(new_lr)

                log.VERBOSE(sys._getframe().f_lineno,
                            __file__, __name__,
                            f'Current learning rate: {optimizer.learning_rate.numpy()}')

                # Start evaluation at the end of epoch
                log.STATUS(sys._getframe().f_lineno,
                           __file__, __name__,
                           f'Evaluating...')

                _, map, map50 = evaluator.eval(data_loader)

                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', avg_loss, step=epoch)
                    tf.summary.scalar('conf_loss', avg_conf_loss, step=epoch)
                    tf.summary.scalar('loc_loss', avg_loc_loss, step=epoch)

                with val_summary_writer.as_default():
                    tf.summary.scalar('mAP', map, step=epoch)

                # Checkpoint
                ckpt.start_epoch.assign_add(1)
                save_path = ckpt.write(os.path.join(
                    model_savedir, f'{model_name}_last.ckpt'))
                log.STATUS(sys._getframe().f_lineno,
                           __file__, __name__,
                           "Saved checkpoint for epoch {}: {}".format(
                               int(ckpt.start_epoch), save_path))

                # Save the best
                if map > best_metric.map:
                    print(f'Best mAP of {model_name}: {map:.4f}')
                    best_metric.map = map

                    self.best_model_path = os.path.join(
                        output_dir, f'best_saved_model')

                    tf.saved_model.save(model, self.best_model_path)
                    log.STATUS(sys._getframe().f_lineno,
                               __file__, __name__,
                               f'Saved the best model as {self.best_model_path}')

                    # save the corresponding default bounding boxes for inference
                    metadata_path = os.path.join(output_dir,
                                                 f'best_saved_model', 'metadata.npz')
                    np.savez(metadata_path, default_boxes=data_loader.default_boxes,
                             input_shape=data_loader.input_shape, model_name=model_name)
                    log.STATUS(sys._getframe().f_lineno,
                               __file__, __name__,
                               f'Saved model metadata as {metadata_path}')

        return ret_values.IXO_RET_SUCCESS, best_metric, self.best_model_path
