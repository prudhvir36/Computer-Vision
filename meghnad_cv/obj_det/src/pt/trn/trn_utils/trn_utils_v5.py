import argparse
import math
import os
import sys
import random
import time
from typing import Tuple
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

from meghnad.core.cv.obj_det.src.utils.general import get_meghnad_repo_dir, get_sync_dir

ROOT = get_meghnad_repo_dir() / 'yolov5'
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import meghnad.repo.obj_det.yolov5.val as validate  # for end-of-epoch mAP
from meghnad.repo.obj_det.yolov5.models.experimental import attempt_load
from meghnad.repo.obj_det.yolov5.models.yolo import Model
from meghnad.repo.obj_det.yolov5.utils.autoanchor import check_anchors
from meghnad.repo.obj_det.yolov5.utils.autobatch import check_train_batch_size
from meghnad.repo.obj_det.yolov5.utils.callbacks import Callbacks
from meghnad.repo.obj_det.yolov5.utils.dataloaders import create_dataloader
from meghnad.repo.obj_det.yolov5.utils.downloads import attempt_download, is_url
from meghnad.repo.obj_det.yolov5.utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                                                       check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                                                       get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                                                       labels_to_image_weights, methods, one_cycle, print_args, strip_optimizer,
                                                       yaml_save)
from meghnad.repo.obj_det.yolov5.utils.loggers import Loggers
from meghnad.repo.obj_det.yolov5.utils.loggers.comet.comet_utils import check_comet_resume
from meghnad.repo.obj_det.yolov5.utils.loss import ComputeLoss
from meghnad.repo.obj_det.yolov5.utils.metrics import fitness
from meghnad.repo.obj_det.yolov5.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                                                           smart_resume, torch_distributed_zero_first)
from meghnad.core.cv.obj_det.cfg import ObjDetConfig

from utils.log import Log
from utils.common_defs import method_header

log = Log()
_config = ObjDetConfig()
_defaults = _config.get('default_constants')['pt']['trn']['trn_utils_v5']


LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = check_git_info()


@method_header(
    description='''
        Training pipeline.''',
    arguments='''
        opt: Config object.''',
    returns='''
        Path to best model.''')
def train(opt: object) -> str:  # hyp is path/to/hyp.yaml or hyp dictionary
    hyp, opt, device, callbacks = _build_opt(opt)
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    callbacks.run('on_pretrain_routine_start')

    save_dir = Path(opt.sync_dir) / save_dir

    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') +
                ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp,
                          LOGGER)  # loggers instance

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(
            data, opt.sync_dir)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = {0: 'item'} if single_cls and len(
        data_dict['names']) != 1 else data_dict['names']  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith(
        _defaults['val_path_suffix'])  # COCO dataset

    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            # download if not found locally
            weights = attempt_download(weights)
        # load checkpoint to CPU to avoid CUDA memory leak
        ckpt = torch.load(weights, map_location='cpu')
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get(
            'anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get(
            'anchors')) and not resume else []  # exclude keys
        # checkpoint state_dict as FP32
        csd = ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(),
                              exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(
            f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report

    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get(
            'anchors')).to(device)  # create
    amp = check_amp(model)  # check AMP

    # Freeze
    freeze = [f'model.{x}.' for x in (freeze if len(
        freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # verify imgsz is gs-multiple
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    nbs = _defaults['nbs']  # nominal batch size
    # accumulate loss before optimizing
    accumulate = max(round(nbs / batch_size), 1)
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(
        model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        def lf(x): return (1 - x / epochs) * \
            (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(
                ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Trainloader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in {-1, 0}:
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=_defaults['create_dataloader_params']['rect'],
                                       rank=_defaults['create_dataloader_params']['rank'],
                                       workers=workers * 2,
                                       pad=_defaults['create_dataloader_params']['pad'],
                                       prefix=colorstr('val: '))[0]

        if not resume:
            if not opt.noautoanchor:
                # run AutoAnchor
                check_anchors(dataset, model=model,
                              thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end', labels, names)

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Model attributes
    # number of detection layers (to scale hyps)
    nl = de_parallel(model).model[-1].nl
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / _defaults['hyp_values']['base_num_cls'] * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / _defaults['hyp_values']['base_img_size']) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(
        dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    # number of warmup iterations, max(3 epochs, 100 iterations)
    nw = max(round(hyp['warmup_epochs'] * nb), 100)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)  # init loss class
    callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    # epoch ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        callbacks.run('on_train_epoch_start')
        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(
                dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(
                range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem',
                    'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
        if RANK in {-1, 0}:
            # progress bar
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)
        optimizer.zero_grad()
        # batch -------------------------------------------------------------
        for i, (imgs, targets, paths, _) in pbar:
            callbacks.run('on_train_batch_start')
            # number integrated batches (since train start)
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / \
                255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(
                    ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(
                        ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(
                            ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(
                    imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    # new shape (stretched to gs-multiple)
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = nn.functional.interpolate(
                        imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(
                    pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / \
                    (i + 1)  # update mean losses
                # (GB)
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', model, ni,
                              imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(
                model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = validate.run(data_dict,
                                                batch_size=batch_size // WORLD_SIZE * 2,
                                                imgsz=imgsz,
                                                half=amp,
                                                model=ema.ema,
                                                single_cls=single_cls,
                                                dataloader=val_loader,
                                                save_dir=save_dir,
                                                plots=False,
                                                callbacks=callbacks,
                                                compute_loss=compute_loss)

            # Update best mAP
            # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi = fitness(np.array(results).reshape(1, -1))
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals,
                          epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,  # {remote, branch, commit} if a git repo
                    'date': datetime.now().isoformat(),
                    'img_size': imgsz}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch,
                              final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            # broadcast 'stop' to all ranks
            dist.broadcast_object_list(broadcast_list, 0)
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        LOGGER.info(
            f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss)  # val best model with plots
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(
                            mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, epoch, results)

    torch.cuda.empty_cache()
    return results, best


@method_header(
    description='''Proceed original config object.''',
    arguments='''
        opt: Config object.
        callbacks: Callback instance''',
    returns='''
    A tuple of essential arguments for the training pipeline''')
def _build_opt(opt: object, callbacks=Callbacks()) -> Tuple:
    opt.sync_dir = get_sync_dir()
    log.STATUS(sys._getframe().f_lineno,
               __file__, __name__,
               f'sync_dir: {opt.sync_dir}')
    opt.data = os.path.join(opt.sync_dir, opt.data)

    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements()

    # Resume (from specified or most recent last.pt)
    if opt.hyp and isinstance(opt.hyp, str):
        opt.hyp = get_meghnad_repo_dir() / 'yolov5' / opt.hyp

    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(
            opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.data, opt.cfg, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), str(
                opt.weights), str(opt.project)  # checks
        if isinstance(opt.hyp, str):
            opt.hyp = check_yaml(opt.hyp)
        assert len(opt.cfg) or len(
            opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            # if default project name, rename to runs/evolve
            if opt.project == str('runs/train'):
                opt.project = str('runs/evolve')
            # pass resume to exist_ok and disable resume
            opt.exist_ok, opt.resume = opt.resume, False
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        # opt.save_dir = str(increment_path(
        #     Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        # opt.save_dir = opt.project / opt.name
        # opt.save_dir = os.path.join(opt.sync_dir, opt.project, opt.name)
        opt.save_dir = Path(opt.output_dir) / opt.name
        if not os.path.isdir(opt.save_dir):
            os.makedirs(opt.save_dir, exist_ok=True)

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != - \
            1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    return opt.hyp, opt, device, callbacks
