# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2025/3/20 13:43
import socket

import math
import os
from pathlib import Path
from typing import Iterable

import wandb

from utils import misc
import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, DistributedSampler

from Datasets.datasets.train.train_data import TrainDataset
from Datasets.datasets.val.val_data import ValDataset
from Datasets.datasets.utils import collate_fn
from Datasets.datasets import transforms as T
import csv
import time
import logging
from utils.wandb_utils import get_viz_img

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def record_csv(filepath, row):
    with open(filepath, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    return


def create_data_loaders(args, eval_train=False):
    if os.name == 'nt':
        path_config = 'Datasets/dataset_config_win.yaml'
    elif os.name == 'posix':
        hostname = socket.gethostname()  # 最简单可靠的方法
        if hostname == 'lq':
            path_config = 'Datasets/dataset_config_lq.yaml'
        else:
            path_config = 'Datasets/dataset_config_wsl.yaml'

    dataset_train = TrainDataset(name_dataset=args.dataset_name, num_frames=args.num_frames, train_size=args.train_size,
                                 path_config=path_config)
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_train.set_epoch(args.start_epoch)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train,
                                   batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn,
                                   num_workers=args.num_workers)

    dataset_val = ValDataset(name_dataset=args.dataset_name, num_frames=args.num_frames, val_size=args.val_size,
                             path_config=path_config)
    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_val = torch.utils.data.BatchSampler(
        sampler_val, args.batch_eval, drop_last=False)
    data_loader_val = DataLoader(dataset_val,
                                 batch_sampler=batch_sampler_val,
                                 collate_fn=collate_fn,
                                 num_workers=args.num_workers)
    if eval_train:
        sampler = torch.utils.data.SequentialSampler(dataset_train)
        batch_sampler = torch.utils.data.BatchSampler(
            sampler, args.batch_eval, drop_last=True)
        eval_trainloader = DataLoader(dataset_train, batch_sampler=batch_sampler, collate_fn=collate_fn,
                                      num_workers=args.num_workers)
        return data_loader_train, data_loader_val, eval_trainloader
    else:
        return data_loader_train, data_loader_val


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    output_viz_dir=Path('./outputs/'), use_wandb: bool = False,
                    viz_freq: int = 700, total_epochs=15, args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train Epoch: [{}/{}]:'.format(epoch, total_epochs)

    inverse_norm_transform = T.InverseNormalizeTransforms()
    if not os.path.exists(output_viz_dir):
        os.makedirs(output_viz_dir)
    _loss_t_csv_fn = os.path.join(output_viz_dir, 'loss.csv')
    if epoch == 0 and os.path.exists(_loss_t_csv_fn):
        os.rename(_loss_t_csv_fn, os.path.join(output_viz_dir, 'loss_{}.csv'.format(time.time())))

    print_freq = 10  # 打印频率，后续打印结果在misc.log_every函数中

    model.train()
    criterion.train()
    weight_dict = {"loss_mask": args.mask_loss_coef, "loss_dice": args.dice_loss_coef}

    i_iter = 0
    loss_sum = 0
    item_count = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        i_iter = i_iter + 1
        # print(fr'{samples:}', samples.shape)
        samples.to(device)
        targets = [
            {k: v.to(device) if k == 'masks' else v for k, v in target.items()}
            for target in targets
        ]
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # 分布式训练
        loss_dict_reduced = misc.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            logger.critical("Loss is {}, skip training for this sample".format(loss_value))
            logger.critical(loss_dict_reduced)
            logger.debug('video_name: {} frame_ids:{} center_frame:{}'.format(targets[0]['video_name'],
                                                                              str(targets[0]['frame_ids']),
                                                                              targets[0]['center_frame']))
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if use_wandb:
            wandb_dict = {'loss': loss_value, 'lr': optimizer.param_groups[0]["lr"]}
            if i_iter % viz_freq == 0:
                viz_img = get_viz_img(samples.frame, targets, outputs, inverse_norm_transform)
                wandb_dict['train_viz_img'] = wandb.Image(viz_img)
            wandb.log(wandb_dict)
        loss_sum += float(loss_value)
        item_count += 1
        if i_iter % 50 == 49:
            loss_avg = loss_sum / item_count
            loss_sum = 0
            item_count = 0
            record_csv(_loss_t_csv_fn, ['%e' % loss_avg])
        # import ipdb;ipdb.set_trace()
        # break
    metric_logger.synchronize_between_processes()
    logger.debug("Averaged stats:{}".format(metric_logger))
    # save_loss_plot(epoch, _loss_t_csv_fn, viz_save_dir=output_viz_dir)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    import argparse
    import logging

    import random
    import sys
    import time

    import numpy as np
    import torch.backends.cudnn as cudnn
    from torch.utils.data import DataLoader
    from torch.utils.data import DataLoader, DistributedSampler

    from Datasets.datasets.train.train_data import TrainDataset
    from Datasets.datasets.utils import collate_fn
    from Datasets.datasets.val.val_data import ValDataset
    from model.criterions import SetCriterion
    from model.net import Net
    from utils import misc
    from utils.torch_poly_lr_decay import PolynomialLRDecay
    from train import get_args_parser, record_csv

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)


    def main(args):
        print('starting main ...')
        # 确定性输出，结果可复现
        cudnn.benchmark = False
        cudnn.deterministic = True
        # fix the seed for reproducibility
        seed = args.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # import ipdb; ipdb.set_trace()
        misc.init_distributed_mode(args)
        logger.debug("git:\n  {}\n".format(misc.get_sha()))
        device = torch.device(args.device)
        model = Net(num_tl=args.num_frames)
        criterion = SetCriterion()
        logger.debug(str(model))
        model.to(device)
        param_dicts = model.configure_param(args.lr, args.lr_backbone)

        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = PolynomialLRDecay(optimizer, max_decay_steps=args.epochs - 1, end_learning_rate=args.end_lr,
                                         power=args.poly_power)
        data_loader_train, data_loader_val = create_data_loaders(parsed_args)
        train_one_epoch(model, criterion, data_loader=data_loader_train, optimizer=optimizer, device=device, epoch=0,
                        args=args)


    args_parser = argparse.ArgumentParser('train script', parents=[get_args_parser()])
    parsed_args = args_parser.parse_args()
    main(parsed_args)
