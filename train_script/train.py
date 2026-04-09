# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2025/3/20 13:43
import argparse
import datetime
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml

from model.criterions import SetCriterion
from model.SGMP_UNet import Net

from utils import misc
from utils.predict import infer_on_dataset_val, infer_on_dataset_train
from utils.torch_poly_lr_decay import PolynomialLRDecay

from train_script.get_args_parser import get_args_parser
from train_script.train_one_epoch import train_one_epoch, create_data_loaders
from utils.wandb_utils import init_or_resume_wandb_run
from utils.config import (
    load_yaml,
    deep_update,
    cli_keys_from_argv,
    resolve_dataset_config,
    apply_config_to_args,
    save_yaml,
)
from utils.output import build_output_dir
from utils.logger import init_logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_VAL_MIOU_RE = re.compile(r"val_miou_([0-9]+(?:\.[0-9]+)?)")


def _parse_val_miou(filename: str) -> float | None:
    match = _VAL_MIOU_RE.search(filename)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _select_best_checkpoints(best_files: list[str], keep_top: int) -> set[str]:
    scored = []
    unscored = []
    for name in best_files:
        score = _parse_val_miou(name)
        if score is None:
            unscored.append(name)
        else:
            scored.append((name, score))

    unique_scores = sorted({score for _, score in scored}, reverse=True)
    if len(unique_scores) <= keep_top:
        keep = {name for name, _ in scored}
        keep.update(unscored)
        return keep

    cutoff = unique_scores[keep_top - 1]
    keep = {name for name, score in scored if score >= cutoff}
    keep.update(unscored)
    return keep


def _get_wandb():
    import wandb
    return wandb


def train(args, device, model, criterion):
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    # import ipdb; ipdb.set_trace()
    param_dicts = model_without_ddp.configure_param(args.lr, args.lr_backbone)
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    # 使用带重启的余弦退火学习率调度
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.cos_cycle, T_mult=args.cos_mult, eta_min=args.end_lr
    )
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.end_lr)
    # lr_scheduler = PolynomialLRDecay(optimizer, max_decay_steps=args.epochs - 1, end_learning_rate=args.end_lr,
    #                                  power=args.poly_power)
    # 预训练权重
    pretrain_path = getattr(args, "pretrain_settings", None)
    if pretrain_path:
        if not os.path.exists(pretrain_path):
            raise FileNotFoundError(f"pretrain_settings not found: {pretrain_path}")
        print(f"loading pretrained model from: {pretrain_path}")
        state_dict = torch.load(pretrain_path, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(state_dict['model'], strict=True)
        optimizer.load_state_dict(state_dict['optimizer'])
        lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        args.start_epoch = state_dict['epoch'] + 1
    model.train()
    # 输出
    output_viz_dir = os.path.join(args.output_dir, 'viz')
    # 导入数据集
    eval_on_main_only = args.distributed and misc.is_main_process()
    eval_trainloader = None
    if args.eval_train:
        if not args.distributed or eval_on_main_only:
            data_loader_train, data_loader_val, eval_trainloader, sampler_train = create_data_loaders(
                args, eval_train=True, eval_on_main_only=eval_on_main_only)
        else:
            data_loader_train, data_loader_val, sampler_train = create_data_loaders(
                args, eval_train=False, eval_on_main_only=False)
    else:
        data_loader_train, data_loader_val, sampler_train = create_data_loaders(
            args, eval_train=False, eval_on_main_only=eval_on_main_only)

    logger.debug("Start training ... ...")
    start_time = time.time()
    best_eval_iou = 0
    best_eval_epoch = 0
    print('log file: ' + os.path.join(args.output_dir, 'train.log'))

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and sampler_train is not None:
            sampler_train.set_epoch(epoch)
        epoch_start_time = time.time()
        logger.debug('epoch: %3d  optimizer.base_lr: %e' % (epoch, optimizer.param_groups[0]['lr']))
        logger.debug('epoch: %3d  optimizer.backbone_lr: %e' % (epoch, optimizer.param_groups[1]['lr']))

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, output_viz_dir, args)
        if train_stats and "loss" in train_stats:
            logger.debug('[Epoch:%2d] train_loss_avg:%0.6f' % (epoch, train_stats["loss"]))
        epoch_end_time = time.time()

        val_start_time = epoch_end_time
        logger.debug('**************************')
        mean_iou = None
        if args.eval_val_interval and epoch % args.eval_val_interval == 0:
            if misc.is_dist_avail_and_initialized():
                dist.barrier()
            if misc.is_main_process():
                mean_iou = infer_on_dataset_val(model, data_loader_val, device,
                                                msc=False, flip=True,
                                                out_dir=os.path.join(output_viz_dir, 'epoch_' + str(epoch)),
                                                use_wandb=args.use_wandb, viz_freq=args.viz_freq)
                logger.debug('[Epoch:%2d] val_mean_iou:%0.3f' % (epoch, mean_iou))
            if misc.is_dist_avail_and_initialized():
                dist.barrier()
            model.train()

        train_iou = None
        if args.eval_train and args.eval_interval and epoch % args.eval_interval == 0:
            if misc.is_dist_avail_and_initialized():
                dist.barrier()
            if misc.is_main_process() and eval_trainloader is not None:
                train_iou = infer_on_dataset_train(model, eval_trainloader, device,
                                                   msc=False, flip=True,
                                                   out_dir=os.path.join(output_viz_dir, 'epoch_' + str(epoch)),
                                                   use_wandb=args.use_wandb,
                                                   viz_freq=args.viz_freq)
                logger.debug('[Epoch:%2d] train_mean_iou:%0.3f' % (epoch, train_iou))
                if args.use_wandb:
                    _get_wandb().log({'train_miou': train_iou})
            if misc.is_dist_avail_and_initialized():
                dist.barrier()
            model.train()

        if args.use_wandb and mean_iou is not None:
            _get_wandb().log({'miou val': mean_iou})
        if mean_iou is not None:
            if mean_iou > best_eval_iou:
                best_eval_iou = mean_iou
                best_eval_epoch = epoch
            logger.debug('Best eval epoch:%03d mean_iou: %0.3f' % (best_eval_epoch, best_eval_iou))

        if epoch > -1:
            lr_scheduler.step(epoch + 1)

        if mean_iou is not None:
            if train_iou is not None:
                metrics = {'train_miou': train_iou, 'val_miou': mean_iou}
            else:
                metrics = {'val_miou': mean_iou}
            save_checkpoint(args, epoch, epoch == best_eval_epoch, model_without_ddp, optimizer, lr_scheduler, metrics)
        else:
            save_checkpoint_last(args, model_without_ddp, optimizer, lr_scheduler, epoch)
        val_endtime = time.time()

        train_time_str = str(datetime.timedelta(seconds=int(epoch_end_time - epoch_start_time)))
        eval_time_str = str(datetime.timedelta(seconds=int(val_endtime - val_start_time)))
        logger.debug(
            'Epoch:{}/{} Training_time:{} Eval_time:{}'.format(epoch, args.epochs, train_time_str, eval_time_str))
        logger.debug('##########################################################')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.debug('Training time {}'.format(total_time_str))
    return model


def setup_output_dir(parsed_args):
    output_root = getattr(parsed_args, "output_root", None)
    params_summary = (
        f"{datetime.datetime.today().strftime('%Y%m%d%H%M%S')}_"  # 时间戳
        f"t{parsed_args.train_size}v{parsed_args.val_size}f{parsed_args.num_frames:0.1f}_"  # 数据规模
        f"lr{parsed_args.lr:0.1e}_"  # 学习率
        f"{parsed_args.lr_backbone:0.1e}_"  # 骨干学习率
        f"aux{parsed_args.aux_loss:0.1f}_"  # 辅助损失
        f"ep{parsed_args.epochs:02d}"  # 训练轮数
    )
    if not parsed_args.experiment_name:
        parsed_args.experiment_name = "exp_{params_summary}"
    parsed_args.experiment_name = parsed_args.experiment_name.replace(
        '{params_summary}', params_summary)  # 动态替换模板变量
    output_path = build_output_dir(
        output_root, parsed_args.dataset_name, parsed_args.experiment_name, "train")
    parsed_args.output_dir = output_path
    return output_path


def make_logger(parsed_args):
    output_path = parsed_args.output_dir
    _, log_path = init_logger(output_path, log_name="train.log")

    if parsed_args.use_wandb:
        _get_wandb()
        wandb_id_file = os.path.join(output_path, str(parsed_args.experiment_name) + '_wandb.txt')
        wandb_id_file = Path(wandb_id_file)
        config = init_or_resume_wandb_run(wandb_id_file,
                                          entity_name=parsed_args.wandb_user,
                                          project_name=parsed_args.wandb_project,
                                          run_name=parsed_args.experiment_name,
                                          args=parsed_args)


def save_checkpoint(args, epoch, is_best, model, optimizer, lr_scheduler, metrics=None):
    """
    保存模型checkpoint，文件名包含mIoU指标
    Args:
        metrics (dict): 包含评估指标如 {'train_mIoU': 0.85, 'val_mIoU': 0.92}
    """
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_dir = args.output_dir
        # 获取当前指标值
        # miou_str = f"miou{metrics['mIoU']:.4f}" if metrics and 'mIoU' in metrics else ""
        if metrics is not None:
            metrics_str = '_'.join([f'{key}_{value:.3f}' for key, value in metrics.items()])
        else:
            metrics_str = ''
        # 定义保存路径（包含mIoU）
        checkpoint_paths = [
            os.path.join(output_dir, f'checkpoint_{epoch:04d}_{metrics_str}.pth'),  # 示例: checkpoint_0010_miou0.8732.pth
            os.path.join(output_dir, 'checkpoint_last.pth')
        ]
        if is_best:
            best_path = os.path.join(output_dir, f'checkpoint_best_{metrics_str}.pth')
            checkpoint_paths.append(best_path)

        # 文件管理策略
        checkpoint_files = [f for f in os.listdir(output_dir)
                            if f.startswith('checkpoint_') and f.endswith('.pth')]

        best_files = [f for f in checkpoint_files if f.startswith('checkpoint_best_')]
        best_keep = set(best_files)
        if is_best:
            best_keep = _select_best_checkpoints(
                best_files + [os.path.basename(best_path)],
                keep_top=3,
            )

        regular_checkpoints = [
            f for f in checkpoint_files
            if f.startswith('checkpoint_')
            and not f.startswith('checkpoint_best_')
            and f != 'checkpoint_last.pth'
        ]
        latest_regular = [
            f for f in sorted(regular_checkpoints,
                              key=lambda x: os.path.getmtime(os.path.join(output_dir, x)),
                              reverse=True)[:3]
        ]

        # 保留文件规则：best + last + 最新3个常规checkpoint
        keep_files = {
            'checkpoint_last.pth',
            *latest_regular,
            *best_keep,
        }

        # 清理旧文件
        if misc.is_main_process():
            for f in checkpoint_files:
                if f not in keep_files:
                    os.remove(os.path.join(output_dir, f))
                    logger.debug(f"Deleted old checkpoint: {f}")

        # 保存模型
        for path in checkpoint_paths:
            logger.info(f"Saving checkpoint to {path}")
            misc.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'metrics': metrics  # 保存评估指标
            }, path)


def save_checkpoint_last(args, model, optimizer, lr_scheduler, epoch):
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, 'checkpoint_last.pth')
        logger.info(f"Saving checkpoint to {path}")
        misc.save_on_master({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
            'metrics': None
        }, path)


def main(args):
    print('starting main ...')
    # import ipdb; ipdb.set_trace()
    misc.init_distributed_mode(args)
    # 确定性输出，结果可复现
    cudnn.benchmark = False
    cudnn.deterministic = True
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if misc.is_main_process():
        make_logger(args)
        cfg = getattr(args, "_cfg", None)
        if cfg and getattr(args, "output_dir", None):
            final_cfg = deep_update(cfg, vars(args))
            save_yaml(final_cfg, os.path.join(args.output_dir, "exp_config.yaml"))
            logger.debug(
                "config: dataset=%s num_frames=%s batch_size=%s lr=%s lr_backbone=%s epochs=%s eval_interval=%s eval_val_interval=%s",
                args.dataset_name,
                args.num_frames,
                args.batch_size,
                args.lr,
                args.lr_backbone,
                args.epochs,
                args.eval_interval,
                args.eval_val_interval,
            )
    logger.debug("git:\n  {}\n".format(misc.get_sha()))
    device = torch.device(args.device)
    model = Net(num_tl=args.num_frames, is_train=args.is_train)
    # wandb.watch(
    #     model,
    #     log="all",  # 监控参数和梯度（可选："gradients"或"parameters"）
    #     log_freq=5,  # 每100步记录一次（避免日志过大）
    #     log_graph=True  # 记录计算图（会占用额外存储空间）
    # )
    criterion = SetCriterion()
    logger.debug(str(model))
    model.to(device)

    train(args, device, model, criterion)
    if args.use_wandb:
        _get_wandb().finish()


if __name__ == '__main__':
    import os

    cwd = os.getcwd()
    print(cwd)
    args_parser = argparse.ArgumentParser('train script', parents=[get_args_parser()])
    parsed_args = args_parser.parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_config_path = parsed_args.default_config or os.path.join(repo_root, "configs", "default.yaml")
    train_config_path = parsed_args.config or os.path.join(repo_root, "configs", "train.yaml")
    cfg = deep_update(load_yaml(default_config_path), load_yaml(train_config_path))
    cli_keys = cli_keys_from_argv(sys.argv)
    apply_config_to_args(parsed_args, cfg, cli_keys)
    if not getattr(parsed_args, "dataset_config", None):
        resolved = resolve_dataset_config(cfg)
        if resolved:
            parsed_args.dataset_config = resolved
    if not getattr(parsed_args, "dataset_config", None):
        raise SystemExit("dataset_config is required. Set it via config or --dataset_config.")
    if not os.path.exists(parsed_args.dataset_config):
        raise SystemExit(f"dataset_config not found: {parsed_args.dataset_config}")

    if cfg.get("eval_interval") is not None and cfg["eval_interval"] < 1:
        cfg["eval_interval"] = 1
    if cfg.get("eval_val_interval") is not None and cfg["eval_val_interval"] < 1:
        cfg["eval_val_interval"] = 1
    parsed_args._cfg = cfg
    setup_output_dir(parsed_args)
    main(parsed_args)
