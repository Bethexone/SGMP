# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2025/3/20 13:43
import argparse
import datetime
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb
import yaml

from model.criterions import SetCriterion
from model.SGMP_UNet import Net

from utils import misc
from utils.evals import infer_on_dataset_val, infer_on_dataset_train
from utils.torch_poly_lr_decay import PolynomialLRDecay

from train_script.get_args_parser import get_args_parser
from train_script.train_one_epoch import train_one_epoch, create_data_loaders
from utils.wandb_utils import init_or_resume_wandb_run

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
    if hasattr(args, 'pretrain_settings'):
        print(f"loading pretrained model from: {args.pretrain_settings}")
        state_dict = torch.load(args.pretrain_settings, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(state_dict['model'], strict=True)
        optimizer.load_state_dict(state_dict['optimizer'])
        lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        args.start_epoch = state_dict['epoch'] + 1
    model.train()
    # 输出
    output_viz_dir = os.path.join(args.output_dir, 'viz')
    # 导入数据集
    if args.eval_train:
        data_loader_train, data_loader_val, eval_trainloader = create_data_loaders(args, args.eval_train)
    else:
        data_loader_train, data_loader_val = create_data_loaders(args)

    logger.debug("Start training ... ...")
    start_time = time.time()
    best_eval_iou = 0
    best_eval_epoch = 0
    print('log file: ' + args.log_file)

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        logger.debug('epoch: %3d  optimizer.base_lr: %e' % (epoch, optimizer.param_groups[0]['lr']))
        logger.debug('epoch: %3d  optimizer.backbone_lr: %e' % (epoch, optimizer.param_groups[1]['lr']))

        train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, output_viz_dir, use_wandb=args.use_wandb,
            total_epochs=args.epochs, args=args)
        epoch_end_time = time.time()

        val_start_time = epoch_end_time
        logger.debug('**************************')
        mean_iou = infer_on_dataset_val(model, data_loader_val, device,
                                          msc=False, flip=True,
                                          out_dir=os.path.join(output_viz_dir, 'epoch_' + str(epoch)),
                                          use_wandb=args.use_wandb, viz_freq=500)
        logger.debug('[Epoch:%2d] val_mean_iou:%0.3f' % (epoch, mean_iou))

        train_iou = None
        if epoch % 5 == 0 and args.eval_train:
            train_iou = infer_on_dataset_train(model, eval_trainloader, device,
                                               msc=False, flip=True,
                                               out_dir=os.path.join(output_viz_dir, 'epoch_' + str(epoch)),
                                               use_wandb=True,
                                               viz_freq=700)
            logger.debug('[Epoch:%2d] train_mean_iou:%0.3f' % (epoch, train_iou))
            if args.use_wandb:
                wandb.log({'train_miou': train_iou})

        if args.use_wandb:
            wandb.log({'miou val': mean_iou})
        if mean_iou > best_eval_iou:
            best_eval_iou = mean_iou
            best_eval_epoch = epoch
        logger.debug('Best eval epoch:%03d mean_iou: %0.3f' % (best_eval_epoch, best_eval_iou))

        if epoch > -1:
            lr_scheduler.step()

        if train_iou is not None:
            metrics = {'train_miou': train_iou, 'val_miou': mean_iou}
        else:
            metrics = {'val_miou': mean_iou}
        save_checkpoint(args, epoch, epoch == best_eval_epoch, model_without_ddp, optimizer, lr_scheduler, metrics)
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


def make_logger(parsed_args):
    params_summary = (
        f"{datetime.datetime.today().strftime('%Y%m%d%H%M%S')}_"  # 时间戳
        f"t{parsed_args.train_size}v{parsed_args.val_size}f{parsed_args.num_frames:0.1f}_"  # 数据规模
        f"lr{parsed_args.lr:0.1e}_"  # 学习率
        f"{parsed_args.lr_backbone:0.1e}_"  # 骨干学习率
        f"aux{parsed_args.aux_loss:0.1f}_"  # 辅助损失
        f"ep{parsed_args.epochs:02d}"  # 训练轮数
    )
    parsed_args.experiment_name = parsed_args.experiment_name.replace(
        '{params_summary}', params_summary)  # 动态替换模板变量

    output_path = os.path.join(  # 构建三级目录结构
        str(parsed_args.output_dir),  # 根目录
        str(parsed_args.dataset_name),  # 数据集名称
        str(parsed_args.experiment_name))  # 实验标识
    os.makedirs(output_path, exist_ok=True)  # 创建目录

    logging.basicConfig(
        filename=os.path.join(output_path, 'out.log'),  # 日志文件路径
        format='%(asctime)s %(levelname)s %(module)s-%(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',  # 时间格式
        level=logging.DEBUG  # 日志级别
    )
    logging.getLogger().addHandler(  # 添加控制台输出
        logging.StreamHandler(sys.stdout))

    if parsed_args.use_wandb:
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

        # 保留文件规则：best + last + 最新3个常规checkpoint
        keep_files = {
            'checkpoint_last.pth',
            *[f for f in sorted(checkpoint_files,
                                key=lambda x: os.path.getmtime(os.path.join(output_dir, x)),
                                reverse=True)[:3]],
            *[f for f in checkpoint_files if f.startswith('checkpoint_best_')]
        }

        # 清理旧文件
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
    args.output_dir = os.path.join(args.output_dir, args.dataset_name)
    make_logger(args)
    # import ipdb; ipdb.set_trace()
    misc.init_distributed_mode(args)
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
    wandb.finish()


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser('train script', parents=[get_args_parser()])
    parsed_args = args_parser.parse_args()
    main(parsed_args)
