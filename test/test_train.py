# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2025/3/20 13:43
# NOTE: Test-only script; use train_script/train.py for official training.
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

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from model.criterions import SetCriterion
from model.SGMP_UNet import Net

from utils import misc
from utils.predict import infer_on_dataset_val
from utils.torch_poly_lr_decay import PolynomialLRDecay

from train_script.get_args_parser import get_args_parser
from utils.config import (
    load_yaml,
    deep_update,
    cli_keys_from_argv,
    resolve_dataset_config,
    apply_config_to_args,
)
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
    # import ipdb;ipdb.set_trace()
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialLRDecay(optimizer, max_decay_steps=args.epochs - 1, end_learning_rate=args.end_lr,
                                     power=args.poly_power)
    # 预训练权重
    if getattr(args, 'pretrain_settings', None):
        print(f"loading pretrained model from: {args.pretrain_settings}")
        state_dict = torch.load(args.pretrain_settings, map_location='cpu')
        model_without_ddp.load_state_dict(state_dict['model'], strict=True)
        optimizer.load_state_dict(state_dict['optimizer'])
        lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        args.start_epoch = state_dict['epoch'] + 1
        model.train()
    # 输出
    output_viz_dir = os.path.join(args.output_dir, 'viz')
    # 导入数据集
    data_loader_train, data_loader_val, _ = create_data_loaders(args)

    logger.debug("Start training ... ...")
    start_time = time.time()
    best_eval_iou = 0
    best_eval_epoch = 0
    log_file = args.log_file or os.path.join(args.output_dir, 'out.log')
    print('log file: ' + log_file)

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        logger.debug('epoch: %3d  optimizer.base_lr: %e' % (epoch, optimizer.param_groups[0]['lr']))
        logger.debug('epoch: %3d  optimizer.backbone_lr: %e' % (epoch, optimizer.param_groups[1]['lr']))

        # train_one_epoch(
        #     model, criterion, data_loader_train, optimizer, device, epoch,
        #     args.clip_max_norm, output_viz_dir, use_wandb=args.use_wandb,
        #     total_epochs=args.epochs, args=args)
        epoch_end_time = time.time()

        val_start_time = epoch_end_time
        # import ipdb;ipdb.set_trace()
        mean_iou = infer_on_dataset_val(model, data_loader_val, device,
                                        msc=False, flip=True, save_pred=False,
                                        out_dir=output_viz_dir + '/src_' + str(epoch), use_wandb=args.use_wandb)
        logger.debug('**************************')
        logger.debug('[Epoch:%2d] val_mean_iou:%0.3f' % (epoch, mean_iou))
        if args.use_wandb:
            wandb.log({'miou val': mean_iou})
        if mean_iou > best_eval_iou:
            best_eval_iou = mean_iou
            best_eval_epoch = epoch
        logger.debug('Best eval epoch:%03d mean_iou: %0.3f' % (best_eval_epoch, best_eval_iou))

        if epoch > -1:
            lr_scheduler.step()

        save_checkpoint(args, epoch, epoch == best_eval_epoch, model_without_ddp, optimizer, lr_scheduler)

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
    if not parsed_args.output_dir:
        parsed_args.output_dir = 'outputs'
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

    parsed_args.log_file = os.path.join(output_path, 'out.log')
    logging.basicConfig(
        filename=parsed_args.log_file,  # 日志文件路径
        format='%(asctime)s %(levelname)s %(module)s-%(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',  # 时间格式
        level=logging.DEBUG  # 日志级别
    )
    # 添加控制台输出
    logging.getLogger().addHandler(
        logging.StreamHandler(sys.stdout))

    if parsed_args.use_wandb:
        wandb_id_file = os.path.join(output_path, str(parsed_args.experiment_name) + '_wandb.txt')
        wandb_id_file = Path(wandb_id_file)
        config = init_or_resume_wandb_run(wandb_id_file,
                                          entity_name=parsed_args.wandb_user,
                                          project_name=parsed_args.wandb_project,
                                          run_name=parsed_args.experiment_name,
                                          args=parsed_args)


def save_checkpoint(args, epoch, is_best, model, optimizer, lr_scheduler):
    if args.output_dir:
        output_dir = args.output_dir
        # 获取所有checkpoint文件并按修改时间排序（最新优先）
        checkpoint_files = [f for f in os.listdir(output_dir)
                            if f.startswith('checkpoint_') and f.endswith('.pth')]
        checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)

        # 定义需要保存的模型路径（当前epoch + best + last）
        checkpoint_paths = [
            os.path.join(output_dir, f'checkpoint_{epoch}.pth'),  # 常规保存
            os.path.join(output_dir, 'checkpoint_last.pth'),  # 新增：最后一个epoch
        ]
        if is_best:
            checkpoint_paths.append(os.path.join(output_dir, 'checkpoint_best.pth'))

        # 保留最新3个 + best + last（避免重复）
        keep_files = {'checkpoint_best.pth', 'checkpoint_last.pth'}  # 必须保留的模型
        keep_files.update(f for f in checkpoint_files[:3])  # 保留最新的3个

        # 删除旧文件（不在keep_files中的checkpoint_*.pth）
        for f in checkpoint_files:
            if f not in keep_files and os.path.exists(os.path.join(output_dir, f)):
                os.remove(os.path.join(output_dir, f))

        # 保存模型
        for checkpoint_path in checkpoint_paths:
            logger.debug(f'Saving checkpoint to: {checkpoint_path}')
            misc.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)


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
    if not args.experiment_name:
        args.experiment_name = 'test_exp_{params_summary}'
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


if __name__ == '__main__':
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
    main(parsed_args)
