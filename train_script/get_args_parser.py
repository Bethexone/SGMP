# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2025/3/21 19:29
import argparse
import torch


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    parser.add_argument('--num_frames', default=None, type=int,
                        help="Number of Frames Sequence")
    parser.add_argument('--pretrain_settings', default=None, type=str, help='pretrain weights path')

    # log Save Paths
    parser.add_argument('--dataset_name', default=None, help="TSIRMT, NUDT-MIRSDT, IRSDT")
    parser.add_argument('--experiment_name', default=None)
    parser.add_argument('--output_dir', default=None, help='save path')
    parser.add_argument('--output_root', default=None, help='output root directory')
    parser.add_argument('--use_wandb', action='store_true', default=None)
    parser.add_argument('--wandb_user', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_resume_id', type=str, default=None)

    # Loss coefficients
    parser.add_argument('--mask_loss_coef', default=None, type=float)
    parser.add_argument('--dice_loss_coef', default=None, type=float)
    parser.add_argument('--bbox_loss_coef', default=None, type=float)
    parser.add_argument('--giou_loss_coef', default=None, type=float)
    parser.add_argument('--eos_coef', default=None, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--remove_difficult', action='store_true', default=None)

    parser.add_argument(
        '--num_fusion_head',
        type=int,
        default=None,
        nargs='*',  # 允许 0 个或多个值
        help="List of integers (default: [2, 2, 2, 2])"
    )

    # Training Params
    parser.add_argument('--is_train', action='store_true', default=None,
                        help='Choose True for train')
    parser.add_argument('--eval_train', action='store_true', default=None, help='Choose True for eval Train_dataset')
    parser.add_argument('--eval_interval', default=None, type=int, help='eval train interval in epochs')
    parser.add_argument('--eval_val_interval', default=None, type=int, help='eval val interval in epochs')
    parser.add_argument('--viz_freq', default=None, type=int, help='visualization frequency')
    parser.add_argument('--train_print_freq', default=None, type=int, help='train log print frequency')
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--lr_backbone', default=None, type=float)
    parser.add_argument('--end_lr', default=None, type=float)
    # parser.add_argument('--lr_drop', default=4, type=int)
    parser.add_argument('--poly_power', default=None, type=float)
    parser.add_argument('--weight_decay', default=None, type=float)
    parser.add_argument('--cos_cycle', default=None, type=int, help='初始重启周期')
    parser.add_argument('--cos_mult', default=None, type=int, help='每次重启后周期长度的倍增因子')

    parser.add_argument('--aux_loss', default=None, type=float)
    parser.add_argument('--aux_loss_norm', default=None, type=float, help='是否对损失归一化')
    parser.add_argument('--start_epoch', default=None, type=int)
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--clip_max_norm', default=None, type=float, help='gradient clipping max norm')
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--batch_eval', default=None, type=int)
    parser.add_argument('--train_size', default=None, type=int)
    parser.add_argument('--val_size', default=None, type=int)

    # Misc
    parser.add_argument(
        '--device',
        default=None,
        help='device to use for training / testing (优先使用 cuda:1)'
    )
    parser.add_argument('--seed', default=None, type=int)
    # parser.add_argument('--resume', default=False, help='resume from checkpoint')
    parser.add_argument('--num_workers', default=None, type=int)
    parser.add_argument('--world_size', default=None, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default=None, help='url used to set up distributed training')

    # log
    parser.add_argument('--log_file', default=None, help='log file')
    parser.add_argument('--config', default=None, type=str, help='train config path')
    parser.add_argument('--default_config', default=None, type=str, help='base config path')
    parser.add_argument('--dataset_config', default=None, type=str, help='dataset config path override')

    return parser
