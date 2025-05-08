# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2025/3/21 19:29
import argparse
import torch


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    parser.add_argument('--num_frames', default=6, type=int,
                        help="Number of Frames Sequence")
    parser.add_argument('--pretrain_settings', default=argparse.SUPPRESS, type=str, help='pretrain weights path')

    # log Save Paths
    parser.add_argument('--dataset_name', default='NUDT-MIRSDT', help="TSIRMT, NUDT-MIRSDT, IRDST")
    parser.add_argument('--experiment_name', default='SGMP_hyperparam_exp_{params_summary}')
    parser.add_argument('--output_dir', default='result/', help='save path')
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--wandb_user', type=str, default='2860873783')
    parser.add_argument('--wandb_project', type=str, default='SGMP')
    parser.add_argument('--wandb_resume_id', type=str, default=None)

    # Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument(
        '--num_fusion_head',
        type=int,
        default=[2, 2, 2, 2],
        nargs='*',  # 允许 0 个或多个值
        help="List of integers (default: [2, 2, 2, 2])"
    )

    # Training Params
    parser.add_argument('--is_train', default=True, type=bool,
                        help='Choose True for train')
    parser.add_argument('--eval_train', default=True, type=bool, help='Choose True for eval Train_dataset')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--end_lr', default=1e-7, type=float)
    # parser.add_argument('--lr_drop', default=4, type=int)
    parser.add_argument('--poly_power', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=3e-4, type=float)
    parser.add_argument('--cos_cycle', default=15, type=int, help='初始重启周期')
    parser.add_argument('--cos_mult', default=1, type=int, help='每次重启后周期长度的倍增因子')

    parser.add_argument('--aux_loss', default=0.5, type=float)
    parser.add_argument('--aux_loss_norm', default=0, type=float, help='是否对损失归一化')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=45, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--batch_eval', default=4, type=int)
    parser.add_argument('--train_size', default=320, type=int)
    parser.add_argument('--val_size', default=320, type=int)

    # Misc
    parser.add_argument(
        '--device',
        default='cuda:1' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 'cuda' if torch.cuda.is_available() else 'cpu',
        help='device to use for training / testing (优先使用 cuda:1)'
    )
    parser.add_argument('--seed', default=42, type=int)
    # parser.add_argument('--resume', default=False, help='resume from checkpoint')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # log
    parser.add_argument('--log_file', default='result/out.log', help='log file')

    return parser
