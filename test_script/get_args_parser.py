# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2025/4/5 17:31
import argparse

import torch


def get_args_parser():
    parser = argparse.ArgumentParser('SAMP_UNet', add_help=False)
    # Model name
    parser.add_argument('--st_model', default='SAMP_UNet', type=str,
                        help="SAMP_UNet")
    # Transformer
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_frames', default=6, type=int,
                        help="Number of frames")
    parser.add_argument('--val_size', default=320, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Init Weights
    parser.add_argument('--is_train', default=False, type=bool,
                             help='Choose False for train')
    # parser.add_argument('--model_path', type=str,
    #                     default='result/TSIRMT/checkpoint_best_val_miou_0.668.pth',
    #                     help="Path to the model weights.")
    # parser.add_argument('--model_path', type=str,
    #                     default='result/IRDST/checkpoint_best_val_miou_0.610.pth',
    #                     help="Path to the model weights.")
    parser.add_argument('--model_path', type=str,
                        default='result/NUDT-MIRSDT/checkpoint_best_val_miou_0.763.pth',
                        help="Path to the model weights.")

    # Segmentation
    parser.add_argument("--save_pred", action="store_true", default=True)
    parser.add_argument('--masks', action='store_true', default=True,
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--dataset', type=str, default='NUDT-MIRSDT', help='TSIRMT,NUDT-MIRSDT,IRDST')
    parser.add_argument('--sequence_names', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='predict',
                        help='path where to save, empty for no saving')
    parser.add_argument(
        '--device',
        default='cuda:1' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 'cuda' if torch.cuda.is_available() else 'cpu',
        help='device to use for training / testing (优先使用 cuda:1)'
    )
    parser.add_argument('--msc', action='store_true')
    parser.add_argument('--flip', action='store_true', default=True)

    # Misc
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser