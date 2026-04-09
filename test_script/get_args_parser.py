# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2025/4/5 17:31
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('SGMP-UNet', add_help=False)
    # Model name
    parser.add_argument('--st_model', default=None, type=str,
                        help="SGMP-UNet")
    # Transformer
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--num_frames', default=None, type=int,
                        help="Number of frames")
    parser.add_argument('--val_size', default=None, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true', default=None)

    # Init Weights
    parser.add_argument('--is_train', action='store_true', default=None,
                        help='Choose False for train')
    # parser.add_argument('--model_path', type=str,
    #                     default='result/TSIRMT/checkpoint_best_val_miou_0.668.pth',
    #                     help="Path to the model weights.")
    # parser.add_argument('--model_path', type=str,
    #                     default='result/IRSDT/checkpoint_best_val_miou_0.610.pth',
    #                     help="Path to the model weights.")
    parser.add_argument('--model_path', type=str,
                        default=None,
                        help="Path to the model weights.")

    # Segmentation
    parser.add_argument("--save_pred", action="store_true", default=None)
    parser.add_argument('--masks', action='store_true', default=None,
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--dataset', type=str, default=None, help='TSIRMT,NUDT-MIRSDT,IRSDT')
    parser.add_argument('--sequence_names', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='path where to save, empty for no saving')
    parser.add_argument('--output_root', default=None, type=str, help='output root directory')
    parser.add_argument(
        '--device',
        default=None,
        help='device to use for training / testing (优先使用 cuda:1)'
    )
    parser.add_argument('--msc', action='store_true', default=None)
    parser.add_argument('--flip', action='store_true', default=None)

    # Misc
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--num_workers', default=None, type=int)
    parser.add_argument('--world_size', default=None, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default=None, help='url used to set up distributed training')
    parser.add_argument('--config', default=None, type=str, help='inference config path')
    parser.add_argument('--default_config', default=None, type=str, help='base config path')
    parser.add_argument('--dataset_config', default=None, type=str, help='dataset config path override')
    return parser
