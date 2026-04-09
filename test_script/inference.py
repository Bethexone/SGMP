"""
Inference code for SGMP-UNet.
Based on VisTR (https://github.com/Epiphqny/VisTR)
and DETR (https://github.com/facebookresearch/detr)
and MED-VT (https://github.com/rkyuca/medvt)
"""
import socket
import sys
import argparse
import logging
import random
import numpy as np
import os
import torch
from torch.utils.data import DataLoader

import utils.misc as utils_misc
from utils.config import (
    load_yaml,
    deep_update,
    cli_keys_from_argv,
    resolve_dataset_config,
    apply_config_to_args,
    save_yaml,
)
from utils.predict import infer_on_dataset_test
from utils.output import build_output_dir
from utils.logger import init_logger
from test_script.get_args_parser import get_args_parser
from utils.metrics import SigmoidMetric, SamplewiseSigmoidMetric, PD0_FA0
from model.SGMP_UNet import Net
from Datasets.datasets.utils import collate_fn
from Datasets.datasets.val.val_data import ValDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger("PIL").setLevel(logging.WARNING)

def run_inference(args, device, model, iou_metric, nIoU_metric, PD0_FA0, load_state_dict=True, out_dir=None):
    if out_dir is None:
        out_dir = args.output_dir
    if out_dir is None or len(out_dir) == 0:
        out_dir = './results'
    os.makedirs(out_dir, exist_ok=True)
    # ### Data Loader #########
    if not hasattr(args, 'save_pred'):
        args.save_pred = False
    if not hasattr(args, 'msc'):
        args.msc = False
    if not hasattr(args, 'flip'):
        args.flip = False

    path_config = getattr(args, 'dataset_config', None)

    dataset_val = ValDataset(name_dataset=args.dataset,
                             num_frames=args.num_frames,
                             val_size=args.val_size,
                             sequence_names=args.sequence_names,
                             max_sc=args.input_max_sc if hasattr(args, 'input_max_sc') and args.msc else None,
                             path_config=path_config)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_val = torch.utils.data.BatchSampler(sampler_val, args.batch_size, drop_last=False)
    data_loader_val = DataLoader(dataset_val, batch_sampler=batch_sampler_val,
                                 collate_fn=collate_fn,
                                 num_workers=args.num_workers)
    with torch.no_grad():
        if load_state_dict:
            state_dict = torch.load(args.model_path, weights_only=False, map_location=device)['model']
            model.load_state_dict(state_dict, strict=True)
        model.eval()
        infer_on_dataset_test(args, model, data_loader_val, device,
                              iou_metric, nIoU_metric, PD0_FA0,
                              msc=args.msc,flip=args.flip,
                              save_pred=args.save_pred,
                              out_dir=args.output_dir,
                              msc_scales=args.msc_scales if hasattr(args, 'msc_scales') else None,
                              )


def main(args):
    device_name = args.device
    if not device_name or device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    utils_misc.init_distributed_mode(args)
    seed = (args.seed or 42) + utils_misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = Net(num_tl=args.num_frames, is_train=args.is_train)
    model.to(device)
    iou_metric = SigmoidMetric()
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0)
    PD_FA = PD0_FA0(nclass=1, thre=0)
    run_inference(args, device, model, iou_metric, nIoU_metric, PD_FA)
    print('Thank You!')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser('inference script', parents=[get_args_parser()])
    parsed_args = args_parser.parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_config_path = parsed_args.default_config or os.path.join(repo_root, "configs", "default.yaml")
    infer_config_path = parsed_args.config or os.path.join(repo_root, "configs", "infer.yaml")
    cfg = deep_update(load_yaml(default_config_path), load_yaml(infer_config_path))
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
    if parsed_args.model_path:
        if not os.path.exists(parsed_args.model_path):
            raise SystemExit(f"model_path not found: {parsed_args.model_path}")
        experiment_name = os.path.splitext(os.path.basename(parsed_args.model_path))[0]
        output_root = getattr(parsed_args, "output_root", None)
        parsed_args.output_dir = build_output_dir(output_root, parsed_args.dataset, experiment_name, "infer")
        os.makedirs(parsed_args.output_dir, exist_ok=True)
        init_logger(parsed_args.output_dir, log_name="infer.log")
        final_cfg = deep_update(cfg, vars(parsed_args))
        save_yaml(final_cfg, os.path.join(parsed_args.output_dir, "exp_config.yaml"))
    else:
        raise SystemExit("model_path is required. Set it via config or --model_path.")

    logger.debug(parsed_args)
    logger.debug('output_dir: ' + str(parsed_args.output_dir))
    logger.debug('experiment_name:%s' % experiment_name)
    logger.debug('log file: ' + str(os.path.join(parsed_args.output_dir, 'infer.log')))
    main(parsed_args)
