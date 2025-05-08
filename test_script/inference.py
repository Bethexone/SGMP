"""
Inference code for LMAFormer.
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
from utils.evals import infer_on_dataset_test
from test_script.get_args_parser import get_args_parser
from utils.metrics import SigmoidMetric, SamplewiseSigmoidMetric, PD0_FA0
from model.net import Net
from Datasets.datasets.utils import collate_fn
from Datasets.datasets.val.val_data import ValDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

    if os.name == 'nt':
        path_config = 'Datasets/dataset_config_win.yaml'
    elif os.name == 'posix':
        hostname = socket.gethostname()
        if hostname == 'lq':
            path_config = 'Datasets/dataset_config_lq.yaml'
        else:
            path_config = 'Datasets/dataset_config_linux.yaml'

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
    device = torch.device(args.device)
    utils_misc.init_distributed_mode(args)
    seed = args.seed + utils_misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = Net(num_tl=args.num_frames, is_train=args.is_train)
    model.to(device)
    iou_metric = SigmoidMetric()
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0)
    PD_FA = PD0_FA0(nclass=1, thre=0)
    args.sequence_names = None
    run_inference(args, device, model, iou_metric, nIoU_metric, PD_FA)
    print('Thank You!')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser('inference script', parents=[get_args_parser()])
    parsed_args = args_parser.parse_args()
    if hasattr(parsed_args, 'output_dir'):
        experiment_name = os.path.splitext(os.path.basename(parsed_args.model_path))[0]
        parsed_args.output_dir = os.path.join(parsed_args.output_dir, parsed_args.dataset, experiment_name)
        os.makedirs(parsed_args.output_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(parsed_args.output_dir, 'out.log'),
        format='%(asctime)s %(levelname)s %(module)s-%(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger.debug(parsed_args)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logger.debug('output_dir: ' + str(parsed_args.output_dir))
    logger.debug('experiment_name:%s' % experiment_name)
    logger.debug('log file: ' + str(os.path.join(parsed_args.output_dir, 'out.log')))
    main(parsed_args)
