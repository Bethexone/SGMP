import csv
import logging
import os
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils.misc import interpolate, is_main_process
from utils.wandb_utils import get_viz_img
from utils.metrics import eval_iou, PRMetric, ROCMetric, TargetROCMetric
from utils.Visual import save_video_iou_csv, save_test_csv, save_pred_images
from Datasets.datasets import transforms as T
from utils.model_inputs import ModelInputs

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_CURVE_BINS = 100


def _save_curve_csv(out_dir, filename, header, rows):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def create_eval_save_dir_name_from_args(_args):
    _dir_name = 'infer_%s%3d%slpp_mode%d_sc%0.2f_%d' % (
        _args.dataset,
        _args.val_size,
        'msc' if _args.msc else 'ssc',
        _args.lprop_mode,
        _args.lprop_scale,
        int(time.time()))
    return _dir_name


def _resize_sequence(frame, frame_seq, size):
    if size is None:
        return frame, frame_seq
    frame = interpolate(frame, size=size, mode="bilinear", align_corners=False)
    b, c, t, h, w = frame_seq.shape
    seq = frame_seq.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    seq = interpolate(seq, size=size, mode="bilinear", align_corners=False)
    _, _, nh, nw = seq.shape
    seq = seq.reshape(b, t, c, nh, nw).permute(0, 2, 1, 3, 4)
    return frame, seq


def compute_predictions_flip_ms(model, samples, gt_shape, ms=True, ms_gather='mean', flip=True,
                                flip_gather='mean', scales=None, sigmoid=True):
    outputs = compute_predictions_ms(model, samples, gt_shape, ms=ms, ms_gather=ms_gather,
                                     scales=scales, sigmoid=sigmoid)
    outputs['pred_masks'] = interpolate(outputs['pred_masks'], size=gt_shape, mode="bilinear",
                                        align_corners=False)
    if flip:
        samples_flipped = augment_flip(samples.copy())

        outputs_flipped = compute_predictions_ms(model, samples_flipped, gt_shape, ms=ms,
                                                 ms_gather=ms_gather, scales=scales, sigmoid=sigmoid)
        outputs_flipped['pred_masks'] = interpolate(outputs_flipped['pred_masks'], size=gt_shape,
                                                    mode="bilinear", align_corners=False)
        if flip_gather == 'max':
            outputs['pred_masks'] = torch.max(outputs_flipped['pred_masks'].flip(-1), outputs['pred_masks'])
        else:
            outputs['pred_masks'] = (outputs_flipped['pred_masks'].flip(-1) + outputs['pred_masks']) / 2.0
    return outputs


def compute_predictions_ms(model, sample, gt_shape, ms=True, ms_gather='mean',
                           scales=None, sigmoid=False):
    if scales is None:
        scales = [1]
    frame, frame_seq = sample.decompose()
    org_shape = frame.shape[-2:]
    mask_list = []
    for scale in scales:
        size = [int(sz * scale) for sz in org_shape]
        size = [max(32, ((sz + 31) // 32) * 32) for sz in size]  # 向上取整到 32 的倍数

        if scale != 1:
            resized_frame, resized_seq = _resize_sequence(frame, frame_seq, size)
            model_input = ModelInputs(frame=resized_frame, frame_sequence=resized_seq)
        else:
            model_input = sample

        model_output = model(model_input)
        pred = interpolate(model_output['pred_masks'], size=gt_shape, mode="bilinear", align_corners=False)
        if sigmoid:
            pred = pred.sigmoid()
        mask_list.append(pred)
    if ms:
        if ms_gather == 'max':
            ms_pred = torch.max(torch.stack(mask_list, dim=0), dim=0)
            output_result = {'pred_masks': ms_pred}
        else:
            ms_pred = torch.mean(torch.stack(mask_list, dim=0), dim=0)
            output_result = {'pred_masks': ms_pred}
    else:
        output_result = {'pred_masks': mask_list[0]}
    return output_result


def augment_flip(samples, dim=-1):
    flipped = samples.copy()
    flipped.frame = flipped.frame.flip(dim)
    flipped.frame_sequence = flipped.frame_sequence.flip(dim)
    return flipped


@torch.no_grad()
def infer_on_dataset_train(model, data_loader, device, msc=False, flip=False, save_pred=False,
                           out_dir='./results/', msc_scales=None, sigmoid=True, use_wandb=False,
                           wandb_str='eval_train', viz_freq=100):
    if msc and msc_scales is not None:
        _scales = msc_scales
    elif msc and msc_scales is None:
        _scales = [0.75, 0.8, 0.9, 1, 1.1, 1.2, 1.3]
    else:
        _scales = [1]
    model.eval()
    i_iter = 0
    vid_iou_dict = {}
    inverse_norm_transform = T.InverseNormalizeTransforms()
    out_dir = os.path.join(out_dir, wandb_str)
    for samples, targets in tqdm(data_loader):
        i_iter = i_iter + 1
        video_names = [t['video_name'] for t in targets]
        frame_ids = [t['frame_ids'] for t in targets]
        center_frame_names = [t['center_frame'] for t in targets]
        center_frame_indexs = [frame_id.index(center_frame_name) for frame_id, center_frame_name in
                               zip(frame_ids, center_frame_names)]
        samples.to(device)
        center_gts = [target['masks'].squeeze().to(torch.uint8).numpy() for target in targets]
        for center_gt in center_gts:
            center_gt[center_gt > 0] = 1
        gt_shapes = [center_gt.shape for center_gt in center_gts]
        gt_shape = gt_shapes[0]

        outputs = compute_predictions_flip_ms(model, samples, gt_shape, ms=msc, ms_gather='mean',
                                              flip=flip, flip_gather='mean', scales=_scales, sigmoid=sigmoid)
        src_masks = outputs["pred_masks"]
        yc_logits = src_masks.squeeze(1).cpu().detach().numpy().copy()
        yc_binmask = yc_logits.copy()
        yc_binmask[yc_binmask > 0.5] = 1
        yc_binmask[yc_binmask <= 0.5] = 0
        outs = yc_binmask.astype(center_gt.dtype)
        ious = [eval_iou(center_gt, out.copy()) for center_gt, out in zip(center_gts, outs)]

        for video_name, center_frame_name, iou in zip(video_names, center_frame_names, ious):
            if video_name not in vid_iou_dict:
                vid_iou_dict[video_name] = {}
            vid_iou_dict[video_name][center_frame_name] = iou
        if use_wandb and is_main_process():
            import wandb
            wandb_dict = {}
            if i_iter % viz_freq == 0:
                viz_img = get_viz_img(samples.frame, targets, outputs, inverse_norm_transform)
                wandb_dict[wandb_str] = wandb.Image(viz_img)
                wandb.log(wandb_dict)

    csv_file_name = f'{wandb_str}_results.csv'
    video_mean_iou = save_video_iou_csv(out_dir, vid_iou_dict, csv_file_name, logger=logger)
    return video_mean_iou


@torch.no_grad()
def infer_on_dataset_val(model, data_loader, device, msc=False, flip=False, save_pred=False,
                         out_dir='./results/', sigmoid=True, msc_scales=None, use_wandb=False,
                         wandb_str='eval_val', viz_freq=100):
    if msc and msc_scales is not None:
        _scales = msc_scales
    elif msc and msc_scales is None:
        _scales = [0.75, 0.8, 0.9, 1, 1.1, 1.2, 1.3]
    else:
        _scales = [1]
    model.eval()
    i_iter = 0
    vid_iou_dict = {}
    inverse_norm_transform = T.InverseNormalizeTransforms()
    for samples, targets in tqdm(data_loader):
        i_iter = i_iter + 1
        video_names = [t['video_name'] for t in targets]
        frame_ids = [t['frame_ids'] for t in targets]
        center_frame_names = [t['center_frame'] for t in targets]
        center_frame_indexs = [frame_id.index(center_frame_name) for frame_id, center_frame_name in
                               zip(frame_ids, center_frame_names)]
        samples.to(device)
        targets = [{k: v.to(device) if k in ['masks'] else v for k, v in target.items()} for target in targets]
        center_gt_paths = [target['mask_paths'][center_frame_index] for target, center_frame_index in
                           zip(targets, center_frame_indexs)]
        center_gts = [cv2.imread(center_gt_path, cv2.IMREAD_GRAYSCALE) for center_gt_path in center_gt_paths]
        for center_gt in center_gts:
            center_gt[center_gt > 0] = 1
        gt_shapes = [center_gt.shape for center_gt in center_gts]
        gt_shape = gt_shapes[0]

        outputs = compute_predictions_flip_ms(model, samples, gt_shape, ms=msc, ms_gather='mean',
                                              flip=flip, flip_gather='mean', scales=_scales, sigmoid=sigmoid)
        src_masks = outputs["pred_masks"]
        yc_logits = src_masks.squeeze(1).cpu().detach().numpy().copy()
        yc_binmask = yc_logits.copy()
        yc_binmask[yc_binmask > 0.5] = 1
        yc_binmask[yc_binmask <= 0.5] = 0
        outs = yc_binmask.astype(center_gt.dtype)
        ious = [eval_iou(center_gt, out.copy()) for center_gt, out in zip(center_gts, outs)]

        for video_name, center_frame_name, iou in zip(video_names, center_frame_names, ious):
            if video_name not in vid_iou_dict:
                vid_iou_dict[video_name] = {}
            vid_iou_dict[video_name][center_frame_name] = iou
        if use_wandb and is_main_process():
            import wandb
            wandb_dict = {}
            if i_iter % viz_freq == 0:
                viz_img = get_viz_img(samples.frame, targets, outputs, inverse_norm_transform)
                wandb_dict[wandb_str] = wandb.Image(viz_img)
                wandb.log(wandb_dict)

    csv_file_name = f'{wandb_str}_results.csv'
    video_mean_iou = save_video_iou_csv(out_dir, vid_iou_dict, csv_file_name, logger=logger)
    return video_mean_iou


@torch.no_grad()
def infer_on_dataset_test(args, model, data_loader, device,
                          iou_metric, nIoU_metric, PD0_FA0,
                          msc=False, flip=False,
                          save_pred=False,
                          sigmoid=True,
                          out_dir='./results/', msc_scales=None):
    if msc and msc_scales is not None:
        _scales = msc_scales
    elif msc and msc_scales is None:
        _scales = [0.8, 0.9, 1, 1.1, 1.2, 1.3]
    else:
        _scales = [1]
    model.eval()
    i_iter = 0
    vid_iou_dict = {}
    pr_metric = PRMetric(1, _CURVE_BINS)
    roc_metric = ROCMetric(1, _CURVE_BINS)
    target_roc_metric = TargetROCMetric(_CURVE_BINS, match_distance=3.0)

    for samples, targets in tqdm(data_loader):
        i_iter = i_iter + 1
        video_names = [t['video_name'] for t in targets]
        frame_ids = [t['frame_ids'] for t in targets]
        center_frame_names = [t['center_frame'] for t in targets]
        center_frame_indexs = [frame_id.index(center_frame_name) for frame_id, center_frame_name in
                               zip(frame_ids, center_frame_names)]
        samples.to(device)
        targets = [{k: v.to(device) if k in ['masks'] else v for k, v in target.items()} for target in targets]
        center_gt_paths = [target['mask_paths'][center_frame_index] for target, center_frame_index in
                           zip(targets, center_frame_indexs)]
        center_gts = [cv2.imread(center_gt_path, cv2.IMREAD_GRAYSCALE) for center_gt_path in center_gt_paths]
        for center_gt in center_gts:
            center_gt[center_gt > 0] = 1
        gt_shapes = [center_gt.shape for center_gt in center_gts]
        gt_shape = gt_shapes[0]

        outputs = compute_predictions_flip_ms(model, samples, gt_shape, ms=msc, ms_gather='mean',
                                              flip=flip, flip_gather='mean', scales=_scales, sigmoid=sigmoid)
        src_masks = outputs["pred_masks"]
        yc_logits = src_masks.squeeze(1).cpu().detach().numpy().copy()
        yc_binmask = yc_logits.copy()
        yc_binmask[yc_binmask > 0.5] = 1
        yc_binmask[yc_binmask <= 0.5] = 0
        outs = yc_binmask.astype(center_gt.dtype)
        ious = [eval_iou(center_gt, out.copy()) for center_gt, out in zip(center_gts, outs)]
        for pred, center_gt in zip(yc_logits, center_gts):
            pr_metric.update(pred, center_gt)
            roc_metric.update(pred, center_gt)
            target_roc_metric.update(pred, center_gt)

        for video_name, center_frame_name, iou in zip(video_names, center_frame_names, ious):
            if video_name not in vid_iou_dict:
                vid_iou_dict[video_name] = {}
            vid_iou_dict[video_name][center_frame_name] = iou

        if save_pred:
            save_pred_images(out_dir, video_names, center_frame_names, outs, yc_logits)

        for center_gt, out in zip(center_gts, outs):
            iou_metric.update(out, center_gt)
            nIoU_metric.update(out, center_gt)
            PD0_FA0.update(out, center_gt)
    if not vid_iou_dict:
        raise ValueError(
            "infer_on_dataset_test did not process any samples. "
            "Check dataset_config, sequence_names, val_size, and data_loader output."
        )
    _, iou, temp_iou = iou_metric.get()
    niou = nIoU_metric.get()
    fa0, pd0 = PD0_FA0.get()
    save_test_csv(out_dir, fa0, pd0, iou, niou)

    csv_file_name = 'results.csv'
    save_video_iou_csv(out_dir, vid_iou_dict, csv_file_name, logger=logger)
    precision, recall, thresholds = pr_metric.get()
    _save_curve_csv(
        out_dir,
        'pr_curve.csv',
        ['threshold', 'precision', 'recall'],
        [(float(t), float(p), float(r)) for t, p, r in zip(thresholds, precision, recall)],
    )
    tpr, fpr, _, _ = roc_metric.get()
    _save_curve_csv(
        out_dir,
        'roc_curve.csv',
        ['threshold', 'tpr', 'fpr'],
        [(float(t), float(tp), float(fp)) for t, tp, fp in zip(thresholds, tpr, fpr)],
    )
    target_pd, target_fa, target_thresholds = target_roc_metric.get()
    _save_curve_csv(
        out_dir,
        'target_roc_curve.csv',
        ['threshold', 'pd', 'fa_pixel_ratio'],
        [(float(t), float(pd), float(fa))
         for t, pd, fa in zip(target_thresholds, target_pd, target_fa)],
    )
    return iou, niou, fa0, pd0
