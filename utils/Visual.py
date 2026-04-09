import os

from utils.misc import is_main_process

import cv2
import numpy as np
import pandas as pd


def ensure_dir(path):
    if path and is_main_process() and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_video_iou_csv(out_dir, vid_iou_dict, csv_file_name, logger=None):
    ensure_dir(out_dir)
    video_names = []
    video_ious = []
    for k, v in vid_iou_dict.items():
        vid_iou = np.mean(list(v.values()))
        video_names.append(k)
        video_ious.append('%0.5f' % vid_iou)
        if logger is not None:
            logger.debug('video_name:%s iou:%0.5f' % (k, vid_iou))
    video_mean_iou = np.mean([np.mean(list(v.values())) for _, v in vid_iou_dict.items()])
    video_names.append('Video Mean')
    video_ious.append('%0.5f' % video_mean_iou)
    df = pd.DataFrame({"video_name": video_names, "video_iou": video_ious})
    csv_path = os.path.join(out_dir, csv_file_name)
    if is_main_process():
        df.to_csv(csv_path, index=False)
        if logger is not None:
            logger.debug('Videos Mean IOU: %0.5f' % video_mean_iou)
            logger.debug('Saved CSV: %s' % csv_path)
    return video_mean_iou


def save_test_csv(out_dir, fa0, pd0, iou, niou, logger=None):
    ensure_dir(out_dir)
    df = pd.DataFrame({
        'FA0': [fa0],
        'PD0': [pd0],
        'IOU': [iou],
        'nIOU': [niou]
    })
    csv_path = os.path.join(out_dir, "PD_FA.csv")
    if is_main_process():
        df.to_csv(csv_path, index=False)
        if logger is not None:
            logger.debug('Saved CSV: %s' % csv_path)


def save_pred_images(out_dir, video_names, center_frame_names, outs, yc_logits, logger=None):
    ensure_dir(out_dir)
    for video_name, out, center_frame_name, yc_logit in zip(video_names, outs, center_frame_names, yc_logits):
        logits_out_dir = os.path.join(out_dir, 'logits', video_name)
        ensure_dir(logits_out_dir)
        logits_path = os.path.join(logits_out_dir, '%s.png' % center_frame_name)
        if is_main_process():
            cv2.imwrite(logits_path, (yc_logit.astype(np.float32) * 255).astype(np.uint8))

        bm_out_dir = os.path.join(out_dir, 'bin_mask', video_name)
        ensure_dir(bm_out_dir)
        bm_path = os.path.join(bm_out_dir, '%s.png' % center_frame_name)
        if is_main_process():
            cv2.imwrite(bm_path, (out.astype(np.float32) * 255).astype(np.uint8))
            if logger is not None:
                logger.debug('Saved logits: %s' % logits_path)
                logger.debug('Saved bin_mask: %s' % bm_path)
