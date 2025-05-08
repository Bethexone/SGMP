import csv
import logging
import os
import time
import wandb
from utils.misc import interpolate
from utils.wandb_utils import get_viz_img
import cv2
import numpy as np
import scipy.io as scio
import torch
import pandas as pd
from Datasets.datasets.val.val_data import ValDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from Datasets.datasets import transforms as T

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def create_eval_save_dir_name_from_args(_args):
    _dir_name = 'infer_%s%3d%slpp_mode%d_sc%0.2f_%d' % (
        _args.dataset,
        _args.val_size,
        'msc' if _args.msc else 'ssc',
        _args.lprop_mode,
        _args.lprop_scale,
        int(time.time()))
    return _dir_name


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_predictions_flip_ms(model, samples, gt_shape, ms=True, ms_gather='mean', flip=True,
                                flip_gather='mean', scales=None, sigmoid=True):
    # import ipdb;ipdb.set_trace()  # 设置断点
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
        if scale != 1:
            frame = interpolate(frame, size=size, mode="bilinear", align_corners=False)
            frame_seq = [interpolate(x, size=size, mode="bilinear", align_corners=False) for x in frame_seq]

        model_output = model(sample)
        # import ipdb;ipdb.set_trace()
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
    samples.frame = samples.frame.flip(dim)
    samples.frame_sequence = samples.frame_sequence.flip(dim)
    return samples


def eval_iou(annotation, segmentation):
    annotation = annotation.astype(np.bool_)
    segmentation = segmentation.astype(np.bool_)

    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
            np.sum((annotation | segmentation), dtype=np.float32)


@torch.no_grad()
def infer_on_dataset_train(model, data_loader, device, msc=False, flip=False, save_pred=False,
                           out_dir='./results/', msc_scales=None, sigmoid=True,use_wandb=False, wandb_str='eval_train',
                           viz_freq=100):
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
        # targets = [{k: v.to(device) if k in ['masks'] else v for k, v in target.items()} for target in targets]
        # center_gt_paths = [target['mask_paths'][center_frame_index] for target, center_frame_index in
        #                    zip(targets, center_frame_indexs)]
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
        if use_wandb:
            wandb_dict = {}
            if i_iter % viz_freq == 0:
                viz_img = get_viz_img(samples.frame, targets, outputs, inverse_norm_transform)
                wandb_dict[wandb_str] = wandb.Image(viz_img)
                wandb.log(wandb_dict)

    video_mean_iou = np.mean(
        [np.mean(list(vid_iou_f.values())) for _, vid_iou_f in vid_iou_dict.items()])  # 算所有序列的mean_IOU
    #  Write the results to CSV
    csv_file_name = f'{wandb_str}_results.csv'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    video_names = []
    video_ious = []
    for k, v in vid_iou_dict.items():
        vid_iou = np.mean(list(v.values()))
        video_names.append(k)
        video_ious.append('%0.5f' % vid_iou)
        logger.debug('video_name:%s iou:%0.5f' % (k, vid_iou))
    video_names.append('Video Mean')
    video_ious.append('%0.5f' % video_mean_iou)
    df = pd.DataFrame({"video_name": video_names, "video_iou": video_ious})
    # 保存为 CSV 文件
    df.to_csv(os.path.join(out_dir, csv_file_name), index=False)
    # with open(os.path.join(out_dir, csv_file_name), 'w') as f:
    #     cf = csv.writer(f)
    #     cf.writerow(video_names)
    #     cf.writerow(video_ious)
    logger.debug('Videos Mean IOU: %0.5f' % video_mean_iou)
    return video_mean_iou


@torch.no_grad()
def infer_on_dataset_val(model, data_loader, device, msc=False, flip=False, save_pred=False,
                         out_dir='./results/', sigmoid=True,msc_scales=None, use_wandb=False, wandb_str='eval_val', viz_freq=100):
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
        if use_wandb:
            wandb_dict = {}
            if i_iter % viz_freq == 0:
                viz_img = get_viz_img(samples.frame, targets, outputs, inverse_norm_transform)
                wandb_dict[wandb_str] = wandb.Image(viz_img)
                wandb.log(wandb_dict)

    video_mean_iou = np.mean(
        [np.mean(list(vid_iou_f.values())) for _, vid_iou_f in vid_iou_dict.items()])  # 算所有序列的mean_IOU
    #  Write the results to CSV
    csv_file_name = f'{wandb_str}_results.csv'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    video_names = []
    video_ious = []
    for k, v in vid_iou_dict.items():
        vid_iou = np.mean(list(v.values()))
        video_names.append(k)
        video_ious.append('%0.5f' % vid_iou)
        logger.debug('video_name:%s iou:%0.5f' % (k, vid_iou))
    video_names.append('Video Mean')
    video_ious.append('%0.5f' % video_mean_iou)
    df = pd.DataFrame({"video_name": video_names, "video_iou": video_ious})
    # 保存为 CSV 文件
    df.to_csv(os.path.join(out_dir, csv_file_name), index=False)
    logger.debug('Videos Mean IOU: %0.5f' % video_mean_iou)
    return video_mean_iou


@torch.no_grad()
def infer_on_dataset_test(args, model, data_loader, device,
                          iou_metric, nIoU_metric, PD0_FA0,
                          msc=False, flip=False,
                          save_pred=False,
                          out_dir='./results/', msc_scales=None):
    if msc and msc_scales is not None:
        _scales = msc_scales
    elif msc and msc_scales is None:
        _scales = [0.75, 0.8, 0.9, 1, 1.1, 1.2, 1.3]
    else:
        _scales = [1]
    model.eval()
    i_iter = 0
    vid_iou_dict = {}

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
                                              flip=flip, flip_gather='mean', scales=_scales, sigmoid=False)
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

        ###############################
        if save_pred:
            for video_name, iou, out, center_frame_name, yc_logit in zip(video_names, ious, outs, center_frame_names,
                                                                         yc_logits):
                save_name = args.dataset + '_' + video_name
                logits_out_dir = os.path.join(out_dir, 'logits', video_name)
                if not os.path.exists(logits_out_dir):
                    os.makedirs(logits_out_dir)
                cv2.imwrite(os.path.join(logits_out_dir, '%s.png' % center_frame_name),
                            (yc_logit.astype(np.float32) * 255).astype(np.uint8))

                bm_out_dir = os.path.join(out_dir, 'bin_mask', video_name)
                if not os.path.exists(bm_out_dir):
                    os.makedirs(bm_out_dir)
                cv2.imwrite(os.path.join(bm_out_dir, '%s.png' % center_frame_name),
                            (out.astype(np.float32) * 255).astype(np.uint8))  # it is 0, 1

        for center_gt, out in zip(center_gts, outs):
            # import ipdb; ipdb.set_trace()
            iou_metric.update(out, center_gt)
            nIoU_metric.update(out, center_gt)
            PD0_FA0.update(out, center_gt)
    _, IoU, temp_iou = iou_metric.get()
    nIoU = nIoU_metric.get()
    FA0, PD0 = PD0_FA0.get()
    # 创建DataFrame
    df = pd.DataFrame({
        'FA0': [FA0],
        'PD0': [PD0],
        'IOU': [IoU],
        'nIOU': [nIoU]
    })
    # 保存CSV
    csv_path = f"{out_dir}/PD_FA.csv"
    df.to_csv(csv_path, index=False)

    video_mean_iou = np.mean(
        [np.mean(list(vid_iou_f.values())) for _, vid_iou_f in vid_iou_dict.items()])  # 算所有序列的mean_IOU
    #  Write the results to CSV
    csv_file_name = f'results.csv'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    video_names = []
    video_ious = []
    for k, v in vid_iou_dict.items():
        vid_iou = np.mean(list(v.values()))
        video_names.append(k)
        video_ious.append('%0.5f' % vid_iou)
        logger.debug('video_name:%s iou:%0.5f' % (k, vid_iou))
    video_names.append('Video Mean')
    video_ious.append('%0.5f' % video_mean_iou)
    df = pd.DataFrame({"video_name": video_names, "video_iou": video_ious})
    # 保存为 CSV 文件
    df.to_csv(os.path.join(out_dir, csv_file_name), index=False)
    logger.debug('Videos Mean IOU: %0.5f' % video_mean_iou)

    return IoU, nIoU, FA0, PD0


if __name__ == '__main__':
    from model.net import Net
    from Datasets.datasets.train.train_data import TrainDataset
    from Datasets.datasets.val.val_data import ValDataset
    from Datasets.datasets.utils import collate_fn

    num_tl = 6
    model = Net(num_tl)
    state_dict = torch.load('checkpoint/checkpoint_last.pth')['model']
    model.load_state_dict(state_dict, strict=False)
    batch_size = 16
    name_dataset = 'TSIRMT_tiny'
    dataset_train = TrainDataset(name_dataset=name_dataset, num_frames=num_tl, train_size=320)
    dataset_val = ValDataset(name_dataset=name_dataset, num_frames=num_tl, val_size=320)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, 4, drop_last=True)
    batch_sampler_val = torch.utils.data.BatchSampler(sampler_val, 4, drop_last=False)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_sampler=batch_sampler_val, collate_fn=collate_fn)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # infer_on_dataset_val(model, data_loader_val, device, epoch=1, msc=False, flip=True, save_pred=False,
    #                      out_dir='./results/', msc_scales=None,use_wandb=
