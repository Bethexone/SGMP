import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure
import cv2
import numpy as np


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


def eval_iou(annotation, segmentation):
    annotation = annotation.astype(np.bool_)
    segmentation = segmentation.astype(np.bool_)

    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1
    return np.sum((annotation & segmentation)) / np.sum((annotation | segmentation), dtype=np.float32)


class PD0_FA0():
    def __init__(self, nclass, thre):
        super(PD0_FA0, self).__init__()
        self.nclass = nclass
        self.thre = thre
        self.image_area_total = []
        self.image_area_match = []
        self.FA0 = 0
        self.PD0 = 0
        self.target = 0
        self.exlment_num = 0

    def update(self, preds, labels):

        predits = (preds > self.thre).astype('int64')
        labelss = labels.astype('int64')  # P

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match = []
        self.dismatch = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)
        label_PD = 0
        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):

                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)

                    del coord_image[m]
                    break
        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
        self.FA0 += np.sum(self.dismatch)
        self.PD0 += len(self.distance_match)
        self.exlment_num += predits.size

    def get(self):
        eps = 1e-12
        Final_FA0 = self.FA0 / max(self.exlment_num, eps)
        Final_PD0 = self.PD0 / max(self.target, eps)

        return Final_FA0, Final_PD0

    def reset(self):
        self.FA0 = 0
        self.PD0 = 0


class SigmoidMetric():
    def __init__(self):
        self.reset()
        self.IoU = []

    def update(self, pred, labels):
        correct, labeled = self.batch_pix_accuracy(pred, labels)
        iou = self.batch_intersection_union(pred, labels)

        self.total_correct += correct
        self.total_label += labeled
        self.IoU.append(iou)

    def get(self):
        """Gets the current evaluation result."""
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        temp_IoU = self.IoU
        iou = temp_IoU[-1]
        mIoU = sum(temp_IoU) / len(temp_IoU)
        return pixAcc, mIoU, iou

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

    def batch_pix_accuracy(self, output, target):
        assert output.shape == target.shape
        predict = (output > 0).astype('int64')  # P
        pixel_labeled = np.sum(target > 0)  # T
        pixel_correct = np.sum((predict == target) * (target > 0))  # TP
        assert pixel_correct <= pixel_labeled
        return pixel_correct, pixel_labeled

    def batch_intersection_union(self, output, target):
        target = target.astype(np.bool_)
        output = output.astype(np.bool_)

        if np.isclose(np.sum(target), 0) and np.isclose(np.sum(output), 0):
            return 1
        else:
            return np.sum((target & output)) / \
                np.sum((target | output), dtype=np.float32)


class SamplewiseSigmoidMetric():
    def __init__(self, nclass, score_thresh=0.5):
        self.nclass = nclass
        self.score_thresh = score_thresh
        self.reset()
        self.single_IoU = []

    def update(self, preds, labels):
        """Updates the internal evaluation result."""
        correct, labeled = self.batch_pix_accuracy(preds, labels)
        inter, union = self.batch_intersection_union(preds, labels)

        self.total_correct.append(correct)
        self.total_label.append(labeled)
        self.total_inter.append(inter)
        self.total_union.append(union)

    def get(self):
        """Gets the current evaluation result."""
        nIoU = np.sum(self.total_inter) / np.sum(self.total_union)
        return nIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = []
        self.total_union = []
        self.total_correct = []
        self.total_label = []

    def batch_pix_accuracy(self, output, target):
        assert output.shape == target.shape
        predict = (output > 0).astype('int64')  # P
        pixel_labeled = np.sum(target > 0)  # T
        pixel_correct = np.sum((predict == target) * (target > 0))  # TP
        assert pixel_correct <= pixel_labeled
        return pixel_correct, pixel_labeled

    def batch_intersection_union(self, output, target):
        mini = 1
        maxi = 1  # nclass
        nbins = 1  # nclass
        predict = (output > 0).astype('int64')
        target = target.astype('int64')  # T
        intersection = predict * (predict == target)  # TP

        # areas of intersection and union
        area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
        area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
        area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
        area_union = area_pred + area_lab - area_inter
        assert (area_inter <= area_union).all()
        return area_inter, area_union


class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass, bins):
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins + 1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, '-th, score_thresh: ', score_thresh)
            i_tp, i_pos, i_fp, i_neg, i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass, score_thresh)
            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg
            self.class_pos[iBin] += i_class_pos

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)

        recall = self.tp_arr / (self.pos_arr + 0.001)
        precision = self.tp_arr / (self.class_pos + 0.001)

        return tp_rates, fp_rates, recall, precision

    def reset(self):
        size = self.bins + 1
        self.tp_arr = np.zeros([size])
        self.pos_arr = np.zeros([size])
        self.fp_arr = np.zeros([size])
        self.neg_arr = np.zeros([size])
        self.class_pos = np.zeros([size])


class TargetROCMetric():
    """Computes target-level ROC (PD vs FA pixel ratio) matching PD0_FA0 definition."""

    def __init__(self, bins, match_distance=3.0):
        super(TargetROCMetric, self).__init__()
        self.bins = bins
        self.match_distance = float(match_distance)
        self.thresholds = np.linspace(0.0, 1.0, self.bins + 1)
        self.tp_arr = np.zeros(self.bins + 1, dtype=np.float64)
        self.fp_pixel_arr = np.zeros(self.bins + 1, dtype=np.float64)
        self.total_targets = 0
        self.total_pixels = 0

    def _match_by_centroid(self, gt_props, pred_props):
        if not pred_props:
            return 0
        pred_centroids = [np.array(prop.centroid) for prop in pred_props]
        pred_used = [False] * len(pred_centroids)
        tp = 0
        for gt in gt_props:
            gt_centroid = np.array(gt.centroid)
            matched_idx = None
            for idx, pred_centroid in enumerate(pred_centroids):
                if pred_used[idx]:
                    continue
                if np.linalg.norm(pred_centroid - gt_centroid) < self.match_distance:
                    matched_idx = idx
                    break
            if matched_idx is not None:
                pred_used[matched_idx] = True
                tp += 1
        return tp

    def update(self, preds, labels):
        labels = labels.astype('int64')
        gt_mask = labels > 0
        gt_image = measure.label(gt_mask, connectivity=2)
        gt_props = measure.regionprops(gt_image)
        self.total_targets += len(gt_props)
        self.total_pixels += preds.size

        for iBin, score_thresh in enumerate(self.thresholds):
            pred_mask = (preds > score_thresh)
            pred_image = measure.label(pred_mask, connectivity=2)
            pred_props = measure.regionprops(pred_image)
            tp = self._match_by_centroid(gt_props, pred_props)
            fp_pixels = np.logical_and(pred_mask, ~gt_mask).sum()
            self.tp_arr[iBin] += tp
            self.fp_pixel_arr[iBin] += fp_pixels

    def get(self):
        eps = 1e-12
        pd = self.tp_arr / max(self.total_targets, eps)
        fa = self.fp_pixel_arr / max(self.total_pixels, eps)
        return pd, fa, self.thresholds

    def reset(self):
        size = self.bins + 1
        self.thresholds = np.linspace(0.0, 1.0, size)
        self.tp_arr = np.zeros([size], dtype=np.float64)
        self.fp_pixel_arr = np.zeros([size], dtype=np.float64)
        self.total_targets = 0
        self.total_pixels = 0


class PRMetric():
    def __init__(self, nclass, bins):
        super(PRMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.thresholds = np.linspace(0.0, 1.0, self.bins + 1)
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)

    def update(self, preds, labels):
        for iBin, score_thresh in enumerate(self.thresholds):
            i_tp, i_pos, _, _, i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass, score_thresh)
            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.class_pos[iBin] += i_class_pos

    def get(self):
        recall = self.tp_arr / (self.pos_arr + 0.001)
        precision = self.tp_arr / (self.class_pos + 0.001)
        return precision, recall, self.thresholds

    def reset(self):
        size = self.bins + 1
        self.thresholds = np.linspace(0.0, 1.0, size)
        self.tp_arr = np.zeros([size])
        self.pos_arr = np.zeros([size])
        self.class_pos = np.zeros([size])


class PD_FA():
    def __init__(self, nclass, bins):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins + 1)
        self.PD = np.zeros(self.bins + 1)
        self.target = np.zeros(self.bins + 1)
        self.exlment_num = 0

    def update(self, preds, labels):

        for iBin in range(self.bins + 1):
            score_thresh = (iBin * ((preds.max() - preds.min()) / self.bins)
                            + preds.min())
            predits = (preds > score_thresh).astype('int64')
            labelss = labels.astype('int64')  # P

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss, connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin] += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match = []
            self.dismatch = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)
            label_PD = 0
            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                connectivity = 8
                ret_label, thresholded_label, stats_label, centroids_label = cv2.connectedComponentsWithStats(
                    (labelss).astype('int8'), connectivity=connectivity)
                num_label_region = (thresholded_label == i + 1)
                label_bool = num_label_region.astype(bool)
                predits_bool = predits.astype(bool)
                intersection = np.logical_and(label_bool.reshape(-1), predits_bool.reshape(-1))
                if np.any(intersection):
                    temp_PD_num = 1
                else:
                    temp_PD_num = 0
                label_PD += temp_PD_num
            false_num_pic = predits - labelss  # 预测图-标签%
            false_num_pic_true = (false_num_pic > 0).astype('int64')
            self.FA[iBin] += false_num_pic_true.sum()
            self.PD[iBin] += label_PD
            self.exlment_num += predits.size

    def get(self):
        eps = 1e-12
        Final_FA = self.FA / max(self.exlment_num, eps)
        Final_PD = self.PD / np.maximum(self.target, eps)

        return Final_FA, Final_PD

    def reset(self):
        self.FA = np.zeros([self.bins + 1])
        self.PD = np.zeros([self.bins + 1])


class mIoU:

    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        iou = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        miou = iou.mean()
        return pixAcc, miou

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):
    output = torch.from_numpy(output).unsqueeze(0).unsqueeze(1)
    target = torch.from_numpy(target).unsqueeze(0).unsqueeze(1)
    predict = (output > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError('Unknown target dimension')

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos = tp + fp

    return tp, pos, fp, neg, class_pos


def batch_pix_accuracy(output, target):
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float()) * ((target > 0)).float()).sum()

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _ = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union
