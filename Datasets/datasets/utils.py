# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2025/3/28 14:49
# 自定义 collate_fn
import copy
import os

import torch


def find_image(base_path, name):
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        full_path = os.path.join(base_path, name + ext)
        if os.path.exists(full_path):
            return full_path
    return None  # 没找到就返回 None


def collate_fn(batch):
    images, labels = zip(*batch)  # 解包批次数据，分别获取图片和标签
    data = NestedTensor(images)
    return data, labels


class NestedTensor(object):
    def __init__(self, images):
        frame, frame_sequence = zip(*images)
        self.frame = torch.stack(frame, dim=0)
        self.frame_sequence = torch.stack(frame_sequence, dim=0)

        self.num_tl = self.frame_sequence.shape[2]
        self.FrameW, self.FrameH = self.frame.shape[-2:]

    def decompose(self):
        return self.frame, self.frame_sequence

    def to(self, device):
        self.frame = self.frame.to(device)
        self.frame_sequence = self.frame_sequence.to(device)

    def copy(self, deep=True):
        """
        创建对象的拷贝
        Args:
            deep: 是否执行深拷贝 (默认True)
        Returns:
            NestedTensor: 新拷贝的对象
        """
        if deep:
            # 深拷贝所有张量和属性
            new_obj = object.__new__(NestedTensor)  # 创建空对象
            new_obj.frame = self.frame.clone()  # 张量需要显式clone
            new_obj.frame_sequence = self.frame_sequence.clone()
            new_obj.num_tl = copy.deepcopy(self.num_tl)
            new_obj.FrameW = copy.deepcopy(self.FrameW)
            new_obj.FrameH = copy.deepcopy(self.FrameH)
            return new_obj
        else:
            # 浅拷贝
            return copy.copy(self)


if __name__ == '__main__':
    from Datasets.datasets.train.train_data import TrainDataset
    from torch.utils.data import DataLoader

    dataset_train = TrainDataset(num_frames=6, train_size=320)
    data_loader_train = DataLoader(dataset_train, batch_size=4, collate_fn=collate_fn)
    for batch in data_loader_train:
        pass
