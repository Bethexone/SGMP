# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2025/3/28 14:49
# 自定义 collate_fn
import os

import torch
from utils.model_inputs import ModelInputs


def find_image(base_path, name):
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        full_path = os.path.join(base_path, name + ext)
        if os.path.exists(full_path):
            return full_path
    return None  # 没找到就返回 None


def collate_fn(batch):
    images, labels = zip(*batch)  # 解包批次数据，分别获取图片和标签
    frame, frame_sequence = zip(*images)
    data = ModelInputs(
        frame=torch.stack(frame, dim=0),
        frame_sequence=torch.stack(frame_sequence, dim=0),
    )
    return data, labels


if __name__ == '__main__':
    from Datasets.datasets.train.train_data import TrainDataset
    from torch.utils.data import DataLoader

    dataset_train = TrainDataset(num_frames=6, train_size=320)
    data_loader_train = DataLoader(dataset_train, batch_size=4, collate_fn=collate_fn)
    for batch in data_loader_train:
        pass
