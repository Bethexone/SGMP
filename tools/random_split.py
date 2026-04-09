# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2025/3/29 19:37
import torch
from os import path
import os
import yaml

if __name__ == '__main__':

    path_config = 'dataset_config.yaml'
    name_dataset = 'TSIRMT_tiny'

    # 读取 YAML 文件
    with open(path_config, "r", encoding="utf-8") as f:
        Datasets_Config = yaml.safe_load(f)

    dataset_config = Datasets_Config[name_dataset]
    dir_path = dataset_config['img_path']
    # 假设 images 是一个包含 21 个图像的列表或张量
    # images = list(range(21))  # 代表 0-20 共 21 个图像序列
    folders = os.listdir(dir_path)
    # 定义划分比例（例如 70% 训练，15% 验证，15% 测试）
    train_size = int(0.7 * len(folders))
    val_size = len(folders) - train_size
    test_size = len(folders) - train_size - val_size

    # 使用 random_split 划分数据集
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        folders, [train_size, val_size, test_size]
    )
    train_list = [folders[i] for i in train_dataset.indices]
    val_list = [folders[i] for i in val_dataset.indices]
    test_list = [folders[i] for i in test_dataset.indices]

    train_seqs_file = dataset_config['train_seqs_file']
    with open(train_seqs_file, "w", encoding="utf-8") as f:
        for x in train_list:
            f.write(x)
            f.write('\n')

    val_seqs_file = dataset_config['val_seqs_file']
    with open(val_seqs_file, "w", encoding="utf-8") as f:
        for x in val_list:
            f.write(x)
            f.write('\n')

    # print("训练集：", train_list)
    # print("验证集：", val_list)
    # print("测试集：", test_list)
