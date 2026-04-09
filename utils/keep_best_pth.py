# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2025/4/25 11:00
import os
import re


def keep_best_checkpoints(checkpoint_dir, top_k=5):
    files = os.listdir(checkpoint_dir)
    val_miou_files = []

    # 提取 val_miou 的值和对应文件名
    for f in files:
        if f.endswith(".pth"):
            match = re.search(r"val_miou_([0-9]+(?:\.[0-9]+)?)", f)
            if match:
                val_miou = float(match.group(1))
                val_miou_files.append((val_miou, f))

    # 按照 val_miou 从大到小排序，保留 top_k 个
    val_miou_files.sort(reverse=True)
    keep_files = set(f for _, f in val_miou_files[:top_k])

    # 保留 checkpoint_last.pth
    if "checkpoint_last.pth" in files:
        keep_files.add("checkpoint_last.pth")

    # 删除不在保留列表中的文件
    for f in files:
        if f.endswith(".pth") and f not in keep_files:
            os.remove(os.path.join(checkpoint_dir, f))
            print(f"已删除: {f}")

    print("保留的文件:")
    for f in keep_files:
        print(f)

if __name__ == '__main__':
    # 使用示例
    checkpoint_dir = '/home/mc/zhangwei/Project/experiment/hyperparam_experiment/result/TSIRMT'
    folders = [os.path.join(checkpoint_dir, name) for name in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, name))]
    for folder in folders:
        keep_best_checkpoints(folder)
    # for c
    # keep_best_checkpoints(checkpoint_dir)
