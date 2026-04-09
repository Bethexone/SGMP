# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2025/3/29 10:09
from Datasets.datasets.train.train_data import TrainDataset
from Datasets.datasets.utils import collate_fn
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from typing import Union, List, Optional


def show_image_mask_pairs(
        frames: List[torch.Tensor],
        masks: List[torch.Tensor],
        masked_frames: List[torch.Tensor] = None,
        denormalize: bool = False,
        figsize: tuple = (12, 6),
        titles: Optional[List[str]] = None,
):
    """
    显示成对的图像和掩码（左图原图，右图掩码）。

    参数:
        frames: 原图列表，每个张量形状为 (C, H, W)
        masks: 掩码列表，每个张量形状为 (1, H, W) 或 (H, W)
        nrow: 每行显示的图片对数
        denormalize_frames: 是否对原图反归一化
        frame_mean: 原图归一化的均值（如 [0.485, 0.456, 0.406]）
        frame_std: 原图归一化的标准差（如 [0.229, 0.224, 0.225]）
        figsize: 图像显示大小
        titles: 每对图像的标题列表（可选）
    """

    assert len(frames) == len(masks), "输入列表长度必须一致"

    # 将输入统一为列表形式并转移到CPU
    frame_list = [frame.detach().cpu() for frame in frames]
    mask_list = [mask.detach().cpu() for mask in masks]
    if masked_frames is not None:
        masked_list = [masked.detach().cpu() for masked in masked_frames]

    # 反归一化原图（如果需要）
    if denormalize:
        frame_list = [_denormalize_tensor(img) for img in frame_list]

    if masked_frames is None:
        masked_list = [x * y for x, y in zip(frame_list, mask_list)]

    # 计算行数和列数
    n = len(frame_list)
    ncols = 3  # 每组显示3张（原图 + 掩码 + 原图×掩码）
    nrows = n  # 根据nrow计算总行数

    # 创建子图
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)  # 确保axes是二维的

    # 遍历所有图像组
    for row in range(nrows):
        frame = frame_list[row]
        mask = mask_list[row]
        masked = masked_list[row]

        # 显示原图
        ax_frame = axes[row, 0]
        ax_frame.imshow(frame.permute(1, 2, 0).clip(0, 1))
        ax_frame.axis('off')
        if titles and row < len(titles):
            ax_frame.set_title(f"Frame {row + 1}")

        # 显示掩码
        ax_mask = axes[row, 1]
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        ax_mask.imshow(mask, cmap='gray')
        ax_mask.axis('off')
        if titles and row < len(titles):
            ax_mask.set_title(f"Mask {row + 1}")

        # 显示原图×掩码
        ax_masked = axes[row, 2]
        ax_masked.imshow(masked.permute(1, 2, 0).clip(0, 1))
        ax_masked.axis('off')
        if titles and row < len(titles):
            ax_masked.set_title(f"Masked {row + 1}")

    plt.tight_layout()
    plt.show()


def _denormalize_tensor(
        tensor: torch.Tensor,
        mean: List[float] = None,
        std: List[float] = None,
) -> torch.Tensor:
    """反归一化张量（支持单张或批次）"""
    if mean == None or std == None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    tensor = tensor.clone()
    if tensor.dim() == 3:  # (C, H, W)
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
    elif tensor.dim() == 4:  # (B, C, H, W)
        for t in tensor:
            for c, m, s in zip(t, mean, std):
                c.mul_(s).add_(m)
    return tensor.clamp_(0, 1)  # 限制到 [0, 1] 范围


if __name__ == '__main__':
    train_dataset = TrainDataset(name_dataset='TSIRMT_tiny',num_frames=6, train_size=320)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    for batch in train_dataloader:
        samples, target = batch
        frame, frame_sequences = samples.frame, samples.frame_sequence
        masks = [t['masks'] for t in target]
        frame_ls = [x for x in frame]
        masked_frames = [frame * mask for frame, mask in zip(frame_ls, masks)]
        show_image_mask_pairs(frame_ls, masks, denormalize=True)
