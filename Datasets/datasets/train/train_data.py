"""
Dataloader with train data.
"""
import math
import torch
import torch.utils.data
import os
from PIL import Image
import cv2
import numpy as np
import glob
import logging

from torch.utils.data import Dataset
import Datasets.datasets.transforms as T
import socket
import yaml
from Datasets.datasets.utils import find_image
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger("PIL").setLevel(logging.INFO)


# name_dataset = 'TSIRMT'
# dataset_config = Datasets_Config[name_dataset]


# # 获取 TSIRMT 配置
# tsirmt_config = config["TSIRMT"]
# train_seqs_file = tsirmt_config["train_seqs_file"]
# val_seqs_file = tsirmt_config["val_seqs_file"]
# img_path = tsirmt_config["img_path"]
# mask_path = tsirmt_config["mask_path"]

# print(train_seqs_file, val_seqs_file, img_path, mask_path)


class TrainDataset(Dataset):
    def __init__(self, name_dataset='TSIRMT', num_frames=6, train_size=320, path_config=None):
        super(TrainDataset, self).__init__()
        assert train_size % 32 == 0

        if path_config is None:
            if os.name == 'nt':
                path_config = 'Datasets/dataset_config_win.yaml'
            elif os.name == 'posix':
                hostname = socket.gethostname()  # 最简单可靠的方法
                if hostname == 'lq':
                    path_config = 'Datasets/dataset_config_lq.yaml'
                else:
                    path_config = 'Datasets/dataset_config_wsl.yaml'

        # 读取 YAML 文件
        with open(path_config, "r", encoding="utf-8") as f:
            Datasets_Config = yaml.safe_load(f)

        dataset_config = Datasets_Config[name_dataset]

        self.num_frames = num_frames
        self.split = 'train'
        self._transforms = make_train_transform(train_size=train_size)

        self.train_seqs_file = dataset_config['train_seqs_file']
        self.img_path = dataset_config['img_path']
        self.mask_path = dataset_config['mask_path']
        self.frames_info = {
            name_dataset: {}
        }
        self.img_ids = []
        logger.debug(f'loading {name_dataset} dataset train seqs...')
        with open(self.train_seqs_file, 'r') as f:
            video_names = f.readlines()
            video_names = [name.strip() for name in video_names]
            logger.debug(f'{name_dataset} dataset-train num of videos: {len(video_names)}')
            extensions = ['png', 'jpg', 'bmp']
            for video_name in video_names:
                frames = []
                for ext in extensions:
                    frames.extend(glob.glob(os.path.join(self.mask_path, video_name, f'*.{ext}')))
                frames = sorted(frames)
                # frames = sorted(glob.glob(os.path.join(self.mask_path, video_name, '*.png')))
                # logger.debug(f"frame_path:{os.path.splitext(os.path.basename(frames[0]))[0]}")
                # self.frames_info[name_dataset][video_name] = [frame_path.split('\\')[-1][:-4] for frame_path in frames]
                self.frames_info[name_dataset][video_name] = [os.path.splitext(os.path.basename(frame_path))[0] for
                                                              frame_path in frames]
                self.img_ids.extend([(name_dataset, video_name, frame_index) for frame_index in range(len(frames))])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_ids_i = self.img_ids[idx]
        dataset, video_name, frame_index = img_ids_i
        vid_len = len(self.frames_info[dataset][video_name])
        center_frame_name = self.frames_info[dataset][video_name][frame_index]

        # 超出循环时回绕
        frame_indices = [(x + vid_len) % vid_len for x in range(frame_index - math.floor(float(self.num_frames) / 2),
                                                                frame_index + math.ceil(float(self.num_frames) / 2), 1)]

        assert len(frame_indices) == self.num_frames
        frame_ids = []
        img = []
        masks = []
        mask_paths = []
        for frame_id in frame_indices:
            frame_name = self.frames_info[dataset][video_name][frame_id]
            frame_ids.append(frame_name)

            img_path = find_image(os.path.join(self.img_path, video_name), frame_name)
            gt_path = find_image(os.path.join(self.mask_path, video_name), frame_name)
            # logger.debug(f'img_path:{img_path},gt_path:{gt_path}')
            # import ipdb;ipdb.set_trace()

            img_i = Image.open(img_path).convert('RGB')
            img.append(img_i)
            # gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            # gt[gt > 0] = 255
            # masks.append(torch.Tensor(np.expand_dims(np.asarray(gt.copy()), axis=0)))
            mask_paths.append(gt_path)
        # import ipdb;ipdb.set_trace()
        # masks = torch.cat(masks, dim=0)
        center_frame_index = frame_ids.index(center_frame_name)
        gt = cv2.imread(mask_paths[center_frame_index], cv2.IMREAD_GRAYSCALE)
        gt[gt > 0] = 255
        masks.append(torch.Tensor(np.expand_dims(np.asarray(gt.copy()), axis=0)))
        masks = torch.cat(masks, dim=0)

        target = {'dataset': dataset, 'video_name': video_name, 'center_frame': center_frame_name,
                  'frame_ids': frame_ids, 'masks': masks, 'vid_len': vid_len, 'mask_paths': mask_paths}

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        frame = img[center_frame_index]
        frame_sequence = torch.stack(img, dim=1)
        return (frame, frame_sequence), target


def make_train_transform(train_size=None):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # T.Normalize([0.5], [0.5])
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResize(scales, max_size=800),
        T.PhotometricDistort(),
        T.Compose([
            T.RandomResize([500, 600, 700]),
            T.RandomSizeCrop(473, 750),
            T.RandomResize([(train_size, train_size)], max_size=int(1.8 * train_size), least_factor=32),
        ]),
        normalize,
    ])


if __name__ == '__main__':
    from matplotlib import pyplot as plt


    def tensor_show(ax, frame, cmap=None):
        frame = frame.float()

        # 调整维度为 [H, W, C] 或 [H, W]
        if frame.dim() == 3:
            frame_np = frame.permute(1, 2, 0).cpu().numpy()
        else:
            frame_np = frame.cpu().numpy()

        # 显示图像
        ax.imshow(frame_np, cmap=cmap)
        ax.axis('off')  # 关闭坐标轴


    name_dataset = 'TSIRMT_tiny'
    dataset = TrainDataset(name_dataset=name_dataset, train_size=320)
    a = dataset[10]
    (frame, frame_sequence), target = a
    print(frame.shape, frame_sequence.shape)
    # 将 Tensor 转为 PIL 格式
    # 创建两个并排的子图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # 显示 RGB 图像
    tensor_show(axes[0], frame)
    # 显示 Mask（单通道用灰度显示）
    tensor_show(axes[1], target['masks'], cmap='gray')
    plt.show()
