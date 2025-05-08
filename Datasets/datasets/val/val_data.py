"""
Datasets dataloader for inference.
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
import yaml
import Datasets.datasets.transforms as T
from torch.utils.data import Dataset

from Datasets.datasets.utils import find_image

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ValDataset(Dataset):
    def __init__(self, name_dataset='TSIRMT', num_frames=6, val_size=320, sequence_names=None, max_sc=None,path_config=None):
        super(ValDataset, self).__init__()

        assert val_size % 32 == 0

        if path_config is None:
            if os.name == 'nt':
                path_config = 'Datasets/dataset_config_win.yaml'
            elif os.name == 'posix':
                path_config = 'Datasets/dataset_config_wsl.yaml'

        # 读取 YAML 文件
        with open(path_config, "r", encoding="utf-8") as f:
            Datasets_Config = yaml.safe_load(f)

        dataset_config = Datasets_Config[name_dataset]

        self.num_frames = num_frames
        self.split = 'val'
        self.im_size = val_size
        self._transforms = make_validation_transforms(min_size=val_size, max_sc=max_sc)

        self.val_seqs_file = dataset_config['val_seqs_file']
        self.img_path = dataset_config['img_path']
        self.mask_path = dataset_config['mask_path']

        self.frames_info = {
            name_dataset: {},
        }
        self.img_ids = []
        logger.debug(f'loading {name_dataset} val seqs...')
        if sequence_names is None or len(sequence_names) == 0:
            with open(self.val_seqs_file, 'r') as f:
                video_names = f.readlines()
                video_names = [name.strip() for name in video_names]
        else:
            video_names = sequence_names
        logger.debug('dataset-val num of videos: {}'.format(len(video_names)))
        extensions = ['png', 'jpg', 'bmp']
        for video_name in video_names:
            frames = []
            for ext in extensions:
                frames.extend(glob.glob(os.path.join(self.mask_path, video_name, f'*.{ext}')))
            frames = sorted(frames)
        # for video_name in video_names:
        #     frames = sorted(glob.glob(os.path.join(self.mask_path, video_name, '*.png')))
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
            # img_path = os.path.join(self.img_path, video_name, frame_name + '.png')
            # gt_path = os.path.join(self.mask_path, video_name, frame_name + '.png')
            img_i = Image.open(img_path).convert('RGB')
            img.append(img_i)
            # gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            # gt[gt > 0] = 255
            # masks.append(torch.Tensor(np.expand_dims(np.asarray(gt.copy()), axis=0)))
            mask_paths.append(gt_path)

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


def make_validation_transforms(min_size=360, max_sc=None):
    if max_sc is None:
        max_sc = 1.8
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return T.Compose([
        T.RandomResize([min_size], max_size=int(max_sc * min_size), least_factor=32),
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


    dataset = ValDataset(val_size=320)
    a = dataset[0]
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
