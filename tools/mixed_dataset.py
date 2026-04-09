# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2025/4/2 20:08
import yaml
from pathlib import Path
import shutil
from tqdm import tqdm


def make_mixed_dataset(path_config, Mixed_Dataset_path='D:\Project\dataset\InfraRed\Mixed_Dataset'):
    Mixed_Dataset_path = Path(Mixed_Dataset_path)
    Mixed_Dataset_path.mkdir(exist_ok=True)
    dst_images = Mixed_Dataset_path / 'images'
    dst_masks = Mixed_Dataset_path / 'masks'
    dst_images.mkdir(exist_ok=True)
    dst_masks.mkdir(exist_ok=True)

    with open(path_config, "r", encoding="utf-8") as f:
        Datasets_Config = yaml.safe_load(f)
    for name_dataset, config in Datasets_Config.items():
        img_path = Path(config['img_path'])
        mask_path = Path(config['mask_path'])
        src_videos = img_path.iterdir()
        for src_video in tqdm(src_videos):
            name_video = src_video.stem
            new_name = f"{name_dataset}_{name_video}"
            shutil.copytree(src_video, dst_images / new_name, dirs_exist_ok=True)

            src_mask = mask_path / name_video
            shutil.copytree(src_mask, dst_masks / new_name, dirs_exist_ok=True)
        print(f'move {name_dataset} success !')
        # pass


if __name__ == '__main__':
    # 读取 YAML 文件
    path_config = 'Datasets/dataset_config_win.yaml'
    make_mixed_dataset(path_config)
