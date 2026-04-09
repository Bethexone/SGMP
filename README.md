# SGMP-UNet

PyTorch implementation of `SGMP-UNet: Spatial-Semantic Guided Motion-Aware UNet for Infrared Small Targets`.

[Project](https://github.com/Bethexone/SGMP.git) - [Paper](https://www.techrxiv.org/users/921880/articles/1296074-sgmp-spatial-semantic-guided-motion-aware-for-infrared-small-targets)

![Architecture](flow_chart.png)

## Overview

This repository contains the training and inference code for SGMP-UNet on infrared small target detection benchmarks.

Included in this repository:

- model code, training code, inference code, utilities
- split files such as `train_new.txt` and `val_new.txt`
- public configuration files under `configs/` and `Datasets/`

Not included in this repository:

- raw datasets
- model checkpoints
- training outputs and prediction results
- local environment files and machine-specific private configs

## Environment

The current codebase is maintained against Python 3.12 and PyTorch 2.6.0 with CUDA 12.4.

```bash
conda create -n sgmp python=3.12
conda activate sgmp
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

All commands below assume the current working directory is the repository root.

## Repository Layout

- `configs/`: base, training, and inference configs
- `Datasets/`: split files, dataset config templates, and dataloader code
- `model/`: SGMP-UNet network definition
- `train_script/`: training entrypoint
- `test_script/`: inference entrypoint
- `utils/`: metrics, logging, config loading, and output helpers

## Data Preparation

Supported dataset names in code:

- `IRDST`
- `NUDT-MIRSDT`
- `TSIRMT`
- `TSIRMT_tiny`

Download the raw datasets from the corresponding project or paper pages, then edit one dataset config file before running anything:

- `Datasets/dataset_config_win.yaml`
- `Datasets/dataset_config_wsl.yaml`
- `Datasets/dataset_config_ubuntu22.yaml`

Rules:

- keep `train_seqs_file` and `val_seqs_file` as the repo-relative split files already provided here
- change `img_path` and `mask_path` to your local dataset directories
- pass the selected config explicitly with `--dataset_config`

Example structure inside a dataset config:

```yaml
IRDST:
  train_seqs_file: Datasets/IRDST/train_new.txt
  val_seqs_file: Datasets/IRDST/val_new.txt
  img_path: /path/to/IRDST/images
  mask_path: /path/to/IRDST/masks
```

## Training

Default training config lives in `configs/train.yaml`. The default dataset in that file is `IRDST`. To train from a fresh clone, pass your dataset config explicitly:

```bash
python -m train_script.train --dataset_config Datasets/dataset_config_win.yaml
```

Train on another dataset by overriding `dataset_name`:

```bash
python -m train_script.train --dataset_config Datasets/dataset_config_win.yaml --dataset_name NUDT-MIRSDT
```

Optional:

- enable Weights & Biases with `--use_wandb`
- override `--output_root`, `--batch_size`, `--epochs`, or other fields from `configs/train.yaml`

Training outputs are written to:

```text
outputs/<dataset_name>/<experiment_name>/train
```

Typical files in the training output directory:

- `train.log`
- `exp_config.yaml`
- `checkpoint_last.pth`
- metric-tagged checkpoints such as `checkpoint_0009_val_miou_0.727.pth`

## Inference

The following examples use repository-relative checkpoint paths under `result/`. The repository does not ship these weight files. Download the checkpoints from Baidu Netdisk and place them at the exact relative paths shown below before running inference.

Expected local checkpoint paths:

- `result/checkpoint_IRDST_val_miou_0.612/checkpoint_best_val_miou_0.612.pth`
- `result/checkpoint_NUDT-MIRSDT_val_miou_0.727/checkpoint_best_val_miou_0.727.pth`
- `result/checkpoint_TSIRMT_val_miou_0.673/checkpoint_best_val_miou_0.673.pth`

Inference entrypoint:

```bash
python -m test_script.inference --dataset_config Datasets/dataset_config_win.yaml --dataset <DATASET_NAME> --model_path <CHECKPOINT_PATH>
```

Example on `IRDST`:

```bash
python -m test_script.inference --dataset_config Datasets/dataset_config_win.yaml --dataset IRDST --model_path result/checkpoint_IRDST_val_miou_0.612/checkpoint_best_val_miou_0.612.pth
```

Example on `NUDT-MIRSDT`:

```bash
python -m test_script.inference --dataset_config Datasets/dataset_config_win.yaml --dataset NUDT-MIRSDT --model_path result/checkpoint_NUDT-MIRSDT_val_miou_0.727/checkpoint_best_val_miou_0.727.pth
```

Example on `TSIRMT`:

```bash
python -m test_script.inference --dataset_config Datasets/dataset_config_win.yaml --dataset TSIRMT --model_path result/checkpoint_TSIRMT_val_miou_0.673/checkpoint_best_val_miou_0.673.pth
```

Inference outputs are written to:

```text
outputs/<dataset>/<checkpoint_basename>/infer
```

Typical files in the inference output directory:

- `infer.log`
- `exp_config.yaml`
- `results.csv`
- `pr_curve.csv`
- `roc_curve.csv`
- `target_roc_curve.csv`

If `save_pred: true` is enabled, prediction masks and logits are also exported there.

## Config Notes

- `configs/default.yaml` provides public dataset-config lookup entries, but the recommended usage is still to pass `--dataset_config` explicitly.
- `configs/infer.yaml` is a template. In public usage, `--model_path` and usually `--dataset` should be provided on the command line.
- `configs/train.yaml` disables Weights & Biases by default for public use.

## Checkpoints And Reference Results

Baidu download links and reference results:

| Dataset  | Checkpoint: extract code                                                                                        | IoU  | nIoU | Pd | Fa |
|-----------|---------------------------------------------------------------------------------------------------|------|------|------|------|
| IRDST | [checkpoint: rgyh](https://pan.baidu.com/s/18xDxPwtBVGlepJ8QllkOPQ?pwd=rgyh)  | 63.63 | 61.18 | 97.96 | 12.74 |
| NUDT-MIRSDT | [checkpoint: eahf](https://pan.baidu.com/s/16YvIt1B5kLWtmH0tQi-1oA?pwd=eahf)  | 74.41 | 75.02 | 99.22 | 0.19 |
| TSIRMT | [checkpoint: he34](https://pan.baidu.com/s/1aUGnLjE5jh1Vj-lLhUNzwA?pwd=he34)  | 69.58 | 69.35 | 88.45 | 81.04 |

Download the checkpoints from Baidu Netdisk, then place the files at these repository-relative paths before running inference:

- `result/checkpoint_IRDST_val_miou_0.612/checkpoint_best_val_miou_0.612.pth`
- `result/checkpoint_NUDT-MIRSDT_val_miou_0.727/checkpoint_best_val_miou_0.727.pth`
- `result/checkpoint_TSIRMT_val_miou_0.673/checkpoint_best_val_miou_0.673.pth`

## Acknowledgement

We would like to thank the open-source projects with  special thanks to [video swin transformer](https://github.com/SwinTransformer/Video-Swin-Transformer.git) and [LMAFormer](https://github.com/lifier/LMAFormer) for making their code public. Part of the code in our project are collected and modified from several open source repositories.

Please consider citing our paper in your publications if the project helps your research.

Cite as: Wei Zhang, Tao Liu, Tianhang Guan, et al. SGMP: Spatial-Semantic Guided Motion-Aware for Infrared Small Targets. TechRxiv. June 05, 2025.

DOI: 10.36227/techrxiv.174909854.43812513/v1
