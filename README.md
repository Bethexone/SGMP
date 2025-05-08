## PyTorch implementation of "SGMP-UNet: Spatial-Semantic Guided Motion-Aware UNet for Infrared Small Targets"
<!-- [Project](https://github.com/lifier/LMAFormer) - [Paper](https://ieeexplore.ieee.org/document/10758760) -->
<!-- <hr> -->

# Abstract

<p align="justify">
In temporal infrared small target detection, the effective exploitation of motion pattern discrepancies between targets and backgrounds is crucial for achieving accurate detection. Although existing convolutional neural network (CNN)-based architectures benefit from local feature perception, their limited receptive fields hinder the modeling of long-range temporal dependencies, especially in scenes with dynamically complexity, leading to a notable degradation in feature discrimination. While Transformer-based architectures have been demonstrated to address the issue of long-sequence dependency through selfattention mechanisms. However the uniform global interaction tends to dilute the representation of subtle local motion patterns, which are crucial for small target detection. To overcome these limitations, we propose a novel backbone network, called Spatial Semantic-Guided Motion Perception (SGMP). SGMP introduces a spatially semantic-guided attention mechanism that (i) leverages spatial features to generate salient semantic query vectors and value maps for extracting potential semantic priors of targets, and (ii) incorporates a Transformer encoder to model longrange motion features, aligning global motion context with local semantic cues across modalities. The design enables it to focus on motion-sensitive spatiotemporal regions associated with small targets. Built upon SGMP, we design a lightweight and efficient detection framework called SGMP-UNet, which fully exploits motion-aware representations for end-to-end small target detection. Extensive experiments on the NUDT-MIRSDT, IRDST and TSIRMT datasets demonstrate that SGMP-UNet consistently outperforms state-of-the-art (SOTA) methods across multiple evaluation metrics.
</p>

# Architecture

<p align="center">
  <img src="flow_chart.png"  style="width: auto;" alt="accessibility text">
</p>

## Environment Setup

The experiments were done on **Ubuntu 20.04.6 LTS** with **python 3** using anaconda environment. Here is details on how to set up the conda environment.

* Create conda environment:

  ```create environment
  conda create -n SGMP python
  conda activate SGMP
  ```

* Install **PyTorch 2.6.0** with **cuda 12.4**

  ```setup
  pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  ```

* Install other requirements:

  ```setup
  pip install -r requirements.txt
  ```

## Datasets

We evaluate network performance using **NUDR-MIRSDT**, **IRDST** and **TSIRMT**

Download the datasets following the corresponding paper/project page and update dataset paths in **'datasets/path_config.py'**.
Here is the list of datasets used.

* [IRDST](https://pan.baidu.com/s/1OA9uFtAzkknn3pFYGO4R0Q?pwd=6m32) (Extraction code: 6m32)
* [NUDT-MIRSDT](https://pan.baidu.com/s/1qrERzVrEYQ7ToRToMuV47Q?pwd=i5ce)(Extraction code: i5ce)
* [TSIRMT](https://pan.baidu.com/s/1-05XbfxNDRHNtDBstZxccg?pwd=r29f)(Extraction code: r29f)

# Training

The models were trained and tested using a single NVIDIA A5000 GPU.  

* Train LMAFormer with Swin backbone on NUDT-MIRSDR, IRDST, TSIRMT datasets:

  ```bash
  python -m train_script.train
  ```

# Inference

## Inference on IRDST

  ```bash
  python -m test_script.inference  --model_path ./result/IRDST/checkpoint_IRDST_val_miou_0.610.pth  --dataset IRDST --flip --msc --output_dir ./predict

  Expected miou: 60.80
  ```

## Inference on NUDT-MIRSDT

  ```bash
  python -m test_script.inference  --model_path ./result/NUDT-MIRSDT/checkpoint_NUDT-MIRSDT_val_miou_0.763.pth  --dataset NUDT-MIRSDT --flip --msc --output_dir ./predict

  Expected miou: 74.87
  ```

## Inference on TSIRMT

  ```bash
  python -m test_script.inference  --model_path ./result/TSIRMT/checkpoint_TSIRMT_val_miou_0.668.pth  --dataset TSIRMT --flip --msc  --output_dir ./predict
  
  Expected miou: 66.81
  ```

## Results Summary

### Results on NUDT-MIRSDT, IRDST and TSIRMT

| Dataset  | Checkpoint: extract code                                                                                        | IoU  | nIoU | Pd | Fa |
|-----------|---------------------------------------------------------------------------------------------------|------|------|------|------|
| IRDST | [checkpoint: uhnj](https://pan.baidu.com/s/10_cUBX2BCjcM6h1nMwcJLw?pwd=uhnj)  | 60.80 | 59.28 | 98.68 | 17.83 |
| NUDT-MIRSDT | [checkpoint: i58c](https://pan.baidu.com/s/1tVov97xd3mHsd61_m7XxUQ?pwd=i58c)  | 74.87 | 75.42 | 99.48 | 2.91 |
| TSIRMT | [checkpoint: epsw](https://pan.baidu.com/s/1FgaZFkYIvZG2gDu8iip2Aw?pwd=epsw)  | 66.81 | 66.81 | 89.01 | 187.74 |

### Acknowledgement

We would like to thank the open-source projects with  special thanks to [video swin transformer](https://github.com/SwinTransformer/Video-Swin-Transformer.git)  and [LMAFormer](https://github.com/lifier/LMAFormer) for making their code public. Part of the code in our project are collected and modified from several open source repositories.

## Citation

Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follow.
