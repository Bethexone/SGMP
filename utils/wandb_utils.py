import pathlib
from typing import Optional, Dict, List
import wandb
import torch
import numpy as np
import random

from utils.misc import interpolate


def init_or_resume_wandb_run(wandb_id_file_path: pathlib.Path,
                             project_name: Optional[str] = None,
                             entity_name: Optional[str] = None,
                             run_name: Optional[str] = None,
                             args: Optional[Dict] = None):
    """Detect the run id if it exists and resume
        from there, otherwise write the run id to file.
        Returns the config, if it's not None it will also update it first
    """
    # if the run_id was previously saved, resume from there

    blacklist = ['experiment_name', 'wandb_resume_id', 'is_train', 'device','pretrain_settings']  # 定义不需要的键
    config = {
        k: v for k, v in vars(args).items()
        if k not in blacklist
    }
    if wandb_id_file_path.exists() or args.wandb_resume_id is not None:

        if wandb_id_file_path.exists():
            resume_id = wandb_id_file_path.read_text()
        elif args.wandb_resume_id is not None:
            resume_id = args.wandb_resume_id
        print('Resuming from wandb path... ', resume_id)
        wandb.init(entity=entity_name,
                   project=project_name,
                   name=run_name,
                   id=resume_id,
                   resume="allow",
                   config=config)
        # settings=wandb.Settings(start_method="thread"))
    else:
        # if the run_id doesn't exist, then create a new run
        # and write the run id the file
        print('Creating new wandb instance...', wandb_id_file_path)
        run = wandb.init(entity=entity_name, project=project_name, name=run_name, config=config)
        wandb_id_file_path.write_text(str(run.id))

    wandb_config = wandb.config
    if config is not None:
        # update the current passed in config with the wandb_config
        wandb.config.update(config)

    return config


def get_viz_img(images: torch.tensor, targets: List[Dict], outputs: Dict[str, torch.tensor],
                itr, soenet_feats=None):
    """
    Generate Image for visualization
    Args:
        images: [T x C x H x W]
        targets: [{'masks': torch.tensor [T x H x W], ...}]
        outputs: {'pred_masks': torch.tensor [B x T x h x w] }
        itr: Inverse transform
        soenet_feats: {layer: torch.tensor}
    """
    src_masks = outputs["pred_masks"]

    # Prepare Groundtrh masks
    target_masks = [t["masks"] for t in targets]
    target_masks = [t.cpu().numpy() for t in target_masks]
    target_masks = [t.repeat(3, axis=0).astype(np.uint8) * 255 for t in target_masks]

    # Generate Predicted Masks with Thresholding
    src_masks = interpolate(src_masks, size=target_masks[0].shape[-2:], mode="bilinear", align_corners=False)
    src_masks = src_masks.cpu().detach().numpy()
    bin_mask_hwt = np.ones_like(src_masks)
    bin_mask_hwt[src_masks < 0.5] = 0

    # import ipdb;ipdb.set_trace()
    pred_masks = [bin.repeat(3, axis=0) * 255 for bin in bin_mask_hwt]
    pred_masks = [bin.transpose([1, 2, 0]) for bin in pred_masks]
    pred_masks = [bin.astype(np.uint8) for bin in pred_masks]
    target_masks = [t.transpose([1, 2, 0]) for t in target_masks]

    # Inverse Transform Images to Denormalize
    B, C, H, W = images.shape
    images_itr = itr(images)
    images_itr = images_itr.cpu().numpy() * 255
    images_itr = images_itr.astype(np.uint8)
    images_itr = np.transpose(images_itr, [0, 2, 3, 1])

    # Create concatenated Image for randomly sampled N frames
    rnd_idx = random.randint(0, B - 1)

    cat_img = np.concatenate((images_itr[rnd_idx], target_masks[rnd_idx], pred_masks[rnd_idx]),
                             axis=1)

    return cat_img
