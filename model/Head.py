# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2025/3/17 23:09
from functools import partial
from typing import List, Optional, Callable

import torch
from einops import rearrange
from torch import nn, Tensor

from model.third_party_models.swin_transformer import SwinTransformerBlockV2, ShiftedWindowAttentionV2
from model.SWTM_Decoder import Patch_Expand


class Multiscale_Former(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 depths: List[int] = None,
                 num_heads: List[int] = None,
                 window_size: List[int] = None,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 attention_dropout: float = 0.0,
                 stochastic_depth_prob: float = 0.1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 block: Optional[Callable[..., nn.Module]] = None,
                 upsample_layer: Callable[..., nn.Module] = Patch_Expand,
                 ):
        super(Multiscale_Former, self).__init__()
        if depths is None:
            depths = [2, 2]
        if num_heads is None:
            num_heads = [6, 3]
        if window_size is None:
            window_size = [7, 7]

        if block is None:
            block = partial(SwinTransformerBlockV2, attn_layer=ShiftedWindowAttentionV2)

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: List[nn.Module] = []
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim // 4 ** i_stage
            # add patch expand layer
            stage.append(upsample_layer(dim))
            dim = dim // 4
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                        attn_layer=ShiftedWindowAttentionV2,
                    )
                )
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        self.features = nn.Sequential(*layers)

        self.fc_s = nn.Sequential()
        for i_stage in range(len(depths) + 1):
            reshape = Reshape('b h w c', 'b c h w')
            dim = embed_dim // 4 ** i_stage
            # fc = nn.Linear(dim, 1)
            fc = nn.Conv2d(dim, 1, kernel_size=(1, 1), stride=1)
            up = nn.Upsample(scale_factor=2 ** (len(depths) - i_stage), mode='bilinear', align_corners=True)
            self.fc_s.append(nn.Sequential(reshape, fc, up))

        self.fc = nn.Conv2d(3, 1, kernel_size=(1, 1), stride=1)

        # self.num_features = embed_dim // 2 ** (len(depths) - 1)
        # self.norm = norm_layer(self.num_features)

        self.apply(self.init_weights)

    def init_weights(self, m):
        """统一初始化逻辑"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)  # 与Swin官方一致
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Parameter):
            if m.dim() > 1:  # 相对位置偏置等
                nn.init.trunc_normal_(m, std=0.02)

    def configure_param(self, lr):
        params = [
            {'params': self.parameters(), 'lr': lr},
        ]
        return params

    def forward_fc(self, result):
        out = []
        for x, fc in zip(result, self.fc_s):
            out.append(fc(x))

        return out

    def forward(self, x: Tensor) -> List[Tensor]:
        result = []
        result.append(x)
        for idx_layer, layer_up in enumerate(self.features):
            # print(layer_up)
            x = layer_up(x)
            result.append(x)
        out = self.forward_fc(result)
        mask = self.fc(torch.concatenate(out, dim=1))
        out.append(mask)
        return out


class Reshape(nn.Module):
    def __init__(self, str1: str, str2: str):
        super(Reshape, self).__init__()
        self.fmt = str1 + '->' + str2

    def forward(self, x: Tensor, ) -> Tensor:
        return rearrange(x, self.fmt)


if __name__ == '__main__':
    x = torch.randn(10, 80, 80, 96)
    Former = Multiscale_Former(96)
    from torchinfo import summary

    summary(Former, input_data=x)
    out = Former(x)
    for y in out:
        print(y.shape)
