# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2025/3/12 13:15
from functools import partial
from pathlib import Path
from typing import List, Optional, Callable
from urllib import request

import torch
from einops import rearrange
from torch import nn, Tensor

from model.third_party_models.swin_transformer import SwinTransformerBlockV2, ShiftedWindowAttentionV2


class Patch_Expand(nn.Module):
    def __init__(self, dim, expand_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        # assert dim % (expand_scale ** 2) == 0
        self.dim = dim
        self.scale = expand_scale
        out_dim = dim // (self.scale ** 2)
        self.upsample = nn.PixelShuffle(self.scale)  # 使用像素Shuffle实现上采样

        self.projection = nn.Sequential(
            nn.Linear(out_dim, out_dim, bias=False),
            norm_layer(out_dim)
        )

    def forward(self, x):
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.upsample(x)
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.projection(x)
        return x


class Decoder(nn.Module):
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
        super(Decoder, self).__init__()
        if depths is None:
            depths = [2, 6, 2]
        if num_heads is None:
            num_heads = [12, 6, 3]
        if window_size is None:
            window_size = [7, 7]

        if block is None:
            block = partial(SwinTransformerBlockV2, attn_layer=ShiftedWindowAttentionV2)

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: List[nn.Module] = []
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        layers.append(upsample_layer(embed_dim))
        embed_dim = embed_dim // 2
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim // 2 ** i_stage
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
            # add patch expand layer
            if i_stage < (len(depths) - 1):
                stage.append(upsample_layer(dim))
            layers.append(nn.Sequential(*stage))

        self.features = nn.Sequential(*layers)

        self.num_features = embed_dim // 2 ** (len(depths) - 1)
        self.norm = norm_layer(self.num_features)

        self.apply(self.init_weights)

    def init_weights(self, m):
        """统一初始化逻辑"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)  # 与Swin官方一致
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='gelu')
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

    def forward(self, x: List[Tensor]) -> Tensor:
        x_ = None
        for idx_layer, (feat, layer_up) in enumerate(zip(x, self.features)):
            if x_ is None:
                x_ = layer_up(feat)
            else:
                x_ = layer_up(torch.cat([x_, feat], dim=-1))  # 沿通道维拼接

        return self.norm(x_)


if __name__ == '__main__':
    # x = torch.randn(1, 28, 28, 768)
    # model = Patch_Expand(dim=768)
    # print(model(x).shape)

    # torch.Size([5, 7, 7, 768])
    # torch.Size([5, 14, 14, 192])
    # torch.Size([5, 28, 28, 96])
    # torch.Size([5, 56, 56, 48])

    tmp = []
    x0 = torch.randn(5, 7, 7, 768)
    x1 = torch.randn(5, 14, 14, 192)
    x2 = torch.randn(5, 28, 28, 96)
    x3 = torch.randn(5, 56, 56, 48)

    tmp.append(x0)
    tmp.append(x1)
    tmp.append(x2)
    tmp.append(x3)

    num_heads = [3, 6, 12, 24]
    window_size = [7, 7]
    decode = Decoder(embed_dim=768)
    decode.eval()
    from torchinfo import summary
    # summary(decode, input_data=[x0, x1, x2, x3])
    input_data = (x0, x1, x2, x3)  # 将列表转换为元组

    # 使用 summary
    summary(decode, input_data=([x0, x1, x2, x3],))

    # torch.Size([5, 14, 14, 192])
    # torch.Size([5, 28, 28, 96])
    # torch.Size([5, 56, 56, 48])
    # torch.Size([5, 56, 56, 96])
