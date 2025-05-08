# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2025/3/17 13:31
from urllib import request
from typing import List, Optional, Callable

import torch
from einops import rearrange
from torch import nn

from model.third_party_models.resnet import _resnet, Bottleneck, resnet50_attn
from model.third_party_models.swintransformer3d import _swin_transformer3d
from model.third_party_models.swin_transformer import PatchMergingV2
from pathlib import Path


class MSCA(nn.Module):
    def __init__(self, inplanes, embed_dim, num_heads=2, num_tl=2):
        """
        param inplanes: x_Spatial 的输入通道数
        param embed_dim:x_Temporal的通道数
        param num_heads:头数
        """
        super(MSCA, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # 定义投影层（调整通道数）
        self.proj_q = nn.Conv2d(inplanes, embed_dim // 2, kernel_size=1)  # 将 q 的通道从in_C投影到embed_dim
        self.downsample = PatchMergingV2(embed_dim // 2)
        self.proj_k = nn.Conv3d(in_channels=num_tl, out_channels=1, kernel_size=(1, 1, 1))
        self.k_layer_norm = nn.LayerNorm(embed_dim)
        # 创建多头注意力层
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        # self.out_proj = nn.Linear(embed_dim, embed_dim)  # 可选
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_Temporal, x_Spatial):
        x_Spatial_ = self.proj_q(x_Spatial)  # (1,192,28,28)
        x_Spatial_ = rearrange(x_Spatial_, 'b c h w -> b h w c')
        q = self.downsample(x_Spatial_)

        # 调整输入维度顺序为 [batch, T, channel, height, width]
        x_Temporal_k = x_Temporal.permute(0, 1, 4, 2, 3)  # (1,4,192,28,28)
        # 在第二个维度上进行 3D 卷积
        k = self.proj_k(x_Temporal_k).squeeze(1)  # (1,192,28,28)
        k = rearrange(k, 'b c h w -> b h w c')
        k = self.k_layer_norm(k)

        v = q.clone()
        # import ipdb;ipdb.set_trace()
        batch_size, t, h, w, _ = x_Temporal.shape
        Q = rearrange(q, 'b h w c -> (h w) b c')  # (784, B, 192)
        K = rearrange(k, 'b h w c -> (h w) b c')
        V = rearrange(v, 'b h w c -> (h w) b c')

        attn_output, attn_weights = self.multihead_attn(Q, K, V)
        attn_output = rearrange(attn_output, "(h w) b c -> b h w c", h=h, w=w)
        #     (
        #     attn_output.permute(1, 0, 2))  # (B, 784, 192)
        # attn_output = attn_output.view(batch_size, h, w, self.embed_dim)  # (1,28,28,192)
        attn_output = attn_output.unsqueeze(1)
        # attn_output = self.out_proj(attn_output)  # 投影到目标维度
        output = x_Temporal + attn_output
        output = self.layer_norm(output)

        return output


class EncoderBasiclayer(nn.Module):
    def __init__(self, spatial_layer, temporal_layer, fusion):
        super(EncoderBasiclayer, self).__init__()
        self.spatial_layer = spatial_layer
        self.temporal_layer = temporal_layer
        self.fusion = fusion

    def forward(self, x_spatial, x_temporal):
        x_spatial = self.spatial_layer(x_spatial)
        x_temporal = self.temporal_layer(x_temporal)

        if self.fusion is None:
            x_temporalspatial = x_temporal
        else:
            x_temporalspatial = self.fusion(x_temporal, x_spatial)  # [1, 2, 28, 28, 192]

        return x_spatial, x_temporalspatial


class SGMP_Encoder(nn.Module):
    def __init__(self,
                 num_tl: int,
                 embed_dim: int = 96,
                 resnet_depths=None,
                 swin_depths=None,
                 num_swin_heads=None,
                 num_msca_heads=None,
                 window_size: List[int] = None,
                 stochastic_depth_prob: float = 0.1,
                 is_train: bool = True):
        super(SGMP_Encoder, self).__init__()

        if resnet_depths is None:
            resnet_depths = [3, 4, 6, 3]
        if swin_depths is None:
            swin_depths = [2, 2, 6, 2]
        if num_swin_heads is None:
            num_swin_heads = [3, 6, 12, 24]
            # num_swin_heads = [3, 6, 12, 12]
        if num_msca_heads is None:
            num_msca_heads = [3, 6, 2, 2]
        if window_size is None:
            window_size = [8, 7, 7]

        self.num_tl = num_tl

        resnet = resnet50_attn()
        # resnet = _resnet(Bottleneck, resnet_depths)
        swin3d = _swin_transformer3d(
            patch_size=[2, 4, 4],
            embed_dim=embed_dim,
            depths=swin_depths,
            num_heads=num_swin_heads,
            window_size=window_size,
            stochastic_depth_prob=stochastic_depth_prob,
        )
        if is_train:
            resnet, swin3d = self.load_checkpoint(resnet, swin3d)
        # self.freeze_weights()

        resnet_stage = []
        resnet_stage.append(resnet.conv_init)
        resnet_stage.append(resnet.layer1)  # layer1
        resnet_stage.append(resnet.layer2)  # layer2
        resnet_stage.append(resnet.layer3)  # layer3
        resnet_stage.append(resnet.layer4)  # layer4

        swin3d_stage = []
        swin3d_stage.append(nn.Sequential(
            swin3d.patch_embed,  # 初始卷积层
            swin3d.pos_drop
        ))

        swin3d_stage.append(swin3d.features[0])

        swin3d_stage.append(nn.Sequential(
            swin3d.features[1],  # 包含第一个PatchMerging
            swin3d.features[2]
        ))
        swin3d_stage.append(nn.Sequential(
            swin3d.features[3],  # 包含第二个PatchMerging
            swin3d.features[4]
        ))
        swin3d_stage.append(nn.Sequential(
            swin3d.features[5],  # 包含第三个PatchMerging
            swin3d.features[6],
        ))

        self.model = nn.Sequential()

        for i in range(len(swin3d_stage)):
            if i == 0:
                self.model.append(self.maker_Basiclayer(i, resnet_stage, swin3d_stage))
            else:
                self.model.append(
                    self.maker_Basiclayer(i, resnet_stage, swin3d_stage, num_heads=num_msca_heads[i - 1]))

        self.norm = swin3d.norm

        # self.freeze_weights()
        # # 验证冻结状态
        # for name, param in self.model.named_parameters():
        #     print(f"{name}: {'可训练' if param.requires_grad else '冻结'}")

    def maker_Basiclayer(self, i, resnet_stage, swin3d_stage, num_heads=2):
        spatial_layer = resnet_stage[i]
        temporal_layer = swin3d_stage[i]
        if i == 0:
            fusion = None
        else:
            in_c = 64 * (2 ** (i - 1))
            embed_dim = 48 * (2 ** i)
            fusion = MSCA(in_c, embed_dim, num_tl=self.num_tl, num_heads=num_heads)

        return EncoderBasiclayer(spatial_layer, temporal_layer, fusion)

    def load_checkpoint(self, resnet, swin3d):
        # 定义路径和 URL
        resnet_path = Path('pretrain/resnet50-11ad3fa6.pth')
        resnet_url = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
        # 检查文件是否存在
        if not resnet_path.exists():
            print("未找到模型文件，正在下载...")
            resnet_path.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录（如果不存在）
            request.urlretrieve(resnet_url, resnet_path)  # 下载文件
            print(f"模型已下载至 {resnet_path}")

        swin3d_path = Path('pretrain/swin3d_t-7615ae03.pth')
        swin3d_url = "https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth"
        if not swin3d_path.exists():
            print("未找到模型文件，正在下载...")
            swin3d_path.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录（如果不存在）
            request.urlretrieve(swin3d_url, swin3d_path)  # 下载文件
            print(f"模型已下载至 {swin3d_path}")

        # 加载自定义训练的权重
        # resnet_checkpoint = torch.load(resnet_path, weights_only=True,map_location='cpu')
        swin3d_checkpoint = torch.load(swin3d_path, weights_only=True, map_location='cpu')
        # keys_to_delete = [k for k in swin3d_checkpoint.keys() if k.startswith('features.6')]
        # for k in keys_to_delete:
        #     print(f"删除参数: {k}")
        #     del swin3d_checkpoint[k]
        del swin3d_checkpoint['head.weight']
        del swin3d_checkpoint['head.bias']

        # resnet.load_state_dict(resnet_checkpoint, strict=False)
        swin3d.load_state_dict(swin3d_checkpoint, strict=False)
        return resnet, swin3d

    def freeze_weights(self):
        # 冻结ResNet前两阶段（conv1到layer2）
        self.model[:2].requires_grad_(False)  # conv1+bn1+maxpool
        # 保持归一化层可训练（适配红外数据分布）
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                m.requires_grad_(True)

        # 强制解冻所有融合层
        for name, module in self.named_modules():
            if "fusion" in name:
                module.requires_grad_(True)

        # 保持核心模块可训练
        for param in self.norm.parameters():
            param.requires_grad = True
        for param in self.model[-2:].parameters():  # 最后两个层级
            param.requires_grad = True

    def configure_param(self, lr, lr_backbone):
        # 分离融合模块参数与其他参数
        fusion_params = []
        backbone_params = []
        # 参数分组策略
        for name, param in self.named_parameters():
            if "fusion" in name and param.requires_grad:
                fusion_params.append(param)
            elif param.requires_grad:
                backbone_params.append(param)
        params = [
            # 空间分支参数组
            {"params": fusion_params, "lr": lr},
            {"params": backbone_params, "lr": lr_backbone},
        ]
        return params

    def forward(self, x_Spatial, x_Temporal):
        """
            torch.Size([5, 2, 56, 56, 96])
            torch.Size([5, 2, 28, 28, 192])
            torch.Size([5, 2, 14, 14, 384])
            torch.Size([5, 2, 7, 7, 768])
        """
        result = []
        for i, layer in enumerate(self.model):
            # summary(layer, input_data=(x_Spatial, x_Temporal))
            x_Spatial, x_Temporal = layer(x_Spatial, x_Temporal)
            if i == len(self.model) - 1:
                x_Temporal = self.norm(x_Temporal)
            if i != 0:
                result.append(x_Temporal)
        return result


if __name__ == '__main__':
    num_tl = 6
    model = SGMP_Encoder(num_tl // 2)
    x = torch.randn(10, 3, 320, 320)
    y = torch.randn(10, 3, num_tl, 320, 320)
    model.eval()
    # print(model)
    tmp = model(x, y)
    for x in tmp:
        print(x.shape)
