# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2025/3/17 18:44
import torch.nn as nn
from einops import rearrange


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        _, H, W, _ = x.shape
        x = rearrange(x, 'b h w c -> b (h w) c')
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        return x


class MSMU(nn.Module):
    def __init__(self, num_tl, in_features, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        num_tl:1/2时间通道数
        in_features:输入通道数
        """
        super().__init__()
        if out_features is None:
            out_features = in_features // 2

        self.glu = ConvolutionalGLU(
            in_features=in_features * num_tl,
            hidden_features=in_features,  # 隐藏层为单帧输入通道数
            out_features=out_features,  # 输出通道1/4单帧输入通道数
            act_layer=act_layer,
            drop=drop
        )
        self.skip = nn.Conv2d(in_features * num_tl, out_features, 1)  # 跳跃连接

    def forward(self, x):
        x = rearrange(x, 'b t h w c -> b h w (t c)')
        # 分支1: GLU处理
        x_glu = self.glu(x)
        # 分支2: 直接通道压缩 (残差连接)
        x = rearrange(x, 'b h w c -> b c h w')
        x_skip = self.skip(x)
        x_skip = rearrange(x_skip, 'b c h w -> b h w c')
        # 融合
        x_out = x_glu + x_skip
        return x_out


if __name__ == '__main__':
    import torch

    # 假设输入张量大小为 B=2, N=49, C=768，H 和 W 将被计算出来
    B, N, C = 2, 49, 768  # Batch size, sequence length, and channels
    H, W = 7, 7  # Height and width for the input image shape

    num_tl = 8
    # 创建一个模拟输入
    x = torch.randn(B, num_tl, H, W, C)

    # 初始化 ConvolutionalGLU
    # conv_glu = ConvolutionalGLU(in_features=C, hidden_features=768, out_features=C, act_layer=nn.GELU, drop=0.1)
    skip = MSMU(num_tl, 768)
    # 执行前向传播
    output = skip(x)

    # 输出结果
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
