# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2025/3/10 16:31
import torch

from model.SGMP_Encoder import SGMP_Encoder
from model.MSMU import MSMU
from model.SWTM_Decoder import Decoder
from torch import nn
from model.Head import Multiscale_Former


class Net(nn.Module):
    def __init__(self, num_tl, embed_dim=96, is_train=True, num_fusion_head=None):
        super(Net, self).__init__()
        self.num_tl = num_tl // 2
        self.encoder = SGMP_Encoder(self.num_tl, embed_dim=embed_dim, num_msca_heads=num_fusion_head, is_train=is_train)

        self.skip_layers = nn.Sequential()

        for i in range(4):
            in_features = embed_dim * 2 ** i
            if i == 3:
                skip = MSMU(self.num_tl, in_features, out_features=in_features)
            else:
                skip = MSMU(self.num_tl, in_features)
            self.skip_layers.append(skip)

        decode_embeddim = embed_dim * 2 ** 3
        self.decoder = Decoder(decode_embeddim)
        self.former = Multiscale_Former(embed_dim)

    def configure_param(self, lr, lr_backbone):
        params = []

        # 获取 encoder, decoder 和 former 的参数组
        encoder_params = self.encoder.configure_param(lr, lr_backbone)
        decoder_params = self.decoder.configure_param(lr)
        former_params = self.former.configure_param(lr)

        # skip_params 包含 'params' 和 'lr'
        skip_params = {'params': self.skip_layers.parameters(), 'lr': lr}

        # 将每个参数组添加到 params 列表中
        params.extend(encoder_params)
        params.extend(decoder_params)
        params.append(skip_params)
        params.extend(former_params)

        return params

    def forward(self, sample):
        # import ipdb;ipdb.set_trace()
        assert sample.num_tl == self.num_tl * 2
        assert sample.FrameH % 16 == 0
        assert sample.FrameW % 16 == 0
        x, y = sample.decompose()
        skip_s = self.encoder(x, y)
        # summary(self.encoder, input_data=[x, y])
        for skip_i in skip_s:
            print(skip_i.shape)
        decode_s = []
        # for skip_layer, skip_tensor in zip(self.skip_layers, skip_s):
        #     decode_s.append(skip_layer(skip_tensor))

        for skip_layer, skip_tensor in zip(self.skip_layers, skip_s):
            # summary(skip_layer, input_data=skip_tensor)
            decode_s.append(skip_layer(skip_tensor))

        decode_s.reverse()
        # summary(self.decoder, input_data=(decode_s,))
        out = self.decoder(decode_s)
        # summary(self.former, input_data=out)
        mask_down4, mask_down2, mask_down1, mask = self.former(out)
        result = {}
        result["pred_masks"] = mask
        result["aux_pred_masks"] = []
        result["aux_pred_masks"].append(mask_down1)
        result["aux_pred_masks"].append(mask_down2)
        result["aux_pred_masks"].append(mask_down4)

        return result


if __name__ == '__main__':
    num_tl = 6
    model = Net(num_tl)
    model.eval()
    model.configure_param(0.1, 0.01)
    # x = torch.randn(10, 3, 320, 384)
    # y = torch.randn(10, 3, num_tl, 320, 384)
    #
    # from torchinfo import summary
    # # summary(model, input_data=[x, y],depth=10)
    # tmp = model(x, y)
    # for i in tmp:
    #     print(i.shape)
    #
    from Datasets.datasets.utils import NestedTensor

    x = torch.randn(1, 3, 320, 256)
    y = torch.randn(1, 3, num_tl, 320, 256)
    img = [(x, y)] * 10
    sample = NestedTensor(img)
    sample.frame = x
    sample.frame_sequence = y
    sample.num_tl = num_tl
    # from torchinfo import summary

    # summary(model, input_data=sample, depth=10)
    tmp = model(sample)
    for k, v in tmp.items():
        print('v:',v[0].shape)

    # tmp = model(x, y)
    # for i in tmp:
    #     print(i.shape)
