# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2025/3/31 12:42
import torch
from torch import nn

from model.SGMP_UNet import Net
from utils.model_inputs import ModelInputs


# 导出时使用的包装器
class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, y):
        sample = ModelInputs(frame=x.unsqueeze(0), frame_sequence=y.unsqueeze(0))
        result = self.model(sample)
        return result["pred_masks"], result["aux_pred_masks"][0], result["aux_pred_masks"][1]


if __name__ == '__main__':
    num_tl = 6
    model = Net(num_tl)
    # model.eval()
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

    x = torch.randn(3, 320, 256)
    y = torch.randn(3, num_tl, 320, 256)
    # 导出代码
    wrapper = ExportWrapper(Net(num_tl=6))
    # torch.onnx.export(wrapper, (x, y), "model.onnx")
    from torchinfo import summary

    summary(wrapper, input_data=[x, y], depth=10)
