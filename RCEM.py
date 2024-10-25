# import torch
# import torch.nn as nn
# from models.EMA import EMA
# from models.wtconv.wtconv2d import WTConv2d
#
# class RCEM(nn.Module):
#     def __init__(self, in_d, out_d):
#         super(RCEM, self).__init__()
#         self.conv_dw = WTConv2d(in_d, out_d, kernel_size=5, wt_levels=3)
#         self.ema = EMA(out_d)
#
#     def forward(self, x1_1, x1_2, x1_3, x1_4, x1_5,x2_1, x2_2, x2_3, x2_4, x2_5):
#         x1_2 = self.conv_dw(x1_2)
#         x1_2 = self.ema(x1_2)
#         x1_3 = self.conv_dw(x1_3)
#         x1_3 = self.ema(x1_3)
#         x1_4 = self.conv_dw(x1_4)
#         x1_4 = self.ema(x1_4)
#         x1_5 = self.conv_dw(x1_5)
#         x1_5 = self.ema(x1_5)
#
#         x2_2 = self.conv_dw(x2_2)
#         x2_2 = self.ema(x2_2)
#         x2_3 = self.conv_dw(x2_3)
#         x2_3 = self.ema(x2_3)
#         x2_4 = self.conv_dw(x2_4)
#         x2_4 = self.ema(x2_4)
#         x2_5 = self.conv_dw(x2_5)
#         x2_5 = self.ema(x2_5)
#
#         return x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5
import torch.nn as nn
from module.EMA import EMA
from models.wtconv.wtconv2d import WTConv2d


class RCEM(nn.Module):
    def __init__(self, channels_list):
        super(RCEM, self).__init__()

        # 根据每个通道数创建独立的WTConv2d和EMA模块
        self.conv_dw_layers = nn.ModuleList(
            [WTConv2d(in_d, in_d, kernel_size=5, wt_levels=3) for in_d in channels_list])
        self.ema_layers = nn.ModuleList([EMA(in_d) for in_d in channels_list])

    def forward(self, x1_1, x1_2, x1_3, x1_4, x1_5, x2_1, x2_2, x2_3, x2_4, x2_5):
        # 对每个输入特征进行处理，调用对应的WTConv2d和EMA模块
        inputs = [x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5]
        outputs = []

        for i, x in enumerate(inputs):
            x = self.conv_dw_layers[i](x)
            x = self.ema_layers[i](x)
            outputs.append(x)

        # 返回处理后的特征
        return outputs

