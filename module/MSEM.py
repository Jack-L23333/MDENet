import torch
import torch.nn as nn
# from module.wtconv.wtconv2d import WTConv2d

class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    def __init__(
            self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.999))

        if relu:
            self.add_module("relu", nn.ReLU())


import torch
import torch.nn as nn


class _ConvBnReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.999))
        self.add_module("relu", nn.ReLU())


# class pdhm(nn.Module):
#     def __init__(self, dim_in, dim_out):
#         super(pdhm, self).__init__()
#         self.relu = nn.ReLU(True)
#
#         self.conv1 = _ConvBnReLU(dim_in, dim_in, 3, 1, 1)
#         self.conv2 = _ConvBnReLU(dim_in, dim_in, 3, 1, 1)
#
#         # Multi-kernel convolution layers with BatchNorm and ReLU
#         self.branch1 = _ConvBnReLU(dim_in, dim_out // 4, 3, 1, 1)  # 3x3卷积
#         self.branch2 = _ConvBnReLU(dim_in, dim_out // 4, 5, 1, 2)  # 5x5卷积
#         self.branch3 = _ConvBnReLU(dim_in, dim_out // 4, 7, 1, 3)  # 7x7卷积
#         self.branch4 = _ConvBnReLU(dim_in, dim_out // 4, 1, 1, 0)  # 1x1卷积
#
#         self.res = _ConvBnReLU(dim_in, dim_out, 1, 1, 0)
#         self.cat = _ConvBnReLU(dim_out * 3, dim_out, 3, 1, 1)
#
#     def forward(self, x1, x2):
#         x_add = x1 + x2
#         x_diff = torch.abs(x1 - x2)
#         y = self.conv1(x_diff) + self.conv2(x_add)
#
#         y1 = self.branch1(y)
#         y2 = self.branch2(y)
#         y3 = self.branch3(y)
#         y4 = self.branch4(y)
#
#         y = self.cat(torch.cat([self.res(x1), self.res(x2), y1, y2, y3, y4], 1))
#         return y


class MSEM(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MSEM, self).__init__()
        self.relu = nn.ReLU(True)

        self.conv1 = _ConvBnReLU(dim_in, dim_in, 3, 1, 1)


        # 多核卷积层
        self.branch1 = _ConvBnReLU(dim_in, dim_out // 4, 3, 1, 1)
        self.branch2 = _ConvBnReLU(dim_in, dim_out // 4, 5, 1, 2)
        self.branch3 = _ConvBnReLU(dim_in, dim_out // 4, 7, 1, 3)
        self.branch4 = _ConvBnReLU(dim_in, dim_out // 4, 1, 1, 0)

        self.res = _ConvBnReLU(dim_in, dim_out, 1, 1, 0)
        self.cat = _ConvBnReLU(dim_out * 2, dim_out, 3, 1, 1)

    def forward(self, x):
        y = self.conv1(x)#512

        y1 = self.branch1(y)#128
        y2 = self.branch2(y)
        y3 = self.branch3(y)
        y4 = self.branch4(y)

        y = self.cat(torch.cat([self.res(x), y1, y2, y3, y4], 1))
        return y



