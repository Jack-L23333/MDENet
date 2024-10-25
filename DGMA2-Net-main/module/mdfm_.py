# import torch
# import torch.nn as nn
#
#
# def calc_men_std(feat,eps =1e-5):
#     size = feat.size()
#     assert (len(size)==4)
#     N,C,H,W=size[:4]
#     feat_var = feat.var(dim=1)+eps
#     feat_var =feat_var.sqrt().view(N,1,H,W)
#     feat_mean = feat.mean(dim=1).view(N,1,H,W)
#     return feat_mean,feat_var
#
# class MDFM(nn.Module):
#     def __init__(self, in_d, out_d):
#         super(MDFM, self).__init__()
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=in_d * 2, out_channels=in_d, kernel_size=3, padding=1, bias=False),
#             nn.Conv2d(in_channels=in_d, out_channels=out_d, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_d),
#             nn.ReLU()
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=in_d * 2, out_channels=in_d, kernel_size=3, padding=1, bias=False),
#             nn.Conv2d(in_channels=in_d, out_channels=out_d, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_d),
#             nn.ReLU()
#         )
#
#     def forward(self, T1, T2):
#         T1_mean, T1_var = calc_men_std(T1)
#         T2_mean, T2_var = calc_men_std(T2)
#
#         # 合并 T1 和 T2
#         T = torch.cat((T1, T2), dim=1)
#
#         # 通过第一个卷积层
#         x = self.conv1(T)
#
#         # 计算均值和方差的差异
#         T_mean = T1_mean - T2_mean
#         T_var = T1_var - T2_var
#
#         # 进行加权
#         x_mean = x * T_mean
#         x_var = x * T_var
#
#         # 合并加权后的均值和方差
#         X = torch.cat((x_mean, x_var), dim=1)
#
#         # 通过第二个卷积层
#         X = self.conv2(X)
#
#         return X


import torch
import torch.nn as nn
from module.EMA import EMA


def calc_men_std(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4
    N, C, H, W = size[:4]
    feat_var = feat.var(dim=1) + eps
    feat_var = feat_var.sqrt().view(N, 1, H, W)
    feat_mean = feat.mean(dim=1).view(N, 1, H, W)
    return feat_mean, feat_var



class MDFM(nn.Module):
    def __init__(self, in_d, out_d):
        super(MDFM, self).__init__()

        # 多尺度膨胀卷积用于T1和T2的初步特征提取
        self.dilated_conv_T_1 = nn.Sequential(
            nn.Conv2d(in_d, in_d, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(in_d),
            nn.ReLU()
        )
        self.dilated_conv_T_3 = nn.Sequential(
            nn.Conv2d(in_d, in_d, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(in_d),
            nn.ReLU()
        )
        self.dilated_conv_T_5 = nn.Sequential(
            nn.Conv2d(in_d, in_d, kernel_size=3, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(in_d),
            nn.ReLU()
        )
        self.dilated_conv_T_7 = nn.Sequential(
            nn.Conv2d(in_d, in_d, kernel_size=3, padding=7, dilation=7, bias=False),
            nn.BatchNorm2d(in_d),
            nn.ReLU()
        )

        # 将多尺度特征合并为一个通道数为 in_d 的特征图
        self.reduce_conv_T = nn.Sequential(
            nn.Conv2d(in_d * 4, in_d, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_d),
            nn.ReLU()
        )

        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_d * 2, out_channels=in_d, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_d),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_d, out_channels=out_d, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_d),
            nn.ReLU()
        )

        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_d * 2, out_channels=in_d, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_d),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_d, out_channels=out_d, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_d),
            nn.ReLU()
        )

        # 添加 EMA 模块
        self.ema = EMA(out_d)

    def forward(self, T1, T2):
        # T1和T2分别通过四个不同膨胀率的卷积
        T1_1 = self.dilated_conv_T_1(T1)
        T1_3 = self.dilated_conv_T_3(T1)
        T1_5 = self.dilated_conv_T_5(T1)
        T1_7 = self.dilated_conv_T_7(T1)

        T2_1 = self.dilated_conv_T_1(T2)
        T2_3 = self.dilated_conv_T_3(T2)
        T2_5 = self.dilated_conv_T_5(T2)
        T2_7 = self.dilated_conv_T_7(T2)

        # 拼接多尺度膨胀卷积的特征
        T1 = torch.cat([T1_1, T1_3, T1_5, T1_7], dim=1)
        T2 = torch.cat([T2_1, T2_3, T2_5, T2_7], dim=1)

        # 使用1x1卷积减少通道数
        T1 = self.reduce_conv_T(T1)
        T1 = self.ema(T1)
        T2 = self.reduce_conv_T(T2)
        T2 = self.ema(T2)

        # 计算均值和方差
        T1_mean, T1_var = calc_men_std(T1)
        T2_mean, T2_var = calc_men_std(T2)

        # 合并T1和T2并通过第一个卷积块
        T = torch.cat((T1, T2), dim=1)
        x = self.conv1(T)

        # 计算均值和方差的差异
        T_mean = T1_mean - T2_mean
        T_var = T1_var - T2_var

        # 加权
        x_mean = x * T_mean
        x_var = x * T_var

        # 合并加权后的均值和方差
        X = torch.cat((x_mean, x_var), dim=1)

        # 通过第二个卷积块
        X = self.conv2(X)

        return X



