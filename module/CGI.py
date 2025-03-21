import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn #(B, 1, H, W)

class ChannelAttention2(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention2, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.map = nn.AdaptiveMaxPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )
        self.ma =nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)

        x_map = self.map(x)

        cattn = self.ca(x_gap)

        mattn = self.ma(x_map)

        return cattn, mattn #(B, C, 1, 1)
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        # print(pattn1.shape)
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        # print(x2.shape)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2

class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):  # dim指的是输入tensor的通道数，该模块输入与输出相同
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.convfusion = nn.Conv2d(dim * 2, dim, 3, padding=1)  # 改为dim * 2和dim
        self.ca = ChannelAttention2(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, color_feat, graph_feat):
        normal_feat = torch.cat((t1, t2), dim=1)
        normal_feat = self.convfusion(normal_feat)

        initial = normal_feat + color_feat + graph_feat
        cattn, mattn = self.ca(initial) #(B, C, 1, 1)
        sattn = self.sa(initial)#(B, 1, H, W)
        pattn1 = sattn + cattn + mattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * color_feat + (1 - pattn2) * graph_feat
        result = self.conv(result)
        return result

