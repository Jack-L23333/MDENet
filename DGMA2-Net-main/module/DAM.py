# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# class DAM(nn.Module):
#     def __init__(self, out_d):
#         super(DAM, self).__init__()
#         self.out_d=out_d
#         self.conv1 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1),
#                                    nn.BatchNorm2d(256),
#                                    nn.ReLU(inplace=True))
#         self.conv2 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1),
#                                    nn.BatchNorm2d(128),
#                                    nn.ReLU(inplace=True))
#         self.conv3 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1),
#                                    nn.BatchNorm2d(64),
#                                    nn.ReLU(inplace=True))
#         self.conv4 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1),
#                                    nn.BatchNorm2d(64),
#                                    nn.ReLU(inplace=True))
#         # self.convT = nn.Sequential(
#         #                  nn.Conv2d(512, 256, kernel_size=1, stride=1),
#         #                  nn.BatchNorm2d(self.out_d),
#         #                  nn.ReLU(inplace=True)
#         # )
#     def forward(self, d5, d4, d3, d2):
#         #d5
#         d5_ = self.conv1(d5)
#         D4 = F.interpolate(d5_, d4.size()[2:], mode='bilinear', align_corners=True)
#         b, c, h, w = D4.shape
#         D4_ = D4.reshape(b, c, h * w)
#         d4_ = D4_.permute(0, 2, 1)
#         d4 = torch.matmul(d4_, d4)
#         d4 = torch.matmul(D4_, d4)
#         d4 = d4.reshape(b, c, h, w)
#         d4 = D4+d4
#         #d4
#         d4_ = self.conv2(d4)
#         D3 = F.interpolate(d4_, d3.size()[2:], mode='bilinear', align_corners=True)
#         b, c, h, w = D3.shape
#         D3_ = D3.reshape(b, c, h * w)
#         d3_ = D3_.permute(0, 2, 1)
#         d3 = torch.matmul(d3_, d3)
#         d3 = torch.matmul(D3_, d3)
#         d3 = d3.reshape(b, c, h, w)
#         d3 = D3 + d3
#         #d3
#         d3_ = self.conv3(d3)
#         D2 = F.interpolate(d3_, d2.size()[2:], mode='bilinear', align_corners=True)
#         b, c, h, w = D2.shape
#         D2_ = D2.reshape(b, c, h * w)
#         d2_ = D2_.permute(0, 2, 1)
#         d2 = torch.matmul(d2_, d2)
#         d2 = torch.matmul(D2_, d2)
#         d2 = d2.reshape(b, c, h, w)
#         d2 = D2 + d2
#         # d2
#         return d5, d4, d3, d2
import torch
import torch.nn as nn
import torch.nn.functional as F

class DAM(nn.Module):
    def __init__(self, out_d):
        super(DAM, self).__init__()
        self.out_d = out_d
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))

    def forward(self, d5, d4, d3, d2):
        # d5 -> d4
        d5_ = self.conv1(d5)
        D4 = F.interpolate(d5_, d4.size()[2:], mode='bilinear', align_corners=True)
        b, c, h, w = D4.shape

        # 自注意力机制
        D4_ = D4.view(b, c, -1)       # (b, c, h * w)
        attention_map = torch.matmul(D4_, D4_.transpose(1, 2))  # (b, c, c)
        attention_map = F.softmax(attention_map, dim=-1)        # 归一化注意力

        d4 = torch.matmul(attention_map, D4_).view(b, c, h, w)  # 将结果转换为 (b, c, h, w)
        d4 = D4 + d4  # 残差连接

        # d4 -> d3
        d4_ = self.conv2(d4)
        D3 = F.interpolate(d4_, d3.size()[2:], mode='bilinear', align_corners=True)
        b, c, h, w = D3.shape

        D3_ = D3.view(b, c, -1)  # (b, c, h * w)
        attention_map = torch.matmul(D3_, D3_.transpose(1, 2))  # (b, c, c)
        attention_map = F.softmax(attention_map, dim=-1)        # 归一化注意力

        d3 = torch.matmul(attention_map, D3_).view(b, c, h, w)  # (b, c, h, w)
        d3 = D3 + d3  # 残差连接

        # d3 -> d2
        d3_ = self.conv3(d3)
        D2 = F.interpolate(d3_, d2.size()[2:], mode='bilinear', align_corners=True)
        b, c, h, w = D2.shape

        D2_ = D2.view(b, c, -1)  # (b, c, h * w)
        attention_map = torch.matmul(D2_, D2_.transpose(1, 2))  # (b, c, c)
        attention_map = F.softmax(attention_map, dim=-1)        # 归一化注意力

        d2 = torch.matmul(attention_map, D2_).view(b, c, h, w)  # (b, c, h, w)
        d2 = D2 + d2  # 残差连接

        return d5, d4, d3, d2



