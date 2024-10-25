import torch
import torch.nn as nn
from einops import rearrange

# class DEAM(nn.Module):
#     def __init__(self, in_d):
#        super(DEAM, self).__init__()
#
#        self.in_d = in_d
#        self.out_d = in_d*3
#
#        self.conv1 = nn.Sequential(
#                     nn.Conv2d(self.in_d, self.out_d, kernel_size=1, stride=1),
#                     nn.BatchNorm2d(self.out_d),
#                     nn.ReLU(inplace=False)
#        )
#
#     def forward(self, f_map, e_map):
#         _, _, he, we = f_map.shape
#
#         e_map = self.conv1(e_map)
#         q, k, v = torch.chunk(e_map, dim=1, chunks=3)
#         e_map_q = rearrange(q, 'b c h w -> b c (h w)')
#         e_map_k = rearrange(k, 'b c h w -> b (h w) c')
#
#         f_map_ = self.conv1(f_map)
#         _, _, V = torch.chunk(f_map_, dim=1, chunks=3)
#         f_map_v = rearrange(V, 'b c h w -> b c (h w)')
#
#         dot = torch.matmul(e_map_q, e_map_k)
#         att = torch.softmax(dot, dim=-1)
#         out1 = torch.matmul(att, f_map_v)
#         Out = rearrange(out1, 'b c (h w) ->b c h w', h=he, w=we)
#         out = Out+f_map
#
#         return out


class DilatedAttention(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=2):
        super(DilatedAttention, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=dilation * (kernel_size - 1) // 2)

    def forward(self, q, k, v, h, w):
        B, dim, N = q.shape
        k = self.unfold(k.contiguous().view(B, dim, h, w)).reshape(B, dim, self.kernel_size ** 2, N)
        v = self.unfold(v.contiguous().view(B, dim, h, w)).reshape(B, dim, self.kernel_size ** 2, N)

        # 使用一致的形状计算注意力
        attn = (q.unsqueeze(-2) * k).sum(dim=1) / (self.dim ** 0.5)
        attn = attn.softmax(dim=-2)

        out = (attn.unsqueeze(1) * v).sum(dim=-2)
        return out
class DEAM(nn.Module):
    def __init__(self, in_d, kernel_size=3, dilation=2):
       super(DEAM, self).__init__()

       self.in_d = in_d
       self.out_d = in_d * 3
       self.dilated_attention = DilatedAttention(in_d, kernel_size, dilation)

       self.conv1 = nn.Sequential(
           nn.Conv2d(self.in_d, self.out_d, kernel_size=1, stride=1),
           nn.BatchNorm2d(self.out_d),
           nn.ReLU(inplace=False)
       )

       self.fc_out = nn.Conv2d(in_d, in_d, kernel_size=1, stride=1)

    def forward(self, f_map, e_map):
        b, _, he, we = f_map.shape

        e_map = self.conv1(e_map)
        q, k, v = torch.chunk(e_map, dim=1, chunks=3)
        e_map_q = rearrange(q, 'b c h w -> b c (h w)')
        e_map_k = rearrange(k, 'b c h w -> b c (h w)')
        e_map_v = rearrange(v, 'b c h w -> b c (h w)')

        f_map_ = self.conv1(f_map)
        _, _, V = torch.chunk(f_map_, dim=1, chunks=3)
        f_map_v = rearrange(V, 'b c h w -> b c (h w)')

        # 使用膨胀注意力
        out1 = self.dilated_attention(e_map_q, e_map_k, e_map_v, he, we)

        # 还原形状并融合
        Out = rearrange(out1, 'b c (h w) -> b c h w', h=he, w=we)
        out = self.fc_out(Out + f_map)

        return out
