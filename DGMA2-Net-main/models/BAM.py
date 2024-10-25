import torch
import torch.nn as nn
from einops import rearrange

class DEAM(nn.Module):

    def __init__(self, in_d, num_heads=8):
        super(DEAM, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_d // num_heads  # 每个头的通道数
        assert in_d % num_heads == 0, "in_d 必须能被 num_heads 整除"

        # 线性变换生成查询、键和值
        self.qkv_conv = nn.Sequential(
            nn.Conv2d(in_d, in_d * 3, 1, 1),  # 同时生成 Q, K, V
            nn.BatchNorm2d(in_d * 3),
            nn.ReLU(inplace=True)
        )

        # 输出的线性变换，用于恢复通道数
        self.fc_out = nn.Conv2d(in_d, in_d, 1, 1)

    def forward(self, f_map, e_map):
        b, c, h, w = f_map.shape

        # 生成查询、键和值
        qkv = self.qkv_conv(e_map)  # 生成 Q, K, V
        q, k, v = torch.chunk(qkv, 3, dim=1)  # 分割为 q, k, v
        QKV = self.qkv_conv(f_map)
        Q, K, V = torch.chunk(QKV, 3, dim=1)
        # 进行多头划分
        q = rearrange(q, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.num_heads)
        k = rearrange(k, 'b (heads head_dim) h w -> b heads (h w) head_dim', heads=self.num_heads)
        #v = rearrange(v, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.num_heads)
        V = rearrange(V, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.num_heads)

        # 计算每个头的点积注意力
        dots = torch.matmul(q, k) / (self.head_dim ** 0.5)  # 缩放点积
        attn = dots.softmax(dim=-1)

        # 使用注意力加权值
        out = torch.matmul(attn, V)

        # 将多头输出合并
        out = rearrange(out, 'b heads head_dim (h w) -> b (heads head_dim) h w', h=h, w=w)

        # 恢复通道数
        out = self.fc_out(out)

        return out
