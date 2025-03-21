import torch.nn.functional as F
import torch
import torch.nn as nn

class HLFE(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."
        head_dim = dim // num_heads  #64
        self.dim = dim  #512

        self.l_heads = int(num_heads * alpha) #4
        self.h_heads = num_heads - self.l_heads #4
        self.l_dim = self.l_heads * head_dim   #256
        self.h_dim = self.h_heads * head_dim   #256
        self.ws = window_size  #2
        self.scale = qk_scale or head_dim ** -0.5

        # Lo-Fi attention layers
        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.l_q = nn.Conv2d(dim, self.l_dim, 1, bias=qkv_bias)
            self.l_kv = nn.Conv2d(dim, self.l_dim, 1, bias=qkv_bias)
            self.l_proj = nn.Conv2d(self.l_dim, self.l_dim, 1)
            # self.l_proj = nn.Conv2d(self.l_dim, self.l_dim * 2, 1)
        # Hi-Fi attention layers
        if self.h_heads > 0:
            self.h_qkv = nn.Conv2d(dim, self.h_dim * 3, 1, bias=qkv_bias)
            self.h_proj = nn.Conv2d(self.h_dim, self.h_dim, 1)
            # self.h_proj = nn.Conv2d(self.h_dim, self.h_dim * 2, 1)
    def lofi(self, x):
        B, C, H, W = x.shape  # (32,512,8,8)
        # q = self.l_q(x).reshape(B, self.l_heads, self.l_dim // self.l_heads, H * W).permute(0, 1, 3, 2)
        q = self.l_q(x)  # (32,256,8,8)
        q = F.interpolate(q, scale_factor=0.5, mode='bilinear', align_corners=True)
        kv = self.l_kv(self.sr(x) if self.ws > 1 else x)
        # Calculate attention
        attn = (q @ kv.transpose(-2, -1)) * self.scale
        # Step 2: 应用注意力权重到 kv（值）上
        # 使用 attn 矩阵与 kv 相乘，得到加权后的输出
        x = attn @ kv
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        # Step 3: 调整 x 的形状，方便后续的卷积操作
        # 将 x 重塑为 (B, self.l_heads, H, W, self.l_dim // self.l_heads)
        x = x.reshape(B, self.l_heads, H, W, self.l_dim // self.l_heads)
        # Step 4: 调整 x 的维度顺序
        # 将 x 的维度顺序调整为 (B, self.l_heads, self.l_dim // self.l_heads, H, W)，以匹配输出要求
        x = x.permute(0, 1, 4, 2, 3)
        x = x.reshape(B, -1, H, W)  # 合并 h_heads 和 head_dim 维度 -> (B, 256, H, W)
        return self.l_proj(x)

    def hifi(self, x):
        B, C, H, W = x.shape  #(32,512,8,8)
        h_group, w_group = H // self.ws, W // self.ws
        # Reshape x to match the dimensions required for qkv computation
        # x = x.reshape(B, self.h_heads, self.h_dim // self.h_heads, H, W).transpose(2, 3)
        qkv = self.h_qkv(x).reshape(B, 3, self.h_heads, h_group * w_group, self.ws, self.ws, self.h_dim // self.h_heads)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        # Calculate attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # 计算结果
        # x 应为 (B, self.h_heads, H * W, self.h_dim // self.h_heads)
        # Step 5: 将 x 重塑为 (B, self.h_heads, H, W, self.h_dim // self.h_heads)
        x = x.reshape(B, self.h_heads, H, W, self.h_dim // self.h_heads)
        # Step 6: 调整维度顺序
        x = x.permute(0, 1, 4, 2, 3)  # 调整到 (B, self.h_heads, self.h_dim // self.h_heads, H, W)

        x = x.reshape(B, -1, H, W)  # 合并 h_heads 和 head_dim 维度 -> (B, 256, H, W)
        # 使用 h_proj 处理输出
        return self.h_proj(x) #(B, 256, H, W)


    def forward(self, x):

        if self.h_heads == 0:
            return self.lofi(x)
        if self.l_heads == 0:
            return self.hifi(x)

        hifi_out = self.hifi(x)
        lofi_out = self.lofi(x)
        return torch.cat((hifi_out, lofi_out), dim=1)
