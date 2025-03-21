import torch
import torch.nn as nn

def calc_men_std(feat,eps =1e-5):
    size = feat.size()
    assert (len(size)==4)
    N,C,H,W=size[:4]
    feat_var = feat.var(dim=1)+eps
    feat_var =feat_var.sqrt().view(N,1,H,W)
    feat_mean = feat.mean(dim=1).view(N,1,H,W)
    return feat_mean,feat_var


class DEM(nn.Module):
    def __init__(self, in_d, out_d):
        super(DEM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_d * 2, out_channels=in_d, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(in_channels=in_d, out_channels=out_d, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_d),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_d * 2, out_channels=in_d, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(in_channels=in_d, out_channels=out_d, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_d),
            nn.ReLU()
        )

    def forward(self, T1, T2):
        T1_mean, T1_var = calc_men_std(T1)
        T2_mean, T2_var = calc_men_std(T2)

        # 合并 T1 和 T2
        T = torch.cat((T1, T2), dim=1)

        # 通过第一个卷积层
        x = self.conv1(T)

        # 计算均值和方差的差异
        T_mean = T1_mean - T2_mean
        T_var = T1_var - T2_var

        # 进行加权
        x_mean = x * T_mean
        x_var = x * T_var

        # 合并加权后的均值和方差
        X = torch.cat((x_mean, x_var), dim=1)

        # 通过第二个卷积层
        X = self.conv2(X)

        return X


