import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import resnet18
from module.MSEM import MSEM
from models.decoder_new import ChangeInformationExtractionModule, GuidedRefinementModule_concat, Decoder
from module.DEM import DEM
from module.HLFE import HLFE
from module.CGI import CGAFusion

class MSDIG_DTFENet(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(MSDIG_DTFENet, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.mid_d = 64

        self.DEM5 = DEM(in_d=512, out_d=512)
        self.DEM4 = DEM(in_d=256, out_d=256)
        self.DEM3 = DEM(in_d=128, out_d=128)
        self.DEM2 = DEM(in_d=64, out_d=64)

        self.HLFE5 = HLFE(dim=512, num_heads=8)
        self.HLFE4 = HLFE(dim=256, num_heads=8)
        self.HLFE3 = HLFE(dim=128, num_heads=8)
        self.HLFE2 = HLFE(dim=64, num_heads=8)

        self.MSEM5 = MSEM(dim_in=512, dim_out=512)
        self.MSEM4 = MSEM(dim_in=256, dim_out=256)
        self.MSEM3 = MSEM(dim_in=128, dim_out=128)
        self.MSEM2 = MSEM(dim_in=64, dim_out=64)

        self.CGI5 = CGAFusion(dim=512)
        self.CGI4 = CGAFusion(dim=256)
        self.CGI3 = CGAFusion(dim=128)
        self.CGI2 = CGAFusion(dim=64)

        #DAM1
        self.CIEM1 = ChangeInformationExtractionModule(64, 64)
        self.GRM1 = GuidedRefinementModule_concat(out_d=[512, 256, 128, 64])   ######################[512, 256, 128, 64]
        #DAM2
        self.CIEM2 = ChangeInformationExtractionModule(64, 64)
        self.GRM2 = GuidedRefinementModule_concat(out_d=[512, 256, 128, 64])   ###################

        self.decoder = Decoder(self.mid_d, output_nc)


    def forward(self, x1, x2):
        # forward backbone resnet
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone.base_forward(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone.base_forward(x2)

        E5 = self.DEM5(x1_5, x2_5)  # 1/32
        E4 = self.DEM4(x1_4, x2_4)  # 1/16
        E3 = self.DEM3(x1_3, x2_3)  # 1/8
        E2 = self.DEM2(x1_2, x2_2)  # 1/4

        H5 = self.HLFE5(E5)
        H4 = self.HLFE4(E4)
        H3 = self.HLFE3(E3)
        H2 = self.HLFE2(E2)

        D5 = self.MSEM5(E5)  # 1/32
        D4 = self.MSEM4(E4)  # 1/16
        D3 = self.MSEM3(E3)  # 1/8
        D2 = self.MSEM2(E2)  # 1/4

        d5 = self.CGI5(x1_5, x2_5, D5, H5)
        d4 = self.CGI4(x1_4, x2_4, D4, H4)
        d3 = self.CGI3(x1_3, x2_3, D3, H3)
        d2 = self.CGI2(x1_2, x2_2, D2, H2)


        # DAM1
        d5_p, d4_p, d3_p, d2_p = self.CIEM1(d5, d4, d3, d2)
        d5, d4, d3, d2 = self.GRM1(d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p)
        ##DAM2
        d5_p, d4_p, d3_p, d2_p = self.CIEM2(d5, d4, d3, d2)
        d5, d4, d3, d2 = self.GRM2(d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p)

        mask = self.decoder(d5, d4, d3, d2, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5)

        mask = F.interpolate(mask, x1.size()[2:], mode='bilinear', align_corners=True)
        mask = torch.sigmoid(mask)

        return mask

if __name__ == '__main__':
    x1 = torch.randn((4, 3, 256, 256)).cuda()
    x2 = torch.randn((4, 3, 256, 256)).cuda()
    model = MSDIG_DTFENet(3, 1).cuda()
    out = model(x1, x2)
    print(out.shape)
