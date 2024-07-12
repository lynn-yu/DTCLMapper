import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18,resnet34
import numpy as np
import torch.nn.functional as F
from projects.mmdet3d_plugin.maptr.modules.resnet import conv1x1,conv3x3
from projects.mmdet3d_plugin.maptr.modules.resnet import BasicBlock
import os
import torch
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        #self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',align_corners=True)
        self.up = nn.Upsample(size=(100,50), mode='bilinear',align_corners=True)
        #self.up = nn.Upsample(size=(50, 100), mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # self.offset = nn.Conv2d(out_channels, 18, kernel_size=3, padding=1, bias=False)
        # self.deconv =DeformConv2D(out_channels, out_channels, kernel_size=3, padding=1)
        # self.bn4 = nn.BatchNorm2d(out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        x3 = self.conv(x1)
        # offsets = self.offset(x3)
        # x4 = F.relu(self.deconv(x3, offsets))
        # x5 = self.bn4(x4)
        return x3

class CamEncode(nn.Module):
    def __init__(self, C):
        super(CamEncode, self).__init__()
        self.C = C
        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        #self.trunk = resnet34(pretrained=False, zero_init_residual=True)
        self.up1 = Up(320+112, self.C)
        # self.up2 = Up(40 + 64, self.C)
        # self.up3 = Up(24 + 64, self.C)
    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()
        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)  # 卷积过程
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x
        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
        # 1 24 16 64 176
        # 2 24 24 32 88
        # 3 24 40 16 44
        # 4 24 112 8 22
        # 5 24 320 4 11
        # out = []
        # out.append(self.up1(endpoints['reduction_5'], endpoints['reduction_4']))   # 24 64 8 22
        # out.append(self.up2(x, endpoints['reduction_3']))  # 24 64 8 22
        # out.append(self.up3(x1, endpoints['reduction_2']))  # 24 64 8 22
        return self.up1(endpoints['reduction_5'], endpoints['reduction_4'])

    def forward(self, x):
        return self.get_eff_depth(x)
class BevEncode(nn.Module):
    def __init__(self, inC, outC,seg_out=False, instance_seg=False, embedded_dim=50, many_instance_seg=False, many_instance_dim=50*6):
        super(BevEncode, self).__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3
        self.seg_out = seg_out
        if seg_out:
            self.up1 = Up(64 + 256, 256, scale_factor=4)
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, outC, kernel_size=1, padding=0),
            )
        self.instance_seg = instance_seg
        if instance_seg:
            self.up1_embedded = Up(64 + 256, 256, scale_factor=4)
            self.up3 =  nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True)
            self.up2_embedded = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, embedded_dim, kernel_size=1, padding=0),
            )
        # self.conv = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
        #         nn.BatchNorm2d(128),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(128, embedded_dim, kernel_size=1, padding=0),
        #                           )
        self.many_instance_seg = many_instance_seg
        if many_instance_seg:
            # self.m1 = Up(64 + 256, 256, scale_factor=4)
            # self.m2 = nn.Upsample(scale_factor=2, mode='bilinear',
            #                 align_corners=True)
            self.up3_embedded = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, many_instance_dim, kernel_size=1, padding=0),
            )
    def forward(self, x):
        x = self.conv1(x)#
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)#64 100 50
        x = self.layer2(x1)#128 50 25
        x2 = self.layer3(x)#256 25 13
        if self.seg_out:
            x = self.up1(x2, x1) #128 100 50
            x = self.up2(x)
        if self.instance_seg:
            x_embedded_fea_1 = self.up1_embedded(x2, x1)
            x_embedded_fea = self.up3(x_embedded_fea_1)#4 256 200 100
            x_embedded = self.up2_embedded(x_embedded_fea)

        else:
            x_embedded_fea = None
            x_embedded = None
        if self.many_instance_seg:
            x_embedded_many = self.up3_embedded(x_embedded_fea)
            return x, x_embedded, x_embedded_fea, x_embedded_many
        else:
            return x, x_embedded, x_embedded_fea


class CLS_Head(nn.Module):
    def __init__(self, inC, embedded_dim=16):
        super(CLS_Head, self).__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1_embedded = Up(64 + 256, 256, scale_factor=4)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=True)
        self.up2_embedded = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, embedded_dim, kernel_size=1, padding=0),
        )
        # self.conv = nn.Sequential(nn.Conv2d(embedded_dim, embedded_dim, kernel_size=3, padding=1, bias=False),
        #                           nn.BatchNorm2d(inC),
        #                           nn.ReLU(inplace=True),
        #                           nn.Conv2d(inC, inC, kernel_size=1, padding=1, bias=False),
        #                           nn.BatchNorm2d(inC),
        #                           )
    def forward(self, x):
        x = self.conv1(x)  #
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)  # 64 100 50
        x = self.layer2(x1)  # 128 50 25
        x2 = self.layer3(x)  # 256 25 13
        x_embedded_fea_1 = self.up1_embedded(x2, x1)
        x_embedded_fea = self.up3(x_embedded_fea_1)  # 4 256 200 100
        x_embedded = self.up2_embedded(x_embedded_fea)
        return x_embedded, x_embedded_fea

class CLS_HeadV2(nn.Module):
    def __init__(self, inC, embedded_dim=16):
        super(CLS_HeadV2, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(inC, inC, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(inC),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(inC, inC, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(inC),
                                  )
        self.relu = nn.ReLU(inplace=True)
        self.cls_head = nn.Sequential(
            nn.Conv2d(inC, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, embedded_dim, kernel_size=1, padding=0)
        )
    def forward(self, x):

       embedding = self.relu(self.conv(x)+x)
       cls = self.cls_head(embedding)

       return cls,embedding
class BevEncode_v1(nn.Module):
    def __init__(self, inC, outC,seg_out=False, instance_seg=False, embedded_dim=16, direction_pred=True, direction_dim=37):
        super(BevEncode_v1, self).__init__()

        self.seg_head = nn.Sequential(
            nn.Conv2d(inC, inC, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC, outC, kernel_size=1, padding=0)
        )
        self.conv = nn.Conv2d(inC, inC, kernel_size=3, padding=1, bias=False)

        self.cls_head = nn.Sequential(
            nn.Conv2d(inC, inC, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC, embedded_dim, kernel_size=1, padding=0)
        )

        self.weight = nn.Conv2d(embedded_dim, embedded_dim, kernel_size=1, padding=0,)

    def forward(self, x):
       seg = self.seg_head(x)
       embedding = self.conv(x)
       cls = self.cls_head(embedding)
       weight = self.weight(cls)
       return seg,cls,embedding,weight


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict
if __name__ == '__main__':
    stn = BevEncode(inC=256, outC=4)
    x = torch.rand((4,256,400,200))
    y = stn(x)
    print(y.shape)