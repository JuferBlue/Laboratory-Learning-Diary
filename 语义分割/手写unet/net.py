"""
@author:JuferBlue
@file:net.py.py
@date:2024/9/22 8:07
@description:
"""
import torch
from torch import nn
from torch.nn import functional as F
class Conv_Block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,stride=1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,padding=1,stride=2,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        return self.layer(x)

class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.c1 = Conv_Block(3,64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64,128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128,256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256,512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512,1024)
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024,512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512,256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256,128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128,64)
        self.out = nn.Conv2d(64,21,3,1,1)


    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        R6 = self.c6(self.u1(R5,R4))
        R7 = self.c7(self.u2(R6,R3))
        R8 = self.c8(self.u3(R7,R2))
        R9 = self.c9(self.u4(R8,R1))
        return self.out(R9)


if __name__ == '__main__':
    x = torch.randn(2,3,512,512)
    net = UNet()
    print(net(x).shape)