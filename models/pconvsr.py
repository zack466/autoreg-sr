from .partialconv2d import PartialConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F


class PConvSR(nn.Module):
    # based on SRCNN architecture, but using PartialConv instead
    def __init__(self):
        super(PConvSR, self).__init__()
        self.conv_1 = PartialConv2d(
            3, 64, 9, padding=4, multi_channel=True, return_mask=True
        )
        self.conv_2 = PartialConv2d(
            64, 32, 5, padding=2, multi_channel=True, return_mask=True
        )
        self.conv_3 = PartialConv2d(
            32, 3, 5, padding=2, multi_channel=True, return_mask=True
        )

    def forward(self, x, mask):
        x, mask = self.conv_1(x, mask)
        x = F.relu(x)
        x, mask = self.conv_2(x, mask)
        x = F.relu(x)
        x, mask = self.conv_3(x, mask)
        x = F.relu(x)
        return x, mask


class PConvResidual(nn.Module):
    def __init__(self, channels):
        super(PConvResidual, self).__init__()
        self.c1 = PartialConv2d(
            channels, channels, 3, 1, 1, multi_channel=True, return_mask=True
        )
        self.c2 = PartialConv2d(
            channels, channels, 3, 1, 1, multi_channel=True, return_mask=True
        )

    def forward(self, x, mask):
        y, mask = self.c1(x, mask)
        y = F.relu(y)
        y, mask = self.c2(y, mask)
        x = F.relu(y + x)
        return x, mask


class PConvResNet(nn.Module):
    # based on resnet
    def __init__(self):
        super(PConvResNet, self).__init__()
        self.first = PartialConv2d(3, 32, 3, 1, 1, multi_channel=True, return_mask=True)
        self.c1 = PConvResidual(32)
        self.c2 = PConvResidual(32)
        self.c3 = PConvResidual(32)
        self.last = PartialConv2d(32, 3, 3, 1, 1, multi_channel=True, return_mask=True)

    def forward(self, x, mask):
        x, mask = self.first(x, mask)
        x, mask = self.c1(x, mask)
        x, mask = self.c2(x, mask)
        x, mask = self.c3(x, mask)
        x, mask = self.last(x, mask)
        return x, mask