from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.upsampling import Upsample
from .partialconv2d import PartialConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MaskedSelfAttention


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


class PartialConvBN(nn.Module):
    def __init__(self, c_in, c_out, ksize, stride, batch_norm):
        super(PartialConvBN, self).__init__()
        self.c1 = PartialConv2d(
            c_in,
            c_out,
            ksize,
            stride,
            padding=ksize // 2,
            multi_channel=True,
            return_mask=True,
        )
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x, mask):
        x, mask = self.c1(x, mask)
        if self.batch_norm:
            x = self.bn(x)
        x = F.relu(x)
        return x, mask


class PartialConvUp(nn.Module):
    def __init__(self, c_in_1, c_in_2, c_out, lrelu=True, bn=True):
        super(PartialConvUp, self).__init__()
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv = PartialConv2d(c_in_1 + c_in_2, c_out, 3, 1)
        self.lrelu = lrelu
        if lrelu:
            self.lrelu = nn.LeakyReLU(0.2)
        self._bn = bn
        if bn:
            self.bn = nn.BatchNorm2d(c_in_1)

    def forward(self, x, y):
        x = self.up1(x)
        x = torch.cat((x, y), dim=1)
        x = self.conv(x)
        if self.lrelu:
            x = self.lrelu(x)
        if self._bn:
            x = self.bn(x)
        return x


class PartialSR(nn.Module):
    # a mini U-net for SR using partial convolutions, based on https://arxiv.org/pdf/1804.07723.pdf
    def __init__(self, batch_norm=True):
        super().__init__()
        self.batch_norm = batch_norm
        self.c1 = PartialConv2d(
            3, 64, 5, 2, padding=2, multi_channel=True, return_mask=True
        )
        if self.batch_norm:
            self.bn1 = BatchNorm2d(64)
        self.c2 = PartialConv2d(
            64, 128, 3, 2, padding=1, multi_channel=True, return_mask=True
        )
        if self.batch_norm:
            self.bn2 = BatchNorm2d(128)
        self.c3 = PartialConv2d(
            128, 128, 3, 1, padding=1, multi_channel=True, return_mask=True
        )
        if self.batch_norm:
            self.bn3 = BatchNorm2d(128)

        self.up4 = Upsample(scale_factor=2)
        self.c4 = PartialConv2d(
            128 + 64, 64, 3, 1, padding=1, multi_channel=True, return_mask=True
        )
        self.up5 = Upsample(scale_factor=2)
        self.c5 = PartialConv2d(
            64 + 3, 3, 3, 1, padding=1, multi_channel=True, return_mask=True
        )

    def forward(self, x, mask):
        # input shape (B, 3, H, W)
        p1, mask = self.c1(x, mask)
        if self.batch_norm:
            p1 = self.bn1(p1)
        p1 = F.relu(p1)
        # print(p1.shape)  # (B, 64, H/2, W/2)

        p2, mask = self.c2(p1, mask)
        if self.batch_norm:
            p2 = self.bn2(p2)
        p2 = F.relu(p2)
        # print(p2.shape)  # (B, 128, H/4, W/4)

        p3, mask = self.c3(p2, mask)
        if self.batch_norm:
            p3 = self.bn3(p3)
        p3 = F.relu(p3)
        # print(p3.shape)  # (B, 128, H/4, W/4)

        p4 = self.up4(p3)  # (B, 128, H/2, W/2)
        p4 = torch.cat((p4, p1), dim=1)
        p4, mask = self.c4(p4)
        p4 = F.leaky_relu(p4, 0.2)
        # print(p4.shape)

        p5 = self.up5(p4)
        p5 = torch.cat((p5, x), dim=1)
        p5, mask = self.c5(p5)
        # print(p5.shape)

        return p5


class ConvSR(nn.Module):
    # a mini U-net for SR without partial convolutions, based on previous
    def __init__(self, batch_norm=True):
        super().__init__()
        self.batch_norm = batch_norm
        self.c1 = Conv2d(3, 64, 5, 2, padding=2)
        if self.batch_norm:
            self.bn1 = BatchNorm2d(64)
        self.c2 = Conv2d(64, 128, 3, 2, padding=1)
        if self.batch_norm:
            self.bn2 = BatchNorm2d(128)
        self.c3 = Conv2d(128, 128, 3, 1, padding=1)
        if self.batch_norm:
            self.bn3 = BatchNorm2d(128)

        self.up4 = Upsample(scale_factor=2)
        self.c4 = Conv2d(128 + 64, 64, 3, 1, padding=1)
        self.up5 = Upsample(scale_factor=2)
        self.c5 = Conv2d(64 + 3, 3, 3, 1, padding=1)

    def forward(self, x):
        # input shape (B, 3, H, W)
        p1 = self.c1(x)
        if self.batch_norm:
            p1 = self.bn1(p1)
        p1 = F.relu(p1)
        # print(p1.shape)  # (B, 64, H/2, W/2)

        p2 = self.c2(p1)
        if self.batch_norm:
            p2 = self.bn2(p2)
        p2 = F.relu(p2)
        # print(p2.shape)  # (B, 128, H/4, W/4)

        p3 = self.c3(p2)
        if self.batch_norm:
            p3 = self.bn3(p3)
        p3 = F.relu(p3)
        # print(p3.shape)  # (B, 128, H/4, W/4)

        p4 = self.up4(p3)  # (B, 128, H/2, W/2)
        p4 = torch.cat((p4, p1), dim=1)
        p4 = self.c4(p4)
        p4 = F.leaky_relu(p4, 0.2)
        # print(p4.shape)

        p5 = self.up5(p4)
        p5 = torch.cat((p5, x), dim=1)
        p5 = self.c5(p5)
        # print(p5.shape)

        return p5