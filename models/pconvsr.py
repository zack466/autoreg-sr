from .partialconv2d import PartialConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F


class PConvSR(nn.Module):
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