from torchvision.models import vgg19, resnet18
import torch
from torch import nn
import torch.nn.functional as F
from .metrics import ssim_loss


class VGG19Loss(nn.Module):
    def __init__(self):
        super(VGG19Loss, self).__init__()
        self.vgg = vgg19(pretrained=True).features
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        alpha = 0.1
        return alpha * ssim_loss(x, y) + (1 - alpha) * F.mse_loss(
            self.vgg(x), self.vgg(y)
        )
