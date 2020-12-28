import torch
import numpy as np
from .pytorch_ssim import ssim


def psnr(lr, hr):
    return -20 * torch.log10(torch.sqrt(torch.mean((hr - lr) ** 2)))


def psnr_loss(lr, hr):
    return 20 * torch.log10(torch.sqrt(torch.mean((hr - lr) ** 2)))


def consistency(pred, original):
    downscaled = torch.resize(pred, original.shape)
    return torch.nn.functional.mse_loss(original, downscaled).item()


def ssim_loss(lr, hr):
    return 1 - ssim(lr, hr)
