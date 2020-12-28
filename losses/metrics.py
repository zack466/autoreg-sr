import torch
import numpy as np
from torch.nn.functional import l1_loss, mse_loss
from .pytorch_ssim import ssim
from .pytorch_msssim import ms_ssim


def psnr(lr, hr):
    return -20 * torch.log10(torch.sqrt(torch.mean((hr - lr) ** 2)))


def psnr_loss(lr, hr):
    return 20 * torch.log10(torch.sqrt(torch.mean((hr - lr) ** 2)))


def consistency(pred, original):
    downscaled = torch.resize(pred, original.shape)
    return torch.nn.functional.mse_loss(original, downscaled).item()


def ssim_loss(lr, hr):
    return 1 - ssim(lr, hr)


def ms_ssim_l1(lr, hr):
    # mixed loss from https://research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/NN_ImgProc.pdf
    alpha = 0.84
    return alpha * ms_ssim(lr, hr) + (1 - alpha) * l1_loss(lr, hr)


def ms_ssim_l2(lr, hr):
    # mixed loss from https://research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/NN_ImgProc.pdf
    alpha = 0.84
    return alpha * ms_ssim(lr, hr) + (1 - alpha) * mse_loss(lr, hr)