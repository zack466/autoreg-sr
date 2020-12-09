import torch
import PIL
import torchvision
import numpy as np
from matplotlib import pyplot as plt


def showImage(img):
    # expects ndarray with shape CHW

    # PIL not working with ndarray image for some reason
    # torchvision.transforms.ToPILImage()((img * 256).astype('uint8')).show()
    # torchvision.transforms.ToPILImage()(img).show()

    if 0.0 < img[0][0][0] < 1.0:  # hacky way of checking for float32
        plt.imshow((img * 256).astype("uint8").transpose(2, 1, 0))


def showImages(imgs):
    fig = plt.figure()
    for i, image in enumerate(imgs):
        fig.add_subplot(1, len(imgs), i + 1)
        plt.imshow((image * 256).astype("uint8").transpose(2, 1, 0))
    plt.show()


def example_batch(example, batch_size):
    # expands an example tensor (C,H,W) to a full batch (B,C,H,W)
    return torch.cat([example.unsqueeze(0)] * batch_size)


# taken from https://github.com/pytorch/examples/blob/1de2ff9338bacaaffa123d03ce53d7522d5dcc2e/imagenet/main.py
# license information in LICENSES file
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)