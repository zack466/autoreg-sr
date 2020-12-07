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

    if 0. < img[0][0][0] < 1.: # hacky way of checking for float32
        plt.imshow((img* 256).astype('uint8').transpose(2,1,0))

def showImages(imgs):
    fig = plt.figure()
    for i, image in enumerate(imgs):
        fig.add_subplot(1, len(imgs), i+1)
        plt.imshow((image* 256).astype('uint8').transpose(2,1,0))
    plt.show()
