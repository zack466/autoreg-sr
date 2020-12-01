import torch
import PIL
import torchvision

def showImage(img):
    torchvision.transforms.ToPILImage()(img).show()