import time
import utils

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import h5py
import os
import numpy as np

class Div2KTiny(Dataset):
    def __init__(self, device):
        ## super init??
        self.data_path = os.path.dirname(__file__)+"/saved/div2k/train-x2/"
        self.trainsets = os.listdir(self.data_path)[:-1]
        self.device = device
        self.current = (None, None) # pointer to images and labels
        self.load_set(0)

    def __len__(self):
        return len(self.trainsets) * 4096

    def load_set(self, set):
        self.set_number = set
        with h5py.File(self.data_path + self.trainsets[set], 'r') as f:
             # for some reason h5 --> np.array --> torch.tensor is much faster than h5 --> torch.tensor
             images = torch.tensor(np.array(f['images'])).permute(0,3,1,2).to(self.device)
             labels = torch.tensor(np.array(f['labels'])).permute(0,3,1,2).to(self.device)
             self.current = (images, labels)

    def __getitem__(self, idx):
        if idx // 4096 == self.set_number: # should already be open
            pass
        else:
            self.load_set(idx // 4096)
        sample = {'image':self.current[0][idx], 'label':self.current[1][idx]}
        return sample

class Div2K(Dataset):
    def __init__(self, size, factor):
        """
        size: pixel dimensions of image to be upscaled
        factor: upscaling/downscaling factor, options: 2, 3, 4
        """
        self.size = size
        self.factor = factor

    def __len__(self):
        return 800
    
    def __getitem__(self, idx):
        img_hr_name = "./datasets/saved/DIV2K_train_HR/" + str(idx+1).zfill(4) + ".png"
        img_lr_name = f"./datasets/saved/DIV2K_train_LR_bicubic/X{self.factor}/" + str(idx+1).zfill(4) + f"x{self.factor}.png"
        # C,H,W
        img_hr = Image.open(img_hr_name)
        img_lr = Image.open(img_lr_name)
        hr_transform = transforms.Compose([
            transforms.CenterCrop(self.size * self.factor),
            transforms.ToTensor()
        ])
        lr_transform = transforms.Compose([
            transforms.CenterCrop(self.size),
            transforms.ToTensor()
        ])
        sample = {'lr': lr_transform(img_lr), 'hr': hr_transform(img_hr)}

        return sample
    
    def test(self):
        t0 = time.time()
        for idx in range(1):
            img_hr_name = "./datasets/saved/DIV2K_train_HR/" + str(idx+1).zfill(4) + ".png"
            img_lr_name = f"./datasets/saved/DIV2K_train_LR_bicubic/X{self.factor}/" + str(idx+1).zfill(4) + f"x{self.factor}.png"
            # C,H,W
            img_hr = transforms.functional.to_tensor(Image.open(img_hr_name))
            img_lr = transforms.functional.to_tensor(Image.open(img_lr_name))
        t1 = time.time()
        return t1 - t0
    
    # TODO: write cache function to convert images to hdf5 files, allow dataset to simply load from the h5
    # bc converting png to tensor is slow (100-150 ms to load a hi-rez image)
    