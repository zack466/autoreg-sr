#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torchvision
import torchsummary
import PIL
from matplotlib import pyplot as plt
import numpy as np

import utils
from utils import AverageMeter
import datasets
from models import SRCNN
from models import PartialConv2d
from models import PConvSR
from losses import VGG16PartialLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[2]:


batch_size = 32
up_factor = 4
lr_res = 16
data_mult = 4
hr_res = lr_res * up_factor
div2k = datasets.Div2K(lr_res, up_factor, data_mult) # lr size, factor
training_set, validation_set, test_set = torch.utils.data.random_split(div2k, [640*data_mult,80*data_mult,80*data_mult])
lr, hr = div2k[6]
print(lr.shape)
print(hr.shape)
utils.showImages([lr, hr])


# In[ ]:





# In[3]:


# x = torch.rand((16, 3, 64, 64)).to(device)
# mask_in = torch.zeros((16, 3, 64, 64)).to(device)
# mask_in[0,0,0,0] = 1
# model(x, mask_in)

def image_mask(x, factor):
    new_dims = tuple([x.shape[0], x.shape[1], x.shape[2]*factor, x.shape[3]*factor])
    out = torch.zeros(new_dims)
    out[:,:,::factor,::factor] = x[:,:,:,:]
    mask = torch.zeros(new_dims)
    mask[:,:,::factor,::factor] = 1
    return out, mask

# x = torch.rand((32, 3, 64, 64))
# image_mask(x, 2)


# In[4]:


training_dataloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size)
validation_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

model = PConvSR().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_func = VGG16PartialLoss().to(device)

torchsummary.summary(model, input_size=[(3, 64, 64), (3, 64, 64)])


# In[5]:


# basic training loop
for epoch in range(4):
    for i, batch in enumerate(training_dataloader):
        # TODO make sure ops happen on GPU, unless the speed isn't affected
        optimizer.zero_grad()

        lr, hr = batch
        lr, hr = lr.to(device), hr.to(device)
        with torch.no_grad():
            upscaled, mask_in = image_mask(lr, up_factor)

        pred, mask_out = model(upscaled.to(device), mask_in.to(device))
        loss, _, _ = loss_func(pred, hr) # VGG style loss

        loss.backward()

        optimizer.step()

        # print(i, loss)
    # basic validation loop
    print(f"Epoch {epoch}")
    validation_loss = ("validation")
    for i, batch in enumerate(validation_dataloader):
        with torch.no_grad():
            lr, hr = batch
            lr, hr = lr.to(device), hr.to(device)
            with torch.no_grad():
                upscaled, mask_in = image_mask(lr, up_factor)

            pred, mask_out = model(upscaled.to(device), mask_in.to(device))
            loss, _, _ = loss_func(pred, hr)
            validation_loss.update(loss.item())
            print(validation_loss)


# In[6]:


# basic testing loop
print("Testing")
test_loss = AverageMeter("testing")
for i, batch in enumerate(test_dataloader):
    with torch.no_grad():
        lr, hr = batch
        lr, hr = lr.to(device), hr.to(device)
        with torch.no_grad():
            upscaled, mask_in = image_mask(lr, up_factor)

        pred, mask_out = model(upscaled.to(device), mask_in.to(device))
        loss, _, _ = loss_func(pred, hr)
        test_loss.update(loss.item())
        print(test_loss)


# In[7]:


for i in range(32,48):
    with torch.no_grad():
        lr, hr = test_set[i]
        batch = utils.example_batch(torch.tensor(lr).to(device), batch_size)
        upscaled, mask_in = image_mask(batch, 4)
        pred, _ = model(upscaled.to(device), mask_in.to(device))
        bicubic = torchvision.transforms.functional.resize(torch.tensor(lr), (64,64))
        utils.showImages([np.asarray(lr), np.array(upscaled[0].cpu()), np.array(bicubic), np.array(pred[0].cpu()), hr])


# In[ ]:




