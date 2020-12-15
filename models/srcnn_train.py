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
import datasets
from models import SRCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[2]:


batch_size = 32
div2k = datasets.Div2K(32, 2)
training_set, validation_set, test_set = torch.utils.data.random_split(div2k, [640,80,80])
# lr, hr = div2k[0]
# print(lr.shape)
# utils.showImages([lr, hr])


# In[3]:


training_dataloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size)
validation_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

model = SRCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
loss_func = torch.nn.MSELoss()

torchsummary.summary(model, input_size=(3, 64, 64))


# In[4]:


# basic training loop
for epoch in range(16):
    for i, batch in enumerate(training_dataloader):
        optimizer.zero_grad()

        lr, hr = batch
        lr, hr = lr.to(device), hr.to(device)
        with torch.no_grad():
            upscaled = torchvision.transforms.functional.resize(lr, (64,64))

        pred = model(upscaled)
        loss = loss_func(pred, hr)
        loss.backward()

        optimizer.step()

        #print(i, loss)
    # basic validation loop
    print(f"Epoch {epoch}")
    for i, batch in enumerate(validation_dataloader):
        with torch.no_grad():
            lr, hr = batch
            lr, hr = lr.to(device), hr.to(device)
            with torch.no_grad():
                upscaled = torchvision.transforms.functional.resize(lr, (64,64))

            pred = model(upscaled)
            loss = loss_func(pred, hr)
            print(i, loss)


# In[5]:


# basic testing loop
print("Testing")
for i, batch in enumerate(test_dataloader):
    with torch.no_grad():
        lr, hr = batch
        lr, hr = lr.to(device), hr.to(device)
        with torch.no_grad():
            upscaled = torchvision.transforms.functional.resize(lr, (64,64))

        pred = model(upscaled)
        loss = loss_func(pred, hr)
        print(i, loss)


# In[16]:


for i in range(8):
    with torch.no_grad():
        lr, hr = test_set[i]
        upscaled = torchvision.transforms.functional.resize(torch.Tensor(lr), (64,64)).to(device)
        pred = np.asarray(model(utils.example_batch(upscaled, batch_size))[0].cpu())
        upscaled = np.asarray(upscaled.cpu())
        utils.showImages([lr, upscaled, pred, hr])


# In[ ]:




