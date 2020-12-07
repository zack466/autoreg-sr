#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import PIL
from matplotlib import pyplot as plt

import utils
import datasets
from models import SRCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[2]:


div2k = datasets.Div2K(32, 2)
# lr, hr = div2k[0]
# print(lr.shape)
# utils.showImages([lr, hr])


# In[3]:


dataloader = torch.utils.data.DataLoader(div2k, batch_size=16, num_workers=4)

model = SRCNN()


# In[7]:


for i, batch in enumerate(dataloader):
    lr, hr = batch
    if i > 4:
        break
    with torch.no_grad():
        print(lr.shape)
        print(hr.shape)
        print(model(lr).shape)


# In[ ]:




