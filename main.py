import torch
from PIL import Image

import datasets


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = datasets.Div2K(device)

img = Image.fromarray(dataset[0]['image'].size())
