import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv_1 = nn.Conv2d(3, 64, 9, padding=4)
        self.conv_2 = nn.Conv2d(64, 32, 5, padding=2)
        self.conv_3 = nn.Conv2d(32, 3, 5, padding=2)
    
    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        return x