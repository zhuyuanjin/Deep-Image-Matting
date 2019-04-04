import torch
from torch import nn
import torch.nn.functional as F







class RefineNet(nn.Module):

    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, 3, padding=1)
        self.BN1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.BN2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.BN3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        x = self.BN1(F.relu(self.conv1(x)))
        x = self.BN2(F.relu(self.conv2(x)))
        x = self.BN3(F.relu(self.conv3(x)))
        x = F.sigmoid(self.conv4(x))
        return x

