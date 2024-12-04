import torch
import torch.nn as nn
from .sparse import SparseBSRLinear
from torchinfo import summary

class SparseRandMLP(nn.Module):
    def __init__(self, p=0.5, block_size=4, size=64):
        super().__init__()

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.fc1 = nn.Linear(28 * 28, size)
        self.sfc = SparseBSRLinear(size, size, p=p, block_size=block_size)
        self.fc2 = nn.Linear(size, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.sfc(x))
        x = self.fc2(x)
        x = self.softmax(x)

        return x

    def summary(self):
        summary(self, input_size=(1, 28, 28))
