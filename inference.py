from tqdm.utils import colorama
import numpy as np
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.sparse_bsr_mlp import SparseBSRLinear
from train import test
from utils import to_bsr

import time

device = 'cuda'

class BlockedMLP(nn.Module):
    def __init__(self, model, block_size):
        """
        converts trained MLP model to blocked MLP model with same weights
        """
        super().__init__()

        crow_indices, col_indices, values = to_bsr(model.fc2.weight.data, block_size)

        self.features = nn.Sequential(
            nn.Flatten(),
            model.fc1,
            nn.ReLU(),
            SparseBSRLinear(model.fc2.in_features, model.fc2.out_features, block_size, crow_indices, col_indices, values),
            nn.ReLU(),
            model.fc3
        )

    def forward(self, x):
        x = self.features(x)

        return x


transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
# test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())
# train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

criterion = nn.CrossEntropyLoss()

model = torch.load("sparse_model.pth")

t1 = time.time()
for _ in range(10):
    acc = test(model, criterion, test_loader)
t2 = time.time()

print(acc, t2 - t1)

blocked_model = BlockedMLP(model, 16)
blocked_model = blocked_model.to(device)

t1 = time.time()
for _ in range(10):
    acc = test(model, criterion, test_loader)
t2 = time.time()

print(acc, t2 - t1)
