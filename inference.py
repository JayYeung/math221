import json
from math import prod
from tqdm.utils import colorama
import numpy as np
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.sparse_bsr_mlp import SparseBSRLinear, CustomLinear
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
            SparseBSRLinear(model.fc2.in_features, model.fc2.out_features, block_size, crow_indices, col_indices, values, model.fc2.bias.data),
            nn.ReLU(),
            model.fc3
        )

    def forward(self, x):
        x = self.features(x)

        return x


class CustomMLP(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.features = nn.Sequential(
            nn.Flatten(),
            CustomLinear(model.fc1.in_features, model.fc1.out_features, model.fc1.weight.data, model.fc1.bias.data),
            nn.ReLU(),
            CustomLinear(model.fc2.in_features, model.fc2.out_features, model.fc2.weight.data, model.fc2.bias.data),
            nn.ReLU(),
            CustomLinear(model.fc3.in_features, model.fc3.out_features, model.fc3.weight.data, model.fc3.bias.data)
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




import matplotlib.pyplot as plt

SPARSITIES = [60, 80, 85, 90, 98]
BLOCK_SIZES = [32, 64]

criterion = nn.CrossEntropyLoss()

results = {
    (s, b): [] for s, b in product(SPARSITIES, BLOCK_SIZES)
}

for sparsity, block_size in product(SPARSITIES, BLOCK_SIZES):
    print(f"\n\nsparsity = {sparsity}")
    model = torch.load(f"./checkpoints/magnitude-based/sparse_model_{block_size}_{sparsity}.pth")

    print(sum(model.fc2.weight.data.flatten() == 0) / (512 * 1024))

    t1 = time.time()
    acc = test(model, criterion, test_loader)
    t2 = time.time()

    reg = t2 - t1
    print(acc, t2 - t1)

    blocked_model = BlockedMLP(model, 16)
    blocked_model = blocked_model.to(device)

    blocked_model(torch.randn(16, 28 * 28).to(device))
    blocked_model(torch.randn(16, 28 * 28).to(device))
    blocked_model(torch.randn(16, 28 * 28).to(device))

    t1 = time.time()
    acc = test(blocked_model, criterion, test_loader)
    t2 = time.time()
    blocked = t2 - t1

    results[(sparsity, block_size)].extend([acc, reg / blocked])

    print(acc, t2 - t1)

print(results)

with open("results.json") as f:
    json.dump(results, f)




plt.plot(SPARSITIES, sorted([results[s][0] for s in product(SPARSITIES, BLOCK_SIZES)], reverse=True), label='accuracy')
plt.savefig("acc.png")
plt.close()

plt.plot(SPARSITIES, sorted([results[s][1] for s in product(SPARSITIES, BLOCK_SIZES)]), label='speedup')
plt.savefig("speedup.png")
plt.close()
