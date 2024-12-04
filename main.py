import numpy as np
from itertools import product
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# from models.cnn import CNN
from models.mlp import MLP
# from models.mlp_rand import MLP_rand

from models.sparse_bsr_mlp import SparseMLP
# from models.sparse_bsr_mask_mlp import SparseMLP  # DOESN'T WORK

from train import train, test

import time
from tqdm import tqdm

from models.mlp_group_lasso import MLP_group_lasso

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64
learning_rate = 0.001
epochs = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# model = MLP()
# model = model.to(device)

model = MLP_group_lasso().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(model, criterion, optimizer, epochs)
acc = test(model, criterion)

print(acc)

torch.save(model, "model.pth")
