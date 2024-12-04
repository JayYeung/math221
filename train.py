import numpy as np
from itertools import product

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

import time
from tqdm import tqdm

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

def train(model, criterion, optimizer, epochs):
    model.train()

    for epoch in tqdm(range(epochs)):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test(model, criterion):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    acc = 100 * correct / total

    return acc

model = MLP()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(model, criterion, optimizer, epochs)
acc = test(model, criterion)

print(acc)

torch.save(model, "model.pth")
