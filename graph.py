import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# from models.cnn import CNN
# from models.mlp import MLP
# from models.mlp_rand import MLP_rand

from models.sparse_bsr_mlp import SparseMLP
# from models.sparse_bsr_mask_mlp import SparseMLP  # DOESN'T WORK

import time
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64
learning_rate = 0.001
epochs = 5

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

plot_df = []

def train(model, criterion, optimizer, epochs):
    model.train()
    start_time = time.time()

    for epoch in tqdm(range(epochs)):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    end_time = time.time()
    return end_time - start_time

def test(model, criterion):
    model.eval()
    start_time = time.time()
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
    end_time = time.time()

    return acc, end_time - start_time


for block_size, p in product(2 ** np.arange(4, 8), np.linspace(0.1, 0.9, 9)):
    print(f"\nBLOCK_SIZE={block_size}, p={p}")

    model = SparseMLP(p=p, block_size=block_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_time = train(model, criterion, optimizer, epochs)
    test_accuracy, inference_time = test(model, criterion)

    plot_df.append({
        'p': p,
        'test_accuracy': test_accuracy,
        'train_time': train_time,
        'inference_time': inference_time
    })

    print(f'p = {p}, Test Accuracy: {test_accuracy:.2f}%, Train Time: {train_time:.2f} seconds, Inference Time: {inference_time:.2f} seconds')

plot_df = pd.DataFrame(plot_df)

fig, ax1 = plt.subplots()

ax1.set_xlabel('proportion of blocked sparsity')
ax1.set_ylabel('Test Accuracy', color='tab:blue')
ax1.plot(plot_df['p'], plot_df['test_accuracy'], color='tab:blue', label='Test Accuracy')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Time (seconds)', color='tab:red')
ax2.plot(plot_df['p'], plot_df['train_time'], color='tab:green', label='Train Time')
ax2.plot(plot_df['p'], plot_df['inference_time'], color='tab:red', label='Inference Time')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

plt.title('Test Accuracy, Train Time, and Inference Time vs p')
plt.savefig('plot.png')
plt.show()
