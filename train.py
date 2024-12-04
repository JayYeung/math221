import numpy as np
from itertools import product
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# from models.cnn import CNN
# from models.mlp import MLP
# from models.mlp_rand import MLP_rand
from models.mlp_accuracy_based import MLPAccuracyPrune
from models.sparse_bsr_mlp import SparseMLP
# from models.sparse_bsr_mask_mlp import SparseMLP  # DOESN'T WORK

import time
from tqdm import tqdm

from models.mlp_group_lasso import MLP_group_lasso

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64
learning_rate = 0.001
epochs = 10

train_dataset = datasets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

test_dataset = datasets.MNIST(root='./data',
                               train=False,
                               transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset,
                         batch_size=batch_size,
                         shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                        batch_size=batch_size,
                        shuffle=False)

def see_weights(model, epoch):
    for i, layer in enumerate([model.fc1, model.fc2, model.fc3]):
        total_weights = layer.weight.data.numel()
        zero_weights = torch.sum(torch.abs(layer.weight.data) < 1e-7).item()
        sparsity = (zero_weights / total_weights) * 100
        print(f"Layer {i+1} sparsity: {sparsity:.2f}% ({zero_weights}/{total_weights} weights are zero)")

    for i, layer in enumerate([model.fc1, model.fc2, model.fc3]):
        plt.figure(figsize=(10, 10))
        weights = layer.weight.data.cpu().numpy()
        plt.imshow(weights, cmap='seismic', interpolation='nearest')
        plt.colorbar()
        plt.title(f'MLP_accuracy_based {epoch} Layer {i+1} Weights')
        plt.savefig(f'weight_map_images/MLP_accuracy_based_{epoch}_Layer{i+1}_weights.png')
        plt.close()
    print("Weight visualizations saved as PNG files")

def train(model, criterion, optimizer, epochs):
    model.train()

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update pruning logic
        model.update_iteration(train_loader, device, criterion)

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        print(f'Accuracy after epoch {epoch + 1}: {test(model, criterion)}')
        see_weights(model, epoch)
    



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

    return acc, time.time() - start_time

model = MLPAccuracyPrune(start_itr=2, pruning_percent=0.5).to(device)

# model.summary()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_time = train(model, criterion, optimizer, epochs)
test_accuracy, inference_time = test(model, criterion)

print(test_accuracy)
