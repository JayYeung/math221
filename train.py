import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.mlp_group_lasso import MLP_group_lasso

import time
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 10

# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def see_weights(model, epoch):
    for i, layer in enumerate([model.fc1, model.fc2, model.fc3]):
        total_weights = layer.weight.data.numel()
        zero_weights = torch.sum(torch.abs(layer.weight.data) < 1e-7).item()        
        sparsity = (zero_weights / total_weights) * 100
        print(f"Layer {i+1} sparsity: {sparsity:.2f}% ({zero_weights}/{total_weights} weights are zero)")

    import matplotlib.pyplot as plt
    for i, layer in enumerate([model.fc1, model.fc2, model.fc3]):
        plt.figure(figsize=(10, 10))
        weights = layer.weight.data.cpu().numpy()
        plt.imshow(weights, cmap='seismic', interpolation='nearest')
        plt.colorbar()
        plt.title(f'MLP_group_lasso {epoch} Layer {i+1} Weights')
        plt.savefig(f'weight_map_images/MLP_group_lasso_{epoch}_Layer{i+1}_weights.png')
        plt.close()
    print("Weight visualizations saved as PNG files")

def train(model, criterion, optimizer, epochs):
    model.train()
    start_time = time.time()

    for epoch in tqdm(range(epochs)):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            loss = model.compute_loss(criterion, outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        see_weights(model, epoch) 
        

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

model = MLP_group_lasso().to(device)

model.summary()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_time = train(model, criterion, optimizer, epochs)
test_accuracy, inference_time = test(model, criterion)

print(test_accuracy)