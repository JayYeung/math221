import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.sparse_mlp import MLP

# Set random seed for reproducibility
print("Setting random seed...")
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
print("\nInitializing hyperparameters...")
num_epochs = 10
batch_size = 100
learning_rate = 0.001
freq = 100  # Frequency parameter
start_itr = 1000  # Start iteration
ramp_itr = 2000  # Ramp iteration
end_itr = 3000  # End iteration

print("Loading CIFAR-10 dataset...")
# CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

test_dataset = datasets.CIFAR10(root='./data',
                               train=False,
                               transform=transforms.ToTensor())

print("Creating data loaders...")
# Data loader
train_loader = DataLoader(dataset=train_dataset,
                         batch_size=batch_size,
                         shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                        batch_size=batch_size,
                        shuffle=False)

# Train model with pruning
print("\n=== Starting training with pruning ===")
model = MLP(pruning_percent=.90, start_itr=2000)
print("Created pruning model")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_step = len(train_loader)
print("\nBeginning main training loop...")
for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = model.compute_loss(criterion, outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update model's iteration counter and apply thresholding
        model.update_iteration()

        if (i+1) % 100 == 0:
            print(f'Training - Epoch [{epoch+1}/{num_epochs}], '
                  f'Step [{i+1}/{total_step}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Theta: {model.theta:.4f}')

    # Test the model
    print("\nEvaluating model...")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the model on the 10000 test images: '
              f'{100 * correct / total}%')
    model.train()

print('\nTraining finished!')
# Print final weights
print("\nFinal weights:")
for i, layer in enumerate([model.fc1, model.fc2, model.fc3]):
    total_weights = layer.weight.data.numel()
    zero_weights = torch.sum(layer.weight.data == 0).item()
    sparsity = (zero_weights / total_weights) * 100
    print(f"Layer {i+1} sparsity: {sparsity:.2f}% ({zero_weights}/{total_weights} weights are zero)")

# Save weights as images
print("\nSaving weights as images...")
import matplotlib.pyplot as plt
for i, layer in enumerate([model.fc1, model.fc2, model.fc3]):
    plt.figure(figsize=(10, 10))
    weights = layer.weight.data.cpu().numpy()
    plt.imshow(weights, cmap='seismic', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Layer {i+1} Weights')
    plt.savefig(f'layer{i+1}_weights.png')
    plt.close()
print("Weight visualizations saved as PNG files")
