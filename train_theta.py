from itertools import groupby
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.sparse_mlp import SparseMLP
from models.mlp_accuracy_based import MLPAccuracyPrune
import matplotlib.pyplot as plt
import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 10
lr = 0.001

# train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
# test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())
# train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# model = SparseMLP(in_dimension=(28 * 28),
#             out_dimension=10,
#             pruning_percent=.95, start_itr=2000,
#             block_size=32
# )
model = MLPAccuracyPrune(pruning_percent=.90, start_itr=2000, group_size=32)

model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
total_step = len(train_loader)
for epoch in range(epochs):
    print(f"\nEpoch [{epoch+1}/{epochs}]")

    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = model.compute_loss(criterion, outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update model's iteration counter and apply thresholding
        model.update_iteration(train_loader, device, criterion)

        if (i+1) % 100 == 0:
            print(f'Training - Epoch [{epoch+1}/{epochs}], '
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


torch.save(model, "acc_sparse_model_32_90.pth")

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
for i, layer in enumerate([model.fc1, model.fc2, model.fc3]):
    plt.figure(figsize=(10, 10))
    weights = layer.weight.data.cpu().numpy()
    plt.imshow(weights, cmap='seismic', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Layer {i+1} Weights')
    plt.savefig(f'layer{i+1}_weights.png')
    plt.close()
print("Weight visualizations saved as PNG files")
