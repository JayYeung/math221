import train
from models.mlp_rand import MLPRand
from models.mlp_group_lasso import MLP_group_lasso
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 10
lr = 0.001

transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for w in [1e-4, 1e-3, 1e-2, 1e-1]:
# for p in [0.6, 0.8, 0.85, 0.9, 0.95, 0.98]:
    # model = MLPRand(p, block_size=32)
    model = MLP_group_lasso(lasso_weight=w)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train.train(model, nn.CrossEntropyLoss(), optimizer, 10, train_loader)

    torch.save(model, f"lasso_32_{p}.pth")
