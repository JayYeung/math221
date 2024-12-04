from tqdm.utils import colorama
import numpy as np
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim

from models.sparse_bsr_mlp import SparseBSRLinear
from train import test

import time
from tqdm import tqdm

model = torch.load("model.pth")

device = 'cuda'

def convert_bsr(weight, block_size):
    return weight.to_sparse_bsr(blocksize=block_size)


class BlockedMLP(nn.Module):
    def __init__(self, model, p, block_size):
        """
        converts trained MLP model to blocked MLP model with same weights
        """
        super().__init__()

        fc2_weights_bsr = convert_bsr(model.fc2.weight.data, block_size)

        crow_indices    = fc2_weights_bsr.crow_indices()
        col_indices     = fc2_weights_bsr.col_indices()
        values          = fc2_weights_bsr.values()

        print(crow_indices, col_indices, values)

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


blocked_model = BlockedMLP(model, 0.5, 16)
blocked_model = blocked_model.to(device)

criterion = nn.CrossEntropyLoss()

# test(model, criterion)
