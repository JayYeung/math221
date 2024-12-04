import numpy as np

import torch
import torch.nn as nn
from torch.sparse import _triton_ops

import warnings

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SparseBSRLinear(nn.Module):
    def __init__(self, in_features, out_features, p, block_size, mask):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.block_size = block_size
        self.mask = mask
        self.p = p

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        assert in_features % block_size == 0, "in_features must be divisible by block_size"
        assert out_features % block_size == 0, "out_features must be divisible by block_size"

    def forward(self, input):
        output = _triton_ops.bsr_dense_mm(
            (self.weight * self.mask.to(device)).to_sparse_bsr(blocksize=self.block_size),
            input.to(device).T
        ).T

        return output + self.bias.to(device)


class SparseMLP(nn.Module):
    def __init__(self, p=0.5, block_size=4, size=256):
        super().__init__()

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.fc1 = nn.Linear(28 * 28, size)
        self.sfc = SparseBSRLinear(
            in_features=size, out_features=size,
            p=p, block_size=block_size,
            mask=self._build_mask((size, size), p, block_size)
        )
        self.fc2 = nn.Linear(size, 10)

    def _build_mask(self, shape, p, block_size):
        mask = torch.ones(shape)

        rows, cols = shape
        row_blocks = rows // block_size
        col_blocks = cols // block_size

        for i in range(row_blocks):
            for j in range(col_blocks):
                if torch.rand(1).item() <= p:
                    mask[
                        i * block_size : (i + 1) * block_size,
                        j * block_size : (j + 1) * block_size,
                    ] = 0

        return mask

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.sfc(x))
        x = self.fc2(x)
        x = self.softmax(x)

        return x
