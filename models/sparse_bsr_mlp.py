import torch
import torch.nn as nn
from torch.sparse import _triton_ops

import warnings

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SparseBSRLinear(nn.Module):
    def __init__(self, in_features, out_features, block_size, crow_indices, col_indices, values):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        assert in_features % block_size == 0, "in_features must be divisible by block_size"
        assert out_features % block_size == 0, "out_features must be divisible by block_size"

        num_blocks_row = out_features // block_size
        num_blocks_col = in_features // block_size

        self.crow_indices = nn.Parameter(torch.tensor(crow_indices, dtype=torch.int64), requires_grad=False)
        self.col_indices = nn.Parameter(torch.tensor(col_indices, dtype=torch.int64), requires_grad=False)
        self.values = nn.Parameter(torch.cat([v.flatten() for v in values]).view(-1, block_size, block_size))

        self.bias = nn.Parameter(torch.zeros(out_features))


    def forward(self, input):
        sparse_bsr = torch.sparse_bsr_tensor(
            self.crow_indices,
            self.col_indices,
            self.values,
            size=(self.out_features, self.in_features),
            requires_grad=True,
        )

        output = _triton_ops.bsr_dense_mm(sparse_bsr, input.to(device).T).T

        return output + self.bias.to(device)


class SparseMLP(nn.Module):
    def __init__(self, p=0.5, block_size=4, size=256):
        super().__init__()

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.fc1 = nn.Linear(28 * 28, size)
        self.sfc = SparseBSRLinear(size, size, p=p, block_size=block_size)
        self.fc2 = nn.Linear(size, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.sfc(x))
        x = self.fc2(x)
        x = self.softmax(x)

        return x
