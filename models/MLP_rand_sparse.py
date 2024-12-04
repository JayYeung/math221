import torch
import torch.nn as nn
from torch.sparse import _triton_ops

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SparseBSRLinear(nn.Module):
    def __init__(self, in_features, out_features, p, block_size):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        assert in_features % block_size == 0, "in_features must be divisible by block_size"
        assert out_features % block_size == 0, "out_features must be divisible by block_size"

        num_blocks_row = out_features // block_size
        num_blocks_col = in_features // block_size

        crow_indices = [0]
        col_indices = []
        values = []

        for i in range(num_blocks_row):
            for j in range(num_blocks_col):
                if torch.rand(1).item() > p:
                    col_indices.append(j)
                    values.append(torch.randn(block_size, block_size))

            crow_indices.append(len(col_indices))

        self.crow_indices = nn.Parameter(torch.tensor(crow_indices, dtype=torch.int64), requires_grad=False)
        self.col_indices = nn.Parameter(torch.tensor(col_indices, dtype=torch.int64), requires_grad=False)
        self.values = nn.Parameter(torch.cat([v.flatten() for v in values]).view(-1, block_size, block_size))

        self.bias = nn.Parameter(torch.zeros(out_features))


    def forward(self, input):
        sparse_bsr = torch.sparse_bsr_tensor(
            self.crow_indices.to(device),
            self.col_indices.to(device),
            self.values.to(device),
            size=(self.out_features, self.in_features),
            requires_grad=True,
        )

        output = _triton_ops.bsr_dense_mm(sparse_bsr, input.to(device).T).T

        # output = torch.mm(sparse_bsr, input.to(device).T).T

        return output + self.bias.to(device)


class SparseMLP(nn.Module):
    def __init__(self, p=0.5, block_size=4, size=128):
        super().__init__()

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

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
