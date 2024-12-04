import torch
import torch.nn as nn

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
            self.crow_indices,
            self.col_indices,
            self.values,
            size=(self.out_features, self.in_features),
            requires_grad=True,
        )

        output = torch.mm(sparse_bsr, input.T).T

        return output + self.bias
