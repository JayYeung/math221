import torch.nn as nn
from torchinfo import summary

class MLP(nn.Module):
    def __init__(self, pruning_percent, start_itr):
        super(MLP, self).__init__()
        self.pruning_percent = pruning_percent
        self.start_itr = start_itr
        self.current_itr = 0

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

        # Apply initial threshold
        self.apply_threshold()

    def update_iteration(self):
        """Update current iteration and apply pruning if needed"""
        self.current_itr += 1
        if self.start_itr <= self.current_itr:
            self.apply_threshold()

    def apply_threshold(self):
        # Apply pruning to all linear layers
        for layer in [self.fc1, self.fc2]:
            self.zero_blocks(layer.weight.data)
            if layer.bias is not None:
                self.zero_blocks(layer.bias.data.view(-1, 1))

    def zero_blocks(self, matrix):
        """Zero out bottom pruning_percent of blocks based on block sums"""
        size = 32
        rows, cols = matrix.size()
        block_sums = []
        block_positions = []

        # Calculate all block sums and their positions
        for row in range(0, rows, size):
            for col in range(0, cols, size):
                end_row = min(row + size, rows)
                end_col = min(col + size, cols)
                block = matrix[row:end_row, col:end_col]
                block_sum = abs(block.sum().item())
                block_sums.append(block_sum)
                block_positions.append((row, col, end_row, end_col))

        # Calculate cutoff value based on percentile
        if block_sums:
            sorted_sums = sorted(block_sums)
            cutoff_idx = int(len(sorted_sums) * self.pruning_percent)
            cutoff_value = sorted_sums[cutoff_idx]

            # Zero out blocks below cutoff
            for sum_val, (row, col, end_row, end_col) in zip(block_sums, block_positions):
                if sum_val <= cutoff_value:
                    matrix[row:end_row, col:end_col].zero_()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def summary(self):
        summary(self, input_size=(1, 28, 28))
