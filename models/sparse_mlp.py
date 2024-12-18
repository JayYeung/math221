import torch
import torch.nn as nn
import math
from torchinfo import summary

class SparseMLP(nn.Module):
    def __init__(self, in_dimension, out_dimension, pruning_percent, start_itr, block_size=32, lasso_weight=1e-4):
        super().__init__()

        self.pruning_percent = pruning_percent
        self.start_itr = start_itr
        self.current_itr = 0
        self.theta = 0.0
        self.block_size = block_size
        self.lasso_weight = lasso_weight

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_dimension, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, out_dimension)
        self.relu = nn.ReLU()

        self.apply_threshold()

    def update_iteration(self):
        self.current_itr += 1
        if self.start_itr <= self.current_itr:
            progress = (self.current_itr - self.start_itr) / 5000  # Assume 1000 iterations for full growth
            current_theta = self.pruning_percent * (1 - math.exp(-5 * progress))  # -5 controls curve steepness
            self.theta = max(0, min(current_theta, self.pruning_percent))  # Clamp between 0 and pruning_percent
            self.apply_threshold()

    def apply_threshold(self):
        for layer in [self.fc2]:
            self.zero_blocks(layer.weight.data)
            if layer.bias is not None:
                self.zero_blocks(layer.bias.data.view(-1, 1))

    def zero_blocks(self, matrix):
        size = 32
        rows, cols = matrix.size()
        block_sums = []
        block_positions = []

        for row in range(0, rows, size):
            for col in range(0, cols, size):
                end_row = min(row + size, rows)
                end_col = min(col + size, cols)
                block = matrix[row:end_row, col:end_col]
                block_sum = abs(block.sum().item())
                block_sums.append(block_sum)
                block_positions.append((row, col, end_row, end_col))

        if block_sums:
            sorted_sums = sorted(block_sums)
            cutoff_idx = int(len(sorted_sums) * self.theta)  # Use current theta instead of pruning_percent
            cutoff_value = sorted_sums[cutoff_idx]

            for sum_val, (row, col, end_row, end_col) in zip(block_sums, block_positions):
                if sum_val <= cutoff_value:
                    matrix[row:end_row, col:end_col].zero_()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))

        torch.cuda.synchronize()
        m1 = torch.cuda.memory_allocated()

        x = self.relu(self.fc2(x))

        torch.cuda.synchronize()
        m2 = torch.cuda.memory_allocated()

        print(m2 - m1)

        x = self.fc3(x)
        return x

    def compute_group_lasso(self):
        """
        Compute the group lasso regularization loss for fc2 weights.
        Group lasso is computed by summing the norms of groups of weights.
        """
        weight = self.fc2.weight
        rows, cols = weight.shape
        group_lasso_loss = 0.0

        # Group weights in blocks along the columns
        num_groups = cols // self.block_size
        for g in range(num_groups):
            group_weights = weight[:, g * self.block_size : (g + 1) * self.block_size]
            group_lasso_loss += torch.norm(group_weights, dim=1).sum()

        return self.lasso_weight * group_lasso_loss

    def compute_loss(self, criterion, outputs, targets, lasso=True):
        base_loss = criterion(outputs, targets)
        if not lasso:
            return base_loss

        group_lasso_loss = self.compute_group_lasso()
        return base_loss + group_lasso_loss
