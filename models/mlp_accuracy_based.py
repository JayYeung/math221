import torch
import torch.nn as nn
import math
from torchinfo import summary
from tqdm import tqdm, trange

class MLPAccuracyPrune(nn.Module):
    def __init__(self, pruning_percent, start_itr, group_size=32, eval_batches=10):
        super(MLPAccuracyPrune, self).__init__()
        self.pruning_percent = pruning_percent
        self.start_itr = start_itr
        self.current_itr = 0
        self.group_size = group_size
        self.eval_batches = eval_batches  # Number of batches to evaluate pruning impact
        self.theta = 0.0

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28 * 1, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

        self.pruned_blocks = []

    def update_iteration(self, data_loader, device, criterion):
        """Update current iteration and prune blocks based on accuracy impact."""
        self.current_itr += 1
        if self.start_itr <= self.current_itr:
            self.prune_least_important_blocks(data_loader, device, criterion)

    def prune_least_important_blocks(self, data_loader, device, criterion):
        """Evaluate block importance and prune least important ones."""
        layer_blocks = []

        # Evaluate block importance for fc1 and fc2
        for layer in [self.fc1, self.fc2]:
            importance_scores, block_positions = self.evaluate_block_importance(layer, data_loader, device, criterion)
            layer_blocks.extend(zip(importance_scores, block_positions, [layer] * len(block_positions)))

        # Sort blocks by importance (lower scores mean less impact)
        layer_blocks.sort(key=lambda x: x[0])  # Sort by importance score

        # Prune the least important blocks
        num_blocks_to_prune = int(len(layer_blocks) * self.pruning_percent)
        for _, (row, col, end_row, end_col), layer in layer_blocks[:num_blocks_to_prune]:
            layer.weight.data[row:end_row, col:end_col].zero_()

    def evaluate_block_importance(self, layer, data_loader, device, criterion):
        """Evaluate the importance of each block by measuring accuracy impact."""
        rows, cols = layer.weight.size()
        block_positions = []
        importance_scores = []
        size = self.group_size

        for row in trange(0, rows, size):
            for col in range(0, cols, size):
                end_row = min(row + size, rows)
                end_col = min(col + size, cols)
                block_positions.append((row, col, end_row, end_col))

                # Temporarily zero out the block
                original_block = layer.weight.data[row:end_row, col:end_col].clone()
                layer.weight.data[row:end_row, col:end_col].zero_()

                # Measure accuracy impact
                impact = self.evaluate_accuracy_impact(data_loader, device, criterion)

                # Restore the block
                layer.weight.data[row:end_row, col:end_col] = original_block

                importance_scores.append(impact)

        return importance_scores, block_positions

    def evaluate_accuracy_impact(self, data_loader, device, criterion):
        """Measure accuracy after zeroing out a block."""
        self.eval()  # Set model to evaluation mode
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                if batch_idx >= self.eval_batches:
                    break  # Limit evaluation to a few batches
                data, target = data.to(device), target.to(device)
                outputs = self(data)
                loss = criterion(outputs, target)
                total_loss += loss.item()
        return total_loss

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def compute_loss(self, criterion, outputs, targets):
        return criterion(outputs, targets)
