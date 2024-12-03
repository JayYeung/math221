import torch
import torch.nn as nn
from torchinfo import summary

class MLP_rand(nn.Module):
    def __init__(self, p=0.5, block_size=4, size=32):
        super(MLP_rand, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, 10)
        # self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

        # Create the blocked sparse mask for fc2
        self.block_size = block_size
        self.mask = self.create_blocked_mask(self.fc2.weight.shape, p, block_size).to(self.fc2.weight.device)

    def create_blocked_mask(self, shape, p, block_size):
        """
        Create a blocked sparse mask for a given shape and proportion p.
        """
        mask = torch.ones(shape)

        # Divide the weight matrix into blocks
        rows, cols = shape
        row_blocks = rows // block_size
        col_blocks = cols // block_size
        
        zeroed = 0

        for i in range(row_blocks):
            for j in range(col_blocks):
                if torch.rand(1).item() < p:
                    mask[
                        i * block_size : (i + 1) * block_size,
                        j * block_size : (j + 1) * block_size,
                    ] = 0
                    zeroed += block_size ** 2
        
        # print(f'Zeroed {zeroed} out of {rows * cols} weights, proportion = {zeroed / (rows * cols)} ({float(p)})')
        
        return mask

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        
        # Apply the blocked sparse mask to fc2 weights
        with torch.no_grad():
            self.fc2.weight.mul_(self.mask)

        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def summary(self):
        summary(self, input_size=(1, 28, 28))