import torch
import torch.nn as nn
from torchinfo import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLPRand(nn.Module):
    def __init__(self, p=0.5, block_size=32):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

        # Create the blocked sparse mask for fc2
        self.block_size = block_size
        self.mask = self.create_blocked_mask(self.fc2.weight.shape, p, block_size).to(self.fc2.weight.device)

    def create_blocked_mask(self, shape, p, block_size):
        """
        Create a blocked sparse mask for a given shape and proportion p.
        """
        mask = torch.ones(shape)
        mask = mask.to(device)

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
        self.mask = self.mask.to(device)
        with torch.no_grad():
            self.fc2.weight.mul_(self.mask)

        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def update_iteration(*args):
        pass
