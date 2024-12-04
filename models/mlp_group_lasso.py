import torch
import torch.nn as nn
from torchinfo import summary

class MLP_group_lasso(nn.Module):
    def __init__(self, group_size=4, size1=512, size2=256, lasso_weight=1e-4):
        super(MLP_group_lasso, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, size1)
        self.fc2 = nn.Linear(size1, size2)
        self.fc3 = nn.Linear(size2, 10)
        self.relu = nn.ReLU()
        self.group_size = group_size
        self.lasso_weight = lasso_weight  

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
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
        num_groups = cols // self.group_size
        for g in range(num_groups):
            group_weights = weight[:, g * self.group_size : (g + 1) * self.group_size]
            group_lasso_loss += torch.norm(group_weights, dim=1).sum()

        return self.lasso_weight * group_lasso_loss

    def compute_loss(self, criterion, outputs, targets):
        base_loss = criterion(outputs, targets)
        group_lasso_loss = self.compute_group_lasso()
        return base_loss + group_lasso_loss

    def summary(self):
        summary(self, input_size=(1, 28, 28))
