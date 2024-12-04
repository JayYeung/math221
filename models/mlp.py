import torch
import torch.nn as nn
from torchinfo import summary

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten() 
        self.fc1 = nn.Linear(28 * 28, 128) 
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x) 
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        x = self.fc3(x)
        return x

    def summary(self):
        summary(self, input_size=(1, 28, 28))
    
