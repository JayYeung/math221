import numpy as np
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.sparse_bsr_mlp import SparseMLP

import time
from tqdm import tqdm


def convert(model):
