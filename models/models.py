import pandas as pd
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch

'''
A simple fully connected neural network
'''
class MLPModel(nn.Module):
  def __init__(self, input_size, hidden_size, output_size=1, n_hidden=1, dropout=0.2):
    super().__init__()

    # Input layer

    # Neural network
    self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

  def forward(self, x):
    x = torch.flatten(x,start_dim=1, end_dim=2)
    x = self.linear_relu_stack(x)
    return x