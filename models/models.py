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

    # Initialize layers
    layers = []

    # Input layer
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout))

    # Hidden layers
    for i in range(n_hidden):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

    # Output layer
    layers.append(nn.Linear(hidden_size, output_size))

    # Combine all layers into a sequential module
    self.layers = nn.Sequential(*layers)
    print(self.layers)

  def forward(self, x):
    x = self.flatten(x)
    x = self.linear_relu_stack(x)
    return x