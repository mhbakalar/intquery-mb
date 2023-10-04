import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

'''
A simple fully connected neural network
'''
class Model(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, n_hidden=0, dropout=0.5):
    super().__init__()
    self.flatten = nn.Flatten()

    # Build a stack of linear layers with dropout
    self.fc_layers = nn.ModuleList()
    self.dropout_layers = nn.ModuleList()

    if n_hidden == 1:
      # Do not add dropout for a single-layer network
      self.fc_layers.append(nn.Linear(input_size, output_size))
      self.dropout_layers.append(nn.Dropout(0))
  
    else:
      # Multi layer netowrk
      self.fc_layers.append(nn.Linear(input_size, hidden_size))
      self.dropout_layers.append(nn.Dropout(dropout))

      # Add the hidden layers
      for i in range(0, n_hidden-1):
          self.fc_layers.append(nn.Linear(hidden_size, hidden_size))
          self.dropout_layers.append(nn.Dropout(dropout))

      # Add the output layer
      self.fc_layers.append(nn.Linear(hidden_size, output_size))
      self.dropout_layers.append(nn.Dropout(0))

  def forward(self, x):
    x = self.flatten(x)

    # Fully connected multilayer network
    for i, layer in enumerate(self.fc_layers):
        x = F.relu(layer(x))
        x = self.dropout_layers[i](x)

    return x