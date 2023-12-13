import torch.nn as nn
import torch.utils.data
import torch
import torch.nn.functional as F
import math

'''
A simple fully connected neural network
'''
class MLPModel(nn.Module):
  def __init__(self, seq_length, vocab_size, hidden_size, output_size=1, n_hidden=1, dropout=0.2):
    super().__init__()

    # Input dimensions
    input_size = seq_length * vocab_size

    # Initialize layers
    layers = []

    # Input layer
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(nn.Hardswish())
    layers.append(nn.Dropout(dropout))

    # Hidden layers
    for i in range(n_hidden):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.Hardswish())
        layers.append(nn.Dropout(dropout))

    # Output layer
    layers.append(nn.Linear(hidden_size, output_size))

    # Combine all layers into a sequential module
    self.layers = nn.Sequential(*layers)

  def forward(self, x):
    x = x.view(x.size(0), -1)
    return self.layers(x)
class TransformerModel(nn.Module):
    
    def __init__(self, seq_length, ntoken, d_model, nhead, nhid, nlayers, dropout=0.5):
        super().__init__()
        try:
          self.d_model = d_model
          self.model_type = 'Transformer'
          self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length)
          encoder_layers = nn.TransformerEncoderLayer(
              d_model=d_model, 
              nhead=nhead, 
              dim_feedforward=nhid, 
              dropout=dropout, 
              batch_first=True,
          )
          self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
          self.encoder = nn.Embedding(ntoken, d_model)
          self.decoder = SequencePooler(d_model)
          self.init_weights()
        except Exception as e:
          print("Exception building transformer:")
          print(e)
    

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        output = torch.mean(output, dim=1)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.0, max_len=5000):
      super().__init__()
      self.dropout = nn.Dropout(p=dropout)
      position = torch.arange(max_len).unsqueeze(1)
      div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
      pe = torch.zeros(max_len, embedding_dim)
      pe[:, 0::2] = torch.sin(position * div_term)
      pe[:, 1::2] = torch.cos(position * div_term)
      pe = pe.unsqueeze(0)
      self.register_buffer('pe', pe)

    def forward(self, x):
      """
      Args:
          x: Tensor, shape [batch_size, seq_len, embedding_dim]
      """
      x = x + self.pe[:x.size(0)]
      return self.dropout(x)    
class SequencePooler(nn.Module):
        """
        Sequence pooling source from:
        https://github.com/SHI-Labs/Compact-Transformers
        """
        def __init__(self, d_model):
            super().__init__()
            self.embedding_dim = d_model
            self.attention_pool = nn.Linear(self.embedding_dim, 1)
            self.projection = nn.Linear(self.embedding_dim, 1)
                
        def forward(self, x):
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
            x = self.projection(x)
            return x


