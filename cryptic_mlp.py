import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import genome_data

translation_dict = {"A":0,"C":1,"T":2,"G":3,"N":4}
reverse_translation_dict = {0:"A",1:"C",2:"T",3:"G",4:"N"}

class MLP(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    x = self.flatten(x)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

class Dataset(torch.utils.data.Dataset):
  def __init__(self, sequences, labels, vocab_size):
    self.sequences = sequences
    self.labels = labels
    self.vocab_size = vocab_size

  def __len__(self):
    return len(self.sequences)

  def __getitem__(self, idx):
    sequence = self.sequences[idx]
    label = self.labels[idx]
    encoding = torch.tensor([translation_dict[c] for c in sequence])
    x = F.one_hot(encoding, num_classes=self.vocab_size).to(torch.float32)
    return x, label

if __name__ == "__main__":
  # Parameters
  dinucleotide = "GC"
  dinucleotide_ids = [translation_dict[c] for c in dinucleotide]
  seq_length = 46
  hidden_size = 128
  vocab_size = 5
  train_test_split = 0.8

  # Read hits from text file
  hits = pd.read_csv('bxb1_cryptic.txt', header=None, names=['seq'])['seq'].tolist()
  hits_labels = np.ones(len(hits))

  # Generate random decoy sequences
  decoys = np.random.randint(0,4,[len(hits),46])
  decoys[:,22:24] = dinucleotide_ids  # Set core dinucleotide for each decoy 
  decoys = ["".join([reverse_translation_dict[c] for c in decoy]) for decoy in decoys]
  decoys_labels = np.zeros(len(decoys))

  # Concatenate hits and decoys
  sequences = np.hstack([hits, decoys])
  labels = np.hstack([hits_labels, decoys_labels])
  dataset = Dataset(sequences, labels, vocab_size=vocab_size)

   # Test and train data split
  train_size = int(train_test_split*len(dataset))
  test_size = len(dataset) - train_size
  train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
  
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
  val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

  # Build model
  model = MLP(input_size=seq_length*vocab_size, hidden_size=hidden_size, output_size=1)
  loss_fn = torch.nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  # Training loop
  for epoch in range(15):
    for i, (data, target) in enumerate(train_dataloader):
      output = model(data).flatten()
      loss = loss_fn(output.flatten(), target.float())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    #print("Train loss: ", loss)

    # Compute validation loss each epoch
    with torch.no_grad():
      loss = 0
      for i, (data, target) in enumerate(val_dataloader):
        output = model(data).flatten()
        loss += loss_fn(output.flatten(), target.float())
      #print("Val loss: ", loss/len(val_dataloader))

# Evaluate on genomic data
genome_dataset = genome_data.DinucleotideDataset('../hg38.fa', dinucleotide="GC", length=46)
genome_dataloader = torch.utils.data.DataLoader(genome_dataset, batch_size=32)

with torch.no_grad():
  count = 0
  for i, (chr, start, end, sequence, data) in enumerate(genome_dataloader):
    try:
      output = model(data).flatten()
      logits = torch.sigmoid(output)
      locs = (logits > 0.95).nonzero()
      if len(locs) > 0:
        for l in locs:
          print(chr[l], start[l].item(), end[l].item(), sequence[l].upper(), '{:.2f}'.format(logits[l].item()))
    except:
      pass  # todo handle exceptions from non 50 character FASTA lines