import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import argparse 
import re
import genome_data
from sklearn.metrics import f1_score, accuracy_score


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

  parser = argparse.ArgumentParser(description='MLP for Cryptic Site Classification')
  parser.add_argument('--reference', default='/data/references/hg38.fa', type=str, help='Path to the reference genome')
  parser.add_argument('--dinucleotide', default='GC', type=str, help='Dinucleotide to use')
  parser.add_argument('--sequence_length', default=46, type=int, help='Length of the sequence')
  parser.add_argument('--cryptic_sequences', default='bxb1_cryptic.txt', type=str, help='File containing cryptic sequences')
  parser.add_argument('--fixed_bases', nargs='*', type=str, default=None, help='Fixed bases in the sequence')
  args = parser.parse_args()
  
  # Then you can access these command line arguments as follows:
  reference = args.reference
  dinucleotide = args.dinucleotide
  seq_length = args.sequence_length
  cryptic_seq_fn = args.cryptic_sequences
  fixed_bases = args.fixed_bases
  
  # Parameters
  dinucleotide_ids = [translation_dict[c] for c in dinucleotide]
  hidden_size = 128
  vocab_size = 5
  train_test_split = 0.8

  # Read hits from text file
  hits = pd.read_csv(cryptic_seq_fn, header=None, names=['seq'])['seq'].tolist()
  hits_labels = np.ones(len(hits))

  # Generate random decoy sequences
  decoys = np.random.randint(0,4,[len(hits),seq_length])  # Don't use N in decoys
  dn_start = int(seq_length/2 - 1)
  decoys[:,dn_start:dn_start+2] = dinucleotide_ids  # Set core dinucleotide for each decoy 

  if fixed_bases != None: 
    fixed_base_dict = {}

    for fixed_base in fixed_bases:
      match = re.match(r'([A-Z])(\d+)', fixed_base)
      base = match.group(1)
      pos_to_change = int(match.group(2))
      base_id = translation_dict[base]
      fixed_base_dict[pos_to_change] = base_id

    # set fixed bases
    for pos in fixed_base_dict:
      base_id = fixed_base_dict[pos]
      decoys[:,pos-1] = base_id
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

# training loop
for epoch in range(15):
    train_loss = 0.0
    train_preds, train_targets = [], []

    # Training Phase
    for i, (data, target) in enumerate(train_dataloader):
        output = model(data).flatten()

        # Convert output probabilities to predicted class
        preds = (output > 0.5).float()
        train_preds.extend(preds.tolist())
        train_targets.extend(target.tolist())

        loss = loss_fn(output, target.float())
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_accuracy = accuracy_score(train_targets, train_preds)
    train_f1 = f1_score(train_targets, train_preds, average='weighted')

    print(f"Epoch {epoch}/{15}")
    print(f"-------------")
    print(f"Train loss: {train_loss/len(train_dataloader)}")
    print(f"Train Accuracy: {train_accuracy}")
    print(f"Train F1 Score: {train_f1}")

    val_loss = 0.0
    val_preds, val_targets = [], []

    # Validation Phase
    with torch.no_grad():
        for i, (data, target) in enumerate(val_dataloader):
            output = model(data).flatten()

            # Convert output probabilities to predicted class
            preds = (output > 0.5).float()
            val_preds.extend(preds.tolist())
            val_targets.extend(target.tolist())

            loss = loss_fn(output, target.float())
            val_loss += loss.item()

        val_accuracy = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')

    print(f"Val loss: {val_loss/len(val_dataloader)}")
    print(f"Val Accuracy: {val_accuracy}")
    print(f"Val F1 Score: {val_f1}")
    print(f"-------------\n")

# Evaluate on genomic data
genome_dataset = genome_data.DinucleotideDataset(reference, dinucleotide, seq_length)
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