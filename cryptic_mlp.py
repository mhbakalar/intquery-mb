import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

translation_dict = {"A":0,"C":1,"T":2,"G":3}
reverse_translation_dict = {0:"A",1:"C",2:"T",3:"G"}

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
  def __init__(self, sequences, labels):
    self.sequences = sequences
    self.labels = labels

  def __len__(self):
    return len(self.sequences)

  def __getitem__(self, idx):
    sequence = self.sequences[idx]
    label = self.labels[idx]
    encoding = torch.tensor([translation_dict[c] for c in sequence])
    x = F.one_hot(encoding).to(torch.float32)
    return x, label

if __name__ == "__main__":
  # Read hits from text file
  hits = pd.read_csv('bxb1_cryptic.txt', header=None, names=['seq'])['seq'].tolist()
  hits_labels = np.ones(len(hits))

  # Generate random decoy sequences
  decoys = np.random.randint(0,4,[len(hits),46])
  decoys = ["".join([reverse_translation_dict[c] for c in decoy]) for decoy in decoys]
  decoys_labels = np.zeros(len(decoys))

  # Concatenate hits and decoys
  sequences = np.hstack([hits, decoys])
  labels = np.hstack([hits_labels, decoys_labels])

  dataset = Dataset(sequences, labels)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True)

  model = MLP(input_size=46*4, hidden_size=128, output_size=1)
  loss_fn = torch.nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  for epoch in range(10):
    for i, (data, target) in enumerate(dataloader):
      output = model(data).flatten()
      loss = loss_fn(output.flatten(), target.float())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    print(loss)