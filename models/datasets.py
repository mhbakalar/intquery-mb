import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import Dataset

import utils.fasta_data

class SequenceDataset(Dataset):
  def __init__(self, sequences, labels):
    self.sequences = sequences
    self.labels = labels

  def __len__(self):
    return len(self.sequences)

  def __getitem__(self, idx):
    seq = self.sequences[idx]
    label = self.labels[idx]
    one_hot = utils.fasta_data.str_to_one_hot(seq)
    return one_hot, label

'''
Return genomic intervals using fasta reference and bed file coordinates. Requires polar.
'''
class GenomeIntervalDataset(Dataset):
  def __init__(
    self,
    bed_file,
    fasta_file,
    filter_df_fn = utils.fasta_data.identity,
    chr_bed_to_fasta_map = dict(),
    context_length = None,
    return_seq_indices = False,
    shift_augs = None,
    rc_aug = False,
    return_augs = False
  ):
    super().__init__()
    bed_path = Path(bed_file)
    assert bed_path.exists(), 'path to .bed file must exist'

    df = pl.read_csv(str(bed_path), separator = '\t', has_header = False)
    df = filter_df_fn(df)
    self.df = df

    # if the chromosome name in the bed file is different than the keyname in the fasta
    # can remap on the fly
    self.chr_bed_to_fasta_map = chr_bed_to_fasta_map

    self.fasta = utils.fasta_data.FastaInterval(
      fasta_file = fasta_file,
      context_length = context_length,
      return_seq_indices = return_seq_indices,
      shift_augs = shift_augs,
      rc_aug = rc_aug
    )

    self.return_augs = return_augs

  def __len__(self):
    return len(self.df)

  def __getitem__(self, ind):
    interval = self.df.row(ind)
    chr_name, start, end = (interval[0], interval[1], interval[2])
    chr_name = self.chr_bed_to_fasta_map.get(chr_name, chr_name)
    one_hot = self.fasta(chr_name, start, end, return_augs = self.return_augs)
    return one_hot
  
class GenomeBoxcarDataset(Dataset):
  def __init__(
    self,
    fasta_file,
    chr_name = None,
    filter_df_fn = utils.fasta_data.identity,
    window_length = 46,
    context_length = None,
    return_seq_indices = False,
    shift_augs = None,
    rc_aug = False,
    return_augs = False,
    read_ahead = 0
  ):
    super().__init__()
    self.window_length = window_length
    self.chr_name = chr_name

    self.fasta = utils.fasta_data.FastaInterval(
      fasta_file = fasta_file,
      context_length = context_length,
      return_seq_indices = return_seq_indices,
      shift_augs = shift_augs,
      rc_aug = rc_aug,
      read_ahead = read_ahead
    )
    
    # Collect fasta index information
    self.length = len(self.fasta.seqs[self.chr_name]) - window_length
    self.length = 10000000
    self.start = 0

    self.return_augs = return_augs
    
  def __len__(self):
    return self.length

  def __getitem__(self, ind):
    chr_name, start, end = (self.chr_name, ind, ind + self.window_length)
    one_hot = self.fasta(chr_name, start, end, return_augs = self.return_augs)

    # Set the dinculeotide to NN
    dn_left = int(self.window_length/2)-1
    dn_one_hot = utils.fasta_data.str_to_one_hot('NN')
    one_hot[dn_left:dn_left+2,:] = dn_one_hot

    return one_hot, ind

