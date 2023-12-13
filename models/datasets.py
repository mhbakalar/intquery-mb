import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


import utils.fasta_data

class SequenceDataset(Dataset):
  def __init__(self, sequences, labels, one_hot=True):
    self.sequences = sequences
    self.labels = labels
    self.one_hot = one_hot

  def __len__(self):
    return len(self.sequences)

  def __getitem__(self, idx):
    seq = self.sequences[idx]
    label = self.labels[idx]

    # Transform to one hot representation if needed
    if self.one_hot:
      seq = utils.fasta_data.str_to_one_hot(seq)
    else:
       seq = utils.fasta_data.str_to_seq_indices(seq)
       
    return seq, label


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
  
class GenomeIterableDataset(IterableDataset):
    def __init__(self, fasta_file, chr_name, window_length, rc_aug):
      super().__init__()
      
      self.fasta_file = fasta_file
      self.chr_name = chr_name
      self.window_length = window_length
      self.rc_aug = rc_aug

      # Peak at fasta file to determine sequence length
      fasta_length = len(
        utils.fasta_data.FastaInterval(
          fasta_file = self.fasta_file,
        ).seqs[self.chr_name]
      )
      
      self.start = 0
      self.end = fasta_length - self.window_length

    def load_fasta(self):
      self.fasta_handle = utils.fasta_data.FastaInterval(
        fasta_file = self.fasta_file,
        rc_aug = self.rc_aug,
        read_ahead = 10000
      )

    def _iter_bounds(self):
      worker_info = torch.utils.data.get_worker_info()
      worker_id = worker_info.id
      total_workers = worker_info.num_workers
      if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
      else:
          world_size = 1
          rank = 0
      total_workers *= world_size

      global_worker_id = worker_id * world_size + rank

      per_worker = int(math.ceil((self.end - self.start) / float(total_workers)))
      iter_start = self.start + (global_worker_id * per_worker)
      iter_end = min(iter_start + per_worker, self.end)
      return iter_start, iter_end

    def __len__(self):
      if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
      else:
          world_size = 1
          rank = 0
      # This might be inaccurate...
      return int((self.end - self.start) / world_size)

    def __iter__(self):
      worker_info = torch.utils.data.get_worker_info()
      iter_start, iter_end = self._iter_bounds()
      for i in range(iter_start, iter_end - self.window_length):
          yield self.fasta_handle(self.chr_name, i, i + self.window_length), i