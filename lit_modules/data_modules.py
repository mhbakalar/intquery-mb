import torch
import lightning as L
import torch.nn.functional as F
import torchmetrics
import genomepy
import pandas as pd
import os
import numpy as np
from models import datasets

'''
Binary classifier. Optionally, use scalar value as sampling weight.
'''
class BinaryDataModule(L.LightningDataModule):
    def __init__(self, 
        data_path, 
        decoy_path=None,
        genomic_reference_file=None, 
        decoy_mul=1, 
        sequence_length=46,
        train_test_split=0.8, 
        batch_size=32,
        smooth_labels=False
    ):
        super().__init__()
        self.data_path = data_path
        self.decoy_path = decoy_path
        self.genomic_reference_file = genomic_reference_file
        self.decoy_mul = decoy_mul
        self.sequence_length = sequence_length
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.smooth_labels = smooth_labels

    def setup(self, stage: str):
        # Select test/train dataset
        fname = stage + '.csv'

        # Load the cryptic seq data and decoys
        sites = pd.read_csv(os.path.join(self.data_path, fname))

        decoys = pd.read_csv(os.path.join(self.decoy_path, 'decoys_100k.csv'))
        decoys = decoys.sample(n=len(sites)*self.decoy_mul, replace=True)

        # Cryptic sites data for training
        hits = sites['seq'].values
        decoys = decoys['seq'].values
        sequences = np.hstack([hits,decoys])
        
        # Soft labels
        label_0 = 0.1 if self.smooth_labels else 0.0
        label_1 = 0.9 if self.smooth_labels else 1.0

        labels = np.hstack([
            np.full(len(hits), label_1, dtype=np.float32),
            np.full(len(decoys), label_0, dtype=np.float32)
        ])

        weights = np.hstack([
            np.full(len(hits), 1./len(hits)), 
            np.full(len(decoys), 1./len(decoys))
        ])

        self.dataset = datasets.SequenceDataset(sequences, labels)

        if stage == 'fit':
            # Test and train data split
            train_size = int(self.train_test_split*len(self.dataset))
            test_size = len(self.dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])

            # Weighted random sampler for upsampling minority class for training
            train_sample_weights = weights[self.train_dataset.indices]
            self.train_sampler = torch.utils.data.WeightedRandomSampler(train_sample_weights, len(train_sample_weights), replacement=True)

        elif stage == 'test':
            self.test_dataset = self.val_dataset

        elif stage == 'predict':
            self.pred_dataset = self.dataset

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.pred_dataset, batch_size=self.batch_size)

'''
Numeric data
'''
class NumericDataModule(L.LightningDataModule):
    def __init__(self, 
        data_path, 
        decoy_path=None,
        genomic_reference_file=None, 
        decoy_mul=1, 
        sequence_length=46,
        train_test_split=0.8, 
        batch_size=32,
        log_transform=False,
        one_hot=True
    ):
        super().__init__()

        self.data_path = data_path
        self.decoy_path = decoy_path
        self.genomic_reference_file = genomic_reference_file
        self.decoy_mul = decoy_mul
        self.sequence_length = sequence_length
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.log_transform = log_transform
        self.one_hot = one_hot

    def setup(self, stage: str):
        # Select test/train dataset
        fname = stage + '.csv'

        # Load the cryptic seq data and decoys
        sites = pd.read_csv(os.path.join(self.data_path, fname))
        hits = sites['seq'].values

        # Load decoys
        if self.decoy_mul > 0:
            decoys = pd.read_csv(os.path.join(self.decoy_path, fname))

            # Sample decoys using decoy_mul factor
            decoys = decoys.sample(n=len(sites)*self.decoy_mul, replace=True)
            decoys = decoys['seq'].values

        # Cryptic sites data for training
        if self.log_transform:
            sites['value'] = np.log(sites['value'])
        
        if self.decoy_mul > 0:
            sequences = np.hstack([hits,decoys])
            labels = np.hstack([
                sites['value'].values.astype(np.float32),
                np.zeros(len(decoys), dtype=np.float32)
            ])
        else:
            sequences = hits
            labels = sites['value'].values.astype(np.float32)

        self.dataset = datasets.SequenceDataset(sequences, labels, one_hot=self.one_hot)

        if stage == 'fit':
            # Test and train data split
            train_size = int(self.train_test_split*len(self.dataset))
            test_size = len(self.dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])

        elif stage == 'test':
            self.test_dataset = self.dataset

        elif stage == 'predict':
            self.pred_dataset = self.dataset

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.pred_dataset, batch_size=self.batch_size)
    
'''
Genome scanning data module.
'''
class GenomeDataModule(L.LightningDataModule):
    def __init__(self, data_file, chr_name, seq_length=46, strand='+', num_workers=0, batch_size=32):
        super().__init__()

        print("Initialize genome data module!")
        self.data_file = data_file
        self.chr_name = chr_name
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.strand = strand
        self.rc_aug = (strand == '-')

    def setup(self, stage: str):
        if stage == 'predict':
            self.pred_dataset = datasets.GenomeIterableDataset(
                fasta_file=self.data_file, 
                chr_name=self.chr_name,
                window_length=self.seq_length,
                rc_aug=self.rc_aug
            )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.pred_dataset, num_workers=self.num_workers, pin_memory=True, batch_size=self.batch_size)
