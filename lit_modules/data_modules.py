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
Multi class data module. Currently designed for three class use – decoy, low, high activity.
Code could be cleaned up here.
'''
class MulticlassDataModule(L.LightningDataModule):
    def __init__(self, 
        data_path, 
        threshold, 
        genomic_reference_file=None, 
        add_decoys=True, 
        n_classes=3, 
        sequence_length=46,
        train_test_split=1.0, 
        batch_size=32
    ):
        super().__init__()
        self.data_path = data_path
        self.threshold = threshold
        self.genomic_reference_file = genomic_reference_file
        self.add_decoys = add_decoys
        self.n_classes = n_classes
        self.sequence_length = sequence_length
        self.train_test_split = train_test_split
        self.batch_size = batch_size

    def setup(self, stage: str):
        # Select test/train dataset
        fname = stage + '.csv'

        # Load the cryptic seq data
        sites = pd.read_csv(os.path.join(self.data_path, fname))

        # Threshold the data to assign a label. This code should live somewhere else...
        sites['label'] = (sites['norm_count'] > self.threshold).astype(int)

        # Cryptic sites data for training
        sequences = sites['seq'].values
        labels = sites['label'].values

        # Add space for decoys with 0 label if used
        if self.add_decoys:
            labels = labels + 1

        # Compute sample weights based on label and class frequency
        class_sample_count = np.array([len(np.where(labels == c)[0]) for c in np.arange(0,self.n_classes)])
        weight = 1. / class_sample_count
        sample_weights = np.array([weight[t] for t in labels])
        self.sample_weights = torch.from_numpy(sample_weights)

        # Convert labels to one hot and build dataset
        one_hot_labels = F.one_hot(torch.tensor(labels), num_classes=self.n_classes)
        self.seq_length = len(sequences[0])
        self.dataset = datasets.SequenceDataset(sequences, one_hot_labels)

        if self.add_decoys:
            # Generate random decoy sequences and update dataset
            n_decoys = len(sites)
            decoy_weight = 1. / n_decoys
            decoy_label = F.one_hot(torch.tensor(0), num_classes=3)
            decoy_dataset = datasets.DecoyDataset(
                fasta_file=self.genomic_reference_file,
                n_decoys=n_decoys,
                length=self.sequence_length,
                label=decoy_label)
            self.dataset = torch.utils.data.ConcatDataset([self.dataset, decoy_dataset])
            self.sample_weights = torch.cat([self.sample_weights, torch.ones(n_decoys)*decoy_weight])

        if stage == 'fit':
            # Test and train data split
            train_size = int(self.train_test_split*len(self.dataset))
            test_size = len(self.dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])

            # Weighted random sampler for upsampling minority class for training
            train_sample_weights = self.sample_weights[self.train_dataset.indices]
            self.train_sampler = torch.utils.data.WeightedRandomSampler(train_sample_weights, len(train_sample_weights), replacement=True)

        elif stage == 'test':
            self.test_dataset = self.dataset

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
Genome scanning data module.
'''
class GenomeDataModule(L.LightningDataModule):
    def __init__(self, data_file, chr, num_workers=0, batch_size=32):
        super().__init__()
        self.data_file = data_file
        self.chr= chr
        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == 'predict':
            self.pred_dataset = datasets.GenomeBoxcarDataset(fasta_file=self.data_file, 
                                                                    chr=self.chr,
                                                                    window_length=46,
                                                                    read_ahead=self.batch_size*46)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.pred_dataset, num_workers=self.num_workers, pin_memory=True, batch_size=self.batch_size)
