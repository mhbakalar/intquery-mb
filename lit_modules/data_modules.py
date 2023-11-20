import torch
import lightning as L
import torch.nn.functional as F
import torchmetrics
import genomepy
import pandas as pd
import os
import numpy as np
from .. import models

'''
Multi class data module. Currently designed for three class use – decoy, low, high activity.
'''
class MulticlassDataModule(L.LightningDataModule):
    def __init__(self, data_path, threshold, genomic_reference_file=None, add_decoys=False, n_classes=2, train_test_split=1.0, batch_size=32):
        super().__init__()
        self.data_path = data_path
        self.threshold = threshold
        self.genomic_reference_file = genomic_reference_file
        self.add_decoys = add_decoys
        self.n_classes = n_classes
        self.train_test_split = train_test_split
        self.batch_size = batch_size

    def setup(self, stage: str):
        # Select test/train dataset
        fname = stage + '.csv'

        # Load the cryptic seq data
        sites = pd.read_csv(os.path.join(self.data_path, fname))

        # Threshold the data to assign a label. This code should live somewhere else...
        sites['label'] = (sites['norm_count'] > self.threshold).astype(int)

        # Compute class frequencies for weighting
        class_sample_count = np.array([len(np.where(sites['label'] == c)[0]) for c in np.unique(sites['label'])])

        # Cryptic sites data for training
        sequences = sites['seq'].values
        labels = sites['label'].values

        # Sample weights based on label and class frequency
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in labels])
        self.samples_weight = torch.from_numpy(samples_weight)

        # Convert labels to one hot and build dataset
        one_hot_labels = F.one_hot(torch.tensor(labels), num_classes=self.n_classes)
        self.seq_length = len(sequences[0])
        self.dataset = models.datasets.SequenceDataset(sequences, one_hot_labels)

        if self.add_decoys:
            # Generate random decoy sequences
            decoy_count = len(sites)
            genome = genomepy.genome.Genome(self.genomic_reference_file)
            samples = genome.get_random_sequences(n=decoy_count, length=self.seq_length-1, max_n=0)
            decoys = pd.Series(list(map(lambda row: genome.get_seq(*row).seq.upper(), samples)))
            decoys_labels = np.zeros(len(decoys))

        if stage == 'fit':
            # Test and train data split
            train_size = int(self.train_test_split*len(self.dataset))
            test_size = len(self.dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])

            # Weighted random sampler for upsampling minority class for training
            train_sample_weights = samples_weight[self.train_dataset.indices]
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
    def __init__(self, data_file, batch_size=64):
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == 'predict':
            self.pred_dataset = models.datasets.GenomeBoxcarDataset(fasta_file=self.data_file)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.pred_dataset, pin_memory=True, batch_size=self.batch_size)
