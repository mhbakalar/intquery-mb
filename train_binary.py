import os
import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from pyfaidx import Fasta
import utils.fasta_data

# Local cryptic module imports
from lit_modules import data_modules, modules
from models import models

if __name__ == "__main__":
    # Set parameters (add CLI interface soon)
    data_path = './data/TB000208a'
    decoy_path = './data/decoys'
    genomic_reference_file = '../data/reference/hg38.fa'
    seq_length = 46
    vocab_size = 5
    input_size = seq_length*vocab_size
    hidden_size = 128
    n_hidden = 2
    train_test_split = 0.8
    batch_size = 64
    decoy_mul = 2
    dropout = 0.5
    lr = 0.001

    # Build the data module
    data_module = data_modules.BinaryDataModule(
        data_path,
        decoy_path=decoy_path,
        decoy_mul=decoy_mul,
        sequence_length=seq_length,
        genomic_reference_file=genomic_reference_file,
        train_test_split=train_test_split,
        batch_size=batch_size,
        smooth_labels=True)

    # Build model
    lit_model = modules.BinaryClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        n_hidden=n_hidden,
        dropout=dropout,
        lr=lr)

    # train the model
    tb_logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs/")
    trainer = pl.Trainer(max_epochs=5, logger=tb_logger, default_root_dir='.')
    trainer.fit(lit_model, data_module)

    # test the model
    from matplotlib import pyplot as plt
    trainer.test(lit_model, data_module)
    batch_preds = trainer.predict(lit_model, data_module)
    preds = torch.vstack([batch[0] for batch in batch_preds]).flatten()
    labels = torch.hstack([batch[1] for batch in batch_preds])
    df = pd.DataFrame({'label':labels, 'pred':preds})
    df[df['label'] == 0.9]['pred'].hist()
    plt.show()

    # Evaluate on chromosome 1
    # Fast prediction code. Currently runs on chrom 1
    chromosomes = list(Fasta(genomic_reference_file).keys())
    chr_name = chromosomes[0]
    pred_data_module = data_modules.GenomeDataModule(genomic_reference_file, chr=chr_name, batch_size=batch_size, num_workers=0)
    batch_preds = trainer.predict(lit_model, pred_data_module)

    # Construct bed file for positive predictions
    pos_indices = []
    pos_preds = []
    for batch in batch_preds:
        preds, inds = batch[0], batch[1]
        hits = torch.nonzero(preds.squeeze() > 0.8)
        pos_preds.append(preds[hits].flatten())
        pos_indices.append(inds[hits].flatten())
    
    flat_indices = torch.flatten(torch.hstack(pos_indices))
    flat_preds = torch.flatten(torch.hstack(pos_preds))
    pred_bed = pd.DataFrame.from_dict({'chr':chr_name, 'start':flat_indices, 'end':flat_indices+seq_length, 'pred':flat_preds})
    pred_bed.to_csv('output/chr1_positive.bed', sep='\t', index=None)

    # Print sequences
    fasta_file = '../data/reference/hg38.fa'
    fasta = utils.fasta_data.FastaInterval(fasta_file=fasta_file, return_seq=True)
    bed_file = pd.read_csv('output/chr1_positive.bed', sep='\t')
    print(bed_file.head())

    for i, row in bed_file.iterrows():
        print(fasta(row['chr'], int(row['start']), int(row['end'])).upper())
    
