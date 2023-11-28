import os
import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from pyfaidx import Fasta
import argparse

# Local cryptic module imports
from lit_modules import data_modules, modules
from models import models

if __name__ == "__main__":
    # Set parameters (add CLI interface soon)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--chrom')
    parser.add_argument('-o', '--outfile')

    args = parser.parse_args()

    data_path = './data/TB000208a'
    decoy_path = './data/decoys'
    genomic_reference_file = '../data/reference/hg38.fa'
    bed_output_file = args.outfile
    seq_length = 46
    vocab_size = 5
    input_size = seq_length*vocab_size
    hidden_size = 2000
    n_hidden = 1
    train_test_split = 0.8
    batch_size = 128
    decoy_mul = 1
    dropout = 0.5
    lr = 0.001
    epochs = 15
    
    # Prediction parameters
    chr_name = args.chrom

    # Build the data module
    data_module = data_modules.NumericDataModule(
        data_path,
        decoy_path=decoy_path,
        decoy_mul=decoy_mul,
        sequence_length=seq_length,
        genomic_reference_file=genomic_reference_file,
        train_test_split=train_test_split,
        batch_size=batch_size,
        log_transform=True)

    # Build model
    lit_model = modules.Regression(
        input_size=input_size,
        hidden_size=hidden_size,
        n_hidden=n_hidden,
        dropout=dropout,
        lr=lr)

    # train the model
    tb_logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs/")
    trainer = pl.Trainer(max_epochs=epochs, logger=tb_logger, default_root_dir='.')
    trainer.fit(lit_model, data_module)

    # Fast prediction code.
    pred_data_module = data_modules.GenomeDataModule(genomic_reference_file, chr=chr_name, batch_size=batch_size, num_workers=10)
    batch_preds = trainer.predict(lit_model, pred_data_module)

    # Construct bed file for positive predictions
    pos_indices = []
    pos_preds = []
    for batch in batch_preds:
        preds, inds = batch[0], batch[1]
        hits = torch.nonzero(preds.squeeze() > 5)
        if len(hits) > 0:
            pos_preds.append(preds[hits].flatten())
            pos_indices.append(inds[hits].flatten())
    
    flat_indices = torch.flatten(torch.hstack(pos_indices))
    flat_preds = torch.flatten(torch.hstack(pos_preds))

    # Save predictions
    pred_bed = pd.DataFrame.from_dict({'chr':chr_name, 'start':flat_indices, 'end':flat_indices+seq_length, 'pred':flat_preds})
    pred_bed.to_csv(bed_output_file, sep='\t', index=None)