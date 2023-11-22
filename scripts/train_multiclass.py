import os
import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from pyfaidx import Fasta

# Set up path to import parent modules
from pathlib import Path
import sys  

# Add to sys.path
sys.path.insert(0, str(Path().resolve().parents[1]))

# Local cryptic module imports
import cryptic.models as models
from cryptic.lit_modules import data_modules, modules
from cryptic.models import models

if __name__ == "__main__":
    # Set parameters (add CLI interface soon)
    data_path = '../data/TB000208a'
    genomic_reference_file = '../../data/reference/hg38.fa'
    n_classes = 2
    seq_length = 22
    vocab_size = 4
    input_size = seq_length*vocab_size
    hidden_size = 8
    n_hidden = 2
    train_test_split = 0.8
    batch_size=128

    # Build the data module
    data_module = data_modules.MulticlassDataModule(data_path, threshold=0.01, n_classes=n_classes, train_test_split=train_test_split, batch_size=batch_size)

    # Build model
    model = models.MLPModel(input_size=input_size, hidden_size=hidden_size, output_size=n_classes, n_hidden=n_hidden, dropout=0.5)
    lit_model = modules.Classifier(model, n_classes)

    # train the model
    tb_logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs/")
    trainer = pl.Trainer(max_epochs=1, logger=tb_logger, default_root_dir='.')
    trainer.fit(lit_model, data_module)

    # Evaluate on chromosome 1
    # Fast prediction code. Currently runs on chrom 1
    chromosomes = list(Fasta(genomic_reference_file).keys())
    chr_name = chromosomes[0]
    pred_data_module = data_modules.GenomeDataModule(genomic_reference_file, chr=chr_name, batch_size=batch_size, num_workers=0)
    batch_preds = trainer.predict(lit_model, pred_data_module)
    
    # Construct bed file for positive predictions
    pos_indices = []
    for batch in batch_preds:
        pred, indices = batch[0], batch[1]
        pos_indices.append(indices[torch.nonzero(pred)])
    
    flat_indices = torch.flatten(torch.vstack(pos_indices))
    pred_bed = pd.DataFrame.from_dict({'chr':chr_name, 'start':flat_indices, 'end':flat_indices+seq_length})
    print(pred_bed)
    
