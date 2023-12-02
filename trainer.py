# main.py
import os
import torch
import pandas as pd
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import BasePredictionWriter

from lit_modules import data_modules, modules

class BedWriter(BasePredictionWriter):

    def __init__(self, output_dir, chr_name, seq_length, strand, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.chr_name = chr_name
        self.seq_length = seq_length
        self.strand = strand

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # Construct bed file for positive predictions
        save_data = []  # Use a list of tuples instead of separate lists
        pos_indices = []
        pos_preds = []
        for batch in predictions:
            preds, inds = batch[0], batch[1]
            hits = torch.nonzero(preds.squeeze() > 1.0)
            pos_preds.append(preds[hits].flatten())
            pos_indices.append(inds[hits].flatten())

        inds = torch.flatten(torch.hstack(pos_indices))
        preds = torch.flatten(torch.hstack(pos_preds))

        # Save predictions
        output_file = os.path.join(self.output_dir, f"{self.chr_name}.bed")
        pred_bed = pd.DataFrame.from_dict({'chr':self.chr_name, 'start':inds, 'end':inds+self.seq_length, 'pred':preds, 'strand':self.strand})

        pred_bed.to_csv(
            output_file,
            sep='\t',
            index=None,
            mode='a',
            header=not os.path.exists(output_file),
        )

        torch.save(predictions, os.path.join(self.output_dir, f"{self.chr_name}_predictions.pt"))

def cli_main():
    cli = LightningCLI(modules.Regression)
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
