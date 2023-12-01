# main.py
import os
import torch
import pandas as pd
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import BasePredictionWriter

from lit_modules import data_modules, modules

class BedWriter(BasePredictionWriter):

    def __init__(self, output_dir, chr_name, seq_length, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.chr_name = chr_name
        self.seq_length = seq_length

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # Construct path for output file
        output_file = os.path.join(self.output_dir, f"{self.chr_name}.bed")

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
        pred_bed = pd.DataFrame.from_dict({'chr':self.chr_name, 'start':inds, 'end':inds+self.seq_length, 'pred':preds})

        pred_bed.to_csv(
            output_file, 
            sep='\t', 
            index=None, 
            header=not os.path.exists(output_file)
        )

        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments('data.chr_name', 'trainer.callbacks.init_args.chr_name')

def cli_main():
    cli = CLI(modules.Regression)
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
