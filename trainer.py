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
        save_path = os.path.join(self.output_dir, str(dataloader_idx))
        os.makedirs(save_path, exist_ok = True)
        # torch.save(prediction, save_path, f"{batch_idx}.pt")

        # Construct path for output file
        output_file = os.path.join(self.output_dir, "predictions.bed")

        # Construct bed file for positive predictions
        save_data = []  # Use a list of tuples instead of separate lists
        preds, inds = prediction[0], prediction[1]
        hits = torch.nonzero(preds.squeeze())

        preds = torch.flatten(preds[hits]).cpu()
        inds = torch.flatten(inds[hits]).cpu()

        # Save predictions
        pred_bed = pd.DataFrame.from_dict({'chr':self.chr_name, 'start':inds, 'end':inds+self.seq_length, 'pred':preds})

        pred_bed.to_csv(
            output_file, 
            sep='\t', 
            index=None, 
            mode='a', 
            header=not os.path.exists(output_file)
        )

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))


def cli_main():
    cli = LightningCLI(modules.Regression)
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
