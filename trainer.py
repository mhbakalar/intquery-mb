# main.py
import os
import torch
import pandas as pd
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import BasePredictionWriter

from lit_modules import data_modules, modules

class BedWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_threshold=0.0, write_interval='epoch'):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.write_threshold = write_threshold

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # Predict on genomic dataset
        if isinstance(trainer.datamodule, data_modules.GenomeDataModule):
            self.write_name = trainer.datamodule.chr_name
            self.seq_length = trainer.datamodule.seq_length
            self.strand = trainer.datamodule.strand

            # Construct bed file for positive predictions
            pos_indices = []
            pos_preds = []
            for batch in predictions:
                preds, inds = batch[0], batch[1]
                hits = torch.nonzero(preds.squeeze() > self.write_threshold)
                pos_preds.append(preds[hits].flatten())
                pos_indices.append(inds[hits].flatten())

            inds = torch.flatten(torch.hstack(pos_indices))
            preds = torch.flatten(torch.hstack(pos_preds))

            # Save predictions
            pred_bed = pd.DataFrame.from_dict({'chr':self.write_name, 'start':inds, 'end':inds+self.seq_length, 'pred':preds, 'strand':self.strand})
            pred_bed = pred_bed.sort_values('start')

        # Predict on numeric dataset
        elif isinstance(trainer.datamodule, data_modules.NumericDataModule):
            self.write_name = os.path.basename(trainer.datamodule.data_path)

            # Construct bed file for positive predictions
            pos_indices = []
            pos_preds = []
            for batch in predictions:
                preds, inds = batch[0], batch[1]
                pos_preds.append(preds.flatten())
                pos_indices.append(inds.flatten())

            inds = torch.flatten(torch.hstack(pos_indices))
            preds = torch.flatten(torch.hstack(pos_preds))

            # Save predictions
            pred_bed = pd.DataFrame.from_dict({'chr':self.write_name, 'ind':inds, 'pred':preds})


        output_file = os.path.join(self.output_dir, f"{self.write_name}.bed")

        pred_bed.to_csv(
            output_file,
            sep='\t',
            index=None,
            mode='a',
            header=not os.path.exists(output_file),
        )
        
        # Uncomment to save all predictions (takes up lots of disk space!)
        # torch.save(predictions, os.path.join(self.output_dir, f"{chr_name}_predictions.pt"))


def cli_main():
    cli = LightningCLI(modules.Regression)
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
