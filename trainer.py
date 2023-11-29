# main.py
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import BasePredictionWriter

from lit_modules import data_modules, modules

class BedWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        torch.save(prediction, os.path.join(self.output_dir, dataloader_idx, f"{batch_idx}.pt"))
        
        # Construct path for output file
        output_file = os.path.join(self.output_dir, "predictions.bed")

        # Construct bed file for positive predictions
        save_data = []  # Use a list of tuples instead of separate lists
        for batch in predictions:
            values, inds = batch[0], batch[1]
            hits = torch.nonzero(preds.squeeze() > 1)
            if len(hits) > 0:
                save_data.extend(zip(values[hits].flatten(), inds[hits].flatten()))

        # Unpack the list of tuples into separate lists if needed
        save_preds, save_inds = zip(*save_data)
        inds = torch.flatten(torch.hstack(pos_indices))
        preds = torch.flatten(torch.hstack(pos_preds))

        # Save predictions
        pred_bed = pd.DataFrame.from_dict({'chr':chr_name, 'start':inds, 'end':inds+seq_length, 'pred':preds})
        
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