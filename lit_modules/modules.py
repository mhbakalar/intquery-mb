import torch
import lightning.pytorch as pl
import torchmetrics

# Need to refactor this directory name
from models import models

'''
Binary classifier
'''
class BinaryClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, n_hidden, dropout=0.5, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.lr = lr

        self.model = models.MLPModel(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            output_size=1, 
            n_hidden=self.n_hidden, 
            dropout=0.5
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task="binary")

    def forward(self, x):
        return self.model(x)
    
    def logging(self, logits, target, loss, stage):
        # Logging to TensorBoard (if installed) by default
        pred = self.sigmoid(logits)
        self.accuracy(torch.round(pred.squeeze()), torch.round(target))
        self.log(f'{stage}_acc', self.accuracy)
        self.log(f'{stage}_loss', loss)

    def training_step(self, batch, batch_idx):
        # Model pass
        data, target = batch
        logits = self(data)
        loss = self.loss_fn(logits.squeeze(), target.float())
        self.logging(logits, target, loss, 'train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Model pass
        data, target = batch
        logits = self(data)
        loss = self.loss_fn(logits.squeeze(), target.float())
        self.logging(logits, target, loss, 'val')
        return loss
    
    def test_step(self, batch, batch_idx):
        # Model pass
        data, target = batch
        logits = self(data)
        preds = self.sigmoid(logits)
        loss = self.loss_fn(logits.squeeze(), target.float())
        self.logging(logits, target, loss, 'test')
        return loss, preds
    
    def predict_step(self, batch, batch_idx):
        # Model pass
        data, ind = batch
        logits = self(data)
        preds = self.sigmoid(logits)
        return preds, ind

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class Regression(pl.LightningModule):
    def __init__(self, seq_length, vocab_size, hidden_size, n_hidden, dropout=0.5, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.lr = lr

        self.model = models.MLPModel(
            seq_length=self.seq_length,
            vocab_size=self.vocab_size, 
            hidden_size=self.hidden_size, 
            output_size=1, 
            n_hidden=self.n_hidden, 
            dropout=self.dropout
        )
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
    def logging(self, logits, target, loss, stage):
        # Logging to TensorBoard (if installed) by default
        self.log(f'{stage}_loss', loss, on_epoch=True)

    def training_step(self, batch, batch_idx):
        # Model pass
        data, target = batch
        output = self(data)
        loss = self.loss_fn(output.squeeze(), target.float())
        self.logging(output, target, loss, 'train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Model pass
        data, target = batch
        output = self(data)
        loss = self.loss_fn(output.squeeze(), target.float())
        self.logging(output, target, loss, 'val')
        return loss
    
    def test_step(self, batch, batch_idx):
        # Model pass
        data, target = batch
        output = self(data)
        loss = self.loss_fn(output.squeeze(), target.float())
        self.logging(output, target, loss, 'test')
        return loss
    
    def predict_step(self, batch, batch_idx):
        # Model pass
        data, ind = batch
        output = self(data)
        return output, ind

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
class RegressionModule(pl.LightningModule):
    def __init__(self, basemodel, lr=1e-3):
        super().__init__()
        #self.save_hyperparameters()
        self.model = basemodel
        self.lr = lr
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
    def logging(self, logits, target, loss, stage):
        # Logging to TensorBoard (if installed) by default
        self.log(f'{stage}_loss', loss, on_epoch=True)

    def training_step(self, batch, batch_idx):
        # Model pass
        data, target = batch
        output = self(data)
        loss = self.loss_fn(output.squeeze(), target.float())
        self.logging(output, target, loss, 'train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Model pass
        data, target = batch
        output = self(data)
        loss = self.loss_fn(output.squeeze(), target.float())
        self.logging(output, target, loss, 'val')
        return loss
    
    def test_step(self, batch, batch_idx):
        # Model pass
        data, target = batch
        output = self(data)
        loss = self.loss_fn(output.squeeze(), target.float())
        self.logging(output, target, loss, 'test')
        return loss
    
    def predict_step(self, batch, batch_idx):
        # Model pass
        data, ind = batch
        output = self(data)
        return output, ind

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer