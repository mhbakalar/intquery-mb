import torch
import lightning.pytorch as pl
import torchmetrics

# Need to refactor this directory name
from models import models

'''
Wraps a multiclass classifier in Lightning
'''
class Classifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, n_hidden, n_classes, dropout=0.5, lr=1e-3):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.dropout = dropout
        self.lr = lr
        
        self.model = models.MLPModel(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            output_size=self.n_classes, 
            n_hidden=self.n_hidden, 
            dropout=0.5
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, x):
        return self.model(x)
    
    def logging(self, logits, target, loss, stage):
        # Logging to TensorBoard (if installed) by default
        pred = self.sigmoid(logits)
        self.accuracy(torch.argmax(pred,1), torch.argmax(target,1))
        self.log(f'{stage}_acc_step', self.accuracy)
        self.log(f'{stage}_loss', loss)

    def training_step(self, batch, batch_idx):
        # Model pass
        data, target = batch
        logits = self(data)
        loss = self.loss_fn(logits, target.float())
        self.logging(logits, target, loss, 'train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Model pass
        data, target = batch
        logits = self(data)
        loss = self.loss_fn(logits, target.float())
        self.logging(logits, target, loss, 'val')
        return loss
    
    def test_step(self, batch, batch_idx):
        # Model pass
        data, target = batch
        logits = self(data)
        output = self.sigmoid(logits)
        preds = torch.argmax(output, 1)

        loss = self.loss_fn(logits, target.float())
        self.logging(logits, target, loss, 'test')
        return loss, preds
    
    def predict_step(self, batch, batch_idx):
        # Model pass
        data, ind = batch
        logits = self(data)
        output = self.sigmoid(logits)
        preds = torch.argmax(output, 1)
        return preds, ind

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
