import torch
import lightning.pytorch as pl
import torchmetrics

'''
Wraps a multiclass classifier in Lightning
'''
class Classifier(pl.LightningModule):
    def __init__(self, model, n_classes):
        super().__init__()
        self.model = model
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
        loss = self.loss_fn(logits, target.float())
        self.logging(logits, target, loss, 'test')
        return loss
    
    def predict_step(self, batch, batch_idx):
        # Model pass
        data = batch
        logits = self(data)
        # output = self.sigmoid(self(data))
        # preds = torch.argmax(output, 1)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer