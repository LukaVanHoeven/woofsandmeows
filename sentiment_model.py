import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics


class SentimentModel(pl.LightningModule):
    def __init__(self, config: dict, model: nn.Module, num_classes: int):
        super().__init__()
        self.config = config
        self.model = model
        self.criterion = config["criterion"]
        self.test_criterion = config["test_criterion"]
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.test_criterion(logits, y)
        self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = self.config["optimizer"].create(self.model)
        sched = self.config["lr_scheduler"].create(
            opt, total_steps=self.trainer.estimated_stepping_batches
        )
        return dict(optimizer=opt, lr_scheduler=sched)