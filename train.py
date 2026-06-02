import pytorch_lightning as pl
import torch.nn as nn

from torch.utils.data import DataLoader
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from sentiment_model import SentimentModel


def train_bcos(
    model: nn.Module,
    config: dict,
    num_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_dir: str = "./experiments",
    experiment_name: str = "experiment",
    max_epochs: int = 50,
    accelerator: str = "auto",
    devices: str = "auto",
    seed: int = 42,
):
    pl.seed_everything(seed, workers=True)

    save_path = Path(save_dir, experiment_name)
    save_path.mkdir(parents=True, exist_ok=True)

    sentiment_model = SentimentModel(config, model, num_classes)

    trainer = pl.Trainer(
        default_root_dir=save_path,
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            ModelCheckpoint(
                dirpath=save_path,
                monitor="val_acc",
                mode="max",
                filename="{epoch}-{val_acc:.4f}",
                save_last=True,
                save_top_k=3,
            ),
            TQDMProgressBar(refresh_rate=5),
        ],
        logger=CSVLogger(save_dir=str(save_path / "logs")),
        num_sanity_val_steps=0,
    )

    trainer.fit(sentiment_model, train_loader, val_loader)
