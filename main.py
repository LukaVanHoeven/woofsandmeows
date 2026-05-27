import math
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from bcos.data.transforms import AddInverse
from bcos.models.resnet import resnet50
from bcos.modules.bcosconv2d import BcosConv2d
from bcos.optim import OptimizerFactory, LRSchedulerFactory
from bcos.training.trainer import (
    setup_callbacks,
    setup_loggers,
    put_trainer_args_into_trainer_config,
)
from bcos.experiments.utils import CHECKPOINT_LAST_FILENAME

# This is the reimplementation of the ClassicLitModel from the bcos repo, but with some changes so it doesnt depend on bcos specific code and config
class LitModel(pl.LightningModule):
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
        self.log("train_acc1", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.test_criterion(logits, y)
        self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc1", self.val_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = self.config["optimizer"].create(self.model)
        sched = self.config["lr_scheduler"].create(
            opt, total_steps=self.trainer.estimated_stepping_batches
        )
        return dict(optimizer=opt, lr_scheduler=sched)


def train_bcos(
    model: nn.Module,
    config: dict,
    num_classes: int,
    datamodule: pl.LightningDataModule = None,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    save_dir: str = "./experiments",
    dataset: str = "custom",
    experiment_name: str = "experiment",
    max_epochs: int = 100,
    accelerator: str = "auto",
    devices: str = "auto",
    resume: bool = True,
    fast_dev_run: bool = False,
    track_grad_norm: bool = False,
    amp: bool = False,
    debug: bool = False,
    wandb_logger: bool = False,
    wandb_project: str = None,
    csv_logger: bool = True,
    refresh_rate: int = 5,
):
    if datamodule is None:
        if train_loader is None or val_loader is None:
            raise ValueError("Provide either datamodule or both train_loader and val_loader")

        class _DataModule(pl.LightningDataModule):
            def train_dataloader(self):
                return train_loader
            def val_dataloader(self):
                return val_loader

        datamodule = _DataModule()

    args = SimpleNamespace(
        base_directory=save_dir,
        dataset=dataset,
        base_network="custom",
        experiment_name=experiment_name,
        resume=resume,
        fast_dev_run=fast_dev_run,
        track_grad_norm=track_grad_norm,
        amp=amp,
        debug=debug,
        wandb_logger=wandb_logger,
        wandb_project=wandb_project,
        wandb_name=None,
        wandb_id=None,
        csv_logger=csv_logger,
        tensorboard_logger=False,
        jit=False,
        cache_dataset=None,
        distributed=False,
        explanation_logging=False,
        explanation_logging_every_n_epochs=1,
        refresh_rate=refresh_rate,
    )

    save_path = Path(save_dir, dataset, "custom", experiment_name)
    save_path.mkdir(parents=True, exist_ok=True)

    loggers = setup_loggers(args)
    pl.seed_everything(config.get("seed", 42), workers=True)

    lit_model = LitModel(config, model, num_classes)

    callbacks = setup_callbacks(args, config)

    trainer_config = dict(config.get("trainer", {}))
    trainer_config.setdefault("max_epochs", max_epochs)
    put_trainer_args_into_trainer_config(args, trainer_config)

    pl_trainer = pl.Trainer(
        default_root_dir=save_path,
        accelerator=accelerator,
        devices=devices,
        logger=loggers,
        callbacks=callbacks,
        **trainer_config,
    )

    ckpt_path = None
    if resume:
        ckpt = save_path / CHECKPOINT_LAST_FILENAME
        ckpt_path = ckpt if ckpt.exists() else None

    pl_trainer.fit(lit_model, datamodule=datamodule, ckpt_path=ckpt_path)


def train_cats_vs_dogs():
    train_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomCrop(50),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        AddInverse(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        AddInverse(),
    ])

    train_dataset = ImageFolder("data/train", transform=train_transform)
    val_dataset = ImageFolder("data/val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

    model = resnet50(
        num_classes=2,
        in_chans=6,
        small_inputs=True,
        conv_layer=partial(BcosConv2d, b=2, max_out=2),
    )

    config = {
        "criterion": nn.CrossEntropyLoss(),
        "test_criterion": nn.CrossEntropyLoss(),
        "optimizer": OptimizerFactory(name="Adam", lr=0.001),
        "lr_scheduler": LRSchedulerFactory(name="cosineannealinglr", epochs=10),
        "trainer": {"max_epochs": 10},
        "seed": 42,
    }

    train_bcos(
        model=model,
        config=config,
        num_classes=2,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset="catsdogs",
        experiment_name="bcos_resnet20",
        max_epochs=10,
        csv_logger=True,
    )


def visualize_explanation(
    image_path: str = "cat.png",
    checkpoint_path: str = "experiments/catsdogs/custom/bcos_resnet20/last.ckpt",
    save_path: str = "explanation.png",
):
    from PIL import Image

    model = resnet50(
        num_classes=2,
        in_chans=6,
        small_inputs=True,
        conv_layer=partial(BcosConv2d, b=2, max_out=2),
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = {
        k.removeprefix("model."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    tensor = AddInverse()(transforms.ToTensor()(img)).unsqueeze(0).requires_grad_(True)

    result = model.explain(tensor)
    print(f"Predicted: {['cat','dog'][result['prediction']]} (class {result['prediction']})")

    explanation_img = result["explanation"]
    Image.fromarray((explanation_img * 255).astype("uint8")).save(save_path)
    print(f"Explanation saved to {save_path}")

    with torch.no_grad(), model.explanation_mode():
        probs = torch.softmax(model(tensor), dim=1)
    print(f"Confidence: cat={probs[0,0]:.3f}, dog={probs[0,1]:.3f}")


if __name__ == "__main__":
    train_cats_vs_dogs()
    visualize_explanation()
