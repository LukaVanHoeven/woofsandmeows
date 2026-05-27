from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from bcos.data.transforms import AddInverse
from bcos.models.resnet import resnet50
from bcos.modules.bcosconv2d import BcosConv2d
from bcos.optim import OptimizerFactory, LRSchedulerFactory


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
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_dir: str = "./experiments",
    experiment_name: str = "experiment",
    max_epochs: int = 100,
    accelerator: str = "auto",
    devices: str = "auto",
    seed: int = 42,
):
    pl.seed_everything(seed, workers=True)

    save_path = Path(save_dir, experiment_name)
    save_path.mkdir(parents=True, exist_ok=True)

    lit_model = LitModel(config, model, num_classes)

    trainer = pl.Trainer(
        default_root_dir=save_path,
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            ModelCheckpoint(
                dirpath=save_path,
                monitor="val_acc1",
                mode="max",
                filename="{epoch}-{val_acc1:.4f}",
                save_last=True,
                save_top_k=3,
            ),
            TQDMProgressBar(refresh_rate=5),
        ],
        logger=CSVLogger(save_dir=str(save_path / "logs")),
        num_sanity_val_steps=0,
    )

    trainer.fit(lit_model, train_loader, val_loader)


def train_cats_vs_dogs():
    pl.seed_everything(41, workers=True)

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
    }

    train_bcos(
        model=model,
        config=config,
        num_classes=2,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name="bcos_resnet20",
        max_epochs=10,
    )


def visualize_explanation(
    image_path: str = "cat.png",
    checkpoint_path: str = "experiments/bcos_resnet20/last.ckpt",
    save_path: str = "explanation.png",
):
    from PIL import Image

    # This is the bcos model that is used in the paper as well
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
