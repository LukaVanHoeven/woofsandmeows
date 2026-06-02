import matplotlib.pyplot as plt
import os
import torch

import config as CONFIG

from audio_dataset import CatDogAudioDataset, REVERSE_LABEL_MAP
from create_logger import create_logger
from data import to_dataloaders
from explain import explain_audio
from train import train_bcos

from bcos.modules.bcosconv2d import BcosConv2d
from bcos.modules.losses import BinaryCrossEntropyLoss
from bcos.models.resnet import resnet18
from bcos.optim import OptimizerFactory, LRSchedulerFactory
from functools import partial
from IPython.display import Audio
from sklearn.model_selection import train_test_split

torch.set_float32_matmul_precision("medium") # or "high", tradeoff between speed and precision


def main()-> None:
    dataset = CatDogAudioDataset("data/cats_dogs/")
    logger.debug(dataset)
    logger.debug(len(dataset))
    logger.debug(dataset[0])
    logger.debug(dataset[0][0].shape)
    plt.imshow(dataset[0][0].squeeze(), aspect="auto", origin="lower")
    plt.show()

    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=CONFIG.TEST_SIZE,
        random_state=42
    )
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    logger.debug(f"{len(train_dataset) = }, {len(val_dataset) = }")

    train_dataloader, val_dataloader = to_dataloaders(
        [train_dataset, val_dataset], 
        batch_sizes=[CONFIG.BATCH_SIZE] * 2, 
        shuffles=[True, False],
        logger=logger,
        num_workers=CONFIG.NUM_DATAWORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    model = resnet18(
        num_classes=2, 
        in_chans=1, 
        small_inputs=True,
        conv_layer=partial(BcosConv2d, b=2, max_out=2)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    config = {
            "criterion": BinaryCrossEntropyLoss(),
            "test_criterion": BinaryCrossEntropyLoss(),
            "optimizer": OptimizerFactory(name="Adam", lr=1e-4),
            "lr_scheduler": LRSchedulerFactory(name="cosineannealinglr", epochs=10),
        }

    train_bcos(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        config=config,
        max_epochs=CONFIG.MAX_EPOCHS,
        num_classes=2,
        accelerator="gpu"
    )

    true_label, pred_label, explanation, wav_input, wav_explain = explain_audio(model, "data/cats_dogs/cat_1.wav")

    plt.imshow(explanation.squeeze(), aspect="auto", origin="lower")
    plt.show()

    # print(f"Original: {REVERSE_LABEL_MAP[true_label]}")
    # display(Audio(wav_input.squeeze().cpu().numpy(), rate=CONFIG.SAMPLE_RATE))

    # print(f"Explanation: {REVERSE_LABEL_MAP[pred_label]}")
    # display(Audio(wav_explain.squeeze().detach().cpu().numpy(), rate=CONFIG.SAMPLE_RATE))


if __name__ == "__main__":
    logger = create_logger(
        name="Deep Learning - Assignment 2", 
    )

    logger.debug(torch.__version__)
    logger.debug(torch.version.hip)
    logger.debug(torch.cuda.is_available())
    logger.debug(torch.cuda.get_device_name(0))

    main()
