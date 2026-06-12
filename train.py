"""
DISCLAIMER: 
This code was previously part of Joris Heemskerk's prior
work for the Computer Vision course, and is being re-used here.
"""

import copy
import logging
import torch

import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from typing import Callable

from tune import OptunaPruningCallback

METRICS = {
    "accuracy": lambda y_hat, y: (y_hat.argmax(dim=-1) == y).float().mean(),
}


def train_cross_validation(
    full_train_dataset: Dataset,
    k_folds: int,
    dataset_to_dataloader_function: Callable,
    model: nn.Module,
    loss_fn: nn.Module,
    optimiser: torch.optim.Optimizer,
    n_epochs: int,
    device: str,
    logger: logging.Logger,
    pruning_callback: OptunaPruningCallback=None,
)-> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, np.ndarray],
    nn.Module,
]:
    """
    Train a model for `n_epochs` epochs using k-fold cross validation.

    :param full_train_dataset: Dataset to train with.
    :type full_train_dataset: Dataset
    :param k_folds: The number of folds to use.
    :type k_folds: int
    :param dataset_to_dataloader_function: Function that converts a 
        Dataset into a DataLoader.
    :type dataset_to_dataloader_function: Callable
    :param model: Model to train.
    :type model: nn.Module
    :param loss_fn: Loss function to update gradients with.
    :type loss_fn: nn.Module
    :param optimiser: Optimiser used for backpropagation.
    :type optimiser: torch.optim.Optimizer
    :param n_epochs: Number of epochs to train for.
    :type n_epochs: int
    :param device: Device to move data to.
    :type device: str
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :param pruning_callback: Optional parameter; used in training for 
        hyperband pruning. (DEFAULT=None)
    :type pruning_callback: OptunaPruningCallback
    :return: Per-fold, per-epoch train losses and metrics and validation
        losses and metrics as numpy arrays, along with the model 
        checkpoint that achieved the best validation loss across all 
        folds.
    :rtype: tuple[
        np.ndarray,
        dict[str, np.ndarray],
        np.ndarray,
        dict[str, np.ndarray],
        nn.Module
    ]
    """
    best = None
    best_val_loss = float("inf")

    train_losses_per_fold: list[list[float]] = []
    val_losses_per_fold: list[list[float]] = []
    train_metrics_per_fold: list[dict[str, list[float]]] = []
    val_metrics_per_fold: list[dict[str, list[float]]] = []

    # Save initial states so every fold starts from the same weights.
    initial_model_state = copy.deepcopy(model.state_dict())
    initial_optimiser_state = copy.deepcopy(optimiser.state_dict())

    s, e = "-----=====", "=====-----"
    for k in range(k_folds):
        logger.info(f"{s}{'#' * (len(str(k+1)) + len(str(k_folds)) + 10)}{e}")
        logger.info(f"{s}# Fold {k+1}/{k_folds} #{e}")
        logger.info(f"{s}{'#' * (len(str(k+1)) + len(str(k_folds)) + 10)}{e}")

        model.load_state_dict(copy.deepcopy(initial_model_state))
        optimiser.load_state_dict(copy.deepcopy(initial_optimiser_state))

        try:
            logger.debug("Attempting to fold per person.")
            # Only works if k_folds == participants.
            train_idx, val_idx = full_train_dataset.get_person_fold_indices(
                k, 
                k_folds, 
                logger
            )
        except NotImplementedError:
            logger.debug("Failed. Now folding per file.")
            train_idx, val_idx = full_train_dataset.get_fold_indices(
                k, 
                k_folds, 
                logger
            )
        full_train_dataset.fit_normalisation(train_idx)

        train_dataloader = dataset_to_dataloader_function(
            Subset(full_train_dataset, train_idx)
        )[0]
        val_dataloader = dataset_to_dataloader_function(
            Subset(full_train_dataset, val_idx)
        )[0]

        train_losses, train_metrics, val_losses, val_metrics, _ = train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimiser=optimiser,
            n_epochs=n_epochs,
            device=device,
            logger=logger,
            pruning_callback=pruning_callback,
            pbar_preamble=f"Fold {k + 1}/{k_folds} -"
        )

        # Track the model from the fold with the lowest mean val loss.
        fold_mean_val_loss = float(np.mean(val_losses))
        if fold_mean_val_loss < best_val_loss:
            best_val_loss = fold_mean_val_loss
            best = copy.deepcopy(model.state_dict())

        train_losses_per_fold.append(train_losses)
        val_losses_per_fold.append(val_losses)
        train_metrics_per_fold.append(train_metrics)
        val_metrics_per_fold.append(val_metrics)

    model.load_state_dict(best)

    train_metrics_combined = {
        k: np.array([fold[k] for fold in train_metrics_per_fold])
        for k in METRICS
    }
    val_metrics_combined = {
        k: np.array([fold[k] for fold in val_metrics_per_fold])
        for k in METRICS
    }

    return (
        np.array(train_losses_per_fold),
        train_metrics_combined,
        np.array(val_losses_per_fold),
        val_metrics_combined,
        model,
    )

def train(
    train_dataloader: DataLoader, 
    val_dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module, 
    optimiser: torch.optim.Optimizer,
    n_epochs: int,
    device: str,
    logger: logging.Logger,
    pruning_callback: OptunaPruningCallback=None,
    pbar_preamble: str=''
)-> tuple[
    list[float],
    dict[str, list[float]],
    list[float],
    dict[str, list[float]],
    nn.Module
]:
    """
    Train a model for `n_epochs` epochs.

    :param train_dataloader: Dataset to train with.
    :type train_dataloader: DataLoader
    :param val_dataloader: Dataset to validate with.
    :type val_dataloader: DataLoader
    :param model: Model to train.
    :type model: nn.Module
    :param loss_fn: Loss function to update gradients with.
    :type loss_fn: nn.Module
    :param optimiser: Optimiser used for backpropagation.
    :type optimiser: torch.optim.Optimizer
    :param n_epochs: Number of epochs to train for.
    :type n_epochs: int
    :param device: Device to move data to.
    :type device: str
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :param pruning_callback: Optional parameter; used in training for 
        hyperband pruning. (DEFAULT=None)
    :type pruning_callback: OptunaPruningCallback
    :param pbar_preamble: Preamble for the progress bar.
    :type pbar_preamble: str
    :return: Per epoch train losses and metrics and validation losses 
        and metrics. Along with the model checkpoint that achieved the 
        best validation loss.
    :rtype: tuple[
        list[float],
        dict[str, list[float]],
        list[float],
        dict[str, list[float]],
        nn.Module
    ]
    """
    best = None
    train_losses_per_epoch, train_metrics_per_epoch = [], [] 
    val_losses_per_epoch, val_metrics_per_epoch = [], []
    for i in tqdm(
        range(n_epochs), 
        f"\033[33m{pbar_preamble}{' ' if len(pbar_preamble) > 0 else ''}Epoch"
    ):
        print("\033[37m", end="") # Reset colour.
        logger.info(f"-----===== Epoch {i} (training) =====-----")
        train_loss, train_metrics = train_epoch(
            train_dataloader, 
            model, 
            loss_fn, 
            optimiser,
            device,
            logger
        )
        train_losses_per_epoch.append(train_loss)
        train_metrics_per_epoch.append(train_metrics)

        logger.info(f"-----===== Epoch {i} (validation) =====-----")
        val_loss, val_metrics = val_epoch(
            val_dataloader, 
            model, 
            loss_fn, 
            device,
            logger
        )

        if val_loss < min(
            val_losses_per_epoch
        ) if len(val_losses_per_epoch) > 0 else float("inf"):
            best = copy.deepcopy(model.state_dict())
        val_losses_per_epoch.append(val_loss)
        val_metrics_per_epoch.append(val_metrics)

        if pruning_callback is not None:
            pruning_callback(val_loss)

    logger.info("Done training")
    model.load_state_dict(best)

    train_metrics = {
        k: [d[k] for d in train_metrics_per_epoch] for k in METRICS
    }
    val_metrics = {k: [d[k] for d in val_metrics_per_epoch] for k in METRICS}
    return \
        train_losses_per_epoch, \
        train_metrics, \
        val_losses_per_epoch, \
        val_metrics, \
        model

def train_epoch(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module, 
    optimiser: torch.optim.Optimizer,
    device: str,
    logger: logging.Logger
)-> tuple[float, dict[str, float]]:
    """
    Train a model for 1 epoch.
 
    :param dataloader: Dataset to train with.
    :type dataloader: DataLoader
    :param model: Model to train.
    :type model: nn.Module
    :param loss_fn: Loss function to update gradients with.
    :type loss_fn: nn.Module
    :param optimiser: Optimiser used for backpropagation.
    :type optimiser: torch.optim.Optimizer
    :param device: Device to move data to.
    :type device: str
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :return: Average training loss and metrics over the epoch.
    :rtype: tuple[float, dict[str, float]]
    """
    scaler = torch.amp.GradScaler(device)

    total_loss = 0
    train_metrics = {metric: [] for metric in METRICS}
    log_points = set(
        np.linspace(0, len(dataloader) - 1, 10, dtype=int)
    )

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device,
            dtype=torch.bfloat16
        ):
            y_hat = model(X).squeeze(-1)
            loss = loss_fn(y_hat, y)

        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        total_loss += loss.item()
        for metric, method in METRICS.items():
            train_metrics[metric].append(method(y_hat, y).item())

        if batch in log_points:
            current = batch * len(y) + len(X)
            metrics_string = ", ".join(
                f"{metric}: {np.mean(train_metrics[metric]):>2f}"
                for metric in METRICS.keys()
            )
            logger.debug(
                f"train loss: {loss.item():>7f} | "
                f"{metrics_string} | "
                f"[{current:>5d}/{len(dataloader.dataset):>5d}]"
            )

    return \
        total_loss / len(dataloader), \
        {key: np.mean(value) for key, value in train_metrics.items()}

def val_epoch(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module,
    device: str,
    logger: logging.Logger
)-> tuple[float, dict[str, float]]:
    """
    Validate loss for a given dataset and model.
 
    :param dataloader: Dataset to validate with.
    :type dataloader: DataLoader
    :param model: Model to validate.
    :type model: nn.Module
    :param loss_fn: Loss function to validate with.
    :type loss_fn: nn.Module
    :param device: Device to move data to.
    :type device: str
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :return: Average validation loss and metrics over the epoch.
    :rtype: tuple[float, dict[str, float]]
    """
    total_loss = 0
    val_metrics = {metric: [] for metric in METRICS}

    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            with torch.autocast(
                device_type=device,
                dtype=torch.bfloat16
            ):
                y_hat = model(X).squeeze(-1)
                loss = loss_fn(y_hat, y)

            total_loss += loss.item()
            for metric, method in METRICS.items():
                val_metrics[metric].append(method(y_hat, y).item())
            
    val_loss = total_loss / len(dataloader)
    metrics_string = ", ".join(
        f"{metric}: {np.mean(val_metrics[metric]):>2f}"
        for metric in METRICS.keys()
    )
    logger.debug(f"Avg loss: {val_loss:>8f} | {metrics_string} |\n")

    return \
        val_loss, \
        {key: np.mean(value) for key, value in val_metrics.items()}

def evaluate(
    dataloader: DataLoader, 
    model: nn.Module,
    device: str,
    logger: logging.Logger,
)-> tuple[float, torch.Tensor, torch.Tensor]:
    """
    Evaluate a model by calculating the METRICS on a dataset.

    :param dataloader: Dataset to validate with.
    :type dataloader: DataLoader
    :param model: Model to evaluate.
    :type model: nn.Module
    :param device: Device to move data to.
    :type device: str
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :return: METRICS on the dataset, predictions made and corresponding
        targets.
    :rtype: tuple[float, torch.Tensor, torch.Tensor]
    """
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating batches"):
            x = x.to(device)
            with torch.autocast(
                device_type=device,
                dtype=torch.bfloat16
            ):
                y_hat = model(x).squeeze(-1)

            targets.append(y.cpu())
            predictions.append(y_hat.cpu())
        
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    return (
        tuple(
            [f(predictions, targets).item() for f in METRICS.values()]
        ),
        predictions,
        targets
    )
