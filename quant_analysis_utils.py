import logging
import torch

from collections import defaultdict
import numpy as np
from torch import nn
from tqdm import tqdm

from esc_audio_dataset import ESCAudioDataset, 


def grid_pointing_game(
    dataset: ESCAudioDataset, 
    model: nn.Module, 
    DEVICE: str,
    logger: logging.Logger,
    first_img_output_dir: str | None=None
)-> np.ndarray:
    """
    Evaluates a B-cos model on the ESC-10 dataset using the grid pointing game
    (Boehle et al., 2022 — https://arxiv.org/abs/2205.10268).

    Protocol
    --------
    For each round r:
    1. Pick one spectrogram per class (label order 0-9), using the r-th sample
        of each class.
    2. Concatenate the 10 spectrograms horizontally →
            x : (1, 1, n_mels, 10 * T)
    3. For each class c, call model.explain(x, idx=c) to get the contribution
        map of the same spatial size as x.
    4. Keep only positively contributing pixels (contribution > 0).
    5. Score = fraction of those positive pixels that fall inside the correct
        column (column c, i.e. the horizontal slice [c*T : (c+1)*T]).

    Scores are collected across all (round, class) pairs.  Mean and std are
    printed at the end.

    Parameters
        model - B-cos model with a `.explain(x, idx)` method
        dataset - ESC10AudioDataset instance
        DEVICE - torch.device
        logger - logger
    """
    # Group dataset indices by class label
    indices_by_class: dict[int, list[int]] = defaultdict(list)
    for _idx in range(len(dataset)):
        indices_by_class[dataset.labels[_idx]].append(_idx)

    n_classes: int = dataset.get_n_classes()

    # Number of complete rounds is capped by the smallest class bucket 
    # so that every round has exactly one sample per class.
    n_rounds: int = min(len(v) for v in indices_by_class.values())

    logger.debug(f"Classes : {n_classes}")
    logger.debug(f"Rounds : {n_rounds} ({n_classes * n_rounds} explain calls total)")

    pointing_scores: list[float] = []

    model.eval()

    for r in tqdm(range(n_rounds), desc="Running rounds"):
        # Build the concatenated input: one spectrogram per class in label
        # order 0..9, so the class-c image always occupies column c.
        imgs = [
            dataset[indices_by_class[cls][r]][0]   # (1, n_mels, T)
            for cls in range(n_classes)
        ]

        # Width (time frames) of a single spectrogram - used to locate columns.
        T: int = imgs[0].shape[-1]

        # Concatenate along the time axis -> (1, n_mels, n_classes * T)
        x_base = torch.cat(imgs, dim=-1)

        for cls_idx in range(n_classes):
            # Recreate the tensor for each explain call so that the computation
            # graph is fresh (model.explain performs its own backward pass).
            x = x_base.unsqueeze(0).to(DEVICE).requires_grad_()
            # x : (1, 1, n_mels, n_classes * T)

            with torch.enable_grad():
                expl_out = model.explain(x, idx=cls_idx)

            # contrib_map : (n_mels, n_classes * T)
            contrib_map = (
                expl_out["contribution_map"].detach().squeeze(0).squeeze(0).cpu()
            )

            positive: torch.Tensor = contrib_map > 0 # (n_mels, n_classes*T)
            total_positive: int = int(positive.sum().item())

            if r == 0 and first_img_output_dir is not None:
                plt.imshow(positive, origin="lower")
                plt.title(f"Explain for {cls_idx}")
                plt.colorbar()
                plt.show()

            if total_positive == 0:
                logger.warning("One of the explanations did not contain any positive contributions.")
                continue

            # Class cls_idx occupies the horizontal band [cls_idx*T : (cls_idx+1)*T]
            correct_col = torch.zeros_like(positive)
            correct_col[:, cls_idx * T : (cls_idx + 1) * T] = True

            in_correct: int = int((positive & correct_col).sum().item())
            pointing_scores.append(in_correct / total_positive)

    return np.array(pointing_scores)
