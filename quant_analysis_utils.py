import logging
import torch
import matplotlib.pyplot as plt

from collections import defaultdict
import numpy as np
from torch import nn
from tqdm import tqdm

from esc_audio_dataset import ESCAudioDataset, REVERSE_LABEL_MAP


def _save_grid_explanation(
    x_base: torch.Tensor,
    contrib_map: torch.Tensor,
    cls_idx: int,
    n_classes: int,
    T: int,
    output_path: str,
) -> None:
    """
    Save a two-panel figure for a single grid pointing game explanation.
 
    Top panel    - log-mel spectrogram (all classes concatenated).
    Bottom panel - signed contribution map from model.explain(), with
                   positive contributions in red and negative in blue.
 
    Both panels share the same column layout:
      • Dashed white vertical lines separate each class column.
      • Each column is labelled with its class name on the x-axis.
      • The column being explained is highlighted with a gold border,
        a faint gold fill, and a bold gold x-axis label.
 
    Parameters
    ----------
    x_base      : (1, n_mels, n_classes * T)  — concatenated spectrogram.
    contrib_map : (n_mels, n_classes * T)      — signed contribution map.
    cls_idx     : index of the class being explained (0-indexed).
    n_classes   : total number of classes.
    T           : time-frame width of one class column.
    output_path : file path to write the PNG to.
    """
    n_mels: int = x_base.shape[-2]
    spec = x_base.squeeze(0).cpu().numpy()
    cmap_data = contrib_map.cpu().numpy()
 
    fig, axes = plt.subplots(
        nrows=2, ncols=1,
        figsize=(max(12, n_classes * 2), 6),
        constrained_layout=True,
    )
 
    # --- top: raw log-mel spectrogram ----------------------------------
    axes[0].imshow(spec, origin="lower", aspect="auto")
    axes[0].set_ylabel("Mel bin")
 
    # --- bottom: signed contribution map (red = positive, blue = negative)
    axes[1].imshow(cmap_data, origin="lower", aspect="auto", cmap="RdBu_r")
    axes[1].set_ylabel("Mel bin")
    axes[1].set_xlabel("Time frame")
 
    # --- per-column annotations (same for both panels) -----------------
    tick_positions = [int((c + 0.5) * T) for c in range(n_classes)]
    tick_labels    = [REVERSE_LABEL_MAP[c] for c in range(n_classes)]
 
    for ax in axes:
        for c in range(n_classes):
            # Dashed vertical separator between columns
            if c > 0:
                ax.axvline(
                    x=c * T - 0.5,
                    color="white", linewidth=1.2, linestyle="--", alpha=0.7,
                    zorder=3,
                )
 
            # Gold highlight on the predicted class column
            if c == cls_idx:
                ax.add_patch(plt.Rectangle(
                    xy=(c * T - 0.5, -0.5),
                    width=T,
                    height=n_mels,
                    linewidth=2,
                    edgecolor="#FFD700",
                    facecolor="#FFD700",
                    alpha=0.12,
                    zorder=2,
                ))
 
        # Class-name tick labels centred in each column
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
 
        # Make the predicted class label gold and bold
        for i, tick in enumerate(ax.get_xticklabels()):
            if i == cls_idx:
                tick.set_color("#FFD700")
                tick.set_fontweight("bold")
 
    axes[0].set_title(
        f"Grid pointing game — explaining: {REVERSE_LABEL_MAP[cls_idx]}",
        pad=6,
    )
 
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def grid_pointing_game(
    dataset: ESCAudioDataset, 
    model: nn.Module, 
    DEVICE: str,
    logger: logging.Logger,
    first_img_output_dir: str | None=None
)-> tuple[np.ndarray, np.ndarray]:
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
    weighted_pointing_scores: list[float] = []

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

            positive: torch.Tensor = contrib_map > 0
            total_positive: int = int(positive.sum().item())

            positive_contrib = contrib_map.clamp(min=0)
            total_positive_weight = positive_contrib.sum().item()

            if r == 0 and first_img_output_dir is not None:
                _save_grid_explanation(
                    x_base=x_base,
                    contrib_map=positive_contrib,
                    cls_idx=cls_idx,
                    n_classes=n_classes,
                    T=T,
                    output_path=f"{first_img_output_dir}/expl_cls_{REVERSE_LABEL_MAP[cls_idx]}.png",
                )

            if total_positive == 0:
                logger.warning("One of the explanations did not contain any positive contributions.")
                continue

            # Class cls_idx occupies the horizontal band [cls_idx*T : (cls_idx+1)*T]
            correct_col = torch.zeros_like(positive)
            correct_col[:, cls_idx * T : (cls_idx + 1) * T] = True

            in_correct: int = int((positive & correct_col).sum().item())
            pointing_scores.append(in_correct / total_positive)

            correct_positive_weight = positive_contrib[correct_col].sum().item()
            weighted_pointing_scores.append(correct_positive_weight / total_positive_weight)

    return np.array(pointing_scores), np.array(weighted_pointing_scores)

def contribution_masking_accuracy(
    dataset: ESCAudioDataset,
    model: nn.Module,
    DEVICE: str,
    logger: logging.Logger,
    first_img_output_dir: str | None
) -> tuple[float, float, float]:
    """
    Evaluates how well the positive contribution regions identified by a B-cos
    explanation preserve predictive performance.

    Protocol
    --------
    For every sample in the dataset:

    1. Run the original image through the model and obtain:
        - predicted class
        - contribution map for the predicted class

    2. Record whether the original prediction matches the ground-truth label.

    3. Construct a binary mask from the contribution map:
            mask = contribution_map > 0

    4. Apply the mask to the original image:
            masked_image = image * mask

       so that only positively contributing pixels remain.

    5. Run the masked image through the model again.

    6. Record whether the masked-image prediction matches the ground-truth
       label.

    At the end, return the classification accuracy before and after masking.

    Interpretation
    --------------
    - Original accuracy measures normal model performance.
    - Masked accuracy measures whether the positive contribution regions alone
      contain sufficient information for correct classification.

    Parameters
    ----------
    dataset:
        Dataset containing spectrograms and labels.

    model:
        B-cos model implementing:
            - forward(x)
            - explain(x, idx)

        where explain(...) returns a dictionary containing
        "contribution_map".

    DEVICE:
        Torch device string.

    logger:
        Logger instance.

    Returns
    -------
    tuple[float, float]
        (
            original_accuracy,
            positive_masked_accuracy,
            negative_masked_accuracy,
        )
    """
    model.eval()

    n_samples = len(dataset)

    correct_original = 0
    correct_positive_masked = 0
    correct_negative_masked = 0

    for idx in tqdm(range(n_samples), desc="Evaluating masks"):
        img, label = dataset[idx]
        x = img.unsqueeze(0).to(DEVICE)

        x_expl = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            expl_out = model.explain(x_expl)

        if expl_out["explained_class_idx"] == label:
            correct_original += 1

        contrib_map = (
            expl_out["contribution_map"].detach().squeeze(0).squeeze(0)
        )

        # positive mask
        positive_mask = contrib_map > 0
        masked_img = img.clone().to(DEVICE)
        masked_img *= positive_mask.unsqueeze(0).to(masked_img.dtype)

        x_masked = masked_img.unsqueeze(0).to(DEVICE)

        if idx == 0:
            plt.imshow(x_masked.squeeze(0).squeeze(0).detach().cpu(), origin="lower")
            plt.title(f"Positive masked image for class: {REVERSE_LABEL_MAP[label]} | prediction: {REVERSE_LABEL_MAP[expl_out["explained_class_idx"]]}")
            plt.colorbar()
            plt.savefig(f"{first_img_output_dir}/positive_masked.png")
            plt.close()
            
            plt.imshow(img.squeeze(0).cpu(), origin="lower")
            plt.title(f"Original image for class: {REVERSE_LABEL_MAP[label]} | prediction: {REVERSE_LABEL_MAP[expl_out["explained_class_idx"]]}")
            plt.colorbar()
            plt.savefig(f"{first_img_output_dir}/original_image.png")
            plt.close()

            plt.imshow(contrib_map.cpu(), origin="lower", cmap="bwr")
            plt.title(f"Contribution map for class: {REVERSE_LABEL_MAP[label]} | prediction: {REVERSE_LABEL_MAP[expl_out["explained_class_idx"]]}")
            plt.colorbar()
            plt.savefig(f"{first_img_output_dir}/contribution_map.png")
            plt.close()

        with torch.no_grad():
            masked_logits = model(x_masked)

        masked_pred = int(masked_logits.argmax(dim=1).item())

        if masked_pred == label:
            correct_positive_masked += 1
        
        # negative mask
        negative_mask = contrib_map <= 0
        masked_img = img.clone().to(DEVICE)
        masked_img *= negative_mask.unsqueeze(0).to(masked_img.dtype)

        x_masked = masked_img.unsqueeze(0).to(DEVICE)
        if idx == 0:
            plt.imshow(x_masked.squeeze(0).squeeze(0).detach().cpu(), origin="lower")
            plt.title(f"Negative masked image for class: {REVERSE_LABEL_MAP[label]} | prediction: {REVERSE_LABEL_MAP[expl_out["explained_class_idx"]]}")
            plt.colorbar()
            plt.savefig(f"{first_img_output_dir}/negative_masked.png")
            plt.close()

        with torch.no_grad():
            masked_logits = model(x_masked)

        masked_pred = int(masked_logits.argmax(dim=1).item())

        if masked_pred == label:
            correct_negative_masked += 1

    original_accuracy = correct_original / n_samples
    positive_masked_accuracy = correct_positive_masked / n_samples
    negative_masked_accuracy = correct_negative_masked / n_samples

    return original_accuracy, positive_masked_accuracy, negative_masked_accuracy
