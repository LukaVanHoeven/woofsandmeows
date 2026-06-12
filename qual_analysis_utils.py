import os
import logging
import matplotlib.pyplot as plt
import torch
import torchaudio
import torch.nn.functional as F
from IPython.display import Audio

from collections import defaultdict
import numpy as np
from torch import nn
from tqdm import tqdm

from esc_audio_dataset import ESCAudioDataset, REVERSE_LABEL_MAP
from quant_analysis_utils import _save_grid_explanation


class STFT:
    def __init__(self, DEVICE, CONFIG):
        self.DEVICE = DEVICE
        self.CONFIG = CONFIG
        self.window = torch.hann_window(self.CONFIG["n_fft"]).to(self.DEVICE)

    def forward(self, wav):
        return torch.stft(
            wav,
            n_fft=self.CONFIG["n_fft"],
            hop_length=self.CONFIG["hop_length"],
            window=self.window,
            return_complex=True
        )

    def inverse(self, spec):
        return torch.istft(
            spec,
            n_fft=self.CONFIG["n_fft"],
            hop_length=self.CONFIG["hop_length"],
            window=self.window
        )

def mel_to_stft(unnormalised_log_mel, stft_spec):
    return F.interpolate(
        input=unnormalised_log_mel,
        size=stft_spec.shape,
        mode="bilinear",
        align_corners=False
    )

def apply_explanation(linear_weights, wav, top_quantile, logger, DEVICE, CONFIG):
    stft = STFT(DEVICE, CONFIG)
    stft_spec = stft.forward(wav).squeeze(0)

    power_mel = torchaudio.functional.DB_to_amplitude(linear_weights, ref=1.0, power=1.0)
    normalised_power_mel = (power_mel - power_mel.min()) / (power_mel.max() - power_mel.min() + 1e-8) # normalize to [0, 1]

    mask = mel_to_stft(normalised_power_mel, stft_spec)
    mask = mask.squeeze(0).squeeze(0)
    
    smoothed_mask = torch.nn.functional.avg_pool2d(
        mask[None, None],
        kernel_size=5,
        stride=1,
        padding=2
    )[0, 0]

    threshold = torch.quantile(smoothed_mask, top_quantile)
    smoothed_mask *= (smoothed_mask >= threshold).float()

    mag = stft_spec.abs() # strength of each time-frequency bin
    mag_weighted = mag * smoothed_mask # Apply weighted top mask (nipple)
    
    phase = stft_spec.angle() # where in the waveform cycle each bin is (important for reconstructing the audio)
    spec_masked = mag_weighted * torch.exp(1j * phase)

    #spec_masked = stft_spec * mask
    wav_out = stft.inverse(spec_masked.unsqueeze(0))

    # print(f"Mask stats - Min: {mask.min()}, Max: {mask.max()}, Mean: {mask.mean()}")
    # print(f"Mean absolute difference: {(mag_masked - mag).abs().mean()}")

    return wav_out

def back_convert_entire_dataset(dataset, model, top_quantile, DEVICE, CONFIG, logger, output_folder)-> None:
    model.eval()

    n_samples = len(dataset)

    for idx in tqdm(range(n_samples), desc="Generating explanations and converting to audio"):
        img, label = dataset[idx]
        x_wav = dataset.load_waveform(idx)
        x = img.unsqueeze(0).to(DEVICE)

        x_expl = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            expl_out = model.explain(x_expl)

        expl_out_wav = apply_explanation(expl_out["contribution_map"].unsqueeze(0), x_wav.to(DEVICE), top_quantile, logger, DEVICE, CONFIG)
        audio_object = Audio(expl_out_wav.squeeze(0).detach().cpu().numpy(), rate=CONFIG["sample_rate"])
        with open(f'{output_folder}/i-{idx}_l-{REVERSE_LABEL_MAP[label]}_p-{REVERSE_LABEL_MAP[expl_out["explained_class_idx"]]}.wav', 'wb') as f:
            f.write(audio_object.data)
 
def grid_pointing_game_qual(
    dataset: ESCAudioDataset, 
    model: nn.Module, 
    DEVICE: str,
    CONFIG: dict,
    logger: logging.Logger,
    top_quantile: float,
    base_output_dir: str
)-> None:
    """
    Perform grid pointing game, but convert back to audio instead of
    calculating scores.
    """
    indices_by_class: dict[int, list[int]] = defaultdict(list)
    for _idx in range(len(dataset)):
        indices_by_class[dataset.labels[_idx]].append(_idx)

    n_classes: int = dataset.get_n_classes()
    n_rounds: int = min(len(v) for v in indices_by_class.values())

    logger.debug(f"Classes : {n_classes}")
    logger.debug(f"Rounds : {n_rounds} ({n_classes * n_rounds} explain calls total)")

    model.eval()

    for r in tqdm(range(n_rounds), desc="Running rounds"):
        imgs = [
            dataset[indices_by_class[cls][r]][0]
            for cls in range(n_classes)
        ]
        imgs_wav = [
            dataset.load_waveform(indices_by_class[cls][r])
            for cls in range(n_classes)
        ]
        T: int = imgs[0].shape[-1]

        x_base = torch.cat(imgs, dim=-1)
        x_wav = torch.cat(imgs_wav, dim=-1)

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

            expl_out_wav = apply_explanation(expl_out["contribution_map"].unsqueeze(0), x_wav.to(DEVICE), top_quantile, logger, DEVICE, CONFIG)
            audio_object = Audio(expl_out_wav.squeeze(0).detach().cpu().numpy(), rate=CONFIG["sample_rate"])
            output_dir = f"{base_output_dir}/grid_{r}/"
            os.makedirs(output_dir, exist_ok=True)
            with open(f'{output_dir}/i-{r}_e-{REVERSE_LABEL_MAP[cls_idx]}.wav', 'wb') as f:
                f.write(audio_object.data)

            positive_contrib = contrib_map.clamp(min=0)
            _save_grid_explanation(
                x_base=x_base,
                contrib_map=positive_contrib,
                cls_idx=cls_idx,
                n_classes=n_classes,
                T=T,
                output_path=f"{output_dir}/i-{r}_e-{REVERSE_LABEL_MAP[cls_idx]}.png",
            )
