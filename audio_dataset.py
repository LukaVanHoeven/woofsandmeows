"""
audio_dataset.py

PyTorch Dataset for binary audio classification from a flat directory of
.wav files whose names are prefixed with ``cat_`` or ``dog_``.

Each sample is pre-processed entirely inside the dataset:

    .wav -> resample -> mono -> pad/crop -> MelSpectrogram -> 
        AmplitudeToDB

``__getitem__`` returns a ``(log_mel_spectrogram, label)`` tuple where 
the label is an integer (0 = cat, 1 = dog).

Example usage::

    from audio_dataset import CatDogAudioDataset
    from torch.utils.data import DataLoader

    dataset = CatDogAudioDataset(root="/data/audio")
    loader  = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=4
    )

    for spectrogram, label in loader:
        # spectrogram : (B, 1, n_mels, time_frames)  float32
        # label       : (B,)                          int64
        ...
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


LABEL_MAP: dict[str, int] = {
    "cat": 0,
    "dog": 1,
}
"""Maps the filename prefix to an integer class index."""

REVERSE_LABEL_MAP: dict[int, str] = {
    0: "cat",
    1: "dog",
}
"""Maps the integer class index to an filename prefix."""

class CatDogAudioDataset(Dataset):
    """
    PyTorch Dataset for cat/dog audio classification.

    Scans *root* for ``.wav`` files whose names begin with ``cat_`` or
    ``dog_``. Sub-directories are intentionally ignored. Each waveform 
    is resampled, converted to mono, zero-padded or centre-cropped to a 
    fixed duration, and transformed to a log-mel spectrogram entirely on
    the fly so that the class is self-contained and requires no separate
    pre-processing step.

    Parameters
    ----------
    root:
        Path to the directory that contains the ``.wav`` files. Only 
        files directly inside *root* are considered; sub-directories 
        are skipped.
    target_sr:
        Target sample rate in Hz. All waveforms are resampled to this
        rate after loading. Defaults to ``22_050``.
    duration:
        Fixed clip length in seconds. Waveforms shorter than this value 
        are zero-padded on the right; longer waveforms are 
        centre-cropped. Defaults to ``3``.
    n_fft:
        FFT window size used by the Short-Time Fourier Transform that 
        underlies ``MelSpectrogram``. Larger values give better 
        frequency resolution at the cost of temporal resolution. 
        Defaults to ``1024``.
    hop_length:
        Number of samples between consecutive STFT frames. Smaller 
        values produce denser time axes. Defaults to ``256``.
    n_mels:
        Number of mel filter-bank channels. This becomes the frequency
        dimension of the output spectrogram. Defaults to ``80``.
    top_db:
        Threshold (in dB) for ``AmplitudeToDB``. Power values more than
        *top_db* below the reference are clipped to ``ref - top_db``.
        Defaults to ``80.0``.
    transform:
        Optional callable applied to the log-mel spectrogram **after** 
        all built-in pre-processing. Useful for data augmentation (e.g.
        ``torchaudio.transforms.FrequencyMasking``). Defaults to 
        ``None``.
    target_transform:
        Optional callable applied to the integer label. Defaults to 
        ``None``.

    Attributes
    ----------
    files:
        Sorted list of ``Path`` objects for every discovered ``.wav`` 
        file.
    labels:
        Integer label (``LABEL_MAP`` value) for each entry in ``files``.
    classes:
        Sorted list of class names found in the directory (subset of
        ``["cat", "dog"]``).

    Raises
    ------
    FileNotFoundError
        If *root* does not exist or is not a directory.
    ValueError
        If *root* contains no ``.wav`` files with a recognised prefix.

    Notes
    -----
    Spectrograms are computed on the fly. For large datasets where
    throughput is a bottleneck, consider pre-computing and caching the
    tensors to disk (e.g. with ``torch.save``) and building a 
    lightweight cached variant on top of this class.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        *,
        target_sr: int = 22_050,
        duration: float = 3.0,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        top_db: float = 80.0,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
        target_transform: Optional[Callable[[int], int]] = None,
    ) -> None:
        super().__init__()

        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(
                f"'root' is not an existing directory: {root}"
            )

        self.target_sr = target_sr
        self.num_samples: int = int(target_sr * duration)
        self.duration = duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.top_db = top_db

        self.transform = transform
        self.target_transform = target_transform

        self._mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self._to_db = torchaudio.transforms.AmplitudeToDB(top_db=top_db)

        self.files: list[Path] = []
        self.labels: list[int] = []

        for path in sorted(root.iterdir()):
            if path.suffix.lower() != ".wav":
                continue
            prefix = path.name.split("_")[0].lower()
            if prefix not in LABEL_MAP:
                continue
            self.files.append(path)
            self.labels.append(LABEL_MAP[prefix])

        if not self.files:
            raise ValueError(
                f"No 'cat_*.wav' or 'dog_*.wav' files found in: {root}"
            )

        self.classes: list[str] = sorted(
            {path.name.split("_")[0].lower() for path in self.files}
        )

    def __len__(self) -> int:
        """Return the total number of audio samples in the dataset."""
        return len(self.files)

    def __repr__(self) -> str:
        """Return a developer-friendly summary string."""
        return (
            f"{self.__class__.__name__}("
            f"n_samples={len(self)}, "
            f"classes={self.classes}, "
            f"target_sr={self.target_sr}, "
            f"duration={self.duration}s, "
            f"n_mels={self.n_mels})"
        )

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """
        Load, pre-process and return a single sample.

        Parameters
        ----------
        idx:
            Index of the sample to retrieve.

        Returns
        -------
        spectrogram : Tensor, shape ``(1, n_mels, time_frames)``, dtype 
            ``float32`` Log-mel spectrogram of the audio clip. 
            The leading channel dimension of ``1`` makes it directly 
            compatible with single-channel CNN inputs 
            (e.g. ResNet with ``in_chans=1``).
        label : int
            Class index: ``0`` for *cat*, ``1`` for *dog*.
        """
        path = self.files[idx]
        label: int = self.labels[idx]

        waveform, sr = torchaudio.load(path) # (channels, samples)

        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.target_sr
            )

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True) # (1, samples)

        waveform = self._fix_length(waveform) # (1, num_samples)

        spectrogram: Tensor = self._to_db(self._mel_transform(waveform))

        if self.transform is not None:
            spectrogram = self.transform(spectrogram)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return spectrogram, label

    def _fix_length(self, waveform: Tensor) -> Tensor:
        """
        Pad or centre-crop *waveform* to exactly ``self.num_samples``.

        Short clips are zero-padded on the right.  Long clips are
        centre-cropped so that the most informative part of the audio
        (typically the middle) is preserved.

        Parameters
        ----------
        waveform:
            Input waveform tensor of shape ``(1, samples)``.

        Returns
        -------
        Tensor
            Waveform of shape ``(1, self.num_samples)``.
        """
        n = waveform.shape[-1]

        if n < self.num_samples:
            # Zero-pad on the right
            pad_amount = self.num_samples - n
            waveform = F.pad(waveform, (0, pad_amount))
        elif n > self.num_samples:
            # Centre-crop
            start = (n - self.num_samples) // 2
            waveform = waveform[:, start : start + self.num_samples]

        return waveform
