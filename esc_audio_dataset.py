"""
audio_dataset.py
 
PyTorch Dataset for 10-class audio classification using the ESC-10 subset
of the ESC-50 Environmental Sound Classification dataset
(https://github.com/karolpiczak/ESC-50).
 
The ESC-10 partition is identified by the ``esc10`` column in the
accompanying metadata CSV (``esc50.csv``).  The 10 classes and their
sequential label indices are defined in ``LABEL_MAP`` below.
 
Each file is split into as many non-overlapping (or overlapping) windows
of ``duration`` seconds as possible using a sliding window with step
``stride``. The final partial window is silently discarded. Each window
is pre-processed entirely inside the dataset:
 
    .wav → resample → mono → window slice → MelSpectrogram →
        AmplitudeToDB → (optional normalisation)
 
``__getitem__`` returns a ``(log_mel_spectrogram, label)`` tuple where
the label is an integer in the range [0, 9].
 
Example usage::
 
    from audio_dataset import ESC10AudioDataset
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
 
    dataset = ESC10AudioDataset(
        data_dirs="/data/ESC-50-master/audio",
        csv_path="/data/ESC-50-master/meta/esc50.csv",
        duration=3.0,
        stride=1.5,
    )
 
    # Fit normalisation on training split only to avoid data leakage
    all_indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(all_indices, test_size=0.2)
    dataset.fit_normalisation(train_idx)
 
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
    )
 
    for spectrogram, label in train_loader:
        # spectrogram : (B, 1, n_mels, time_frames)  float32
        # label       : (B,)                          int64
        ...
"""
 
from __future__ import annotations
 
import csv
import os
from pathlib import Path
from typing import Callable, Optional, Tuple
 
import wave
import numpy as np
import torchaudio
import torch
from torch import Tensor
from torch.utils.data import Dataset
import soundfile as sf
 
# ESC-10 categories mapped to sequential class indices 0-9, ordered by
# their original ESC-50 target number (dog=0, rooster=1, …, chainsaw=41).
LABEL_MAP: dict[str, int] = {
    "dog":            0,
    "rooster":        1,
    "rain":           2,
    "sea_waves":      3,
    "crackling_fire": 4,
    "crying_baby":    5,
    "sneezing":       6,
    "clock_tick":     7,
    "helicopter":     8,
    "chainsaw":       9,
}
"""Maps the ESC-10 category name to a sequential integer class index."""
 
REVERSE_LABEL_MAP: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}
"""Maps the sequential integer class index back to a category name."""
 
class ESCAudioDataset(Dataset):
    """
    PyTorch Dataset for ESC-10 audio classification.
 
    Reads the ESC-50 metadata CSV (*csv_path*) to identify which ``.wav``
    files belong to the ESC-10 subset (rows where ``esc10 == True``), then
    locates those files inside the directory/directories given by
    *data_dirs*.  Sub-directories are intentionally ignored.
 
    Each file is split into as many fixed-length windows as possible using
    a sliding window of ``duration`` seconds stepped by ``stride`` seconds;
    any trailing partial window is discarded (no padding).  Each window is
    resampled, converted to mono, and transformed to a normalised log-mel
    spectrogram on the fly.
 
    Parameters
    ----------
    data_dirs:
        Path, or list of paths, to the directory/directories that contain
        the ESC-50 ``.wav`` audio files.  For the standard ESC-50 release
        this is the single ``audio/`` folder.  Only files directly inside
        each directory are considered; sub-directories are skipped.
    csv_path:
        Path to the ESC-50 metadata CSV file (``esc50.csv``).  Rows with
        ``esc10 == True`` are used; all other rows are ignored.
    folds:
        List of fold numbers (1–5) to include.  Only files whose ``fold``
        value appears in this list are added to the dataset.  Pass
        ``[1, 2, 3, 4]`` for training and ``[5]`` for the held-out test
        partition, for example.  ``None`` (default) includes all five
        folds.
    target_sr:
        Target sample rate in Hz. All waveforms are resampled to this
        rate after loading. Defaults to ``22_050``.
    duration:
        Window length in seconds. Each sample returned by
        ``__getitem__`` corresponds to exactly this many seconds of
        audio. Defaults to ``3.0``.
    stride:
        Step between consecutive window start positions, in seconds.
        Use ``stride == duration`` for non-overlapping windows (the
        default behaviour).  Use ``stride < duration`` for overlapping
        windows.  The final partial window is always discarded.
        Defaults to ``duration`` (no overlap).
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
        all built-in pre-processing and normalisation. Useful for data
        augmentation (e.g. ``torchaudio.transforms.FrequencyMasking``).
        Defaults to ``None``.
    target_transform:
        Optional callable applied to the integer label. Defaults to
        ``None``.
 
    Attributes
    ----------
    files:
        ``Path`` of the source file for every window.
    window_starts:
        Start sample index (in *target_sr* samples) for every window.
    labels:
        Integer label (``LABEL_MAP`` value) for each window.
    folds:
        The fold numbers actually used by this instance, as a sorted list.
    classes:
        Sorted list of category names found across all audio files
        (subset of the 10 ESC-10 class names).
    mean:
        Global mean of the log-mel spectrogram values, set by
        ``fit_normalisation``. ``None`` until fitted.
    std:
        Global standard deviation of the log-mel spectrogram values,
        set by ``fit_normalisation``. ``None`` until fitted.
 
    Raises
    ------
    FileNotFoundError
        If any path in *data_dirs* does not exist or is not a directory,
        or if *csv_path* does not exist.
    ValueError
        If no valid windows are found across all directories.
 
    Notes
    -----
    Spectrograms are computed on the fly. File metadata (sample rate,
    frame count) is read once during ``__init__`` via ``wave`` to build
    the window index without loading every waveform up front.
 
    For large datasets where throughput is a bottleneck, consider
    pre-computing and caching the spectrogram tensors to disk (e.g.
    with ``torch.save``) and building a lightweight cached variant on
    top of this class.
    """
 
    def __init__(
        self,
        data_dirs: str | os.PathLike | list[str | os.PathLike],
        csv_path: str | os.PathLike,
        *,
        folds: Optional[list[int]] = None,
        target_sr: int = 22_050,
        duration: float = 3.0,
        stride: Optional[float] = None,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        top_db: float = 80.0,
        post_transform: Optional[Callable[[Tensor], Tensor]] = None,
        pre_transform: Optional[Callable[[Tensor], Tensor]] = None,
        target_transform: Optional[Callable[[int], int]] = None,
    ) -> None:
        super().__init__()
 
        if isinstance(data_dirs, (str, os.PathLike)):
            data_dirs = [data_dirs]
        dirs: list[Path] = [Path(d) for d in data_dirs]
        for d in dirs:
            if not d.is_dir():
                raise FileNotFoundError(
                    f"'data_dirs' entry is not an existing directory: {d}"
                )
 
        csv_path = Path(csv_path)
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"'csv_path' does not point to an existing file: {csv_path}"
            )
 
        self.target_sr = target_sr
        self.num_samples: int = int(target_sr * duration)
        self.duration = duration
        self.stride: float = duration if stride is None else stride
 
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.top_db = top_db
 
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.target_transform = target_transform
 
        self._mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self._to_db = torchaudio.transforms.AmplitudeToDB(top_db=top_db)
 
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
 
        self.folds: list[int] = sorted(folds) if folds is not None else list(range(1, 6))
 
        self.files: list[Path] = []
        self.window_starts: list[int] = []
        self.labels: list[int] = []
 
        stride_samples: int = int(self.stride * target_sr)
 
        # Build a lookup of {filename: label} for all ESC-10 entries in the
        # CSV that belong to the requested folds.  The 'esc10' column uses
        # the string 'True'/'False'; 'fold' is an integer string 1-5.
        # When folds is None every fold is included.
        fold_set: Optional[set[int]] = set(folds) if folds is not None else None
        esc10_label: dict[str, int] = {}
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["esc10"] != "True":
                    continue
                if fold_set is not None and int(row["fold"]) not in fold_set:
                    continue
                esc10_label[row["filename"]] = LABEL_MAP[row["category"]]
 
        for d in dirs:
            for path in sorted(d.iterdir()):
                if path.suffix.lower() != ".wav":
                    continue
                if path.name not in esc10_label:
                    continue
 
                # Read metadata only; no full waveform load at init time.
                # Use floor conversion so every window start is within the
                # samples that will actually be available after resampling
                # (torchaudio.functional.resample produces at least ceil(N
                # * new / orig) samples, which is >= floor).
                with wave.open(str(path), "rb") as wf:
                    orig_frames = wf.getnframes()
                    orig_sr    = wf.getframerate()
                total_samples = int(orig_frames / orig_sr * target_sr)
 
                start = 0
                while start + self.num_samples <= total_samples:
                    self.files.append(path)
                    self.window_starts.append(start)
                    self.labels.append(esc10_label[path.name])
                    start += stride_samples
 
        if not self.files:
            dirs_str = ", ".join(str(d) for d in dirs)
            raise ValueError(
                f"No valid ESC-10 windows found in: {dirs_str} "
                f"(folds={self.folds}). "
                f"Ensure the audio directory contains the ESC-50 .wav files "
                f"listed in {csv_path} and that 'duration' ({duration}s) is "
                f"not longer than the audio files."
            )
 
        self.classes: list[str] = sorted(
            {REVERSE_LABEL_MAP[label] for label in self.labels}
        )

    def __len__(self) -> int:
        """Return the total number of windows across all files."""
        return len(self.files)

    def get_n_classes(self) -> int:
        """Return the total number of possible classes."""
        return len(LABEL_MAP)

    def __repr__(self) -> str:
        """Return a developer-friendly summary string."""
        norm = (
            f"mean={self.mean:.3f}, std={self.std:.3f}"
            if self.mean is not None
            else "unnormalised"
        )
        return (
            f"{self.__class__.__name__}("
            f"n_windows={len(self)}, "
            f"classes={self.classes}, "
            f"target_sr={self.target_sr}, "
            f"duration={self.duration}s, "
            f"stride={self.stride}s, "
            f"n_mels={self.n_mels}, "
            f"{norm})"
        )

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """
        Load, pre-process and return a single window.

        Parameters
        ----------
        idx:
            Index of the window to retrieve.

        Returns
        -------
        spectrogram : Tensor, shape ``(1, n_mels, time_frames)``, dtype
            ``float32``
            Log-mel spectrogram of the window, normalised when
            ``fit_normalisation`` has been called.
        label : int
            Class index: ``0`` for *cat*, ``1`` for *dog*.
        """
        waveform = self.load_waveform(idx)

        if self.pre_transform is not None:
            waveform = self.pre_transform(waveform)

        spectrogram = self._to_db(self._mel_transform(waveform))

        if self.mean is not None and self.std is not None:
            spectrogram = (spectrogram - self.mean) / self.std

        if self.post_transform is not None:
            spectrogram = self.post_transform(spectrogram)

        # AddInverse
        # spectrogram = torch.cat(
        #     [spectrogram, 1.0 - spectrogram],
        #     dim=0
        # )

        label: int = self.labels[idx]

        if self.target_transform is not None:
            label = self.target_transform(label)

        return spectrogram, label
    
    def fit_normalisation(self, indices: list[int]) -> None:
        """
        Compute mean and std from the log-mel spectrograms of the given
        sample indices and store them as ``self.mean`` / ``self.std``.

        After calling this method, ``__getitem__`` will automatically
        z-score normalise every spectrogram: ``(x - mean) / std``.

        Parameters
        ----------
        indices:
            Indices into the dataset used to compute statistics.
            Should correspond to **training samples only** to avoid
            data leakage into validation or test splits.

        Raises
        ------
        ValueError
            If the computed standard deviation is zero (e.g. all
            windows are silent).

        Notes
        -----
        Spectrograms are computed on the fly for every index in
        *indices*, so this method may take a moment for large datasets.
        User-supplied ``transform`` callables are intentionally
        **not** applied during this step so that augmentation does not
        bias the statistics.
        """
        values: list[np.ndarray] = []
        for i in indices:
            waveform = self.load_waveform(i)
            if self.pre_transform is not None:
                waveform = self.pre_transform(waveform)
            spec = self._to_db(self._mel_transform(waveform))
            values.append(spec.numpy().ravel())

        all_values = np.concatenate(values)
        mean = float(all_values.mean())
        std = float(all_values.std())

        if std == 0.0:
            raise ValueError(
                "Standard deviation is zero across the provided indices; "
                "cannot normalise. Check that the audio windows are not silent."
            )

        self.mean = mean
        self.std = std

    def load_waveform(self, idx: int) -> Tensor:
        """
        Load the waveform for window *idx*, slice it, and return it **without** 
        any transformations, normalisation or user transforms applied.

        Used internally by both ``__getitem__`` and
        ``fit_normalisation``.

        Parameters
        ----------
        idx:
            Index of the window to compute.

        Returns
        -------
        Tensor
            Waveform of shape ``(1, num_samples)``.
        """
        path = self.files[idx]
        start = self.window_starts[idx]

        ### Replace torchaudio.load with soundfile.read for ROCm compatibility ###
        #waveform, sr = torchaudio.load(path)
        waveform, sr = sf.read(path, dtype="float32")
        waveform = torch.from_numpy(waveform)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T
        ### TODO: maybe try look for a fix on ROCm ###

        # Resample if the original sample rate is different from the target
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.target_sr
            )

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # (1, samples)

        # Slice the window — always exactly num_samples long because
        # start + num_samples <= total_samples was enforced in __init__.
        waveform = waveform[:, start : start + self.num_samples]

        return waveform
