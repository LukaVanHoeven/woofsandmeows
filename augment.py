import torch
import random
from torch import Tensor
import torch.nn.functional as F


class SpecAugment:
    def __init__(self, F=27, T=100, mF=2, mT=2):
        self.F = F
        self.T = T
        self.mF = mF
        self.mT = mT

    def __call__(self, x):
        """
        Accepts:
            [F, T] or [1, F, T]

        Returns:
            same shape as input
        """

        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a Tensor")

        original_shape = x.shape

        # ---- normalize shape ----
        if x.dim() == 3 and x.shape[0] == 1:
            x = x.squeeze(0)   # [F, T]
        elif x.dim() != 2:
            raise ValueError(
                f"Expected [F, T] or [1, F, T], got {tuple(x.shape)}"
            )

        x = x.clone()

        F, T = x.shape

        # ---- frequency masking ----
        for _ in range(self.mF):
            f = random.randint(0, min(self.F, F))
            f0 = random.randint(0, max(0, F - f))
            x[f0:f0 + f, :] = 0

        # ---- time masking ----
        for _ in range(self.mT):
            t = random.randint(0, min(self.T, T))
            t0 = random.randint(0, max(0, T - t))
            x[:, t0:t0 + t] = 0

        # ---- restore shape ----
        if len(original_shape) == 3:
            x = x.unsqueeze(0)

        return x


class AudioAugment:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def __call__(self, x: Tensor) -> Tensor:
        """
        Accepts:
            [T] or [1, T]

        Returns:
            same shape as input
        """

        if not isinstance(x, Tensor):
            raise TypeError("Input must be a Tensor")

        original_shape = x.shape

        # ---- normalize to [T] ----
        if x.dim() == 2 and x.shape[0] == 1:
            x = x.squeeze(0)
        elif x.dim() != 1:
            raise ValueError(
                f"Expected [T] or [1, T], got {tuple(x.shape)}"
            )

        x = x.clone()

        # ---- augmentations ----
        if random.random() < 0.5:
            x = self._time_shift(x)

        if random.random() < 0.5:
            x = self._add_noise(x)

        if random.random() < 0.3:
            x = self._time_stretch(x)

        # ---- restore shape ----
        if len(original_shape) == 2:
            x = x.unsqueeze(0)

        return x

    def _time_shift(self, audio: Tensor, shift_max: float = 0.2):
        shift = int(random.uniform(-shift_max, shift_max) * self.sample_rate)

        if shift == 0:
            return audio

        return torch.roll(audio, shifts=shift)

    def _add_noise(self, audio: Tensor, noise_factor: float = 0.005):
        return audio + noise_factor * torch.randn_like(audio)

    def _time_stretch(self, audio: Tensor, stretch_range=(0.9, 1.1)):
        rate = random.uniform(*stretch_range)

        T = audio.shape[0]
        new_T = max(1, int(T / rate))

        audio = audio.unsqueeze(0).unsqueeze(0)  # [1,1,T]

        stretched = F.interpolate(
            audio,
            size=new_T,
            mode="linear",
            align_corners=False
        ).squeeze()

        # crop / pad back to original length
        if stretched.numel() > T:
            stretched = stretched[:T]
        else:
            stretched = F.pad(stretched, (0, T - stretched.numel()))

        return stretched