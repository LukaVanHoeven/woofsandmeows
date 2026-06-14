import torch
import torch.nn.functional as F

from bcos.models.resnet import BcosResNet


class BcosExplainShell():

    def __init__(self, model: BcosResNet):
        self.model = model

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        self.model.cuda(device)
        return self

    def cpu(self):
        self.model.cpu()
        return self

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def explain(self, *args, **kwargs):
        expl_out = self.model.explain(*args, **kwargs)

        smooth_contrib_map = self.smooth_contribution_map(expl_out["contribution_map"])
        expl_out["contribution_map"] = smooth_contrib_map

        return expl_out

    def smooth_contribution_map(
        self,
        contribs: torch.Tensor,
        smooth: int=15,
        alpha_percentile: float=99.5,
    ) -> torch.Tensor:
        """
        Applies the B-cos alpha smoothing pipeline to a 1-channel contribution map.

        Parameters
        ----------
        contribs : torch.Tensor
            Pixel-wise contribution map, shape (B, 1, H, W).
            Positive = supports prediction, negative = suppresses it.
        smooth : int
            Box-blur kernel size. Must be odd for symmetric padding. 0 to skip.
        alpha_percentile : float
            Percentile for robust normalisation into [0, 1]. Default: 99.5.

        Returns
        -------
        torch.Tensor
            Smoothed, normalised contribution map, shape (B, 1, H, W), values in [0, 1].
        """
        # For 1 channel the L2 norm is just the absolute value
        alpha = contribs.abs()  # (B, 1, H, W)

        # Zero out locations with a net negative contribution
        alpha = torch.where(contribs < 0, torch.full_like(alpha, 0), alpha)

        # Box blur: spreads local signal to neighbours, suppresses isolated noise
        if smooth:
            alpha = F.avg_pool2d(
                alpha,
                kernel_size=smooth,
                stride=1,
                padding=(smooth - 1) // 2,  # same-size output
            )

        # Robust per-sample normalisation to [0, 1]
        B = alpha.shape[0]
        alpha_flat = alpha.view(B, -1) # (B, H*W)
        quant = torch.quantile(alpha_flat, alpha_percentile / 100, dim=1) # (B,)
        quant = quant.view(B, 1, 1, 1) # broadcast
        alpha = (alpha / (quant + 1e-12)).clamp(0, 1)

        return alpha  # (B, 1, H, W)
