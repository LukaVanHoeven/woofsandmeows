import torch
import logging
import torch.nn.functional as F
from torch import nn


class GradCAM:
    """
    Wraps a model and a target layer to produce signed GradCAM explanations
    with the same ``.explain()`` interface as a B-cos model.
 
    **Why signed?**
    Standard GradCAM applies a final ReLU, zeroing negative values.  We omit
    that ReLU so that the resulting contribution map has both positive *and*
    negative regions, matching the B-cos semantics:
 
      positive (> 0)  — spatial locations that *increase* the target-class score
      negative (≤ 0)  — locations that *suppress* it or are irrelevant
 
    This makes the grid pointing game and masking experiments directly
    comparable between B-cos and GradCAM without any protocol changes.
 
    Reference
    ---------
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization", ICCV 2017.
 
    Parameters
    ----------
    model        : the network to explain.
    target_layer : layer at whose *output* the map is computed.
                   For ResNets ``model.layer4[-1]`` is the standard choice
                   (highest-resolution semantic feature map).
 
    Usage
    -----
    Always use as a context manager so hooks are removed automatically::
 
        target = model.layer4[-1]
        with GradCAM(model, target) as gradcam:
            out = gradcam.explain(x, idx=3)
            cam = out["contribution_map"]   # (1, 1, H, W), signed float
    """
 
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._hooks: list = []
        self._register_hooks()
 
    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------
    def _register_hooks(self) -> None:
        def _fwd_hook(module, inp, out):
            # Capture the feature-map produced by target_layer.
            self._activations = out
 
        def _bwd_hook(module, grad_in, grad_out):
            # grad_out[0] is dL / d(target_layer output).
            self._gradients = grad_out[0]
 
        self._hooks.append(
            self.target_layer.register_forward_hook(_fwd_hook)
        )
        # register_full_backward_hook is the modern, non-deprecated API.
        self._hooks.append(
            self.target_layer.register_full_backward_hook(_bwd_hook)
        )
 
    def remove_hooks(self) -> None:
        """Detach all registered hooks.  Called automatically by __exit__."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
 
    def __enter__(self) -> "GradCAM":
        return self
 
    def __exit__(self, *args) -> None:
        self.remove_hooks()
    
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

    # ------------------------------------------------------------------
    # explain — mirrors B-cos model.explain()
    # ------------------------------------------------------------------
    def explain(
        self,
        x: torch.Tensor,
        idx: int | None = None,
    ) -> dict:
        """
        Compute a signed GradCAM map for *x* with respect to class *idx*.
 
        The map is bilinearly upsampled to the full spatial size of *x* so
        that its shape matches the B-cos contribution map exactly, and all
        downstream scoring / masking code can be reused without changes.
 
        ``torch.enable_grad()`` is entered internally, so this method is safe
        to call inside a ``torch.no_grad()`` block.
 
        Parameters
        ----------
        x   : ``(1, C, H, W)`` input tensor on the model's device.
        idx : class index to explain.  ``None`` -> argmax of logits, matching
              the default behaviour of ``model.explain()``.
 
        Returns
        -------
        dict
            ``"contribution_map"``    - ``(1, 1, H, W)`` signed float tensor.
            ``"explained_class_idx"`` - ``int``.
        """
        with torch.enable_grad():
            # Detach from any outer graph; create a fresh leaf tensor so each
            # call has its own independent computation graph — same pattern as
            # the B-cos grid loop: ``x = x_base.unsqueeze(0)…requires_grad_()``.
            x_leaf = x.detach().requires_grad_(True)
 
            self.model.zero_grad()
            logits = self.model(x_leaf) # (1, n_classes)
 
            if idx is None:
                idx = int(logits.argmax(dim=1).item())
 
            # Single scalar backward on the target class score.
            logits[0, idx].backward()
 
        # Detach captured buffers before arithmetic - outside the graph.
        acts  = self._activations.detach() # (1, K, H_f, W_f)
        grads = self._gradients.detach() # (1, K, H_f, W_f)
 
        # Per-channel importance: global-average-pool over spatial dims.
        alphas = grads.mean(dim=(-2, -1), keepdim=True) # (1, K, 1, 1)
 
        # Signed weighted combination — no ReLU, preserves negative regions.
        cam = (alphas * acts).sum(dim=1, keepdim=True) # (1, 1, H_f, W_f)
 
        # Bilinear upsample to full input resolution.
        cam_up = F.interpolate(
            cam,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ) # (1, 1, H, W)
 
        return {
            "contribution_map": cam_up,
            "explained_class_idx": idx,
        }
 
 
# ------------------------------------------------------------------
# Convenience helper
# ------------------------------------------------------------------
 
def _get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """
    Return a sensible default GradCAM target layer for *model*.
 
    For ResNet-style architectures (those with a ``layer4`` attribute) the
    last residual block ``model.layer4[-1]`` is returned — this is the
    standard choice from the original Grad-CAM paper and gives the best
    trade-off between spatial resolution and semantic abstraction.
 
    For other architectures the last ``nn.Conv2d`` found in a depth-first
    traversal is returned as a fallback.
 
    Raises
    ------
    ValueError
        If no suitable layer can be found automatically.
    """ 
    last_conv: nn.Module | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
 
    if last_conv is None:
        raise ValueError(
            "Could not automatically identify a GradCAM target layer. "
            "Please pass target_layer explicitly."
        )
    return last_conv
