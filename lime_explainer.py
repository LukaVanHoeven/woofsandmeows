import numpy as np
import torch
from torch import nn

from lime_explainer import lime_image
from skimage.segmentation import slic


class LimeExplainer:
    """
    LIME wrapper exposing the same .explain() interface as
    B-cos models and the GradCAM wrapper.

    Produces a signed dense contribution map:
        positive values -> support target class
        negative values -> oppose target class
    """

    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 1000,
        num_segments: int = 50,
        batch_size: int = 32,
    ):
        self.model = model
        self.num_samples = num_samples
        self.num_segments = num_segments
        self.batch_size = batch_size

        self._explainer = lime_image.LimeImageExplainer()

    # -------------------------------------------------------------
    # mimic nn.Module behaviour
    # -------------------------------------------------------------
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

    def _predict_fn(self, images):
        """
        images:
            (N, H, W, C)
        returns:
            (N, num_classes)
        """

        x = torch.from_numpy(images).permute(0, 3, 1, 2).float()

        device = next(self.model.parameters()).device
        x = x.to(device)

        with torch.no_grad():
            logits = self.model(x)

        probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()

    def explain(
        self,
        x: torch.Tensor,
        idx: int | None = None,
    ):
        """
        Parameters
        ----------
        x:
            (1, C, H, W)

        Returns
        -------
        {
            "contribution_map": (1,1,H,W),
            "explained_class_idx": idx
        }
        """

        device = x.device

        with torch.no_grad():
            logits = self.model(x)

        if idx is None:
            idx = int(logits.argmax(dim=1).item())

        # LIME expects HWC numpy
        image = (
            x[0]
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )

        segmentation_fn = lambda img: slic(
            img,
            n_segments=self.num_segments,
            compactness=0.1,
            start_label=0,
        )

        explanation = self._explainer.explain_instance(
            image=image,
            classifier_fn=self._predict_fn,
            labels=(idx,),
            top_labels=None,
            hide_color=0,
            num_samples=self.num_samples,
            segmentation_fn=segmentation_fn,
            batch_size=self.batch_size,
        )

        segments = explanation.segments

        dense_map = np.zeros(
            segments.shape,
            dtype=np.float32,
        )

        # signed weights
        for seg_id, weight in explanation.local_exp[idx]:
            dense_map[segments == seg_id] = weight

        contribution_map = (
            torch.from_numpy(dense_map)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        return {
            "contribution_map": contribution_map,
            "explained_class_idx": idx,
        }
