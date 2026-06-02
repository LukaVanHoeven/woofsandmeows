from interpretability.explanation_methods.explainers.captum import IxG, IntGrad, DeepLIFT, GradCam
from interpretability.explanation_methods.explainers.rise import RISE
from interpretability.explanation_methods.explainers.lime import Lime
from functools import partial

import torch
import torch.nn as nn
from torchvision import transforms

from PIL import Image

from bcos.data.transforms import AddInverse
from bcos.models.resnet import resnet50
from bcos.modules.bcosconv2d import BcosConv2d

import numpy as np

def to_numpy(tensor):
    """
    Converting tensor to numpy.
    Args:
        tensor: torch.Tensor

    Returns:
        Tensor converted to numpy.

    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.detach().cpu().numpy()

def to_numpy_img(tensor):
    """
    Converting tensor to numpy image. Expects a tensor of at most 3 dimensions in the format (C, H, W),
    which is converted to a numpy array with (H, W, C) or (H, W) if C=1.
    Args:
        tensor: torch.Tensor

    Returns:
        Tensor converted to numpy.

    """
    return to_numpy(tensor.permute(1, 2, 0)).squeeze()

class Trainer(nn.Module):
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def predict(self, x):
        return self.model(x).sigmoid()
    
    def forward(self, x):
        return self.model(x)
    
def visualize_explanation(
    image_path: str = "cat.webp",
    checkpoint_path: str = "experiments/bcos_resnet20/last.ckpt",
    save_path: str = "explanation.png",
):

    # This is the bcos model that is used in the paper as well
    model = resnet50(
        num_classes=2,
        in_chans=6,
        small_inputs=True,
        # We can tune b and maxout, the paper says that maxout does improve accuracy
        conv_layer=partial(BcosConv2d, b=2, max_out=1),
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = {
        k.removeprefix("model."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)
    model.eval()

    img = Image.open(image_path).convert("RGB").resize((360,360))
    tensor = AddInverse()(transforms.ToTensor()(img)).unsqueeze(0).requires_grad_(True)

    result = model.explain(tensor)
    print(f"Predicted: {['cat','dog'][result['prediction']]} (class {result['prediction']})")

    explanation_img = result["explanation"]
    Image.fromarray((explanation_img * 255).astype("uint8")).save(save_path)
    print(f"Explanation saved to {save_path}")

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    print(f"Confidence: cat={probs[0,0]:.3f}, dog={probs[0,1]:.3f}")


def visualize_lime(
        image_path: str = "cat.webp",
        checkpoint_path: str = "experiments/norm50/last.ckpt",
        save_path: str = "lime.png"
):
    model = resnet50(
        num_classes=2,
        in_chans=6,
        small_inputs=True,
        conv_layer=nn.Conv2d,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = {
        k.removeprefix("model."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)
    model.eval()

    lime_expl = Lime(model, kernel_size=4, batch_size=32, num_samples=500, num_classes=2)

    img = Image.open(image_path).convert("RGB").resize((128, 128))
    tensor = AddInverse()(transforms.ToTensor()(img)).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    target = probs[0].argmax().item()
    print(f"Predicted: {['cat', 'dog'][target]} (class {target})")

    model.zero_grad()
    print("Running LIME (this may take a while)...")
    mask = lime_expl.attribute(tensor, target)
    mask_np = to_numpy(mask[0, 0])

    img_np = np.array(img) / 255.0
    cutout = img_np * mask_np[..., None]
    overlay = np.concatenate([cutout, mask_np[..., None]], axis=-1)

    Image.fromarray((overlay * 255).astype("uint8"), mode="RGBA").save(save_path)
    print(f"LIME explanation saved to {save_path}")


if __name__ == "__main__":
    visualize_lime(
        image_path="cat.jpg",
    )
    # visualize_explanation()