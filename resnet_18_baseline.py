import torch
from torch import nn
from torchvision.models import resnet18
import os

class BaselineResNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = resnet18(pretrained=False)
        
        # Adjust first convolution to accept 1-channel spectrograms
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            bias=self.model.conv1.bias is not None
        )
        
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def save(self, path):
        torch.save(
            self.state_dict(),
            os.path.join(path, "baseline_resnet_best.pth")
    )