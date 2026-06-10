from torchvision.models import resnet18
import torch
import os

class BaselineResNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(pretrained=False, num_classes=num_classes)
        # Adjust first conv to accept 1-channel spectrograms
        self.model.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            bias=self.model.conv1.bias is not None
        )

    def forward(self, x):
        return self.model(x)

    def save(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, "baseline_resnet_best.pth"))
