import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class NCameraCNN(nn.Module):
    """A CNN which assumes N cameras are available in the scene.

    The inputs are N images of dimension HxW and the outputs are the pose of the cube which is located in the scene.
    In particular, the outputs are 6d vectors in se(3) which must be sent to SE(3) via the exponential map.
    """

    def __init__(self, n_cams: int = 2, H: int = 376, W: int = 672) -> None:
        """Initialize the CNN.

        Args:
            n_cams: The number of cameras in the scene.
            H: The height of the input images.
            W: The width of the input images.
        """
        super().__init__()
        self.resnet = models.resnet18(
            weights="DEFAULT"
        )  # finetune a pretrained ResNet-18
        self.num_channels = (
            3 * n_cams
        )  # RGB-only for each cam, all channels concatenated
        self.H = H
        self.W = W

        # adjust the first convolutional layer to match the correct number of input channels
        self.resnet.conv1 = nn.Conv2d(
            self.num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace the average pooling and the final fully connected layer
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN."""
        return self.resnet(x)
