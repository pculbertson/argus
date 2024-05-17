from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


@dataclass(frozen=True)
class NCameraCNNConfig:
    """Configuration for the NCameraCNN model.

    Fields:
        n_cams: The number of cameras in the scene.
        W: The width of the input images.
        H: The height of the input images.
    """

    n_cams: int = 2
    W: int = 672
    H: int = 376


class NCameraCNN(nn.Module):
    """A CNN which assumes N cameras are available in the scene.

    The inputs are N images of dimension HxW and the outputs are the pose of the cube which is located in the scene.
    In particular, the outputs are 6d vectors in se(3) which must be sent to SE(3) via the exponential map.
    """

    def __init__(self, cfg: NCameraCNNConfig) -> None:
        """Initialize the CNN."""
        super().__init__()
        self.resnet = models.resnet18(weights="DEFAULT")  # finetune a pretrained ResNet-18
        self.num_channels = 3 * cfg.n_cams  # RGB-only for each cam, all channels concatenated
        self.H = cfg.H
        self.W = cfg.W

        # adjust the first convolutional layer to match the correct number of input channels
        self.resnet.conv1 = nn.Conv2d(self.num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the average pooling and the final fully connected layer
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN.

        Args:
            x: The input images of shape (B, 3 * n_cams, W, H), concatenated along the channel dimension.

        Returns:
            pose: The predicted pose of the cube in the scene expressed in se(3). To get the pose in SE(3), apply the
                exponential map to it, e.g., `pose.Exp()`.
        """
        return self.resnet(x)
