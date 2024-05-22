from dataclasses import dataclass
from typing import Optional

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
        output_type: The type of the output pose pose representation. Either "cts_6d" or "se3".
    """

    n_cams: int = 2
    W: int = 672
    H: int = 376
    output_type: str = "se3"

    def __post_init__(self) -> None:
        """Sanity checks on inputs."""
        assert self.n_cams > 0, "The number of cameras must be positive!"
        assert self.W > 0, "The width of the images must be positive!"
        assert self.H > 0, "The height of the images must be positive!"
        assert self.output_type in ["cts_6d", "se3"], "The output type must be either 'cts_6d' or 'se3'!"


class NCameraCNN(nn.Module):
    """A CNN which assumes N cameras are available in the scene.

    The inputs are N images of dimension HxW and the outputs are the pose of the cube which is located in the scene.

    The outputs of this network depend on the `output_type` field of the config.

    * If the output type is "cts_6d," then the output dimension is 9. The first 3 outputs are the predicted position of
    the cube with respect to the world frame. The last 6 represent the first two columns of a rotation matrix before the
    process of normalization and orthogonalization via Gram-Schmidt. It is show in "On the Continuity of Rotation
    Representations in Neural Networks" that this is a continuous representation for learning rotations.
    * If the output type is "se3," then the output dimension is 6 and represents a vector in se(3) which must be sent to
    SE(3) via the exponential map. The main reason for passing a 6-vector instead of a pypose SE(3) object is because
    then we can torch compile the model without error.
    """

    def __init__(self, cfg: Optional[NCameraCNNConfig] = None) -> None:
        """Initialize the CNN.

        Args:
            cfg: The configuration for the model. If None, the default configuration is used.
        """
        super().__init__()
        self.resnet = models.resnet18(weights="DEFAULT")  # finetune a pretrained ResNet-18

        if cfg is None:
            cfg = NCameraCNNConfig()
        self.num_channels = 3 * cfg.n_cams  # RGB-only for each cam, all channels concatenated
        self.H = cfg.H
        self.W = cfg.W
        self.output_type = cfg.output_type

        # adjust the first convolutional layer to match the correct number of input channels
        self.resnet.conv1 = nn.Conv2d(self.num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # replace the average pooling and the final fully connected layer
        if self.output_type == "cts_6d":
            output_dim = 9
        elif self.output_type == "se3":
            output_dim = 6
        else:
            raise ValueError("The output type must be either 'cts_6d' or 'se3'!")
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN.

        Args:
            x: The input images of shape (B, 3 * n_cams, W, H), concatenated along the channel dimension.

        Returns:
            pose: The predicted pose of the cube in the scene expressed in se(3). To get the pose in SE(3), apply the
                exponential map to it, e.g., `pose.Exp()`.
        """
        assert len(x.shape) == 4, "The input images must be of shape (B, C, W, H)! If B=1, add a dummy dimension."
        return self.resnet(x)
