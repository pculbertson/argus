from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from robomimic.models.obs_core import VisualCore
from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax


@dataclass(frozen=True)
class NCameraCNNConfig:
    """Configuration for the NCameraCNN model.

    Fields:
        n_cams: The number of cameras in the scene.
        W: The width of the input images.
        H: The height of the input images.
        resnet_output_dim: The output dimension of the ResNet model (before final FC layer).
    """

    n_cams: int = 2
    resnet_output_dim: int = 1024


class NCameraCNN(nn.Module):
    """A CNN which assumes N cameras are available in the scene.

    The inputs are N images of dimension HxW and the outputs are the pose of the cube which is located in the scene.
    In particular, the outputs are 6d vectors in se(3) which must be sent to SE(3) via the exponential map.

    The main reason for passing a 6-vector instead of a pypose SE(3) object is because then we can torch compile
    the model without error.
    """

    def __init__(self, cfg: Optional[NCameraCNNConfig] = None, W: int = 672, H: int = 376) -> None:
        """Initialize the CNN.

        Args:
            cfg: The configuration for the model. If None, the default configuration is used.
        """
        super().__init__()
        self.resnet = models.resnet34(weights="DEFAULT")

        if cfg is None:
            cfg = NCameraCNNConfig()
        self.num_channels = 3 * cfg.n_cams  # RGB-only for each cam, all channels concatenated
        self.H = H
        self.W = W
        self.resnet_output_dim = cfg.resnet_output_dim

        self.n_cams = cfg.n_cams

        # adjust the first convolutional layer to match the correct number of input channels

        # replace the average pooling and the final fully connected layer
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.resnet_output_dim)

        self.output_mlp = nn.Sequential(
            nn.Linear(self.n_cams * self.resnet_output_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN.

        Args:
            x: The input images of shape (B, 3 * n_cams, W, H), concatenated along the channel dimension.

        Returns:
            pose: The predicted pose of the cube in the scene expressed in se(3). To get the pose in SE(3), apply the
                exponential map to it, e.g., `pose.Exp()`.
        """
        assert len(x.shape) == 4, "The input images must be of shape (B, C, W, H)! If B=1, add a dummy dimension."

        B, _, _, _ = x.shape

        # Split the input images into n_cams.
        x = x.reshape(-1, 3, self.W, self.H)

        # Forward pass through the resnet.
        x = self.resnet(x)

        # Reshape the output to (B, n_cams * resnet_output_dim).
        x = x.reshape(B, self.n_cams * self.resnet_output_dim)
        x = nn.GELU()(x)

        return self.output_mlp(x)


class NCameraEncoder(nn.Module):
    def __init__(self, cfg: Optional[NCameraCNNConfig] = None) -> None:
        """Initialize the CNN.

        Args:
            cfg: The configuration for the model. If None, the default configuration is used.
        """
        super().__init__()
        self.backbone = VisualCore(
            input_shape=(3, cfg.W, cfg.H),
            backbone_class="ResNet18Conv",  # use ResNet18 as the visualcore backbone
            backbone_kwargs={"pretrained": True, "input_coord_conv": False},  # kwargs for the ResNet18Conv class
            pool_class="SpatialSoftmax",  # use spatial softmax to regularize the model output
            pool_kwargs={"num_kp": 32},  # kwargs for the SpatialSoftmax --- use 32 keypoints
            flatten=True,  # flatten the output of the spatial softmax layer
            feature_dimension=cfg.resnet_output_dim,  # project the flattened feature into a 64-dim vector through a linear layer
        )

        if cfg is None:
            cfg = NCameraCNNConfig()

        # Store config.
        self.H = cfg.H
        self.W = cfg.W
        self.resnet_output_dim = cfg.resnet_output_dim
        self.n_cams = cfg.n_cams

        self.output_mlp = nn.Sequential(
            nn.Linear(self.n_cams * cfg.resnet_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 4, "The input images must be of shape (B, C, W, H)! If B=1, add a dummy dimension."

        B, _, _, _ = x.shape

        # Split the input images into n_cams.
        x = x.reshape(-1, 3, self.W, self.H)

        # Forward pass through the resnet.
        x = self.backbone(x)

        # Reshape the output to (B, n_cams * resnet_output_dim).
        x = x.reshape(B, self.n_cams * self.resnet_output_dim)
        x = nn.ReLU()(x)

        return self.output_mlp(x)
