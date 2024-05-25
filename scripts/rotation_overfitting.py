# flake8: noqa
import pypose as pp
import torch
import torchvision.models as models


class MLP(torch.nn.Module):
    def __init__(self, in_features, hidden_layers, out_features):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(in_features, hidden_layers[0]))
        self.layers.append(torch.nn.ReLU())
        for i in range(1, len(hidden_layers)):
            self.layers.append(torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(hidden_layers[-1], out_features))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def geometric_loss(output: pp.SE3_type, target: pp.SE3_type) -> torch.Tensor:
    # Compute the rotation matrix from the predicted quaternion
    pose_err = (output.Inv() @ target).Log()
    return pose_err.norm(dim=-1).mean()


def train_MLP():
    # Create a simple MLP model
    model = MLP(3, [256, 256, 256, 256], 6)

    num_examples = 100

    # Create a random input tensor
    x = torch.rand(num_examples, 3)

    # Create random pose labels.
    y = pp.randn_SE3(num_examples)

    # Create optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for ii in range(10000):
        optimizer.zero_grad()
        outputs = model(x)
        loss = geometric_loss(pp.se3(outputs).Exp(), y)
        loss.backward()
        optimizer.step()

        if ii % 100 == 0:
            print(f"Iteration {ii}, Loss: {loss.item()}")


def train_resnet():
    # Load a simple resnet
    model = models.resnet18(pretrained="ResNet18_Weights.IMAGENET1K_V1")
    model.fc = torch.nn.Linear(model.fc.in_features, 6)

    num_examples = 100

    # Create a random input images
    H = 32
    W = 32
    x = torch.rand(num_examples, 3, H, W)

    # Create random pose labels.
    y = pp.randn_SE3(num_examples)

    # Create optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Device transfer.
    model = model.to("cuda")
    x = x.to("cuda")
    y = y.to("cuda")

    # Training loop
    for ii in range(10000):
        optimizer.zero_grad()
        outputs = model(x)
        loss = geometric_loss(pp.se3(outputs).Exp(), y)
        loss.backward()
        optimizer.step()

        if ii % 100 == 0:
            print(f"Iteration {ii}, Loss: {loss.item()}")


if __name__ == "__main__":
    train_resnet()
