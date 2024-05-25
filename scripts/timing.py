import numpy as np
import torch

from argus.models import NCameraCNN, NCameraCNNConfig
from argus.utils import time_torch_fn

torch.set_float32_matmul_precision("high")


def main():
    """Runs the timing test."""
    # timing/model parameters
    N_TRIALS = 100  # run the forward pass 100 times
    N_CAMS = 2  # the number of cameras
    W = 256  # resolution of VGA images from ZED
    H = 256

    # setting up model + compiling
    cfg = NCameraCNNConfig(n_cams=N_CAMS, W=W, H=H)
    model = NCameraCNN(cfg).to(torch.float32).cuda()
    model = torch.compile(model, mode="reduce-overhead")

    # Fixes 'Function definition does not bind loop variable 'x''.
    # stackoverflow.com/a/54947488
    def make_fn(x):

        def _():
            return model(x)

        return _

    # timing
    times = []
    for i in range(N_TRIALS + 1):
        # drawing random input
        x = torch.rand((2, N_CAMS * 3, 256, 256), device="cuda")

        # timing the forward pass
        with torch.no_grad():
            _, runtime = time_torch_fn(make_fn(x))
        if i > 0:
            times.append(runtime)
        else:
            print(f"Compilation took {runtime} seconds.")
    print(f"Forward pass took {np.mean(runtime)} seconds on average.")


if __name__ == "__main__":
    main()
