from typing import Callable

import torch

# ###################### #
# CONVENTION CONVERSIONS #
# ###################### #


def xyzwxyz_to_xyzxyzw_SE3(xyzwxyz: torch.Tensor) -> torch.Tensor:
    """Converts a tensor of 7d poses with quats from (w, x, y, z) to (x, y, z, w) order.

    Args:
        xyzwxyz: The tensor of poses of shape (..., 7) whose quats are in (w, x, y, z) order.

    Returns:
        xyzxyzw: The tensor now with (x, y, z, w) order.
    """
    return torch.cat(
        (
            xyzwxyz[..., :3],  # translations
            xyzwxyz[..., -3:],  # the (qx, qy, qz) components
            xyzwxyz[..., -4:-3],  # the qw component
        ),
        dim=-1,
    )


def xyzxyzw_to_xyzwxyz_SE3(xyzxyzw: torch.Tensor) -> torch.Tensor:
    """Converts a tensor of 7d poses with quats from (x, y, z, w) to (w, x, y, z) order.

    Args:
        xyzxyzw: The tensor of poses of shape (..., 7) whose quats are in (x, y, z, w) order.

    Returns:
        xyzwxyz: The tensor now with (w, x, y, z) order.
    """
    return torch.cat(
        (
            xyzxyzw[..., :3],  # translations
            xyzxyzw[..., -1:],  # the qw component
            xyzxyzw[..., -4:-1],  # the (qx, qy, qz) components
        ),
        dim=-1,
    )


# ########## #
# EVALUATION #
# ########## #


def time_torch_fn(fn: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, float]:
    """Time a torch function.

    Source: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#demonstrating-speedups

    Args:
        fn: The function to time.

    Returns:
        result: The result of the function.
        time: The time taken to execute the function.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000
