from typing import Callable

import torch


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
