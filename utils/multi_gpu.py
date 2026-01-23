"""Multi-GPU training utilities.

Provides wrappers for DataParallel and utilities for managing
multi-GPU training setups.
"""


import torch
import torch.nn as nn


def setup_multi_gpu(
    model: nn.Module,
    device: str = "cuda",
    device_ids: list[int] | None = None
) -> tuple[nn.Module, str]:
    """Setup model for multi-GPU training.

    Automatically detects available GPUs and wraps the model in DataParallel
    if multiple GPUs are available.

    Args:
        model: PyTorch model to setup
        device: Target device ("cuda" or "cpu")
        device_ids: List of GPU IDs to use (None = all available)

    Returns:
        (wrapped_model, device_str): Model wrapped for multi-GPU if applicable,
                                     and the device string to use for tensors

    Example:
        >>> model = MyModel()
        >>> model, device = setup_multi_gpu(model, device="cuda")
        >>> # Use device for tensor operations
        >>> x = x.to(device)
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()

        if gpu_count > 1:
            if device_ids is None:
                device_ids = list(range(gpu_count))

            print(f"[multi-gpu] Using DataParallel on {len(device_ids)} GPUs: {device_ids}")
            model = nn.DataParallel(model, device_ids=device_ids)
            device_str = f"cuda:{device_ids[0]}"  # Primary device
        else:
            print("[multi-gpu] Single GPU detected")
            device_str = "cuda:0"

        model = model.to(device_str)
    else:
        print("[multi-gpu] Using CPU")
        device_str = "cpu"
        model = model.to(device_str)

    return model, device_str


def get_unwrapped_model(model: nn.Module) -> nn.Module:
    """Get underlying model from DataParallel wrapper.

    Use this before saving model state_dict to avoid saving with
    'module.' prefix in keys.

    Args:
        model: Potentially wrapped model

    Returns:
        Unwrapped model

    Example:
        >>> torch.save({
        ...     'model_state': get_unwrapped_model(model).state_dict()
        ... }, checkpoint_path)
    """
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def get_gpu_memory_info() -> dict[int, dict[str, float]]:
    """Get memory usage information for all available GPUs.

    Returns:
        Dictionary mapping GPU ID to memory info in MB:
        {
            0: {'allocated': 1024.0, 'reserved': 2048.0, 'free': 6144.0},
            1: {'allocated': 512.0, 'reserved': 1024.0, 'free': 7168.0},
            ...
        }

    Example:
        >>> mem_info = get_gpu_memory_info()
        >>> for gpu_id, info in mem_info.items():
        ...     print(f"GPU {gpu_id}: {info['allocated']:.0f}MB allocated")
    """
    if not torch.cuda.is_available():
        return {}

    info = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**2
        reserved = torch.cuda.memory_reserved(i) / 1024**2
        total = torch.cuda.get_device_properties(i).total_memory / 1024**2
        free = total - reserved

        info[i] = {
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'total': total,
        }

    return info


def print_gpu_utilization() -> None:
    """Print GPU memory utilization for all available GPUs.

    Example:
        >>> print_gpu_utilization()
        GPU 0: 1024.0 MB allocated / 2048.0 MB reserved / 8192.0 MB total
        GPU 1: 512.0 MB allocated / 1024.0 MB reserved / 8192.0 MB total
    """
    mem_info = get_gpu_memory_info()
    if not mem_info:
        print("No GPUs available")
        return

    for gpu_id, info in mem_info.items():
        print(
            f"GPU {gpu_id}: {info['allocated']:.0f} MB allocated / "
            f"{info['reserved']:.0f} MB reserved / {info['total']:.0f} MB total"
        )
