import torch

try:
    import warptrack_cuda_ext
except ImportError as exc:
    raise ImportError(
        "Could not import warptrack_cuda_ext. "
        "Build it first with: python setup.py build_ext --inplace"
    ) from exc


def pairwise_distance(positions: torch.Tensor) -> torch.Tensor:
    """Return squared pairwise spatial distances for CUDA hit positions."""
    return warptrack_cuda_ext.pairwise_distance(positions)
