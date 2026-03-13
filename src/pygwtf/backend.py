from abc import ABC, abstractmethod
from typing import Any

import numpy as np

try:
    import cupy as cp

    gpu_available = True
except ImportError:
    gpu_available = False


class Backend(ABC):
    """
    Backend base class. Defines the interface for the array module to be used for computations (e.g. numpy or cupy)
    and any other backend-specific functionality (such as kernel selection).
    """

    @property
    @abstractmethod
    def xp(self) -> Any:
        """
        Returns the array module to be used for computations. This will be `numpy` for CPU computations and `cupy` for GPU computations.
        """

    @property
    def uses_gpu(self) -> bool:
        """
        Returns True if the backend uses GPU computations, False otherwise.
        """
        return False


class CPUBackend(Backend):
    @property
    def xp(self):
        return np


class GPUBackend(Backend):
    @property
    def xp(self):
        if not gpu_available:
            raise ImportError(
                "CuPy is not available. Please install CuPy to use the GPU backend."
            )
        return cp

    @property
    def uses_gpu(self) -> bool:
        return True


def get_backend(backend_name: str) -> Backend:
    """
    Factory function to get the appropriate backend instance based on the provided backend name.
    """
    if backend_name == "cpu":
        return CPUBackend()
    elif backend_name == "gpu":
        return GPUBackend()
    else:
        raise ValueError(
            f"Invalid backend name '{backend_name}'. Valid options are 'cpu' and 'gpu'."
        )
