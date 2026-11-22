"""
Compute backends for the v2 compiled core.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np


class Backend:
    """Base class: array module + compiled core."""

    name: str = ""

    @property
    def xp(self) -> Any:
        """Array module used for allocation (numpy or cupy)."""
        raise NotImplementedError

    @property
    def core(self) -> Any:
        """Compiled extension module implementing the kernels."""
        raise NotImplementedError

    @property
    def uses_gpu(self) -> bool:
        return False

    def asnumpy(self, array: Any) -> np.ndarray:
        """Copy (if needed) a backend array to a NumPy array."""
        return np.asarray(array)

    def stream_ptr(self) -> int:
        """Raw stream handle kernels are launched on (0 for the CPU)."""
        return 0

    def synchronize(self) -> None:
        """Block until all work queued by pygwtf on this backend is done."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class CPUBackend(Backend):
    """Serial host execution with NumPy arrays."""

    name = "cpu"

    @property
    def xp(self):
        return np

    @functools.cached_property
    def core(self):
        from . import _core_cpu

        return _core_cpu


class GPUBackend(Backend):
    """HIP/CUDA execution with CuPy arrays on the current CuPy device and stream."""

    name = "gpu"

    @functools.cached_property
    def xp(self):
        try:
            import cupy
        except ImportError as err:  # pragma: no cover - depends on install
            raise ImportError(
                "CuPy is required for the GPU backend: install the CuPy build for "
                "your platform (e.g. cupy-cuda12x, cupy-cuda13x or cupy-rocm-*)."
            ) from err
        return cupy

    @functools.cached_property
    def core(self):
        try:
            from . import _core_gpu
        except ImportError as err:  # pragma: no cover - depends on build
            raise ImportError(
                "pygwtf was built without the GPU backend (no CUDA/HIP compiler was "
                "found at build time). Rebuild with GWTF_GPU=CUDA or GWTF_GPU=HIP."
            ) from err
        return _core_gpu

    @property
    def uses_gpu(self) -> bool:
        return True

    def asnumpy(self, array: Any) -> np.ndarray:
        return self.xp.asnumpy(array)

    def stream_ptr(self) -> int:
        return int(self.xp.cuda.get_current_stream().ptr)

    def synchronize(self) -> None:
        self.xp.cuda.get_current_stream().synchronize()


_BACKENDS = {"cpu": CPUBackend, "gpu": GPUBackend}


@functools.lru_cache(maxsize=None)
def _cached_backend(name: str) -> Backend:
    return _BACKENDS[name]()


def get_backend(backend: str | Backend | None = "cpu") -> Backend:
    """Return a backend instance from a name (``'cpu'`` / ``'gpu'``) or pass one through."""
    if backend is None:
        backend = "cpu"
    if isinstance(backend, Backend):
        return backend
    if backend not in _BACKENDS:
        raise ValueError(
            f"Invalid backend name '{backend}'. Valid options are 'cpu' and 'gpu'."
        )
    return _cached_backend(backend)


def gpu_available() -> bool:
    """True if the GPU backend is built and a device is usable."""
    try:
        b = get_backend("gpu")
        b.core  # noqa: B018
        return b.xp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False
