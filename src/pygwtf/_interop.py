"""
Array-API interoperability.
"""

from __future__ import annotations

from typing import Any

import array_api_compat as aac
import numpy as np

# DLPack device types that denote GPU memory (CUDA, ROCm, CUDA managed).
_DLPACK_GPU = {2, 10, 13}


def _is_array(x: Any) -> bool:
    return x is not None and (
        isinstance(x, np.ndarray) or aac.is_array_api_obj(x)
    )


def _on_gpu(x: Any) -> bool:
    dev = getattr(x, "__dlpack_device__", None)
    if dev is None:
        return False
    try:
        return int(dev()[0]) in _DLPACK_GPU
    except Exception:
        return False


def to_backend(x: Any, backend, dtype=None):
    """Return ``x`` as a C-contiguous array of the backend's array module."""
    xp = backend.xp
    if backend.uses_gpu:
        if isinstance(x, xp.ndarray):
            arr = x
        elif _on_gpu(x):
            arr = xp.from_dlpack(x)
        else:
            arr = xp.asarray(np.asarray(x))
        if arr.device.id != xp.cuda.Device().id:
            raise ValueError(
                f"Array is on GPU {arr.device.id} but the current device is "
                f"{xp.cuda.Device().id}; use `with cupy.cuda.Device(...)`."
            )
    else:
        if isinstance(x, np.ndarray):
            arr = x
        elif _on_gpu(x):
            get = getattr(x, "get", None)  # CuPy
            arr = (
                get()
                if get is not None
                else np.asarray(aac.to_device(x, "cpu"))
            )
        else:
            arr = np.asarray(x)
    return xp.ascontiguousarray(arr, dtype=dtype)


class ArrayFormat:
    """Records the array format of the user's inputs so outputs can match it."""

    def __init__(self, sample: Any = None):
        self.kind = "numpy"
        self.ns = np
        self.on_gpu = False
        if sample is None:
            return
        self.on_gpu = _on_gpu(sample)
        if isinstance(sample, np.ndarray):
            return
        try:
            ns = aac.array_namespace(sample)
        except TypeError:
            return
        if aac.is_numpy_namespace(ns):
            return
        self.kind = "cupy" if aac.is_cupy_namespace(ns) else "other"
        self.ns = ns
        self.device = aac.device(sample)

    @classmethod
    def from_inputs(cls, *candidates: Any) -> "ArrayFormat":
        """Format of the first genuine array among the candidates (dict values included)."""
        for c in candidates:
            if isinstance(c, dict):
                for v in c.values():
                    if _is_array(v) and getattr(v, "ndim", 0) > 0:
                        return cls(v)
            elif _is_array(c):
                return cls(c)
        return cls(None)

    def wrap(self, arr: Any, backend) -> Any:
        """Convert a backend array to the recorded format."""
        if arr is None:
            return None
        if self.kind == "numpy":
            return backend.asnumpy(arr)
        if self.kind == "cupy":
            if backend.uses_gpu:
                return arr
            import cupy

            return cupy.asarray(arr)
        # Any other array-API library (torch, jax, ...).
        if backend.uses_gpu and self.on_gpu:
            return self.ns.from_dlpack(arr)
        host = backend.asnumpy(arr)
        out = self.ns.asarray(host)
        if self.on_gpu:
            out = aac.to_device(out, self.device)
        return out
