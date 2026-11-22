"""
Analytic model interface (v2).

In v2 the numerical work of a model -- unit conversion and derived
parameters, time bounds, and the phase/frequency/amplitude track -- lives in
a C++ header compiled into the backend (``src/cpp/include/gwtf/models``), so
it can be fused into the GPU kernels. The Python class only declares the
parameter names (in kernel order) and which compiled model to use.

Defining a new analytic model
-----------------------------
1. Write ``src/cpp/include/gwtf/models/<name>.hpp`` implementing the model
   concept documented in ``model_base.hpp`` (see ``taylort2ecc.hpp``).
2. Register it in ``registry.hpp`` with a new id and rebuild
   (``pip install -e .`` / ``pip install .``).
3. Subclass :class:`AnalyticModel` with ``kernel_name`` equal to the C++
   ``name`` and the parameter lists in the same order as the C++ indices.

Models that are easier to express numerically (or are produced by another
code) can instead feed per-segment tracks to
:class:`pygwtf.multimode.MultipleHarmonicTimeFrequencyWaveform`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..backend import Backend, get_backend


class AnalyticModel:
    """Base class for analytic waveform models backed by a compiled kernel."""

    #: Name of the compiled C++ model (``Model::name``) this class drives.
    kernel_name: str = ""

    def __init__(self, backend: str | Backend | None = None):
        self.backend = get_backend(backend)
        self._info = self._lookup()

    def _lookup(self):
        for info in self.backend.core.models():
            if info.name == self.kernel_name:
                if info.n_params != self.num_parameters:
                    raise ValueError(
                        f"{type(self).__name__} declares {self.num_parameters} parameters "
                        f"but the compiled model {info.name} expects {info.n_params}."
                    )
                if info.n_derived != self.num_derived_parameters:
                    raise ValueError(
                        f"{type(self).__name__} declares {self.num_derived_parameters} derived "
                        f"parameters but the compiled model {info.name} has {info.n_derived}."
                    )
                return info
        names = [m.name for m in self.backend.core.models()]
        raise ValueError(
            f"No compiled model named '{self.kernel_name}' (available: {names}). "
            "Add it to src/cpp/include/gwtf/models/registry.hpp and rebuild."
        )

    @property
    def kernel_id(self) -> int:
        """Integer id of the compiled model."""
        return self._info.id

    @property
    def num_modes(self) -> int:
        """Number of harmonics produced by the compiled model."""
        return self._info.n_modes

    @property
    def parameters(self) -> list[str]:
        """Input parameter names, in kernel order."""
        raise NotImplementedError

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    @property
    def derived_parameters(self) -> list[str]:
        """Derived parameter names, appended after the inputs in kernel order."""
        return []

    @property
    def num_derived_parameters(self) -> int:
        return len(self.derived_parameters)

    def compute_derived_parameters(self, parameters: Any) -> None:
        """
        Fill derived parameters in place (v1-compatible helper).

        ``parameters`` is a backend array of shape
        ``(n_sources, n_parameters + n_derived_parameters)`` whose first
        ``n_parameters`` columns hold the raw inputs. As in v1, the inputs are
        converted to the kernel's internal units in place.
        """
        xp = self.backend.xp
        params = xp.asarray(parameters)
        n = params.shape[0]
        raw = xp.ascontiguousarray(
            params[:, : self.num_parameters], dtype=np.float64
        )
        out = xp.zeros(
            (n, self.num_parameters + self.num_derived_parameters), np.float64
        )
        seg = xp.zeros(n, np.int32), xp.zeros(n, np.int32)
        grid = self.backend.core.Grid()
        grid.nT, grid.nF, grid.dT, grid.dF, grid.fmin = 1, 1, 1.0, 1.0, 1.0
        self.backend.core.prepare(
            self.kernel_id,
            raw,
            out,
            seg[0],
            seg[1],
            grid,
            1.0,
            1.0,
            self.backend.stream_ptr(),
        )
        parameters[...] = out
