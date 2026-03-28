from typing import Callable

import numpy as np

from ...constants import MTsun, clight, pc
from ..base import AnalyticModel
from .common import (
    _get_amplitude,
    _get_hplus_hcross,
    _get_phi_f_fdot,
    _get_time_to_coalescence_cpu_wrap,
    _get_time_to_coalescence_gpu_wrap,
    _get_time_to_f_cpu_wrap,
    _get_time_to_f_gpu_wrap,
)

THREADS_PER_BLOCK = 128


class TaylorT2Ecc(AnalyticModel):
    @property
    def parameters(self) -> list[str]:
        return [
            "M",
            "eta",
            "cosi",
            "e0",
            "D",
            "f0",
            "phi_coal",
        ]

    @property
    def derived_parameters(self) -> list[str]:
        return [
            "t_coal",
        ]

    def compute_derived_parameters(self, parameters: np.ndarray) -> None:
        # dimensionalise parameters
        parameters[:, 0] *= MTsun  # M
        parameters[:, 4] *= pc / clight  # D

        if self.backend.uses_gpu:
            n_sources = parameters.shape[0]
            bpg = (n_sources + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
            _get_time_to_coalescence_gpu_wrap[bpg, THREADS_PER_BLOCK](
                parameters[:, -1], parameters
            )
        else:
            _get_time_to_coalescence_cpu_wrap(parameters[:, -1], parameters)

    @property
    def amplitude_function(self) -> Callable:
        return _get_amplitude

    @property
    def phi_f_fdot_function(self) -> Callable:
        return _get_phi_f_fdot

    @property
    def get_TT_polarisations_function(self) -> Callable:
        return _get_hplus_hcross

    def get_time_bounds(
        self, parameters: np.ndarray, frequency_band: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the start and end times of the waveform for each source in the input parameter array.
        This is used to determine the time segments over which to compute the waveform for each source.
        """
        t_coal = parameters[:, -1]

        t_start = self.backend.xp.zeros_like(t_coal)
        t_end = self.backend.xp.zeros_like(t_coal)
        if self.backend.uses_gpu:
            n_sources = parameters.shape[0]
            bpg = (n_sources + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
            _get_time_to_f_gpu_wrap[bpg, THREADS_PER_BLOCK](
                t_end, frequency_band[1], t_coal, parameters
            )
        else:
            _get_time_to_f_cpu_wrap(
                t_end, frequency_band[1], t_coal, parameters
            )

        return t_start, t_end
