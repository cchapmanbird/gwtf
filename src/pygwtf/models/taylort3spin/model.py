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


def etaM_to_m1m2(eta, M):
    """
    Convert the symmetic mass ratio and total mass of a binary system to the individual masses, assuming m1 >= m2.
    Args:
        eta: Symmetric mass ratio.
        M: Total mass.

    Returns:
        m1: Mass of the first body.
        m2: Mass of the second body.
    """
    q = (1.0 + (1.0 - 4.0 * eta) ** 0.5) / (2.0 * eta) - 1.0
    mu = M * q / (1.0 + q) ** 2
    sqrdet = (1 - 4 * mu / M) ** 0.5
    m1 = M / 2 * (1 + sqrdet)
    m2 = M / 2 * (1 - sqrdet)
    return m1, m2


class TaylorT3Spin(AnalyticModel):
    @property
    def parameters(self) -> list[str]:
        return [
            "M",
            "eta",
            "cosi",
            "D",
            "f0",
            "s1",
            "s2",
            "phi_coal",
        ]

    @property
    def derived_parameters(self) -> list[str]:
        return [
            "t_coal",
            "delta",
            "sigma",
            "s",
        ]

    def compute_derived_parameters(self, parameters: np.ndarray) -> None:
        # dimensionalise parameters
        parameters[:, 0] *= MTsun  # M
        parameters[:, 3] *= pc / clight  # D

        if self.backend.uses_gpu:
            _get_time_to_coalescence_gpu_wrap(parameters[:, 8], parameters)
        else:
            _get_time_to_coalescence_cpu_wrap(parameters[:, 8], parameters)

        M = parameters[:, 0]
        m1, m2 = etaM_to_m1m2(parameters[:, 1], M)

        parameters[:, 9] = (m1 - m2) / M  # delta
        parameters[:, 10] = (
            m2 * parameters[:, 6] - m1 * parameters[:, 5]
        ) / M  # sigma
        parameters[:, 11] = (
            m2 * parameters[:, 6] - m1 * parameters[:, 5]
        ) / M  # M

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
        t_coal = parameters[:, 8]

        t_start = self.backend.xp.zeros_like(t_coal)
        t_end = self.backend.xp.zeros_like(t_coal)
        if self.backend.uses_gpu:
            _get_time_to_f_gpu_wrap(
                t_end, frequency_band[1], t_coal, parameters
            )
        else:
            _get_time_to_f_cpu_wrap(
                t_end, frequency_band[1], t_coal, parameters
            )

        return t_start, t_end
