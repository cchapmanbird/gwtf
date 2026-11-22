from .base import AnalyticModel


class TaylorT2Ecc(AnalyticModel):
    """TaylorT2 with small eccentricity (non-spinning), (2,2) mode.

    Parameters: ``M`` [Msun], ``eta``, ``cosi``, ``e0``, ``D`` [pc], ``f0`` [Hz],
    ``phi_coal``. Derived: ``t_coal`` [s]. Kernel: ``models/taylort2ecc.hpp``.
    """

    kernel_name = "TaylorT2Ecc"

    @property
    def parameters(self) -> list[str]:
        return ["M", "eta", "cosi", "e0", "D", "f0", "phi_coal"]

    @property
    def derived_parameters(self) -> list[str]:
        return ["t_coal"]
