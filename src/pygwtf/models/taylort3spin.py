from .base import AnalyticModel


class TaylorT3Spin(AnalyticModel):
    """TaylorT3, 3.5PN aligned spins (circular), (2,2) mode.

    Parameters: ``M`` [Msun], ``eta``, ``cosi``, ``D`` [pc], ``f0`` [Hz], ``s1``,
    ``s2``, ``phi_coal``. Derived: ``t_coal`` [s], ``delta``, ``sigma``, ``s``.
    Kernel: ``models/taylort3spin.hpp``.
    """

    kernel_name = "TaylorT3Spin"

    @property
    def parameters(self) -> list[str]:
        return ["M", "eta", "cosi", "D", "f0", "s1", "s2", "phi_coal"]

    @property
    def derived_parameters(self) -> list[str]:
        return ["t_coal", "delta", "sigma", "s"]
