from abc import ABC, abstractmethod
from typing import Callable

from numpy.typing import ArrayLike

from ..backend import Backend, get_backend


class AnalyticModel(ABC):
    """
    AnalyticModel is the base class for all analytic waveform models in pygwtf.

    The model class is responsible for
     - Defining the input parameters of the model
     - Computing any derived parameters (such as the time to coalescence) from the input parameters
     - Providing access to the numba-compiled functions required for the waveform generation kernel

    See the docstrings for each required method for more details.
    """

    def __init__(self, backend: str | Backend | None = None):  # type: ignore
        if backend is None:
            self.backend = get_backend("cpu")
        elif isinstance(backend, str):
            self.backend = get_backend(backend)
        else:
            self.backend = backend

    @property
    @abstractmethod
    def parameters(self) -> list[str]:
        """
        Returns a list of the input parameters required for this model.
        The order of the parameters in this list must match the order of the parameters expected by the numba-compiled functions.
        """

    @property
    def num_parameters(self) -> int:
        """
        Returns the number of input parameters required for this model.
        This is inferred from the length of the list returned by the `parameters` method.
        """
        return len(self.parameters)

    @property
    def derived_parameters(self) -> list[str]:
        """
        Returns a list of any derived parameters that are computed from the input parameters.
        The order of the parameters in this list must match the order of the derived parameters expected by the numba-compiled functions.
        These parameters will be appended to the input parameters in the parameter array passed to the numba-compiled functions.
        """
        return []

    @property
    def num_derived_parameters(self) -> int:
        """
        Returns the number of derived parameters computed by this model.
        This is inferred from the length of the list returned by the `derived_parameters` method.
        """
        return len(self.derived_parameters)

    def compute_derived_parameters(self, parameters: ArrayLike) -> None:
        """
        Computes any derived parameters from the input parameters.
        The input parameter array will have shape (n_sources, n_parameters + n_derived_parameters) and the derived parameters
        should be computed in place and stored in the appropriate columns of this array (i.e. the columns after the input parameters).
        The default implementation of this method does nothing, as not all models will have derived parameters.
        """

    @property
    @abstractmethod
    def amplitude_function(self) -> Callable:
        """
        The numba-compiled function that computes the amplitude of the waveform at a given time, frequency and frequency derivative.
        This function should have the signature `amplitude(t, f, fdot, parameters)` where `parameters` is a 1D array containing the input parameters
          for a single source (in the order specified by the `parameters` method).
        """

    @property
    @abstractmethod
    def phi_f_fdot_function(self) -> Callable:
        """
        The numba-compiled function that computes the phase, frequency and frequency derivative of the waveform at a given time.
        This function should have the signature `phi_f_fdot(t, parameters)` where `parameters` is a 1D array containing the input parameters
          for a single source (in the order specified by the `parameters` method).
        """

    @property
    def get_TT_polarisations_function(self) -> Callable | None:
        """
        The numba-compiled function that computes the plus and cross polarisations of the waveform at a given time, frequency and frequency derivative.
        This function is only required for non-TDI waveform generation.
        This function should have the signature `hplus_hcross(hlm, parameters)` where `parameters` is a 1D array containing the input parameters
          for a single source (in the order specified by the `parameters` method) and `hlm` is the multipole.
        The default implementation of this method returns None for flexibility, as not all models will require this function.
        """
        return None

    def get_time_bounds(
        self, parameters: ArrayLike, frequency_band: tuple[float, float]
    ) -> tuple[ArrayLike, ArrayLike] | None:
        """
        Returns the start and end times of the waveform for each source.
        For non-stationary signals (e.g. compact binary coalescences), this method should be overridden to return the appropriate time bounds based on the input parameters.
        The returned start and end times should be 1D arrays of shape (n_sources,) containing the start and end times for each source, or None

        The default implementation of this method returns None for all sources, i.e. the signal remains in-band for all relevant times.

        Parameters
        ----------
        parameters: ArrayLike
            A 2D array of shape (n_sources, n_parameters + n_derived_parameters)
        frequency_band: tuple[float]
            A tuple containing the lower and upper bounds of the frequency band of interest.

        Returns
        -------
        tuple[ArrayLike, ArrayLike] | None
            A tuple containing the start and end times of the waveform for each source, or None if the signal is assumed to be in-band for all relevant times.
        """
        return None
