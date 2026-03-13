from typing import Callable, Type

import numpy as np

from .backend import Backend, get_backend
from .fresnel.kernel import analytic_kernel_constructor
from .models.base import AnalyticModel


class AnalyticTimeFrequencyWaveform:
    """
    Analytic time-frequency waveform generator.

    Constructs Fresnel-approximation kernels (from ``pygwtf.fresnel``) for both
    waveform generation and inner-product statistic evaluation, driven by a
    physical model.  Four compiled kernels are built at construction time — CPU
    and GPU variants for each of the waveform and statistic paths.  The active
    pair is exposed via the ``waveform_kernel`` / ``statistic_kernel`` properties
    according to the chosen backend.

    Parameters
    ----------
    model_class : type[AnalyticModel]
        The model class to instantiate (e.g. ``TaylorT2Ecc``).  Initialised
        inside this constructor with the chosen backend.
    config : dict
        Kernel configuration.  Required keys: ``'dT'``, ``'nF'``, ``'dF'``,
        ``'kernel_width'``.
    backend : str or Backend, optional
        Compute backend — ``'cpu'`` (default) or ``'gpu'``.
    tdi_type : {None, 1, 2}, optional
        TDI generation order.  ``None`` produces TT polarisations.
    channel_fn : callable, optional
        Channel function forwarded to the kernel constructor.
        For TDI use e.g. ``get_AET_TFs`` or ``get_XYZ_TFs``; for TT
        polarisations use e.g. ``_get_hplus_hcross``.
    n_channels : int, optional
        Number of output channels.  Defaults to 3 for TDI, 2 otherwise.
    """

    def __init__(
        self,
        model_class: Type[AnalyticModel],
        config: dict,
        backend: str | Backend = "cpu",
        tdi_type: int | None = None,
        channel_fn: Callable | None = None,
        n_channels: int | None = None,
    ):
        self.backend = (
            get_backend(backend) if isinstance(backend, str) else backend
        )
        self.model = model_class(backend=self.backend)
        self.config = config
        self.tdi_type = tdi_type

        if n_channels is not None:
            self.n_channels = n_channels
        elif tdi_type is not None:
            self.n_channels = 3
        else:
            self.n_channels = 2

        kernel_config = {
            **config,
            "nparams": self.model.num_parameters
            + self.model.num_derived_parameters,
        }

        if channel_fn is None:
            raise ValueError(
                "channel_fn must be provided (e.g. get_AET_TFs or _get_hplus_hcross)."
            )

        self.waveform_kernel_cpu, self.waveform_kernel_gpu = (
            analytic_kernel_constructor(
                kernel_config,
                self.model.amplitude_function,
                self.model.phi_f_fdot_function,
                channel_fn,
                compute_statistic=False,
                tdi_type=tdi_type,
            )
        )
        self.statistic_kernel_cpu, self.statistic_kernel_gpu = (
            analytic_kernel_constructor(
                kernel_config,
                self.model.amplitude_function,
                self.model.phi_f_fdot_function,
                channel_fn,
                compute_statistic=True,
                tdi_type=tdi_type,
            )
        )

        self._param_cache = None

    # ------------------------------------------------------------------
    # active kernel properties
    # ------------------------------------------------------------------

    @property
    def waveform_kernel(self):
        """Active waveform kernel for the selected backend."""
        return (
            self.waveform_kernel_gpu
            if self.backend.uses_gpu
            else self.waveform_kernel_cpu
        )

    @property
    def statistic_kernel(self):
        """Active statistic kernel for the selected backend."""
        return (
            self.statistic_kernel_gpu
            if self.backend.uses_gpu
            else self.statistic_kernel_cpu
        )

    # ------------------------------------------------------------------
    # parameter management
    # ------------------------------------------------------------------

    def _load_parameters(self, parameters) -> np.ndarray:
        """Copy *parameters* into the internal cache, realloc if source count changes.

        Parameters
        ----------
        parameters : dict[str, array_like] or array_like, shape (n_sources, n_params)
            Physical parameters as a named dictionary or a pre-ordered 2-D array.

        Returns
        -------
        ndarray, shape (n_sources, n_params + n_derived)
            The parameter cache with derived parameters filled in.
        """
        xp = self.backend.xp
        n_phys = self.model.num_parameters
        n_total = n_phys + self.model.num_derived_parameters

        if isinstance(parameters, dict):
            raw = xp.stack(
                [
                    xp.asarray(parameters[name], dtype=np.float64)
                    for name in self.model.parameters
                ],
                axis=1,
            )
        else:
            raw = xp.asarray(parameters, dtype=np.float64)

        n_sources = raw.shape[0]

        if (
            self._param_cache is None
            or self._param_cache.shape[0] != n_sources
        ):
            self._param_cache = xp.zeros(
                (n_sources, n_total), dtype=np.float64
            )

        self._param_cache[:, :n_phys] = raw
        self.model.compute_derived_parameters(self._param_cache)

        return self._param_cache

    def _compute_segment_indices(self, parameters_cache):
        """Derive per-source time-segment indices from the model's time bounds."""
        xp = self.backend.xp
        dT = self.config["dT"]
        dF = self.config["dF"]
        nF = self.config["nF"]
        frequency_band = (dF, nF * dF)

        bounds = self.model.get_time_bounds(parameters_cache, frequency_band)
        if bounds is None:
            raise ValueError(
                "Model.get_time_bounds returned None. "
                "Pass segment_start_inds and segment_end_inds explicitly."
            )
        t_start, t_end = bounds
        segment_start_inds = xp.floor(
            xp.asarray(t_start, dtype=np.float64) / dT
        ).astype(np.int64)
        segment_end_inds = xp.floor(
            xp.asarray(t_end, dtype=np.float64) / dT
        ).astype(np.int64)
        return segment_start_inds, segment_end_inds

    # ------------------------------------------------------------------
    # call
    # ------------------------------------------------------------------

    def __call__(
        self,
        parameters,
        channels_or_data=None,
        psds=None,
        segment_start_inds=None,
        segment_end_inds=None,
        parameters_response=None,
        spacecraft_orbits=None,
        statistic=None,
        return_statistic: bool = False,
    ):
        """Evaluate the analytic waveform or inner-product statistic.

        Parameters
        ----------
        parameters : dict or array_like, shape (n_sources, n_params)
            Source parameters as a named dict or a 2-D array in model order.
        channels_or_data : array_like or None
            *Waveform mode*: pre-allocated output of shape
            ``(n_sources, nT, nF, n_channels)``; auto-allocated if ``None``.
            *Statistic mode*: data array of shape ``(nT, nF, n_channels)``.
        psds : array_like, optional
            Power spectral densities, shape ``(nT, nF, n_channels)``.
            Required for statistic mode.
        segment_start_inds, segment_end_inds : array_like of int, optional
            Per-source start/end tranche indices.  Derived automatically from
            ``model.get_time_bounds`` if not provided.
        parameters_response : array_like, shape (n_sources, 4), optional
            Response parameters ``[cosi, pol, ecliptic_long, ecliptic_lat]``.
            Required for TDI mode.
        spacecraft_orbits : array_like, shape (nT, 3, 3), optional
            Pre-computed orbit positions.  Required for TDI mode.
        statistic : array_like or None
            Pre-allocated output of shape ``(n_sources, nT, 2)``.
            Auto-allocated if ``None`` when ``return_statistic=True``.
        return_statistic : bool, optional
            Return the inner-product statistic instead of the channels array.

        Returns
        -------
        ndarray
            Channels ``(n_sources, nT, nF, n_channels)`` when
            ``return_statistic=False``, or statistic ``(n_sources, nT, 2)``
            when ``return_statistic=True``.
        """
        xp = self.backend.xp
        params = self._load_parameters(parameters)
        n_sources = params.shape[0]

        if segment_start_inds is None or segment_end_inds is None:
            segment_start_inds, segment_end_inds = (
                self._compute_segment_indices(params)
            )
        else:
            segment_start_inds = xp.asarray(segment_start_inds, dtype=np.int64)
            segment_end_inds = xp.asarray(segment_end_inds, dtype=np.int64)

        nT = int(segment_end_inds.max()) + 1
        nF = self.config["nF"]

        # Build placeholder arrays for arguments unused in the current mode.
        _dummy_response = xp.zeros((n_sources, 4), dtype=np.float64)
        _dummy_orbits = xp.zeros((nT, 3, 3), dtype=np.float64)
        _dummy_psds = xp.zeros((nT, nF, self.n_channels), dtype=np.float64)
        _dummy_stat = xp.zeros((1, 1, 2), dtype=np.complex128)

        _params_response = (
            xp.asarray(parameters_response, dtype=np.float64)
            if parameters_response is not None
            else _dummy_response
        )
        _orbits = (
            xp.asarray(spacecraft_orbits, dtype=np.float64)
            if spacecraft_orbits is not None
            else _dummy_orbits
        )
        _psds = (
            xp.asarray(psds, dtype=np.float64)
            if psds is not None
            else _dummy_psds
        )

        if return_statistic:
            if statistic is None:
                statistic = xp.zeros((n_sources, nT, 2), dtype=np.complex128)
            self.statistic_kernel(
                channels_or_data,
                segment_start_inds,
                segment_end_inds,
                params,
                _params_response,
                _orbits,
                statistic,
                _psds,
            )
            return statistic
        else:
            if channels_or_data is None:
                channels_or_data = xp.zeros(
                    (n_sources, nT, nF, self.n_channels), dtype=np.complex128
                )
            self.waveform_kernel(
                channels_or_data,
                segment_start_inds,
                segment_end_inds,
                params,
                _params_response,
                _orbits,
                _dummy_stat,
                _dummy_psds,
            )
            return channels_or_data
