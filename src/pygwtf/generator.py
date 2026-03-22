from typing import Type

import numpy as np

from .backend import Backend, get_backend
from .fresnel.kernel import (
    analytic_kernel_constructor,
    semi_coherent_statistic_sum_cpu_wrap,
    semi_coherent_statistic_sum_gpu_wrap,
)
from .models.base import AnalyticModel
from .response.orbits import get_analytic_ltts, get_analytic_orbits
from .response.transfer import get_AET_TFs

THREADS_PER_BLOCK = 128


class AnalyticTimeFrequencyWaveform:
    """
    Analytic time-frequency waveform generator.

    Constructs generative kernels of the chosen prescription (only Fresnel currently supported)
    for both waveform generation and inner-product evaluation in the time-frequency domain,
    driven by a physical model. CPU or GPU operation is specified by the chosen backend.

    Parameters
    ----------
    model_class:
        The model class to instantiate (e.g. ``TaylorT2Ecc``).  Initialised
        inside this constructor with the chosen backend.
    config : dict
        Kernel configuration, with kwargs specific to the prescription used.
    prescription : str, optional
        The prescription to use for the kernel construction. Only ``'fresnel'`` is
        currently supported.
    backend : str or Backend, optional
        Compute backend — ``'cpu'`` (default) or ``'gpu'``.
    tdi_type : {None, 1, 2}, optional
        TDI generation order.  ``None`` produces TT polarisations.
        ``1`` produces first-generation TDI, and ``2`` produces second-generation TDI.
    channels : array_like, optional
        Pre-computed channels array of shape ``(nT, nF, n_channels)
        ``. Required for statistic evaluation if not supplied at call time.
    psds : array_like, optional
        Pre-computed PSD array of shape ``(nT, nF, n_channels)
        ``. Required for statistic evaluation if not supplied at call time.
    spacecraft_orbits : array_like, optional
        Pre-computed spacecraft orbits array of shape ``(nT, 3,
        3)``. Required for TDI generation if not supplied at initialisation, in which case analytic orbits will be used.
    spacecraft_ltts: array_like, optional
        Pre-computed spacecraft light travel times array of shape ``(nT, 3)``.
        Required for TDI generation if not supplied at initialisation, in which case will be calculated analytically from positions
        (Need in metre units not seconds)
    """

    def __init__(
        self,
        model_class: Type[AnalyticModel],
        config: dict,
        prescription: str = "fresnel",
        backend: str | Backend = "cpu",
        tdi_type: int | None = None,
        channels: np.ndarray | None = None,
        psds: np.ndarray | None = None,
        spacecraft_orbits: np.ndarray | None = None,
        spacecraft_ltts: np.ndarray | None = None,
    ):
        self.backend = (
            get_backend(backend) if isinstance(backend, str) else backend
        )
        self.model = model_class(backend=self.backend)
        self.config = config

        assert np.all(
            [key in self.config.keys() for key in ["nT", "nF", "dT", "dF"]]
        ), "Config must contain 'nT', 'nF', 'dT', and 'dF' keys."

        self.config["dt"] = 1 / (self.config["nF"] * self.config["dF"])

        # Setting up time and frequency grid used by the kernels.
        self.t_tranche = (
            self.backend.xp.arange(self.config["nT"]) * self.config["dT"]
        )

        self.f_tranche = (
            self.backend.xp.arange(self.config["nF"]) * self.config["dF"]
        )

        self.tdi_type = tdi_type

        # If TDI type is not specified, do not compute TDI channels, instead go for the h_plus/h_cross polarisations.
        if self.tdi_type is None:
            channel_fn = self.model.get_TT_polarisations_function
            self.n_channels = 2

            # Fill in dummy orbits for non-TDI generation, as the kernel expects an array of this shape regardless.
            spacecraft_orbits = self.backend.xp.zeros(
                (self.config["nT"], 3, 3), dtype=np.float64
            )
        else:
            # Only TDI-2 now.
            channel_fn = get_AET_TFs
            self.n_channels = 3
            if spacecraft_orbits is None:
                print(
                    "Spacecraft orbits not supplied. Falling back to analytic orbits"
                )
                spacecraft_orbits = get_analytic_orbits(self.t_tranche)
            else:
                assert spacecraft_orbits.shape == (self.config["nT"], 3, 3), (
                    f"Spacecraft orbits array must have shape {(self.config['nT'], 3, 3)}"
                )

        self.spacecraft_orbits = spacecraft_orbits

        if spacecraft_ltts is None:
            print(
                "Spacecraft light travel times not supplied. Falling back to analytic calculation"
            )
            spacecraft_ltts = get_analytic_ltts(self.spacecraft_orbits)
        else:
            assert spacecraft_ltts.shape == (self.config["nT"], 3), (
                f"Spacecraft light travel times array must have shape {(self.config['nT'], 3)}"
            )

        self.spacecraft_ltts = spacecraft_ltts

        kernel_config = {
            **self.config,
            "nparams": self.model.num_parameters
            + self.model.num_derived_parameters,
        }

        constructor_args = (
            kernel_config,
            self.model.amplitude_function,
            self.model.phi_f_fdot_function,
            channel_fn,
        )

        # Construct waveform kernels, these are mainly used for debugging.
        self.waveform_kernel_cpu, self.waveform_kernel_gpu = (
            analytic_kernel_constructor(  # type: ignore
                *constructor_args,
                compute_statistic=False,
                tdi_type=tdi_type,
            )
        )

        # Construct inner-product kernels, these are the kernels used for the statistic evaluation, and to build search statistics/likelhoods.
        self.statistic_kernel_cpu, self.statistic_kernel_gpu = (
            analytic_kernel_constructor(  # type: ignore
                *constructor_args,
                compute_statistic=True,
                tdi_type=tdi_type,
            )
        )

        self._param_cache = None

        if channels is not None:
            assert channels.shape == (
                self.config["nT"],
                self.config["nF"],
                self.n_channels,
            ), (
                f"Channels array must have shape {(self.config['nT'], self.config['nF'], self.n_channels)}"
            )

        self.channels = channels

        if psds is not None:
            assert psds.shape == (
                self.config["nT"],
                self.config["nF"],
                self.n_channels,
            ), (
                f"PSDs array must have shape {(self.config['nT'], self.config['nF'], self.n_channels)}"
            )

        self.psds = psds

    def waveform_kernel(self, n_sources, *args):
        # TODO: this feels clunky. Maybe we need a thin Python class for each prescription that handles kernel dispatch?
        """Call waveform kernel for the selected backend."""
        if self.backend.uses_gpu:
            bpg = n_sources + (THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            self.waveform_kernel_gpu[bpg, THREADS_PER_BLOCK](*args)
        else:
            self.waveform_kernel_cpu(*args)

    def statistic_kernel(self, n_sources, *args):
        """Active statistic kernel for the selected backend."""
        if self.backend.uses_gpu:
            bpg = n_sources + (THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            self.statistic_kernel_gpu[bpg, THREADS_PER_BLOCK](*args)
        else:
            self.statistic_kernel_cpu(*args)

    def semi_coherent_statistic_sum_kernel(self, n_sources, *args):
        """Active semi-coherent statistic sum kernel for the selected backend."""
        if self.backend.uses_gpu:
            bpg = n_sources + (THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            semi_coherent_statistic_sum_gpu_wrap[bpg, THREADS_PER_BLOCK](*args)
        else:
            semi_coherent_statistic_sum_cpu_wrap(*args)

    def _load_parameters(
        self, parameters: dict | np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """Copy parameters into the internal cache, realloc if source count changes.

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

        single_source = False
        if isinstance(parameters, dict):
            try:
                if len(parameters[self.model.parameters[0]]) == 0:
                    single_source = True
            except TypeError:
                single_source = True

            raw = xp.stack(
                [
                    xp.atleast_1d(
                        xp.asarray(parameters[name], dtype=np.float64)
                    )
                    for name in self.model.parameters
                ],
                axis=1,
            )
        else:
            if parameters.ndim == 1:
                single_source = True
            raw = xp.atleast_2d(xp.asarray(parameters, dtype=np.float64))

        n_sources = raw.shape[0]

        if (
            self._param_cache is None
            or self._param_cache.shape[0] != n_sources
        ):
            self._param_cache = xp.zeros(
                (n_sources, n_total), dtype=np.float64
            )

        # Assign the physical parameters, then compute the derived parameters in-place in the cache.
        self._param_cache[:, :n_phys] = raw
        self.model.compute_derived_parameters(self._param_cache)

        return self._param_cache, single_source

    def _compute_segment_indices(self, parameters_cache):
        """Derive per-source time-segment indices from the model's time bounds."""
        xp = self.backend.xp
        nT = self.config["nT"]
        dT = self.config["dT"]
        dF = self.config["dF"]
        nF = self.config["nF"]
        frequency_band = (dF, nF * dF)

        # Fill in segment start and end indices based on the model's time bounds for the given parameters and frequency band. If no bounds are returned, default to the full segment.
        bounds = self.model.get_time_bounds(parameters_cache, frequency_band)

        if bounds is None:
            segment_start_inds = xp.zeros(
                parameters_cache.shape[0], dtype=np.int64
            )
            segment_end_inds = xp.full(
                parameters_cache.shape[0], nT - 1, dtype=np.int64
            )

        else:
            t_start, t_end = bounds

            segment_start_inds = xp.floor(
                xp.asarray(t_start, dtype=np.float64) / dT
            ).astype(np.int64)

            segment_end_inds = xp.floor(
                xp.asarray(t_end, dtype=np.float64) / dT
            ).astype(np.int64)

            segment_start_inds = xp.clip(segment_start_inds, 0, nT - 1)

            segment_end_inds = xp.clip(segment_end_inds, 0, nT - 1)

        return segment_start_inds, segment_end_inds

    def __call__(
        self,
        parameters: dict | np.ndarray,
        channels: np.ndarray | None = None,
        psds: np.ndarray | None = None,
        parameters_response: np.ndarray | None = None,
        out: np.ndarray | None = None,
        compute_statistic: bool = False,
        N_seg: int | None = None,
    ):
        """Evaluate the analytic waveform or inner-product statistic.

        Parameters
        ----------
        parameters : dict or array_like, shape (n_sources, n_params)
            Source parameters as a named dict or a 2-D array in model order.
        channels : array_like or None
            Data array of shape ``(nT, nF, n_channels)``. If not supplied
            will use the stored channels from initialisation.
        psds : array_like, optional
            Power spectral densities, shape ``(nT, nF, n_channels)``.
            Required for statistic output. If not supplied
            will use the stored psds from initialisation.
        parameters_response : array_like, shape (n_sources, 4), optional
            Response parameters ``[cosi, pol, ecliptic_long, ecliptic_lat]``.
            Required when using TDI.
        out : array_like or None
            *Waveform output*: pre-allocated output of shape
            ``(n_sources, nT, nF, n_channels)``; auto-allocated if ``None``.
            *Statistic output*: pre-allocated output of shape ``(n_sources, nT, 2)``.
            Auto-allocated if ``None``.
        compute_statistic : bool, optional
            Return the inner-product statistics instead of the channels array.
        N_seg : int, optional
            If not None, the number of segments to divide the data into for statistic computation.
            Ignored if ``compute_statistic=False``.

        Returns
        -------
        ndarray
            Channels ``(n_sources, nT, nF, n_channels)`` when
            ``return_statistic=False``, or statistic ``(n_sources, nT, 2)``
            when ``return_statistic=True``.
        """
        xp = self.backend.xp
        params, single_source = self._load_parameters(parameters)
        n_sources = params.shape[0]

        segment_start_inds, segment_end_inds = self._compute_segment_indices(
            params
        )

        nT = self.config["nT"]
        nF = self.config["nF"]

        if parameters_response is None:
            if self.tdi_type is None:
                parameters_response = xp.zeros(
                    (n_sources, 4), dtype=np.float64
                )
            else:
                raise ValueError(
                    "parameters_response must be supplied for TDI generation."
                )
        else:
            assert parameters_response.shape == (n_sources, 4), (
                f"parameters_response must have shape {(n_sources, 4)}"
            )

        # Compute likelihood/detection statistics
        if compute_statistic:
            if channels is None:
                channels = self.channels
                assert channels is not None, (
                    "Channels must be supplied to compute the statistic."
                )
            if psds is None:
                psds = self.psds
                assert psds is not None, (
                    "PSDs must be supplied to compute the statistic."
                )
            if parameters_response is None:
                parameters_response = xp.zeros(
                    (n_sources, 4), dtype=np.float64
                )
            if out is None:
                # Allocate output array for statistic, shape (n_sources, nT, 2) for d_h, h_h.
                out = xp.zeros((n_sources, nT, 2), dtype=np.complex128)

            self.statistic_kernel(
                n_sources,
                channels,
                segment_start_inds,
                segment_end_inds,
                params,
                parameters_response,
                self.spacecraft_orbits,
                self.spacecraft_ltts,
                out,
                psds,
            )

            if N_seg is not None:
                self.inner_product_array = (
                    out  # cache inner product array for later use
                )

                search_statistic = xp.zeros(n_sources, dtype=np.float64)

                self.semi_coherent_statistic_sum_kernel(
                    n_sources,
                    out,  # contains the per-segment statistics (d|h and h|h) for each source.
                    N_seg,
                    segment_end_inds,
                    segment_start_inds,
                    search_statistic,  # contains <d|h> and <h|h> per source per tranche.
                )
                # output the semi-coherent statistic instead of the per-segment statistics.
                out = search_statistic

        # Compute waveforms
        else:
            if out is None:
                out = xp.zeros(
                    (n_sources, nT, nF, self.n_channels), dtype=np.complex128
                )
            self.waveform_kernel(
                n_sources,
                out,
                segment_start_inds,
                segment_end_inds,
                params,
                parameters_response,
                self.spacecraft_orbits,
                self.spacecraft_ltts,
                xp.zeros(1, dtype=np.complex128),
                xp.zeros(1, dtype=np.float64),
            )

        return out[0] if single_source else out
