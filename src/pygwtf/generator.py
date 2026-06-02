from typing import Type

import numpy as np

from .backend import Backend, get_backend
from .fresnel.kernel import analytic_kernel_constructor
from .fresnel.search_kernels import analytic_kernel_constructor_semi_coherent
from .models.base import AnalyticModel
from .response.orbits import get_analytic_ltts, get_analytic_orbits
from .response.transfer import get_AET_TFs

# Number of threads per block for GPU kernels, common choice. 
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
    tdi_type : {None, 2}, optional
        TDI generation order.  ``None`` produces TT polarisations.
        ``2`` produces second-generation TDI.
    channels : array_like, optional
        Pre-computed data array of shape ``(nT, nF, n_channels)
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
        self.config["nT"] = int(self.config["nT"])
        self.config["nF"] = int(self.config["nF"])
        self.config["dT"] = float(self.config["dT"])
        self.config["dF"] = float(self.config["dF"])
        assert np.all(
            [key in self.config.keys() for key in ["nT", "nF", "dT", "dF"]]
        ), "Config must contain 'nT', 'nF', 'dT', and 'dF' keys."

        self.config["dt"] = 1 / (self.config["nF"] * self.config["dF"])

        # Setting up time and frequency grid used by the kernels.
        self.t_tranche = (
            self.backend.xp.arange(self.config["nT"]) * self.config["dT"]
        )

        # Note: includes the f=0 DC bin. 
        self.f_tranche = (
            self.backend.xp.arange(self.config["nF"]) * self.config["dF"]
        )

        self.tdi_type = tdi_type

        assert self.tdi_type== 2 or self.tdi_type is None, "Only TDI-2 generation is currently supported. Set tdi_type to 2 or None."

        # If TDI type is not specified, do not compute TDI channels, instead go for the h_plus/h_cross polarisations.
        if self.tdi_type is None:
            channel_fn = self.model.get_TT_polarisations_function
            self.n_channels = 2

            # Fill in dummy orbits for non-TDI generation, as the kernel expects an array of this shape regardless.
            spacecraft_orbits = self.backend.xp.zeros(
                (self.config["nT"], 3, 3), dtype=np.float64
            )
        
        # TDI case
        else:
            # Only TDI-2 now.
            channel_fn = get_AET_TFs
            self.n_channels = 3
            if spacecraft_orbits is None:
                print(
                    "Spacecraft orbits not supplied. Falling back to analytic orbits"
                )
                spacecraft_orbits = self.backend.xp.asarray(
                    get_analytic_orbits(self.backend.asnumpy(self.t_tranche))
                )
            else:
                spacecraft_orbits = self.backend.xp.asarray(spacecraft_orbits)
                assert spacecraft_orbits.shape == (self.config["nT"], 3, 3), (
                    f"Spacecraft orbits array must have shape {(self.config['nT'], 3, 3)}"
                )

        self.spacecraft_orbits = spacecraft_orbits

        if spacecraft_ltts is None: # in *metres*
            print(
                "Spacecraft light travel times not supplied. Falling back to analytic calculation"
            )
            spacecraft_ltts = self.backend.xp.asarray(
                get_analytic_ltts(self.backend.asnumpy(self.spacecraft_orbits)) # computed in metres
            )
        else:
            spacecraft_ltts = self.backend.xp.asarray(spacecraft_ltts)# in metres
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

        # Construct waveform kernels, these are used to return the h_plys/h_cross or TDI channels for given parameters. 
        self.waveform_kernel_cpu, self.waveform_kernel_gpu = (
            analytic_kernel_constructor(  
                *constructor_args,
                compute_statistic=False,
                tdi_type=tdi_type,
            )
        )

        # Construct inner-product kernels, these are the kernels used for the statistic evaluation, and to build search statistics/likelhoods.
        # Note that the statistic kernel does not return the waveform and vice-versa. 
        self.statistic_kernel_cpu, self.statistic_kernel_gpu = (
            analytic_kernel_constructor(  
                *constructor_args,
                compute_statistic=True,
                tdi_type=tdi_type,
            )
        )

        # Construct direct semi-coherent statistic kernels that write one value per source.
        # 'Special' kernels used to do do the sobbh search. 
        (
            self.semi_coherent_statistic_kernel_cpu,
            self.semi_coherent_statistic_kernel_gpu,
        ) = analytic_kernel_constructor_semi_coherent(*constructor_args) 

        self._param_cache = None

        # Corresponds to the 'data' array against which inner-products are computed. Can be left as None if only waveform generation is desired.
        if channels is not None:
            assert channels.shape == (
                self.config["nT"],
                self.config["nF"],
                self.n_channels,
            ), (
                f"Channels array must have shape {(self.config['nT'], self.config['nF'], self.n_channels)}"
            )

        self.channels = channels

        # Corresponds to the PSD array used for inner-product/statistic computation. Can be left as None if only waveform generation is desired.
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
        """Call waveform kernel for the selected backend.

        NOTE: There are no explicit returns from this function as the kernels are filling in the pre-allocated output arrays. 
        
        Parameters
        ----------
        n_sources : int
            Number of sources to process, used to determine GPU grid size if applicable.
        *args : tuple
            Arguments to pass to the kernel, excluding n_sources which is passed separately for GPU grid sizing.
            See the kernel construction functions for details on the expected arguments.
        """
        if self.backend.uses_gpu:
            ## bpg: Blocks per grid
            ## Effectively ceil(n_sources / THREADS_PER_BLOCK) but written without needing to import math.ceil.
            # Ensures we have enough blocks (each with THREADS_PER_BLOCK threads) to cover all sources, even if n_sources is not a multiple of THREADS_PER_BLOCK.
            bpg = (n_sources + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
            self.waveform_kernel_gpu[bpg, THREADS_PER_BLOCK](*args)
        else:
            self.waveform_kernel_cpu(*args)

    def statistic_kernel(self, n_sources, *args):
        """Active statistic kernel for the selected backend.
           Only returns the per-segment d_h and h_h values.
        
        NOTE: There are no explicit returns from this function as the kernels are filling in the pre-allocated output arrays.

        Parameters
        ----------
        n_sources : int
            Number of sources to process, used to determine GPU grid size if applicable.
        *args : tuple
            Arguments to pass to the kernel, excluding n_sources which is passed separately for GPU grid sizing.
            See the kernel construction functions for details on the expected arguments.    
        """
        if self.backend.uses_gpu:
            bpg = (n_sources + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
            self.statistic_kernel_gpu[bpg, THREADS_PER_BLOCK](*args)
        else:
            self.statistic_kernel_cpu(*args)

    def semi_coherent_statistic_kernel(self, n_sources, *args):
        """Active direct semi-coherent statistic kernel for the selected backend.
           Returns the final semi-coherent statistic value per source, after summing over segments internally in the kernel.
           NOTE: Only really used for the SoBBH search 

        Parameters
        ----------
        n_sources : int
            Number of sources to process, used to determine GPU grid size if applicable.
        *args : tuple
            Arguments to pass to the kernel, excluding n_sources which is passed separately for GPU grid sizing
            See the kernel construction functions for details on the expected arguments.
        """
        if self.backend.uses_gpu:
            bpg = (n_sources + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
            self.semi_coherent_statistic_kernel_gpu[bpg, THREADS_PER_BLOCK](
                *args
            )
        else:
            self.semi_coherent_statistic_kernel_cpu(*args)

    def _load_parameters(
        self,
        parameters: dict | np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Copy parameters into the internal cache, realloc if source count changes.
        Internal cache is helpful as filling in an array is cheaper than reallocating array, specially on GPU. 

        Parameters
        ----------
        parameters : dict[str, array_like] or array_like, shape (n_sources, n_params)
            Physical parameters as a named dictionary or a pre-ordered 2-D array.

        Returns
        ----------
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

            try:
                dtype = parameters[self.model.parameters[0]].dtype
            except (KeyError, AttributeError):
                dtype = np.float64

            raw = xp.stack(
                [
                    xp.atleast_1d(xp.asarray(parameters[name], dtype=dtype))
                    for name in self.model.parameters
                ],
                axis=1,
            )
        else:
            if parameters.ndim == 1:
                single_source = True
            dtype = parameters.dtype
            raw = xp.atleast_2d(xp.asarray(parameters, dtype=dtype))

        n_sources = raw.shape[0]

        # Re-allocating parameter cache if n_sources changes, or if the dtyle changes (e.g. from array to dict)
        if (
            self._param_cache is None
            or self._param_cache.shape[0] != n_sources
            or self._param_cache.dtype != dtype
        ):
            self._param_cache = xp.zeros((n_sources, n_total), dtype=dtype)

        # Assign the physical parameters, then compute the derived parameters in-place in the cache.
        self._param_cache[:, :n_phys] = raw

        # Method of the model
        self.model.compute_derived_parameters(self._param_cache)

        return self._param_cache, single_source

    def _compute_segment_indices(self, parameters_cache):
        """
        Derive per-source time-segment indices from the model's time bounds.

        Used for computing the time-index range the source is in the LISA band for each source in nsources, 
        
        Parameters
        ----------
        parameters_cache : ndarray, shape (n_sources, n_params + n_derived)
            The parameters with derived parameters filled in.
        
        """
        xp = self.backend.xp
        nT = self.config["nT"]
        dT = self.config["dT"]
        dF = self.config["dF"]
        nF = self.config["nF"]
        frequency_band = (dF, nF * dF)

        # Fill in segment start and end indices based on the model's time bounds for the given parameters and frequency band. 
        #   If no bounds are returned, default to the full segment.
        #   Used to work out which time-segments the statistics/waveforms should actually be computed for. 
        bounds = self.model.get_time_bounds(parameters_cache, frequency_band)

        if bounds is None:
            segment_start_inds = xp.zeros(
                parameters_cache.shape[0], dtype=np.int32
            )
            segment_end_inds = xp.full(
                parameters_cache.shape[0], nT - 1, dtype=np.int32
            )

        else:
            t_start, t_end = bounds

            segment_start_inds = xp.floor(
                xp.asarray(t_start, dtype=np.float64) / dT
            ).astype(np.int32)

            segment_end_inds = xp.floor(
                xp.asarray(t_end, dtype=np.float64) / dT
            ).astype(np.int32)

            # Cliping begin and end segments to be within the valid range of [0, nT-1].
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
        search_statistic: np.ndarray | None = None,
        N_seg: int | None = None,
        mixed_precision: bool = False,
        use_midpoint: bool = True,
    ):
        """Evaluate the analytic waveform/inner-product statistic/search statistic.

        Parameters
        ----------
        parameters : dict or array_like, shape (n_sources, n_params)
            Source parameters as a named dict or a 2-D array in model order.
        channels : array_like or None
            Data array of shape ``(nT, nF, n_channels)``.
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
            NOTE: Not search statistic values but the per-segment d_h and h_h values. If ``True``, requires PSDs and channels to be supplied either at call time or initialisation.
        search_statistic : ndarray or None
            If not None, the output array to store the final semi-coherent search statistic (upsilon) after summing over segments. Must be of shape (n_sources,). Ignored if ``N_seg`` is None.
        N_seg : int, optional
            If not None, the number of segments to divide the data into for statistic computation.
            Ignored if ``compute_statistic=False``.
        mixed_precision : bool, optional
            If True, compute the statistic in mixed precision. The precision of the inputs is maintanied to compute
            waveform-level quantities (such as phase, frequency, amplitude), but single precision is used to construct
            the time-frequency representation and compute inner products. This can reduce memory usage and increase speed,
            particularly on GPU, typically with minimal impact on accuracy.
        use_midpoint : bool, optional
            Whether to evaluate the mode parameters at the midpoint of the segment, or at the beginning. (Default: True)
            (Should be set to True in almost all cases)
        Returns
        -------
        ndarray
            out ``(n_sources, nT, nF, n_channels)`` when
            ``return_statistic=False``, or statistic ``(n_sources, nT, 2)``
            when ``return_statistic=True``.
        """
        xp = self.backend.xp
        params, single_source = self._load_parameters(parameters)
        n_sources = params.shape[0]

        segment_start_inds, segment_end_inds = self._compute_segment_indices(
            params
        )
        out_dtype = np.complex64 if mixed_precision else np.complex128

        nT = self.config["nT"]
        nF = self.config["nF"]

    # Response parameters ``[cosi, pol, ecliptic_long, ecliptic_lat]``
        if parameters_response is None:
            
            if self.tdi_type is None:
                # These variables are still required for the evaluation, so fill them in as dummy empty arrays. 
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

        # Branch: Compute either d_h and h_h per segment, or the semi-coherent search statistics. 
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

            # Branch: Compute d_h and h_h per segment. 
            if N_seg is None:
                # If out is not supplied, allocate an array for the per-segment statistics (d_h and h_h) with shape (n_sources, nT, 2). 
                #   If out is supplied, it will be used to store the per-segment statistics, and should have shape (n_sources, nT, 2).
                if out is None:
                    # Allocate output array for statistic, shape (n_sources, nT, 2) for d_h, h_h.
                    # This array will be filled in by the kernel
                    out = xp.zeros((n_sources, nT, 2), dtype=out_dtype)
                else:
                    # If output array is supplied, zero it as a safety measure to ensure no uninitialised values are used.
                    out.fill(0.0)
                
                # Statistic kernel is responsible for computing waveforms->response->TDI->inner-products->statistics. 
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
                    mixed_precision,
                    use_midpoint,
                )
            # Branch: Compute the semi-coherent search-statistic.
            #   NOTE: This branch/option does not return d_h and h_h at a time-segment level
            #   It returns the direct summed semi-coherent statistic for the whole signal evalution for each source. 
            else:
                # If search_statistic output array is not supplied, allocate an array to store the semi-coherent search statistic (upsilon) with shape (n_sources,).
                if search_statistic is None:
                    if mixed_precision:
                        search_statistic = xp.zeros(
                            n_sources, dtype=np.float32
                        )
                    else:
                        search_statistic = xp.zeros(
                            n_sources, dtype=np.float64
                        )

                self.semi_coherent_statistic_kernel(
                    n_sources,
                    channels,
                    segment_start_inds,
                    segment_end_inds,
                    params,
                    parameters_response,
                    self.spacecraft_orbits,
                    self.spacecraft_ltts,
                    search_statistic,
                    psds,
                    N_seg,
                    mixed_precision,
                    use_midpoint,
                )
                # Output is search statistic in this case.
                out = search_statistic

        # Branch: Compute waveforms/TDI channels. No statistic computation.
        else:
            if out is None:
                out = xp.zeros(
                    (n_sources, nT, nF, self.n_channels), dtype=out_dtype
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
                xp.zeros(1, dtype=out_dtype),
                xp.zeros(1, dtype=np.float32)
                if mixed_precision
                else xp.zeros(1, dtype=np.float64),
                mixed_precision,
                use_midpoint,
            )

        return out[0] if single_source else out
