from math import sqrt
from typing import Callable

import numpy as np
from numba import cuda, jit

from ..response.transfer import fill_k, fill_P_lm, fill_P_lm_given_Y_lm
from ..utils import complex_inner_product
from .common import _fresnel_kernel


def analytic_kernel_constructor(
    config: dict,
    _get_amplitude: Callable,
    _get_phi_f_fdot: Callable,
    _get_channels: Callable,
    compute_statistic: bool = False,
    tdi_type: int | None = None,
    block_vectorised_gpu: bool = False,
):
    """
    Constructor method used to generate a kernel function for computing Fresnel waveforms and derived inner-product statistics (d_h and h_h).

    Parameters:
    ----------
    config: dict
        Configuration dictionary containing the following keys:
        - dT: float, the duration of each time segment.
        - nF: int, the number of frequency bins.
        - dF: float, the width of each frequency bin.
        - kernel_width: int, the number of frequency bins on either side of the central frequency to include in the kernel computation.
        - nparams: int, the number of parameters describing each source. This is used to allocate local arrays in the GPU kernel.

    Following are waveform specific functions, and can be substituted for any waveform written in the same format.
    _get_amplitude: Callable
        Function that takes in time, frequency, frequency derivative, and source parameters, and returns the amplitude of h_22 at that time.
    _get_phi_f_fdot: Callable
        Function that takes in time and source parameters, and returns the phase, frequency, and frequency derivative of h_22 at that time.

    Generalised "Response function/Channel construction" method.
    Can be either a frequency domain response function, or a function within the waveform itself that returns the waveform polarizations.
    Frequency domain response function can in principle be substituted for any response function in the frequency domain (within time-segment)
    _get_channels: Callable
        Function that takes in frequency-domain (within each time-segment) waveforms, in the format h_lm and
            - either transforms to TDI channels using the detector response.
            - or converts to waveform polarizations h_+ and h_x in the frequency domain and returns that.

    kernel_cpu: Callable
        A CPU kernel function that can be called to compute Fresnel waveforms and statistics.
    kernel_gpu: Callable
        A GPU kernel function that can be called to compute Fresnel waveforms and statistics.

    Returns:
    -------
    kernel_cpu: Callable
        A CPU kernel function that can be called to compute Fresnel waveforms and statistics.
    kernel_gpu: Callable
        A GPU kernel function that can be called to compute Fresnel waveforms and statistics.

    """
    dT = config["dT"]
    nF = config["nF"]
    dF = config["dF"]
    kernel_width = config["kernel_width"]
    nparams = config["nparams"]

    if block_vectorised_gpu and kernel_width != 16:
        raise ValueError(
            f"Block vectorised GPU kernel requires kernel_width=16 for now, but got kernel_width={kernel_width}."
        )

    # Only TDI-2 supported for now.
    if tdi_type is None:
        tdi = False
        tdi2 = False
    else:
        tdi = True
        if tdi_type == 1:
            tdi2 = False
        elif tdi_type == 2:
            tdi2 = True
        else:
            raise ValueError(
                f"Invalid tdi_type: {tdi_type}. Must be 1, 2, or None."
            )

    @jit
    def kernel_inner(
        src_num,
        channels,
        segment_start_inds,
        segment_end_inds,
        parameters,
        params_source,
        parameters_response,
        params_source_response,
        P_lm,
        k,
        n,
        p,
        Ls,
        spacecraft_ltts,
        spacecraft_orbits,
        statistic,
        inv_psds,
        mixed_precision,
        use_midpoint,
    ):
        """
        NOTE: HARDWARE AGNOSTIC

        Kernel to compute to compute the fresnel waveforms or statistics (d_h and h_h) for a single source (indexed by src_num) across all time-segments.
        This kernel is called by both the CPU and GPU kernels, with appropriate local array allocations for each hardware type.

        Kernel output is the filled-in statistic array (d_h and h_h for each time-segment) if compute_statistic is True, or the computed waveforms if compute_statistic is False.

        Parameters:
        ----------
        src_num: int
            The index of the source for which to compute the waveforms/statistics.
        channels: array (n_sources, nT, nF, n_channels)
            Either the array to be filled in with the computed waveforms for each time-segment and frequency bin (if compute_statistic is False)
            Or the data array to be used to compute statistics (d_h and h_h) if compute_statistic is True.
        segment_start_inds: array (n_sources,)
            The starting time-segment index for each source. Used to determine which time-segments to compute over for the given source.
        segment_end_inds: array (n_sources,)
            The ending time-segment index for each source. Used to determine which time-segments to compute over for the given source.
        parameters: array (n_sources, nparams)
            The parameters describing each source. This is used to compute the waveform for the given source.
        parameters_response: array (n_sources, 4) or None
            The parameters describing the response for each source, used if tdi is True. This is used to compute the TDI response for the given source.
        P_lm: array (3, 3) (array to be filled in within the kernel)
            Local array to store the response function coefficients for TDI computation.
        k: array (3,) (array to be filled in within the kernel)
            Local array to store the wavevector for TDI computation.
        n: array (3 (nSpacecraft), 3 (nDimensionsPerSpacecraft)) (array to be filled in within the transfer function call within the kernel)
            Unit vectors pointing between the spacecraft, used for TDI response computation.
            Setup in the convention of n[0] (2->3), n[1] (1->3), n[2] (1->2).
        p: array (3, 3) (array to be filled in within the kernel)
            Position vectors for each spacecraft at each time-segment, used for TDI response computation.
        Ls: array (3) (array to be filled in within the kernel)
            Arm lengths for each spacecraft at each time-segment, used for TDI response computation.
        spacecraft_ltts: array (nT, 3) or None
            Light travel times for each spacecraft, used for TDI response computation if tdi is True
            Used to fill in the Ls array within the kernel for TDI response computation.
            NOTE: Precomputed, not computed in the kernel at runtime.
        spacecraft_orbits: array (nT, 3, 3) or None
            Spacecraft orbits, used for TDI response computation if tdi is True.
            Used to fill in the p array within the kernel for TDI response computation.
            NOTE: Precomputed, not computed in the kernel at runtime.
        statistic: array (n_sources, nT, 2) (array to be filled in within the kernel if compute_statistic is True)
            Local array to store the computed statistics (d_h and h_h) for each time-segment for the given source.
        inv_psds: array (nT, nF, n_channels) or None
            The reciprocal (1/psd) of the noise PSD, prefolded outside the kernel so the inner-product loop multiplies instead of dividing.
        mixed_precision: bool
            Whether to use mixed precision (float32) for the computations within the kernel, to save memory and speed up computations.
        use_midpoint : bool
            Whether to evaluate the mode parameters at the midpoint of the segment, or at the beginning.
            (Should be set to True in almost all circumstances)
        """
        # Grab parameters for specified source.
        for i in range(nparams):
            params_source[i] = parameters[src_num, i]

        # Mixed precision operations to save memory and speed up if desired.
        if mixed_precision:
            dT_prec = np.float32(dT)
            dF_prec = np.float32(dF)
        else:
            dT_prec = dT
            dF_prec = dF

        if tdi:
            # Grab response parameters for specified source if TDI response computation is needed.
            for i in range(4):
                params_source_response[i] = parameters_response[src_num, i]

            # Fill in P_lm tensor and wavevector k for TDI response computation for this source.
            fill_P_lm(P_lm, params_source_response)
            fill_k(k, params_source_response)

        for t_idx in range(
            segment_start_inds[src_num], segment_end_inds[src_num] + 1
        ):
            t_tranche = dT * t_idx
            # If use_midpoint is True, evaluate the mode parameters at the midpoint of the segment, rather than the beginning.

            if use_midpoint:
                t_tranche += dT / 2

            phi0_mode, f0_mode, fdot_mode = _get_phi_f_fdot(
                t_tranche, params_source
            )
            amp_mode = _get_amplitude(
                t_tranche, f0_mode, fdot_mode, params_source
            )

            # Start index in frequency bins for the given mode frequency, used to determine which frequency bins to compute over in the kernel.
            start_ind = int(f0_mode / dF)
            d_h = 0.0 + 0.0j
            h_h = 0.0 + 0.0j

            if tdi:
                # Fill in LTTs for this time step.
                for i in range(3):
                    Ls[i] = spacecraft_ltts[t_idx, i]

                # Fill in position vectors for this time step
                for i in range(3):
                    for j in range(3):
                        p[i, j] = spacecraft_orbits[t_idx, i, j]
                transfer_functions = _get_channels(
                    f0_mode, P_lm, k, p, Ls, n, tdi2
                )

            # Cheap way to check how many extra frequency bins to compute over for the given fdot
            extra_fdot_bins = int((fdot_mode * dT) / dF)

            if mixed_precision:
                amp_mode = np.float32(amp_mode)
                phi0_mode = np.float32(phi0_mode % (2 * np.pi))
                f0_mode = np.float32(f0_mode)
                fdot_mode = np.float32(fdot_mode)

                if tdi:
                    transfer_functions = (
                        np.complex64(transfer_functions[0]),
                        np.complex64(transfer_functions[1]),
                        np.complex64(transfer_functions[2]),
                    )

            # Fresnel quantities (common to all frequency bins for a given source at a given time-segment)
            # NOTE: math.sqrt, not np.sqrt -- numpy scalar ufuncs are not supported inside CUDA kernels by this numba version.
            sqrt2fdot = sqrt(2 * fdot_mode)
            amp_mode_prefac = amp_mode / sqrt2fdot
            one_over_fdot = 1 / fdot_mode

            # For each frequeny bin within the time-segment, compute the fresnel.
            for f_rel_idx in range(
                -kernel_width,
                kernel_width + extra_fdot_bins,  # + 1
            ):
                f_idx = start_ind + f_rel_idx
                if f_idx > 0 and f_idx < nF:
                    f_bin = (f_idx + 1) * (dF_prec)
                    h_f_pos = _fresnel_kernel(
                        f_bin,
                        amp_mode_prefac,
                        sqrt2fdot,
                        phi0_mode,
                        f0_mode,
                        fdot_mode,
                        one_over_fdot,
                        dT_prec,
                        use_midpoint,
                    )
                    # Generate just the polarizations
                    if not tdi:
                        h_TT = _get_channels(h_f_pos, params_source)

                    # Generate AET channels via multiplicative, frequency-domain transfer function.
                    for i in range(channels.shape[-1]):
                        if tdi:
                            h = h_f_pos * transfer_functions[i]
                        else:
                            h = h_TT[i]

                        # If compute_statistic is True, compute the inner products for d_h and h_h using the data in channels and the computed waveform h, and the inverse psd in inv_psds.
                        if compute_statistic:
                            d = channels[t_idx, f_idx, i]
                            inv_psd = inv_psds[t_idx, f_idx, i]
                            d_h += complex_inner_product(d, h, inv_psd)
                            h_h += complex_inner_product(h, h, inv_psd)
                        else:
                            channels[src_num, t_idx, f_idx, i] = h

            if compute_statistic:
                statistic[src_num, t_idx, 0] = d_h
                statistic[src_num, t_idx, 1] = h_h

    @cuda.jit
    def kernel_gpu(
        channels,
        segment_start_inds,
        segment_end_inds,
        parameters,
        parameters_response,
        spacecraft_orbits,
        spacecraft_ltts,
        statistic,
        inv_psds,
        mixed_precision,
        use_midpoint,
    ):
        """
        Wrapper for the Fresnel_kernel to launch on GPU.
        Each thread computes the Fresnel waveforms and statistics for a single source across all time-segments.
        This function instantiates the local arrays needed for the kernel_inner function, and calls kernel_inner for each source.

        Parameters:
        ----------
        channels: array (n_sources, nT, nF, n_channels)
            Either the array to be filled in with the computed waveforms for each time-segment and frequency bin (if compute_statistic is False)
            Or the data array to be used to compute statistics (d_h and h_h) if compute_statistic is True.
        segment_start_inds: array (n_sources,)
            The starting time-segment index for each source. Used to determine which time-segments to compute over for the given source.
        segment_end_inds: array (n_sources,)
            The ending time-segment index for each source. Used to determine which time-segments to compute over for the given source.
        parameters: array (n_sources, nparams)
            The parameters describing each source. This is used to compute the waveform for the given source.
        parameters_response: array (n_sources, 4) or None
            The parameters describing the response for each source, used if tdi is True. This is used to compute the TDI response for the given source.
        spacecraft_orbits: array (nT, 3, 3) or None
            Spacecraft orbits, used for TDI response computation if tdi is True.
            Used to fill in the p array within the kernel for TDI response computation.
            NOTE: Precomputed, not computed in the kernel at runtime.
        spacecraft_ltts: array (nT, 3) or None
            Light travel times for each spacecraft, used for TDI response computation if tdi is True
            Used to fill in the Ls array within the kernel for TDI response computation.
        statistic: array (n_sources, nT, 2) (array to be filled in within the kernel if compute_statistic is True)
            Local array to store the computed statistics (d_h and h_h for each time-segment for the given source.
        inv_psds: array (nT, nF, n_channels) or None
            The reciprocal (1/psd) of the noise PSD, prefolded outside the kernel so the inner-product loop multiplies instead of dividing. Used to compute the inner products for the statistics if compute_statistic is True.
        mixed_precision: bool
            Whether to use mixed precision (float32) for the computations within the kernel, to save memory and speed up computations.
        use_midpoint : bool
            Whether to evaluate the mode parameters at the midpoint of the segment, or at the beginning.
            (Should be set to True in almost all circumstances)
        """
        # one source per thread
        src_num = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        # one source per thread
        if src_num < parameters.shape[0]:
            # These are all the local arrays that will be filled in within the kernel_inner function.
            params_source = cuda.local.array(nparams, dtype=parameters.dtype)
            P_lm = cuda.local.array((3, 3), dtype=np.complex128)
            k = cuda.local.array((3,), dtype=parameters_response.dtype)
            n = cuda.local.array((3, 3), dtype=parameters_response.dtype)
            p = cuda.local.array((3, 3), dtype=spacecraft_orbits.dtype)
            Ls = cuda.local.array((3,), dtype=spacecraft_ltts.dtype)
            params_source_response = cuda.local.array(
                4, dtype=parameters_response.dtype
            )
            kernel_inner(
                src_num,
                channels,
                segment_start_inds,
                segment_end_inds,
                parameters,
                params_source,
                parameters_response,
                params_source_response,
                P_lm,
                k,
                n,
                p,
                Ls,
                spacecraft_ltts,
                spacecraft_orbits,
                statistic,
                inv_psds,
                mixed_precision,
                use_midpoint,
            )

    @jit
    def kernel_cpu(
        channels,
        segment_start_inds,
        segment_end_inds,
        parameters,
        parameters_response,
        spacecraft_orbits,
        spacecraft_ltts,
        statistic,
        inv_psds,
        mixed_precision,
        use_midpoint,
    ):
        """
        Wrapper for the Fresnel_kernel to launch on CPU.
        NOTE: No signficant parallelisation for CPU version, just a loop over sources.

        Parameters:
        ----------
        channels: array (n_sources, nT, nF, n_channels)
            Either the array to be filled in with the computed waveforms for each time-segment and frequency bin (if compute_statistic is False)
            Or the data array to be used to compute statistics (d_h and h_h) if compute_statistic is True.
        segment_start_inds: array (n_sources,)
            The starting time-segment index for each source. Used to determine which time-segments to compute over for the given source.
        segment_end_inds: array (n_sources,)
            The ending time-segment index for each source. Used to determine which time-segments to compute over for the given source.
        parameters: array (n_sources, nparams)
            The parameters describing each source. This is used to compute the waveform for the given source.
        parameters_response: array (n_sources, 4) or None
            The parameters describing the response for each source, used if tdi is True. This is used to compute the TDI response for the given source.
        spacecraft_orbits: array (nT, 3, 3) or None
            Spacecraft orbits, used for TDI response computation if tdi is True.
            Used to fill in the p array within the kernel for TDI response computation.
            NOTE: Precomputed, not computed in the kernel at runtime.
        spacecraft_ltts: array (nT, 3) or None
            Light travel times for each spacecraft, used for TDI response computation if tdi is True
            Used to fill in the Ls array within the kernel for TDI response computation.
        statistic: array (n_sources, nT, 2) (array to be filled in within the kernel if compute_statistic is True)
            Local array to store the computed statistics (d_h and h_h for each time-segment for the given source.
        inv_psds: array (nT, nF, n_channels) or None
            The reciprocal (1/psd) of the noise PSD. Used to compute the inner products for the statistics if compute_statistic is True.
        mixed_precision: bool
            Whether to use mixed precision (float32) for the computations within the kernel, to save memory and speed up computations.
        use_midpoint : bool
            Whether to evaluate the mode parameters at the midpoint of the segment, or at the beginning.
            (Should be set to True in almost all circumstances)
        """

        for src_num in range(parameters.shape[0]):
            params_source = np.zeros(nparams, dtype=parameters.dtype)
            P_lm = np.zeros((3, 3), dtype=np.complex128)
            k = np.zeros((3,), dtype=parameters_response.dtype)
            n = np.zeros((3, 3), dtype=parameters_response.dtype)
            p = np.zeros((3, 3), dtype=spacecraft_orbits.dtype)
            Ls = np.zeros((3,), dtype=spacecraft_ltts.dtype)
            params_source_response = np.zeros(
                4, dtype=parameters_response.dtype
            )
            kernel_inner(
                src_num,
                channels,
                segment_start_inds,
                segment_end_inds,
                parameters,
                params_source,
                parameters_response,
                params_source_response,
                P_lm,
                k,
                n,
                p,
                Ls,
                spacecraft_ltts,
                spacecraft_orbits,
                statistic,
                inv_psds,
                mixed_precision,
                use_midpoint,
            )

    # Block-vectorised version for smaller batch sizes
    @cuda.jit
    def gpu_kernel_blocked(
        channels,
        segment_start_inds,
        segment_end_inds,
        parameters,
        parameters_response,
        spacecraft_orbits,
        spacecraft_ltts,
        statistic,
        inv_psds,
        mixed_precision,
        use_midpoint,
    ):
        src_num = cuda.blockIdx.x  # one source per block
        t_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        f_idx = cuda.threadIdx.x
        if src_num < parameters.shape[0]:
            if (
                t_idx >= segment_start_inds[src_num]
                and t_idx <= segment_end_inds[src_num]
            ):
                params_source = cuda.shared.array(nparams, dtype=np.float64)
                P_lm = cuda.local.array((3, 3), dtype=np.complex128)
                k = cuda.local.array((3,), dtype=parameters_response.dtype)
                n = cuda.local.array((3, 3), dtype=parameters_response.dtype)
                p = cuda.local.array((3, 3), dtype=spacecraft_orbits.dtype)
                Ls = cuda.local.array((3,), dtype=spacecraft_ltts.dtype)
                params_source_response = cuda.local.array(
                    4, dtype=parameters_response.dtype
                )

                # multi-thread load of parameters
                if f_idx < nparams and cuda.threadIdx.y == 0:
                    params_source[f_idx] = parameters[src_num, f_idx]

                cuda.syncthreads()

                if tdi:
                    # Grab response parameters for specified source if TDI response computation is needed.
                    for i in range(4):
                        params_source_response[i] = parameters_response[
                            src_num, i
                        ]

                    # Fill in P_lm tensor and wavevector k for TDI response computation for this source.
                    fill_P_lm(P_lm, params_source_response)
                    fill_k(k, params_source_response)

                t_tranche = dT * t_idx
                # If use_midpoint is True, evaluate the mode parameters at the midpoint of the segment, rather than the beginning.

                if use_midpoint:
                    t_tranche += dT / 2

                phi0_mode, f0_mode, fdot_mode = _get_phi_f_fdot(
                    t_tranche, params_source
                )
                amp_mode = _get_amplitude(
                    t_tranche, f0_mode, fdot_mode, params_source
                )

                freq_ind = int(f0_mode / dF) + f_idx - kernel_width
                d_h_here = 0.0 + 0.0j
                h_h_here = 0.0 + 0.0j

                if tdi:
                    # Fill in LTTs for this time step.
                    for i in range(3):
                        Ls[i] = spacecraft_ltts[t_idx, i]

                    # Fill in position vectors for this time step
                    for i in range(3):
                        for j in range(3):
                            p[i, j] = spacecraft_orbits[t_idx, i, j]
                    transfer_functions = _get_channels(
                        f0_mode, P_lm, k, p, Ls, n, tdi2
                    )

                sqrt2fdot = sqrt(2 * fdot_mode)
                amp_mode_prefac = amp_mode / sqrt2fdot
                one_over_fdot = 1 / fdot_mode

                # NOTE: no extra fdot bins here due to block form
                if freq_ind > 0 and freq_ind < nF:
                    f_bin = (freq_ind + 1) * (dF)
                    h_f_pos = _fresnel_kernel(
                        f_bin,
                        amp_mode_prefac,
                        sqrt2fdot,
                        phi0_mode,
                        f0_mode,
                        fdot_mode,
                        one_over_fdot,
                        dT,
                        use_midpoint,
                    )

                    if not tdi:
                        h_TT = _get_channels(h_f_pos, params_source)

                    # Generate AET channels via multiplicative, frequency-domain transfer function.
                    for i in range(channels.shape[-1]):
                        if tdi:
                            h = h_f_pos * transfer_functions[i]
                        else:
                            h = h_TT[i]

                        # If compute_statistic is True, compute the inner products for d_h and h_h using the data in channels and the computed waveform h, and the inverse psd in inv_psds.
                        if compute_statistic:
                            d = channels[t_idx, freq_ind, i]
                            inv_psd = inv_psds[t_idx, freq_ind, i]
                            d_h_here += complex_inner_product(d, h, inv_psd)
                            h_h_here += complex_inner_product(h, h, inv_psd)
                        else:
                            channels[src_num, t_idx, freq_ind, i] = h

                if compute_statistic:
                    # TODO: not restrict to 32 bins
                    reduce = cuda.shared.array(32, dtype=np.complex128)

                    # # d_h
                    cuda.syncthreads()
                    reduce[f_idx] = d_h_here
                    # print("Value at f_idx, d_h_here:", f_idx, d_h_here.real, d_h_here.imag, reduce[f_idx].real, reduce[f_idx].imag)
                    cuda.syncthreads()  # ensure all threads have written before reducing

                    stride = cuda.blockDim.x // 2
                    while stride > 0:
                        if f_idx < stride:
                            reduce[f_idx] += reduce[f_idx + stride]
                        cuda.syncthreads()  # sync after EVERY stride step
                        stride //= 2

                    statistic[src_num, t_idx, 0] = reduce[0]

                    # h_h
                    cuda.syncthreads()
                    reduce[f_idx] = h_h_here
                    cuda.syncthreads()  # ensure all threads have written before reducing

                    stride = cuda.blockDim.x // 2
                    while stride > 0:
                        if f_idx < stride:
                            reduce[f_idx] += reduce[f_idx + stride]
                        cuda.syncthreads()  # sync after EVERY stride step
                        stride //= 2

                    statistic[src_num, t_idx, 1] = reduce[0]

        cuda.syncthreads()

    if block_vectorised_gpu:
        gpu_kernel = gpu_kernel_blocked
    else:
        gpu_kernel = kernel_gpu

    return kernel_cpu, gpu_kernel


def multimode_direct_kernel_constructor(
    config: dict,
    _get_channels: Callable,
    compute_statistic: bool = False,
    tdi_type: int | None = None,
):
    """
    Constructor method used to generate a kernel function for computing Fresnel waveforms and derived inner-product statistics (d_h and h_h).

    The kernels constructed here directly take arrays of amplitude, phase, frequency, and frequency derivative for each mode
    for each time-segment point, rather than computing them from a waveform model.

    Parameters:
    ----------
    config: dict
        Configuration dictionary containing the following keys:
        - dT: float, the duration of each time segment.
        - nF: int, the number of frequency bins.
        - dF: float, the width of each frequency bin.
        - kernel_width: int, the number of frequency bins on either side of the central frequency to include in the kernel computation.
        - nparams: int, the number of parameters describing each source. This is used to allocate local arrays in the GPU kernel.

    Generalised "Response function/Channel construction" method.
    Can be either a frequency domain response function, or a function within the waveform itself that returns the waveform polarizations.
    Frequency domain response function can in principle be substituted for any response function in the frequency domain (within time-segment)
    _get_channels: Callable
        Function that takes in frequency-domain (within each time-segment) waveforms, in the format h_lm and
            - either transforms to TDI channels using the detector response.
            - or converts to waveform polarizations h_+ and h_x in the frequency domain and returns that.

    kernel_cpu: Callable
        A CPU kernel function that can be called to compute Fresnel waveforms and statistics.
    kernel_gpu: Callable
        A GPU kernel function that can be called to compute Fresnel waveforms and statistics.

    Returns:
    -------
    kernel_cpu: Callable
        A CPU kernel function that can be called to compute Fresnel waveforms and statistics.
    kernel_gpu: Callable
        A GPU kernel function that can be called to compute Fresnel waveforms and statistics.

    """
    dT = config["dT"]
    nF = config["nF"]
    dF = config["dF"]
    kernel_width = config["kernel_width"]

    # Only TDI-2 supported for now.
    if tdi_type is None:
        tdi = False
        tdi2 = False
    else:
        tdi = True
        if tdi_type == 1:
            tdi2 = False
        elif tdi_type == 2:
            tdi2 = True
        else:
            raise ValueError(
                f"Invalid tdi_type: {tdi_type}. Must be 1, 2, or None."
            )

    @jit
    def kernel_inner(
        src_num,
        mode_num,
        channels,
        segment_start_inds,
        segment_end_inds,
        phase_amp_information,
        y_lms,
        parameters_response,
        params_source_response,
        P_lm,
        k,
        n,
        p,
        Ls,
        spacecraft_ltts,
        spacecraft_orbits,
        statistic,
        inv_psds,
        mixed_precision,
        use_midpoint,
    ):
        """
        NOTE: HARDWARE AGNOSTIC

        Kernel to compute to compute the fresnel waveforms or statistics (d_h and h_h) for a single source (indexed by src_num) across all time-segments.
        This kernel is called by both the CPU and GPU kernels, with appropriate local array allocations for each hardware type.

        Kernel output is the filled-in statistic array (d_h and h_h for each time-segment) if compute_statistic is True, or the computed waveforms if compute_statistic is False.

        Parameters:
        ----------
        src_num: int
            The index of the source for which to compute the waveforms/statistics.
        mode_num: int
            The index of the mode for which to compute the waveforms/statistics.
        channels: array (n_sources, nT, nF, n_channels)
            Either the array to be filled in with the computed waveforms for each time-segment and frequency bin (if compute_statistic is False)
            Or the data array to be used to compute statistics (d_h and h_h) if compute_statistic is True.
        segment_start_inds: array (n_sources,)
            The starting time-segment index for each source. Used to determine which time-segments to compute over for the given source.
        segment_end_inds: array (n_sources,)
            The ending time-segment index for each source. Used to determine which time-segments to compute over for the given source.
        phase_amp_information: array (n_sources, n_modes, nT, 4)
            The phase and amplitude information for each mode for each source at each time-segment.
            The 4 values are (amp, phi0, f0, fdot) for each mode at each time-segment.
        y_lms: array (n_sources, n_modes, 2)
            The spin-weighted spherical harmonic values for each mode in each source.
            First element is the positive-m component, second element is the negative-m component. For m = 0,
            each component should be halved, so that the sum of the two components is equal to the m=0 component.
            The negative component should include symmetry-related prefactors e.g. (-1)^l.
        parameters_response: array (n_sources, 4) or None
            The parameters describing the response for each source, used if tdi is True. This is used to compute the TDI response for the given source.
        P_lm: array (3, 3) (array to be filled in within the kernel)
            Local array to store the response function coefficients for TDI computation.
        k: array (3,) (array to be filled in within the kernel)
            Local array to store the wavevector for TDI computation.
        n: array (3 (nSpacecraft), 3 (nDimensionsPerSpacecraft)) (array to be filled in within the transfer function call within the kernel)
            Unit vectors pointing between the spacecraft, used for TDI response computation.
            Setup in the convention of n[0] (2->3), n[1] (1->3), n[2] (1->2).
        p: array (3, 3) (array to be filled in within the kernel)
            Position vectors for each spacecraft at each time-segment, used for TDI response computation.
        Ls: array (3) (array to be filled in within the kernel)
            Arm lengths for each spacecraft at each time-segment, used for TDI response computation.
        spacecraft_ltts: array (nT, 3) or None
            Light travel times for each spacecraft, used for TDI response computation if tdi is True
            Used to fill in the Ls array within the kernel for TDI response computation.
            NOTE: Precomputed, not computed in the kernel at runtime.
        spacecraft_orbits: array (nT, 3, 3) or None
            Spacecraft orbits, used for TDI response computation if tdi is True.
            Used to fill in the p array within the kernel for TDI response computation.
            NOTE: Precomputed, not computed in the kernel at runtime.
        statistic: array (n_sources, n_modes, nT, 2) (array to be filled in within the kernel if compute_statistic is True)
            Local array to store the computed statistics (d_h and h_h for each time-segment for the given source for each mode.
        inv_psds: array (nT, nF, n_channels) or None
            The reciprocal (1/psd) of the noise PSD, prefolded outside the kernel so the inner-product loop multiplies instead of dividing.
        mixed_precision: bool
            Whether to use mixed precision (float32) for the computations within the kernel, to save memory and speed up computations.
        use_midpoint : bool
            Whether to evaluate the mode parameters at the midpoint of the segment, or at the beginning.
            (Should be set to True in almost all circumstances)
        """

        # Mixed precision operations to save memory and speed up if desired.
        if mixed_precision:
            dT_prec = np.float32(dT)
            dF_prec = np.float32(dF)
        else:
            dT_prec = dT
            dF_prec = dF

        Y_lm_pos = y_lms[src_num, mode_num, 0]
        Y_lm_neg = y_lms[src_num, mode_num, 1]

        if tdi:
            # Grab response parameters for specified source if TDI response computation is needed.
            for i in range(3):
                params_source_response[i + 1] = parameters_response[src_num, i]

            # Fill in P_0 tensor and wavevector k for TDI response computation for this source.
            fill_P_lm_given_Y_lm(
                P_lm, params_source_response, Y_lm_pos, Y_lm_neg
            )
            fill_k(k, params_source_response)

        for t_idx in range(
            segment_start_inds[src_num], segment_end_inds[src_num] + 1
        ):
            t_tranche = dT * t_idx
            # If use_midpoint is True, assume mode parameters evaluated at the midpoint of the segment, rather than the beginning.

            if use_midpoint:
                t_tranche += dT / 2

            amp_mode = phase_amp_information[src_num, mode_num, t_idx, 0]
            phi0_mode = phase_amp_information[src_num, mode_num, t_idx, 1]
            f0_mode = phase_amp_information[src_num, mode_num, t_idx, 2]
            fdot_mode = phase_amp_information[src_num, mode_num, t_idx, 3]

            if f0_mode <= 0:
                continue  # Skip this time-segment if the mode frequency is non-positive.

            # Start index in frequency bins for the given mode frequency, used to determine which frequency bins to compute over in the kernel.
            start_ind = int(f0_mode / dF)
            d_h = 0.0 + 0.0j
            h_h = 0.0 + 0.0j

            if tdi:
                # Fill in LTTs for this time step.
                for i in range(3):
                    Ls[i] = spacecraft_ltts[t_idx, i]

                # Fill in position vectors for this time step
                for i in range(3):
                    for j in range(3):
                        p[i, j] = spacecraft_orbits[t_idx, i, j]
                transfer_functions = _get_channels(
                    f0_mode, P_lm, k, p, Ls, n, tdi2
                )

            # Cheap way to check how many extra frequency bins to compute over for the given fdot
            extra_fdot_bins = int((fdot_mode * dT) / dF)

            if mixed_precision:
                amp_mode = np.float32(amp_mode)
                phi0_mode = np.float32(phi0_mode % (2 * np.pi))
                f0_mode = np.float32(f0_mode)
                fdot_mode = np.float32(fdot_mode)

                if tdi:
                    transfer_functions = (
                        np.complex64(transfer_functions[0]),
                        np.complex64(transfer_functions[1]),
                        np.complex64(transfer_functions[2]),
                    )

            # Fresnel quantities (common to all frequency bins for a given source at a given time-segment)
            # NOTE: math.sqrt, not np.sqrt -- numpy scalar ufuncs are not supported inside CUDA kernels by this numba version.
            sqrt2fdot = sqrt(2 * abs(fdot_mode))
            amp_mode_prefac = amp_mode / sqrt2fdot
            one_over_fdot = 1 / fdot_mode

            # For each frequeny bin within the time-segment, compute the fresnel.
            for f_rel_idx in range(
                -kernel_width,
                kernel_width + extra_fdot_bins,  # + 1
            ):
                f_idx = start_ind + f_rel_idx
                if f_idx > 0 and f_idx < nF:
                    f_bin = (f_idx + 1) * (dF_prec)
                    h_f_pos = _fresnel_kernel(
                        f_bin,
                        amp_mode_prefac,
                        sqrt2fdot,
                        phi0_mode,
                        f0_mode,
                        fdot_mode,
                        one_over_fdot,
                        dT_prec,
                        use_midpoint,
                    )
                    # Generate just the polarizations
                    if not tdi:
                        h_TT = (
                            0.5 * (Y_lm_pos + Y_lm_neg.conjugate()) * h_f_pos,
                            0.5j * (Y_lm_pos - Y_lm_neg.conjugate()) * h_f_pos,
                        )

                    # Generate AET channels via multiplicative, frequency-domain transfer function.
                    for i in range(channels.shape[-1]):
                        if tdi:
                            h = h_f_pos * transfer_functions[i]
                        else:
                            h = h_TT[i]

                        # If compute_statistic is True, compute the inner products for d_h and h_h using the data in channels and the computed waveform h, and the inverse psd in inv_psds.
                        if compute_statistic:
                            d = channels[t_idx, f_idx, i]
                            inv_psd = inv_psds[t_idx, f_idx, i]
                            d_h += complex_inner_product(d, h, inv_psd)
                            h_h += complex_inner_product(h, h, inv_psd)
                        else:
                            channels[src_num, t_idx, f_idx, i] += (
                                h  # TODO: atomics might be needed here
                            )

            if compute_statistic:
                statistic[src_num, mode_num, t_idx, 0] = d_h
                statistic[src_num, mode_num, t_idx, 1] = h_h

    @cuda.jit
    def kernel_gpu(
        channels,
        segment_start_inds,
        segment_end_inds,
        phase_amp_information,
        y_lms,
        parameters_response,
        spacecraft_orbits,
        spacecraft_ltts,
        statistic,
        inv_psds,
        mixed_precision,
        use_midpoint,
    ):
        """
        Wrapper for the Fresnel_kernel to launch on GPU.
        Each thread computes the Fresnel waveforms and statistics for a single source across all time-segments.
        This function instantiates the local arrays needed for the kernel_inner function, and calls kernel_inner for each source.

        Parameters:
        ----------
        channels: array (n_sources, nT, nF, n_channels)
            Either the array to be filled in with the computed waveforms for each time-segment and frequency bin (if compute_statistic is False)
            Or the data array to be used to compute statistics (d_h and h_h) if compute_statistic is True.
        segment_start_inds: array (n_sources,)
            The starting time-segment index for each source. Used to determine which time-segments to compute over for the given source.
        segment_end_inds: array (n_sources,)
            The ending time-segment index for each source. Used to determine which time-segments to compute over for the given source.
        phase_amp_information: array (n_sources, n_modes, nT, 4)
            The phase and amplitude information for each source and mode. This is used to compute the waveform for the given source.
        y_lms: array (n_sources, n_modes, 2)
            The spin-weighted spherical harmonic values for each mode in each source.
            First element is the positive-m component, second element is the negative-m component. For m = 0,
            each component should be halved, so that the sum of the two components is equal to the m=0 component.
            The negative component should include symmetry-related prefactors e.g. (-1)^l.
        parameters_response: array (n_sources, 4) or None
            The parameters describing the response for each source, used if tdi is True. This is used to compute the TDI response for the given source.
        spacecraft_orbits: array (nT, 3, 3) or None
            Spacecraft orbits, used for TDI response computation if tdi is True.
            Used to fill in the p array within the kernel for TDI response computation.
            NOTE: Precomputed, not computed in the kernel at runtime.
        spacecraft_ltts: array (nT, 3) or None
            Light travel times for each spacecraft, used for TDI response computation if tdi is True
            Used to fill in the Ls array within the kernel for TDI response computation.
        statistic: array (n_sources, n_modes, nT, 2) (array to be filled in within the kernel if compute_statistic is True)
            Local array to store the computed statistics (d_h and h_h for each time-segment for the given source for each mode.
        inv_psds: array (nT, nF, n_channels) or None
            The reciprocal (1/psd) of the noise PSD, prefolded outside the kernel so the inner-product loop multiplies instead of dividing. Used to compute the inner products for the statistics if compute_statistic is True.
        mixed_precision: bool
            Whether to use mixed precision (float32) for the computations within the kernel, to save memory and speed up computations.
        use_midpoint : bool
            Whether to evaluate the mode parameters at the midpoint of the segment, or at the beginning.
            (Should be set to True in almost all circumstances)
        """
        # one source per thread
        modes_per_source = channels.shape[1]
        total = int(phase_amp_information.shape[0] * modes_per_source)
        thread_num = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

        src_num = thread_num // modes_per_source
        mode_num = thread_num % modes_per_source

        # one source per thread
        if thread_num < total:
            # These are all the local arrays that will be filled in within the kernel_inner function.
            P_lm = cuda.local.array((3, 3), dtype=np.complex128)
            k = cuda.local.array((3,), dtype=parameters_response.dtype)
            n = cuda.local.array((3, 3), dtype=parameters_response.dtype)
            p = cuda.local.array((3, 3), dtype=spacecraft_orbits.dtype)
            Ls = cuda.local.array((3,), dtype=spacecraft_ltts.dtype)
            params_source_response = cuda.local.array(
                4, dtype=parameters_response.dtype
            )
            kernel_inner(
                src_num,
                mode_num,
                channels,
                segment_start_inds,
                segment_end_inds,
                phase_amp_information,
                parameters_response,
                y_lms,
                params_source_response,
                P_lm,
                k,
                n,
                p,
                Ls,
                spacecraft_ltts,
                spacecraft_orbits,
                statistic,
                inv_psds,
                mixed_precision,
                use_midpoint,
            )

    @jit
    def kernel_cpu(
        channels,
        segment_start_inds,
        segment_end_inds,
        phase_amp_information,
        y_lms,
        parameters_response,
        spacecraft_orbits,
        spacecraft_ltts,
        statistic,
        inv_psds,
        mixed_precision,
        use_midpoint,
    ):
        """
        Wrapper for the Fresnel_kernel to launch on CPU.
        NOTE: No signficant parallelisation for CPU version, just a loop over sources.

        Parameters:
        ----------
        channels: array (n_sources, nT, nF, n_channels)
            Either the array to be filled in with the computed waveforms for each time-segment and frequency bin (if compute_statistic is False)
            Or the data array to be used to compute statistics (d_h and h_h) if compute_statistic is True.
        segment_start_inds: array (n_sources,)
            The starting time-segment index for each source. Used to determine which time-segments to compute over for the given source.
        segment_end_inds: array (n_sources,)
            The ending time-segment index for each source. Used to determine which time-segments to compute over for the given source.
        phase_amp_information: array (n_sources, n_modes, nT, 4)
            Array containing phase and amplitude information for each mode of each source.
        y_lms: array (n_sources, n_modes, 2)
            The spin-weighted spherical harmonic values for each mode in each source.
            First element is the positive-m component, second element is the negative-m component. For m = 0,
            each component should be halved, so that the sum of the two components is equal to the m=0 component.
            The negative component should include symmetry-related prefactors e.g. (-1)^l.
        parameters_response: array (n_sources, 4) or None
            The parameters describing the response for each source, used if tdi is True. This is used to compute the TDI response for the given source.
        spacecraft_orbits: array (nT, 3, 3) or None
            Spacecraft orbits, used for TDI response computation if tdi is True.
            Used to fill in the p array within the kernel for TDI response computation.
            NOTE: Precomputed, not computed in the kernel at runtime.
        spacecraft_ltts: array (nT, 3) or None
            Light travel times for each spacecraft, used for TDI response computation if tdi is True
            Used to fill in the Ls array within the kernel for TDI response computation.
        statistic: array (n_sources, n_modes, nT, 2) (array to be filled in within the kernel if compute_statistic is True)
            Local array to store the computed statistics (d_h and h_h for each time-segment for the given source for each mode.
        inv_psds: array (nT, nF, n_channels) or None
            The reciprocal (1/psd) of the noise PSD. Used to compute the inner products for the statistics if compute_statistic is True.
        mixed_precision: bool
            Whether to use mixed precision (float32) for the computations within the kernel, to save memory and speed up computations.
        use_midpoint : bool
            Whether to evaluate the mode parameters at the midpoint of the segment, or at the beginning.
            (Should be set to True in almost all circumstances)
        """

        for src_num in range(phase_amp_information.shape[0]):
            for mode_num in range(phase_amp_information.shape[1]):
                P_lm = np.zeros((3, 3), dtype=np.complex128)
                k = np.zeros((3,), dtype=parameters_response.dtype)
                n = np.zeros((3, 3), dtype=parameters_response.dtype)
                p = np.zeros((3, 3), dtype=spacecraft_orbits.dtype)
                Ls = np.zeros((3,), dtype=spacecraft_ltts.dtype)
                params_source_response = np.zeros(
                    4, dtype=parameters_response.dtype
                )
                kernel_inner(
                    src_num,
                    mode_num,
                    channels,
                    segment_start_inds,
                    segment_end_inds,
                    phase_amp_information,
                    y_lms,
                    parameters_response,
                    params_source_response,
                    P_lm,
                    k,
                    n,
                    p,
                    Ls,
                    spacecraft_ltts,
                    spacecraft_orbits,
                    statistic,
                    inv_psds,
                    mixed_precision,
                    use_midpoint,
                )

    return kernel_cpu, kernel_gpu
