from typing import Callable

import numpy as np
from numba import cuda, jit, njit

from ..response.transfer import fill_k, fill_P_lm
from ..utils import complex_inner_product
from .common import _fresnel_kernel


@jit
def semi_coherent_statistic_sum(
    src_num,
    statistics,
    N_seg,
    segment_end_inds,
    segment_start_inds,
):
    """
    Sum the per-segment statistics to get the semi-coherent statistic (upsilon).
    NOTE: Unused right now, used to post-process the per-segment statistics computed by the original kernel, but for now we compute the semi-coherent statistic directly in the kernel.

    Args:
        src_num (int): Index of the source for which to compute the statistic.
        statistics (array): Array of shape (n_sources, n_time_bins, 2) containing the per-segment statistics (d|h and h|h).
        N_seg (int): Number of segments that were summed over to get each per-segment statistic.
        segment_end_inds (array): Array of shape (n_sources,) containing the end indices of the segments for each source.
        segment_start_inds (array): Array of shape (n_sources,) containing the start indices of the segments for each source.

    Returns:
        statistic: Array of shape (n_sources,) containing the semi-coherent statistic for each source.
    """

    segment_end = segment_end_inds[src_num]
    segment_start = segment_start_inds[src_num]

    nT_in_band = (
        segment_end - segment_start + 1
    )  # not sure about this +1, check this. CCB: I think +1 is right?
    nT_per_seg = int(nT_in_band // N_seg)

    semicoherent_statistic = 0.0

    for seg_num in range(N_seg):
        d_h_seg = 0.0 + 0.0j
        h_h_seg = 0.0 + 0.0j

        for t_idx in range(nT_per_seg):
            # Segment start usually zero
            t_idx_global = segment_start + seg_num * nT_per_seg + t_idx

            d_h_seg += statistics[src_num, t_idx_global, 0]
            h_h_seg += statistics[src_num, t_idx_global, 1]

        semicoherent_statistic += (abs(d_h_seg) ** 2) / h_h_seg.real

    return semicoherent_statistic


@cuda.jit
def semi_coherent_statistic_sum_gpu_wrap(
    statistics, N_seg, segment_end_inds, segment_start_inds, search_statistic
):
    src_num = cuda.grid(1)  # one thread for each source
    # Check if the thread index is within the bounds of the statistic array (checking for edge case where the thread grid might be larger than the number of sources)
    # If a thread has a index higher than the number of sources, it should not do anything.
    if src_num < statistics.shape[0]:
        search_statistic[src_num] = semi_coherent_statistic_sum(
            src_num, statistics, N_seg, segment_end_inds, segment_start_inds
        )


@njit
def semi_coherent_statistic_sum_cpu_wrap(
    statistics, N_seg, segment_end_inds, segment_start_inds, search_statistic
):
    # For each source
    for src_num in range(statistics.shape[0]):
        search_statistic[src_num] = semi_coherent_statistic_sum(
            src_num, statistics, N_seg, segment_end_inds, segment_start_inds
        )


def analytic_kernel_constructor_semi_coherent(
    config: dict,
    _get_amplitude: Callable,
    _get_phi_f_fdot: Callable,
    _get_channels: Callable,
):
    '''
    Constructor method used to generate a kernel function for computing the semi-coherent detection statistic directly per source.

    Unlike analytic_kernel_constructor, this constructor builds kernels that evaluate the semi-coherent
    detection statistic in a single pass, without first storing a full (n_sources, nT, 2) array of
    per-time-bin inner products. The time-segments for each source are divided into Nseg sub-segments,
    and the semi-coherent statistic is accumulated as sum_seg(|d_h_seg|^2 / h_h_seg.real).

    NOTE: TDI-2 only -- the semi-coherent search kernel does not support waveform-polarization output mode.

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
    Frequency domain response function can in principle be substituted for any response function in the frequency domain (within time-segment).
    _get_channels: Callable
        Function that takes in frequency-domain (within each time-segment) waveforms, in the format h_lm and
        transforms to TDI channels using the detector response.

    kernel_cpu: Callable
        A CPU kernel function that can be called to compute the per-source semi-coherent statistic.
    kernel_gpu: Callable
        A GPU kernel function that can be called to compute the per-source semi-coherent statistic.

    Returns:
    -------
    kernel_cpu: Callable
        A CPU kernel function that can be called to compute the per-source semi-coherent statistic.
    kernel_gpu: Callable
        A GPU kernel function that can be called to compute the per-source semi-coherent statistic.

    '''
    dT = config["dT"]
    nF = config["nF"]
    dF = config["dF"]
    kernel_width = config["kernel_width"]
    nparams = config["nparams"]

    # Semi-coherent constructor is for TDI statistics only.
    tdi2 = True

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
        psds,
        Nseg,
        mixed_precision,
    ):
        '''
        NOTE: HARDWARE AGNOSTIC

        Kernel to compute the semi-coherent detection statistic for a single source (indexed by src_num),
        accumulated across all time-segments split into Nseg sub-segments.
        This kernel is called by both the CPU and GPU kernels, with appropriate local array allocations for each hardware type.

        Kernel output is the filled-in statistic[src_num], a single semi-coherent statistic value per source,
        computed as sum_seg(|d_h_seg|^2 / h_h_seg.real).

        Parameters:
        ----------
        src_num: int
            The index of the source for which to compute the statistic.
        channels: array (nT, nF, n_channels)
            The data array to be used to compute the inner products (d_h and h_h) for the semi-coherent statistic.
        segment_start_inds: array (n_sources,)
            The starting time-segment index for each source. Used to determine which time-segments to compute over for the given source.
        segment_end_inds: array (n_sources,)
            The ending time-segment index for each source. Used to determine which time-segments to compute over for the given source.
        parameters: array (n_sources, nparams)
            The parameters describing each source. This is used to compute the waveform for the given source.
        params_source: array (nparams,) (array to be filled in within the kernel)
            Local array to store the parameters of the current source.
        parameters_response: array (n_sources, 4)
            The parameters describing the response for each source. This is used to compute the TDI response for the given source.
        params_source_response: array (4,) (array to be filled in within the kernel)
            Local array to store the response parameters of the current source.
        P_lm: array (3, 3) (array to be filled in within the kernel)
            Local array to store the response function coefficients for TDI computation.
        k: array (3,) (array to be filled in within the kernel)
            Local array to store the wavevector for TDI computation.
        n: array (3 (nSpacecraft), 3 (nDimensionsPerSpacecraft)) (array to be filled in within the transfer function call within the kernel)
            Unit vectors pointing between the spacecraft, used for TDI response computation.
            Setup in the convention of n[0] (2->3), n[1] (1->3), n[2] (1->2).
        p: array (3, 3) (array to be filled in within the kernel)
            Position vectors for each spacecraft at each time-segment, used for TDI response computation.
        Ls: array (3,) (array to be filled in within the kernel)
            Arm lengths for each spacecraft at each time-segment, used for TDI response computation.
        spacecraft_ltts: array (nT, 3)
            Light travel times for each spacecraft, used for TDI response computation.
            Used to fill in the Ls array within the kernel for TDI response computation.
            NOTE: Precomputed, not computed in the kernel at runtime.
        spacecraft_orbits: array (nT, 3, 3)
            Spacecraft orbits, used for TDI response computation.
            Used to fill in the p array within the kernel for TDI response computation.
            NOTE: Precomputed, not computed in the kernel at runtime.
        statistic: array (n_sources,) (filled in within the kernel)
            Array to be filled in with the semi-coherent statistic for the given source.
        psds: array (nT, nF, n_channels)
            The power spectral density of the noise, used to compute the inner products for the semi-coherent statistic.
        Nseg: int
            The number of sub-segments into which the in-band time-segments are divided for the semi-coherent statistic.
        mixed_precision: bool
            Whether to use mixed precision (float32) for the computations within the kernel, to save memory and speed up computations.
        '''
        # Mixed precision operations to save memory and speed up if desired.
        if mixed_precision:
            dT_prec = np.float32(dT)
            dF_prec = np.float32(dF)
        else:
            dT_prec = dT
            dF_prec = dF

        # Grab parameters for specified source.
        for i in range(nparams):
            params_source[i] = parameters[src_num, i]

        # Grab response parameters for specified source for TDI response computation.
        for i in range(4):
            params_source_response[i] = parameters_response[src_num, i]
        # Fill in P_lm tensor and wavevector k for TDI response computation for this source.
        fill_P_lm(P_lm, params_source_response)
        fill_k(k, params_source_response)

        # Divide the in-band time-segments for this source into Nseg sub-segments for the semi-coherent statistic.
        nT_in_band = (
            segment_end_inds[src_num] - segment_start_inds[src_num] + 1
        )  # not sure about this +1, check this. CCB: I think +1 is right?
        nT_per_seg = np.int32(nT_in_band // Nseg)

        semi_coherent_stat = 0

        # Outer loop over semi-coherent sub-segments. Each sub-segment contributes |d_h_seg|^2 / h_h_seg.real to the statistic.
        for seg_num in range(Nseg):
            # Per-segment statistic accumulators (over all tranches in the sub-segment).
            d_h_seg = 0.0 + 0.0j
            h_h_seg = 0.0 + 0.0j

            # Inner loop over time-segments (tranches) within the current sub-segment.
            for t_idx_in_seg in range(nT_per_seg):
                # Global time-segment index. Segment start usually zero.
                t_idx = (
                    segment_start_inds[src_num]
                    + seg_num * nT_per_seg
                    + t_idx_in_seg
                )
                t_tranche = dT_prec * t_idx

                phi0_mode, f0_mode, fdot_mode = _get_phi_f_fdot(
                    t_tranche, params_source
                )
                amp_mode = (
                    _get_amplitude(
                        t_tranche, f0_mode, fdot_mode, params_source
                    )
                )

                # Start index in frequency bins for the given mode frequency, used to determine which frequency bins to compute over in the kernel.
                start_ind = int(f0_mode / dF)

                # Per-tranche inner product accumulators.
                d_h = 0.0 + 0.0j
                h_h = 0.0 + 0.0j

                # Fill in LTTs for this time step.
                for i in range(3):
                    Ls[i] = spacecraft_ltts[t_idx, i]

                # Fill in position vectors for this time step.
                for i in range(3):
                    for j in range(3):
                        p[i, j] = spacecraft_orbits[t_idx, i, j]
                transfer_functions = _get_channels(
                    f0_mode, P_lm, k, p, Ls, n, tdi2
                )

                if mixed_precision:
                    amp_mode = np.float32(amp_mode)
                    phi0_mode = np.float32(phi0_mode % (2 * np.pi))
                    f0_mode = np.float32(f0_mode)
                    fdot_mode = np.float32(fdot_mode)

                    transfer_functions = (
                        np.complex64(transfer_functions[0]),
                        np.complex64(transfer_functions[1]),
                        np.complex64(transfer_functions[2]),
                    )

                # Cheap way to check how many extra frequency bins to compute over for the given fdot.
                extra_fdot_bins = int((fdot_mode * dT) / dF)

                # For each frequency bin within the time-segment, compute the fresnel.
                for f_rel_idx in range(
                    -kernel_width, kernel_width + extra_fdot_bins + 1
                ):
                    f_idx = start_ind + f_rel_idx
                    if f_idx > 0 and f_idx < nF:
                        f_bin = (f_idx + 1) * (dF_prec)

                        h_f_pos = _fresnel_kernel(
                            f_bin,
                            amp_mode,
                            phi0_mode,
                            f0_mode,
                            fdot_mode,
                            dT_prec,
                        )

                        # Generate AET channels via multiplicative, frequency-domain transfer function.
                        for i in range(channels.shape[-1]):
                            h = h_f_pos * transfer_functions[i]

                            # Extract data and psd for this time-frequency bin and channel, and accumulate the per-tranche statistic.
                            d = channels[t_idx, f_idx, i]
                            psd = psds[t_idx, f_idx, i]
                            d_h += complex_inner_product(d, h, psd, dF_prec)
                            h_h += complex_inner_product(h, h, psd, dF_prec)

                # Add the per-tranche statistics to the per-segment statistic.
                d_h_seg += d_h
                h_h_seg += h_h

            # Accumulate the semi-coherent statistic: sum_seg(|d_h_seg|^2 / h_h_seg.real).
            semi_coherent_stat += (abs(d_h_seg) ** 2) / h_h_seg.real

        statistic[src_num] = semi_coherent_stat

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
        psds,
        Nseg,
        mixed_precision,
    ):
        '''
        Wrapper for the semi-coherent Fresnel_kernel to launch on GPU.
        Each thread computes the semi-coherent statistic for a single source across all time-segments.
        This function instantiates the local arrays needed for the kernel_inner function, and calls kernel_inner for each source.

        Parameters:
        ----------
        channels: array (nT, nF, n_channels)
            The data array to be used to compute the inner products (d_h and h_h) for the semi-coherent statistic.
        segment_start_inds: array (n_sources,)
            The starting time-segment index for each source. Used to determine which time-segments to compute over for the given source.
        segment_end_inds: array (n_sources,)
            The ending time-segment index for each source. Used to determine which time-segments to compute over for the given source.
        parameters: array (n_sources, nparams)
            The parameters describing each source. This is used to compute the waveform for the given source.
        parameters_response: array (n_sources, 4)
            The parameters describing the response for each source. This is used to compute the TDI response for the given source.
        spacecraft_orbits: array (nT, 3, 3)
            Spacecraft orbits, used for TDI response computation.
            Used to fill in the p array within the kernel for TDI response computation.
            NOTE: Precomputed, not computed in the kernel at runtime.
        spacecraft_ltts: array (nT, 3)
            Light travel times for each spacecraft, used for TDI response computation.
            Used to fill in the Ls array within the kernel for TDI response computation.
        statistic: array (n_sources,) (filled in within the kernel)
            Array to be filled in with the semi-coherent statistic for each source.
        psds: array (nT, nF, n_channels)
            The power spectral density of the noise, used to compute the inner products for the semi-coherent statistic.
        Nseg: int
            The number of sub-segments into which the in-band time-segments are divided for the semi-coherent statistic.
        mixed_precision: bool
            Whether to use mixed precision (float32) for the computations within the kernel, to save memory and speed up computations.
        '''
        # one source per thread
        src_num = (
            cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        )
        # one source per thread
        if src_num < parameters.shape[0]:
            # These are all the local arrays that will be filled in within the kernel_inner function.
            params_source = cuda.local.array(nparams, dtype=parameters.dtype)
            P_lm = cuda.local.array((3, 3), dtype=channels.dtype)
            k = cuda.local.array((3,), dtype=parameters_response.dtype)
            n = cuda.local.array((3, 3), dtype=parameters_response.dtype)
            p = cuda.local.array((3, 3), dtype=spacecraft_orbits.dtype)
            Ls = cuda.local.array((3,), dtype=spacecraft_ltts.dtype)
            params_source_response = cuda.local.array(
                4, dtype=parameters.dtype
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
                psds,
                Nseg,
                mixed_precision,
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
        psds,
        Nseg,
        mixed_precision,
    ):
        '''
        Wrapper for the semi-coherent Fresnel_kernel to launch on CPU.
        NOTE: No significant parallelisation for CPU version, just a loop over sources.

        Parameters:
        ----------
        channels: array (nT, nF, n_channels)
            The data array to be used to compute the inner products (d_h and h_h) for the semi-coherent statistic.
        segment_start_inds: array (n_sources,)
            The starting time-segment index for each source. Used to determine which time-segments to compute over for the given source.
        segment_end_inds: array (n_sources,)
            The ending time-segment index for each source. Used to determine which time-segments to compute over for the given source.
        parameters: array (n_sources, nparams)
            The parameters describing each source. This is used to compute the waveform for the given source.
        parameters_response: array (n_sources, 4)
            The parameters describing the response for each source. This is used to compute the TDI response for the given source.
        spacecraft_orbits: array (nT, 3, 3)
            Spacecraft orbits, used for TDI response computation.
            Used to fill in the p array within the kernel for TDI response computation.
            NOTE: Precomputed, not computed in the kernel at runtime.
        spacecraft_ltts: array (nT, 3)
            Light travel times for each spacecraft, used for TDI response computation.
            Used to fill in the Ls array within the kernel for TDI response computation.
        statistic: array (n_sources,) (filled in within the kernel)
            Array to be filled in with the semi-coherent statistic for each source.
        psds: array (nT, nF, n_channels)
            The power spectral density of the noise, used to compute the inner products for the semi-coherent statistic.
        Nseg: int
            The number of sub-segments into which the in-band time-segments are divided for the semi-coherent statistic.
        mixed_precision: bool
            Whether to use mixed precision (float32) for the computations within the kernel, to save memory and speed up computations.
        '''
        for src_num in range(parameters.shape[0]):
            params_source = np.zeros(nparams, dtype=parameters.dtype)
            P_lm = np.zeros((3, 3), dtype=channels.dtype)
            k = np.zeros((3,), dtype=parameters_response.dtype)
            n = np.zeros((3, 3), dtype=parameters_response.dtype)
            p = np.zeros((3, 3), dtype=spacecraft_orbits.dtype)
            Ls = np.zeros((3,), dtype=spacecraft_ltts.dtype)
            params_source_response = np.zeros(4, dtype=parameters.dtype)
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
                psds,
                Nseg,
                mixed_precision,
            )

    return kernel_cpu, kernel_gpu
