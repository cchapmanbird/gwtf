from typing import Callable

import numpy as np
from numba import cuda, jit, njit

from ..response.transfer import fill_k, fill_P_lm
from ..utils import complex_inner_product
from .common import _fresnel_kernel


def analytic_kernel_constructor(
    config: dict,
    _get_amplitude: Callable,
    _get_phi_f_fdot: Callable,
    _get_channels: Callable,
    compute_statistic: bool = False,
    tdi_type: int | None = None,
):
    dT = config["dT"]
    nF = config["nF"]
    dF = config["dF"]
    kernel_width = config["kernel_width"]
    nparams = config["nparams"]

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
        psds,
    ):
        for i in range(nparams):
            params_source[i] = parameters[src_num, i]

        if tdi:
            for i in range(4):
                params_source_response[i] = parameters_response[src_num, i]
            fill_P_lm(P_lm, params_source_response)
            fill_k(k, params_source_response)

        for t_idx in range(
            segment_start_inds[src_num], segment_end_inds[src_num] + 1
        ):
            t_tranche = dT * t_idx

            phi0_mode, f0_mode, fdot_mode = _get_phi_f_fdot(
                t_tranche, params_source
            )
            amp_mode = (
                _get_amplitude(t_tranche, f0_mode, fdot_mode, params_source)
                / 2
            )

            start_ind = int(f0_mode / dF)
            d_h = 0.0 + 0.0j
            h_h = 0.0 + 0.0j

            if tdi:
                # Fill in LTTs for this time step.
                for i in range(3):
                    Ls[i] = spacecraft_ltts[t_idx, i]

                for i in range(3):
                    for j in range(3):
                        p[i, j] = spacecraft_orbits[t_idx, i, j]
                transfer_functions = _get_channels(
                    f0_mode, P_lm, k, p, Ls, n, tdi2
                )

            for f_rel_idx in range(-kernel_width, kernel_width):
                f_idx = start_ind + f_rel_idx
                if f_idx > 0 and f_idx < nF:
                    f_bin = (f_idx + 1) * (dF)

                    h_f_pos = _fresnel_kernel(
                        f_bin,
                        amp_mode,
                        phi0_mode,
                        f0_mode,
                        fdot_mode,
                        dT,
                    )

                    if not tdi:
                        h_TT = _get_channels(h_f_pos, params_source)

                    for i in range(channels.shape[-1]):
                        if tdi:
                            h = h_f_pos * transfer_functions[i]
                        else:
                            h = h_TT[i]

                        if compute_statistic:
                            d = channels[t_idx, f_idx, i]
                            psd = psds[t_idx, f_idx, i]
                            d_h += complex_inner_product(d, h, psd, dF)
                            h_h += complex_inner_product(h, h, psd, dF)
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
        psds,
    ):
        src_num = (
            cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        )  # one source per thread
        if src_num < parameters.shape[0]:
            params_source = cuda.local.array(nparams, dtype=np.float64)
            P_lm = cuda.local.array((3, 3), dtype=np.complex128)
            k = cuda.local.array((3,), dtype=np.float64)
            n = cuda.local.array((3, 3), dtype=np.float64)
            p = cuda.local.array((3, 3), dtype=np.float64)
            Ls = cuda.local.array((3,), dtype=np.float64)
            params_source_response = cuda.local.array(4, dtype=np.float64)
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
    ):
        for src_num in range(parameters.shape[0]):
            params_source = np.zeros(nparams, dtype=np.float64)
            P_lm = np.zeros((3, 3), dtype=np.complex128)
            k = np.zeros((3,), dtype=np.float64)
            n = np.zeros((3, 3), dtype=np.float64)
            p = np.zeros((3, 3), dtype=np.float64)
            Ls = np.zeros((3,), dtype=np.float64)
            params_source_response = np.zeros(4, dtype=np.float64)
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
            )

    return kernel_cpu, kernel_gpu

