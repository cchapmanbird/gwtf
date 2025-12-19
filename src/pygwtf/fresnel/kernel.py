from .common import _fresnel_kernel
from ..utils import complex_inner_product
from ..response.transfer import fill_k, fill_P_lm
from numba import cuda, njit
import numpy as np


def analytic_kernel_constructor(
        config,
        _get_amplitude,
        _get_time_to_coalescence,
        _get_phi_f_fdot,
        _get_channels,
        gpu=True,
        compute_statistic=False,
        tdi_type=None
    ):
    nT = config['nT']
    dT = config['dT']
    nF = config['nF']
    dF = config['dF']
    kernel_width = config['kernel_width']
    nparams = config['nparams']

    if tdi_type is None:
        tdi = False
        tdi2 = False
    else:
        tdi = True
        tdi2 = tdi_type
        assert isinstance(tdi2, int)

    @cuda.jit
    def kernel_gpu(channels, parameters, parameters_response, spacecraft_orbits, statistic, psds):
        src_num = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x  # one source per thread
        if src_num < parameters.shape[0]:
            params_source = cuda.local.array(nparams, dtype=np.float64)
            for i in range(nparams):
                params_source[i] = parameters[src_num, i]
            
            if tdi:
                P_lm = cuda.local.array((3,3), dtype=np.complex128)
                k = cuda.local.array((3,), dtype=np.float64)
                n = cuda.local.array((3,3), dtype=np.float64)
                p = cuda.local.array((3,3), dtype=np.float64)
                params_source_response = cuda.local.array(4, dtype=np.float64)
                for i in range(4):
                    params_source_response[i] = parameters_response[src_num, i]
                fill_P_lm(P_lm, params_source_response)
                fill_k(k, params_source_response)
        
            time_to_coalescence = _get_time_to_coalescence(params_source)
            
            for t_idx in range(nT):
                t_tranche = dT * t_idx
                if t_tranche + dT > time_to_coalescence:
                    break

                phi0_mode, f0_mode, fdot_mode = _get_phi_f_fdot(t_tranche, time_to_coalescence, params_source)
                amp_mode = _get_amplitude(t_tranche, f0_mode, fdot_mode, params_source)

                start_ind = int(f0_mode / dF)
                d_h = 0.0 + 0.0j
                h_h = 0.0 + 0.0j

                if tdi:
                    for i in range(3):
                        for j in range(3):
                            p[i,j] = spacecraft_orbits[t_idx, i, j]
                    transfer_functions = _get_channels(f0_mode, P_lm, k, p, n, tdi2)

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
                            dT
                        ) / 2

                        if not tdi:
                            h_TT = _get_channels(h_f_pos, parameters)

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


    @njit
    def kernel_cpu(channels, parameters, parameters_response, spacecraft_orbits, statistic, psds):
        for src_num in range(parameters.shape[0]):
            params_source = np.zeros(nparams, dtype=np.float64)
            for i in range(nparams):
                params_source[i] = parameters[src_num, i]
            
            if tdi:
                P_lm = np.zeros((3,3), dtype=np.complex128)
                k = np.zeros((3,), dtype=np.float64)
                n = np.zeros((3,3), dtype=np.float64)
                p = np.zeros((3,3), dtype=np.float64)
                params_source_response = np.zeros(4, dtype=np.float64)
                for i in range(4):
                    params_source_response[i] = parameters_response[src_num, i]
                fill_P_lm(P_lm, params_source_response)
                fill_k(k, params_source_response)
        
            time_to_coalescence = _get_time_to_coalescence(params_source)
            
            for t_idx in range(nT):
                t_tranche = dT * t_idx
                if t_tranche + dT > time_to_coalescence:
                    break

                phi0_mode, f0_mode, fdot_mode = _get_phi_f_fdot(t_tranche, time_to_coalescence, params_source)
                amp_mode = _get_amplitude(t_tranche, f0_mode, fdot_mode, params_source)

                start_ind = int(f0_mode / dF)
                d_h = 0.0 + 0.0j
                h_h = 0.0 + 0.0j

                if tdi:
                    for i in range(3):
                        for j in range(3):
                            p[i,j] = spacecraft_orbits[t_idx, i, j]
                    transfer_functions = _get_channels(f0_mode, P_lm, k, p, n, tdi2)

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
                            dT
                        ) / 2

                        if not tdi:
                            h_TT = _get_channels(h_f_pos, parameters)

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

    if gpu:
        return kernel_gpu
    else:
        return kernel_cpu

