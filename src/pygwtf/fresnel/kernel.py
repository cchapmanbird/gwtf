from .common import _fresnel_kernel
from ..utils import complex_inner_product
from ..response.transfer import fill_k, fill_P_lm
from numba import cuda, njit
import numpy as np

def analytic_waveform_kernel_constructor(
        nT,
        dT,
        nF,
        dF,
        kernel_width,
        _get_amplitude,
        _get_time_to_coalescence,
        _get_phi_f_fdot,
        _get_channels,
        nparams,
        gpu=True,
        tdi=True,
        tdi2=True,
    ):
    if tdi is None:
        kernel = analytic_waveform_kernel_constructor_no_response(
            nT,
            dT,
            nF,
            dF,
            kernel_width,
            _get_amplitude,
            _get_time_to_coalescence,
            _get_phi_f_fdot,
            _get_channels,
            nparams,
            gpu=gpu,
        )
    else:
        kernel = analytic_waveform_kernel_constructor_response(
            nT,
            dT,
            nF,
            dF,
            kernel_width,
            _get_amplitude,
            _get_time_to_coalescence,
            _get_phi_f_fdot,
            _get_channels,
            nparams,
            tdi2,
            gpu=gpu,
        ) 

    return kernel

def analytic_waveform_kernel_constructor_no_response(
        nT,
        dT,
        nF,
        dF,
        kernel_width,
        _get_amplitude,
        _get_time_to_coalescence,
        _get_phi_f_fdot,
        _get_channels,
        nparams,
        gpu=True,
    ):

    @cuda.jit
    def _gpu_kernel(channels, parameters):
        src_num = cuda.blockIdx.x  # one source per block
        if src_num < parameters.shape[0]:  # parameters has shape (num_sources, num_parameters)
            params_source = cuda.local.array(nparams, dtype=np.float64)
            for p_ind in range(nparams):
                params_source[p_ind] = parameters[src_num, p_ind]

            time_to_coalescence = _get_time_to_coalescence(params_source)

            for tranche_ind in range(cuda.threadIdx.x, nT, cuda.blockDim.x):
                t_tranche = dT * tranche_ind
                if t_tranche + dT > time_to_coalescence:
                    break
                phi0_mode, f0_mode, fdot_mode = _get_phi_f_fdot(t_tranche, time_to_coalescence, params_source)
                amp_mode = _get_amplitude(t_tranche, f0_mode, fdot_mode, params_source)
                start_ind = int(f0_mode / dF)    
                for freq_ind in range(start_ind - kernel_width, start_ind + kernel_width):
                    if freq_ind < 0 or freq_ind >= nF:
                        continue
                    f_bin = (freq_ind + 1) * (dF)
                    h_f_pos = _fresnel_kernel(
                        f_bin,
                        amp_mode,
                        phi0_mode,
                        f0_mode,
                        fdot_mode,
                        dT
                    ) / 2

                    h_chan = _get_channels(h_f_pos, params_source)
                    for i in range(channels.shape[3]):
                        channels[src_num, tranche_ind, freq_ind, i] += h_chan[i]

                cuda.syncthreads()
    
    @njit
    def _cpu_kernel(channels, parameters):
        for src_num in parameters.shape[0]:
            params_source = np.zeros(nparams, dtype=np.float64)
            for p_ind in range(nparams):
                params_source[p_ind] = parameters[src_num, p_ind]

            time_to_coalescence = _get_time_to_coalescence(params_source)

            for tranche_ind in range(nT):
                t_tranche = dT * tranche_ind
                if t_tranche + dT > time_to_coalescence:
                    break

                phi0_mode, f0_mode, fdot_mode = _get_phi_f_fdot(t_tranche, time_to_coalescence, params_source)
                amp_mode = _get_amplitude(t_tranche, f0_mode, fdot_mode, params_source)
                start_ind = int(f0_mode / dF)
                for freq_ind in range(start_ind - kernel_width, start_ind + kernel_width):
                    if freq_ind < 0 or freq_ind >= nF:
                        continue
                    f_bin = (freq_ind + 1) * (dF)
                    h_f_pos = _fresnel_kernel(
                        f_bin,
                        amp_mode,
                        phi0_mode,
                        f0_mode,
                        fdot_mode,
                        dT
                    ) / 2
                    h_chan = _get_channels(h_f_pos, params_source)
                    for i in range(channels.shape[3]):
                        channels[src_num, tranche_ind, freq_ind, i] += h_chan[i]

    if gpu:
        return _gpu_kernel
    else:
        return _cpu_kernel

def analytic_waveform_kernel_constructor_response(
        nT,
        dT,
        nF,
        dF,
        kernel_width,
        _get_amplitude,
        _get_time_to_coalescence,
        _get_phi_f_fdot,
        _get_transfer_functions,
        nparams,
        tdi2,
        gpu=True,
    ):
        
    @cuda.jit
    def gpu_kernel(channels, parameters, parameters_response, spacecraft_orbits):
        src_num = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x  # one source per thread
        if src_num < parameters.shape[0]:

            P_lm = cuda.local.array((3,3), dtype=np.complex128)
            k = cuda.local.array((3,), dtype=np.float64)
            n = cuda.local.array((3,3), dtype=np.float64)
            p = cuda.local.array((3,3), dtype=np.float64)

            params_source = cuda.local.array(nparams, dtype=np.float64)
            params_source_response = cuda.local.array(4, dtype=np.float64)
            for i in range(nparams):
                params_source[i] = parameters[src_num, i]
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
                for i in range(3):
                   for j in range(3):
                        p[i,j] = spacecraft_orbits[t_idx, i, j]

                transfer_functions = _get_transfer_functions(f0_mode, P_lm, k, p, n, tdi2)

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
                        for i in range(channels.shape[3]):
                            channels[src_num, t_idx, f_idx, i] += h_f_pos * transfer_functions[i]
        
        cuda.syncthreads()

    if gpu:
        return gpu_kernel

def analytic_statistic_kernel_constructor(
        nT,
        dT,
        nF,
        dF,
        kernel_width,
        _get_amplitude,
        _get_time_to_coalescence,
        _get_phi_f_fdot,
        _get_channels,
        nparams,
        gpu=True,
        tdi=True,
        tdi2=True,
    ):
    if tdi is None:
        kernel = analytic_statistic_kernel_constructor_no_response(
            nT,
            dT,
            nF,
            dF,
            kernel_width,
            _get_amplitude,
            _get_time_to_coalescence,
            _get_phi_f_fdot,
            _get_channels,
            nparams,
            gpu=gpu,
        )
    else:
        kernel = analytic_statistic_kernel_constructor_response(
            nT,
            dT,
            nF,
            dF,
            kernel_width,
            _get_amplitude,
            _get_time_to_coalescence,
            _get_phi_f_fdot,
            _get_channels,
            nparams,
            tdi2,
            gpu=gpu,
        ) 

    return kernel

def analytic_statistic_kernel_constructor_response(
        nT,
        dT,
        nF,
        dF,
        kernel_width,
        _get_amplitude,
        _get_time_to_coalescence,
        _get_phi_f_fdot,
        _get_transfer_functions,
        nparams,
        tdi2,
        gpu=True,
        
    ):

    @cuda.jit
    def _gpu_kernel(statistic, channels, psds, parameters, parameters_response, spacecraft_orbits):
        src_num = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x  # one source per thread
        if src_num < parameters.shape[0]:

            P_lm = cuda.local.array((3,3), dtype=np.complex128)
            k = cuda.local.array((3,), dtype=np.float64)
            n = cuda.local.array((3,3), dtype=np.float64)
            p = cuda.local.array((3,3), dtype=np.float64)

            params_source = cuda.local.array(nparams, dtype=np.float64)
            params_source_response = cuda.local.array(4, dtype=np.float64)
            for i in range(nparams):
                params_source[i] = parameters[src_num, i]
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

                for i in range(3):
                   for j in range(3):
                        p[i,j] = spacecraft_orbits[t_idx, i, j]
                transfer_functions = _get_transfer_functions(f0_mode, P_lm, k, p, n, tdi2)

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

                        for i in range(channels.shape[2]):
                            d = channels[t_idx, f_idx, i]
                            psd = psds[t_idx, f_idx, i]
                            h = h_f_pos * transfer_functions[i]
                            d_h += complex_inner_product(d, h, psd, dF)
                            h_h += complex_inner_product(h, h, psd, dF)

                statistic[src_num, t_idx, 0] = d_h
                statistic[src_num, t_idx, 1] = h_h

    if gpu:
        return _gpu_kernel

def analytic_statistic_kernel_constructor_no_response(
        nT,
        dT,
        nF,
        dF,
        kernel_width,
        _get_amplitude,
        _get_time_to_coalescence,
        _get_phi_f_fdot,
        _get_channels,
        nparams,
        gpu=True,
    ):

    # This is a block-vectorised version that is depressingly just slower
    # @cuda.jit
    # def _gpu_kernel(statistic, channels, psds, parameters):
    #     src_num = cuda.blockIdx.x  # one source per block
    #     t_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    #     f_idx = cuda.threadIdx.x
    #     if src_num < parameters.shape[0] and t_idx < nT:
    #         params_source = cuda.shared.array(nparams, dtype=np.float64)
    #         if f_idx < nparams and cuda.threadIdx.y == 0:
    #             params_source[f_idx] = parameters[src_num, f_idx]
    #         cuda.syncthreads()

    #         t_seg_here = dT * t_idx
    #         phi0_mode, f0_mode, fdot_mode = _get_phi_f_fdot(t_seg_here, params_source)
    #         amp_mode = _get_amplitude(t_seg_here, f0_mode, fdot_mode, params_source)
            
    #         freq_ind = int(f0_mode / dF) + f_idx - kernel_width
    #         d_h_here = 0.0 + 0.0j
    #         h_h_here = 0.0 + 0.0j
    #         if freq_ind > 0 or freq_ind < nF:
    #             f_bin = (freq_ind + 1) * (dF)
    #             h_f_pos = _fresnel_kernel(
    #                 f_bin,
    #                 amp_mode,
    #                 phi0_mode,
    #                 f0_mode,
    #                 fdot_mode,
    #                 dT
    #             ) / 2

    #             h_TT = _get_channels(h_f_pos, params_source)
    #             for i in range(channels.shape[2]):
    #                 d = channels[t_idx, freq_ind, i]
    #                 psd = psds[t_idx, freq_ind, i]
    #                 h = h_TT[i]
    #                 d_h_here += complex_inner_product(d, h, psd, dF)
    #                 h_h_here += complex_inner_product(h, h, psd, dF)

    #         partial = d_h_here.real - 0.5 * h_h_here.real

    #         cuda.syncthreads()

    #         reduce = cuda.shared.array(32, dtype=np.float64)           
    #         reduce[cuda.threadIdx.x] = partial

    #         stride = cuda.blockDim.x // 2
    #         while stride > 0:
    #             if cuda.threadIdx.x < stride:
    #                 reduce[cuda.threadIdx.x] += reduce[cuda.threadIdx.x + stride]
    #             cuda.syncthreads()
    #             stride //= 2
            
    #         statistic[src_num, t_idx] = reduce[0]

    #     cuda.syncthreads()

    @cuda.jit
    def _gpu_kernel(statistic, channels, psds, parameters):
        src_num = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x  # one source per thread
        if src_num < parameters.shape[0]:
            params_source = cuda.local.array(nparams, dtype=np.float64)
            for i in range(nparams):
                params_source[i] = parameters[src_num, i]
            
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

                        h_TT = _get_channels(h_f_pos, params_source)
                        for i in range(channels.shape[2]):
                            d = channels[t_idx, f_idx, i]
                            psd = psds[t_idx, f_idx, i]
                            h = h_TT[i]
                            d_h += complex_inner_product(d, h, psd, dF)
                            h_h += complex_inner_product(h, h, psd, dF)

                statistic[src_num, t_idx, 0] = d_h
                statistic[src_num, t_idx, 1] = h_h

    if gpu:
        return _gpu_kernel
