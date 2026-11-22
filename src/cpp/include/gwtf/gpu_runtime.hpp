// GPU runtime portability. Source files are written against the HIP runtime
// API. On AMD (or any HIP installation) this simply includes HIP. When
// compiled by nvcc without HIP headers, the small set of HIP names used in
// the code base are mapped onto the CUDA runtime, so the identical source
// builds natively on NVIDIA hardware.
//
#pragma once

#include "gwtf/config.hpp"

#if defined(GWTF_HIP)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>

using hipStream_t = cudaStream_t;
using hipError_t = cudaError_t;
#define hipSuccess cudaSuccess
#define hipGetLastError cudaGetLastError
#define hipPeekAtLastError cudaPeekAtLastError
#define hipGetErrorString cudaGetErrorString
#define hipStreamSynchronize cudaStreamSynchronize
#define hipMemsetAsync cudaMemsetAsync
#define hipGetDevice cudaGetDevice
#define hipLaunchKernelGGL(kernel, grid, block, shmem, stream, ...) \
    kernel<<<(grid), (block), (shmem), (stream)>>>(__VA_ARGS__)
#endif

#include <stdexcept>
#include <string>

namespace gwtf {
namespace gpu {

inline void check(hipError_t err, const char* what)
{
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("pygwtf GPU error in ") + what + ": "
                                 + hipGetErrorString(err));
    }
}

} // namespace gpu
} // namespace gwtf
