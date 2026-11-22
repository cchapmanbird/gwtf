// Portability layer shared by the host (plain C++) and device (HIP / CUDA) builds.
//
// Every numerical routine in pygwtf's v2 backend is written once as a
// GWTF_HD (host + device) function. The CPU extension compiles these headers
// with the host compiler; the GPU extension compiles the same headers with
// hipcc (AMD, native HIP) or nvcc (NVIDIA, via the HIP->CUDA shim in
// gpu_runtime.hpp). Headers must therefore stay free of host-only library
// calls on device paths, and should avoid heavyweight std headers so they
// remain usable under runtime compilation (hipRTC / NVRTC) in future.
#pragma once

#include <cmath>
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define GWTF_GPU_COMPILER 1
#define GWTF_HD __host__ __device__
#define GWTF_DEVICE __device__
#define GWTF_GLOBAL __global__
#define GWTF_RESTRICT __restrict__
#else
#define GWTF_HD
#define GWTF_DEVICE
#define GWTF_GLOBAL
#define GWTF_RESTRICT __restrict__
#endif

// True only while compiling device code (second pass of nvcc / hipcc).
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define GWTF_DEVICE_PASS 1
#endif

namespace gwtf {

// Polarisation convention applied to every detector response (TT and TDI).
// +1: arXiv:2003.00357 convention, -1: Mojito convention (v1 default).
constexpr double pol_convention = -1.0;

} // namespace gwtf
