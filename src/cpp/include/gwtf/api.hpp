// Backend entry points and their plain-old-data argument structs.
//
// The nanobind layer (compiled by the host compiler) validates arrays and
// fills these structs; each compute backend (cpu_ops.cpp, gpu_ops.hip)
// implements the run_* functions. Keeping this boundary free of templates
// and device code is what lets the same bindings drive both backends.
//
// All arrays are C-contiguous. Pointers to "Real" data are float or double
// according to `precision`; complex arrays are interleaved (re, im).
#pragma once

#include <cstdint>

#include "gwtf/fresnel.hpp"
#include "gwtf/response.hpp"

namespace gwtf {

enum class Precision : int32_t { Single = 0, Double = 1 };
enum class NoiseKind : int32_t { Diagonal = 0, Full = 1 };

// Largest number of modes per source the fused multi-mode statistic kernel
// handles (cross-mode terms are kept in registers/local memory). Larger
// counts go through waveform generation + inner_product in Python.
// TODO relax this restriction!
constexpr int32_t max_fused_modes = 16;

// How the statistic of a multi-mode numerical source is evaluated. Single-
// mode sources (and every analytic model) always use the CrossTerms kernel,
// for which the three are identical.
enum class StatisticMode : int32_t {
    // Exact <h|h> of the summed signal, cross-mode terms included. One thread
    // per (source, segment); at most max_fused_modes modes.
    CrossTerms = 0,
    // Self terms only, one thread per (source, segment) looping over modes.
    SegmentLoop = 1,
    // Self terms only, one thread per (source, mode, segment); contributions
    // are summed into the output with atomics. No limit on the mode count.
    ModeParallel = 2,
};

// Which sources to evaluate and where their parameters live.
struct SourceArgs {
    int32_t model;              // analytic model id, or numerical_model_id
    int64_t n_src;
    // Analytic models: prepared parameters (n_src, param_stride), double.
    const double* params;
    int32_t param_stride;
    // Numerical model: tracks (n_src, n_modes, nT, 4) as (amp, phi, f, fdot),
    // double; ylms (n_src, n_modes, 2) complex128.
    const double* tracks;
    const void* ylms;
    int32_t n_modes;
    // Per-source in-band segment range [seg_start, seg_end] (int32, n_src).
    const int32_t* seg_start;
    const int32_t* seg_end;
    // Response parameters (n_src, 4): cosi, pol, ecliptic longitude/latitude.
    const double* response;
};

// Detector geometry, only read by TDI responses. Real-typed.
struct DetectorArgs {
    ResponseKind kind;
    const void* orbits;  // (nT, 3, 3) spacecraft positions [m]
    const void* ltts;    // (nT, 3) arm lengths [m]
};

struct PrepareArgs {
    int32_t model;
    int64_t n_src;
    const double* raw;  // (n_src, n_raw) user parameters, model order
    int32_t n_raw;
    double* params;     // (n_src, stride) out: dimensionful + derived
    int32_t stride;
    int32_t* seg_start; // (n_src,) out
    int32_t* seg_end;   // (n_src,) out
    Grid grid;
    double f_lo, f_hi;  // frequency band used for the time bounds
    void* stream;
};

struct WaveformArgs {
    Grid grid;
    SourceArgs src;
    DetectorArgs det;
    WindowKind window;
    Precision precision;
    double amp_scale;  // multiplies every mode amplitude (1 for physical output)
    void* out;  // (n_src, nT, nF, n_channels) complex<Real>, zeroed by caller
    void* stream;
};

struct StatisticArgs {
    Grid grid;
    SourceArgs src;
    DetectorArgs det;
    WindowKind window;
    Precision precision;
    NoiseKind noise;
    StatisticMode mode;
    // Exact rescaling for the single-precision path: the caller passes
    // data * s and weights / s^2 and sets amp_scale = s.
    double amp_scale;
    const void* data;     // (nT, nF, n_channels) complex<Real>
    const void* weights;  // diagonal: (nT, nF, n_channels) Real = 4 dF / psd
                          // full:     (nT, nF, n_channels, n_channels) complex<Real> = 4 dF invC
    void* out;            // (n_src, nT, 2) complex<Real>: <d|h>, <h|h> per segment
    void* stream;
};

struct InnerProductArgs {
    int64_t n_src;
    int32_t nT, nF, n_channels;
    Precision precision;
    NoiseKind noise;
    const void* h;        // (n_src, nT, nF, n_channels) complex<Real>
    const void* data;
    const void* weights;
    void* out;            // (n_src, nT, 2) complex<Real>
    void* stream;
};

struct TrackArgs {
    int32_t model;        // analytic id
    int64_t n_src;
    const double* params; // prepared (n_src, stride)
    int32_t stride;
    Grid grid;
    double* out;          // (n_src, n_modes, nT, 4): amp, phi, f, fdot
    void* stream;
};

namespace backend {
void run_prepare(const PrepareArgs& a);
void run_waveform(const WaveformArgs& a);
void run_statistic(const StatisticArgs& a);
void run_inner_product(const InnerProductArgs& a);
void run_tracks(const TrackArgs& a);
void synchronize(void* stream);
} // namespace backend

} // namespace gwtf
