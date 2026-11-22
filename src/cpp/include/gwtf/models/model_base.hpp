// Analytic model concept.
//
// An analytic model is a struct with only static members:
//
//   static constexpr const char* name;   // matches the Python class' kernel_name
//   static constexpr int n_params;       // user-facing parameters
//   static constexpr int n_derived;      // derived parameters appended per source
//   static constexpr int n_modes;        // harmonics produced (compile time)
//   struct State;                        // per-source values needed by track()
//
//   GWTF_HD static void compute_derived(double* p);
//       In place on one row of length n_params + n_derived: convert units and
//       fill the derived parameters. Runs once per source per call.
//   GWTF_HD static bool time_bounds(const double* p, double f_lo, double f_hi,
//                                   double& t_start, double& t_end);
//       Times between which the signal is inside [f_lo, f_hi]. Return false
//       if the signal is in band throughout (no bounds).
//   GWTF_HD static State load(const double* p);
//   GWTF_HD static bool track(const State& s, int mode, double t, ModeTrack& tr);
//       Phase, frequency, fdot and amplitude of `mode` at time t (seconds
//       from the start of the data). Return false to skip the mode here.
//   GWTF_HD static void ylm(const State& s, int mode, double cosi,
//                           cplx<double>& Yp, cplx<double>& Yn);
//       Spin-weighted spherical harmonics (Y_lm, Y_l-m) of `mode`. The
//       negative-m entry carries any (-1)^l symmetry factor.
//
// Everything is evaluated in double precision: mode phases reach ~1e7 rad,
// so the model is the one stage that must not run in single precision.
//
// Adding a model: write a header implementing this concept (see
// taylort2ecc.hpp), add it to GWTF_ANALYTIC_MODELS in registry.hpp, rebuild,
// and add a Python AnalyticModel subclass with the same kernel_name and
// parameter order. See docs/v2_backend.md.
#pragma once

#include "gwtf/complex.hpp"
#include "gwtf/constants.hpp"
#include "gwtf/fresnel.hpp"
#include "gwtf/math.hpp"
#include "gwtf/response.hpp"
