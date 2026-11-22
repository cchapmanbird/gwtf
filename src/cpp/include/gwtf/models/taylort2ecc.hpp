// TaylorT2 with small eccentricity, non-spinning, (2,2) mode only.
// Parameters: M [Msun], eta, cosi, e0, D [pc], f0 [Hz], phi_coal.
// Derived:    t_coal [s].  (Port of legacy/models/taylort2ecc.)
#pragma once

#include "gwtf/models/model_base.hpp"
#include "gwtf/models/taylort2ecc_pn.hpp"

namespace gwtf {
namespace models {

struct TaylorT2Ecc {
    static constexpr const char* name = "TaylorT2Ecc";
    static constexpr int n_params = 7;
    static constexpr int n_derived = 1;
    static constexpr int n_modes = 1;
    enum { M = 0, ETA, COSI, E0, D, F0, PHI_COAL, T_COAL };

    struct State {
        double M, eta, e0, D, phi_coal, tc, v0, theta_0;
    };

    GWTF_HD static double time_to_coalescence(double M, double eta, double e0, double f0)
    {
        const double v0 = pow(constants::pi * M * f0, 1.0 / 3.0);
        return 5.0 / 256.0 * M / eta / ipow<8>(v0) * pn::taylort2ecc::_calculate_T(v0, v0, e0, eta);
    }

    GWTF_HD static void compute_derived(double* p)
    {
        p[M] *= constants::MTsun;
        p[D] *= constants::pc / constants::clight;
        p[T_COAL] = time_to_coalescence(p[M], p[ETA], p[E0], p[F0]);
    }

    GWTF_HD static bool time_bounds(const double* p, double /*f_lo*/, double f_hi,
                                    double& t_start, double& t_end)
    {
        const double M_ = p[M], eta = p[ETA];
        const double v0 = pow(constants::pi * M_ * p[F0], 1.0 / 3.0);
        const double v = pow(constants::pi * M_ * f_hi, 1.0 / 3.0);
        const double tc_from_f
            = 5.0 / 256.0 * M_ / eta / ipow<8>(v) * pn::taylort2ecc::_calculate_T(v, v0, p[E0], eta);
        t_start = 0.0;
        t_end = p[T_COAL] - tc_from_f;
        return true;
    }

    GWTF_HD static State load(const double* p)
    {
        State s;
        s.M = p[M];
        s.eta = p[ETA];
        s.e0 = p[E0];
        s.D = p[D];
        s.phi_coal = p[PHI_COAL];
        s.tc = p[T_COAL];
        s.v0 = pow(constants::pi * s.M * p[F0], 1.0 / 3.0);
        s.theta_0 = pow(s.tc * s.eta / (5.0 * s.M), -1.0 / 8.0);
        return s;
    }

    GWTF_HD static bool track(const State& s, int /*mode*/, double t, ModeTrack& tr)
    {
        using namespace pn::taylort2ecc;
        const double pi = constants::pi;
        const double tau = (s.tc - t) * s.eta / (5.0 * s.M);
        // Powers of tau^(1/8) instead of double-precision pow (costly on GPUs).
        const double tau_r8 = std::sqrt(std::sqrt(std::sqrt(tau)));
        const double theta = 1.0 / tau_r8;  // tau^(-1/8)
        const double dtau_dt = -s.eta / (5.0 * s.M);
        const double dtheta_dt = (-1.0 / 8.0) / (tau * tau_r8) * dtau_dt;  // tau^(-9/8)

        const double Xi = _Xi(theta, s.theta_0, s.e0, s.eta);
        const double tau_prefac = 1.0 / 8.0 / ipow<3>(tau_r8) / (pi * s.M);
        const double freq = tau_prefac * Xi;
        const double dXi_dt = _dXi_dtheta(theta, s.theta_0, s.e0, s.eta, false) * dtheta_dt;
        const double d_prefactor_dt = -3.0 / 8.0 * dtau_dt / tau;
        const double fdot = tau_prefac * (d_prefactor_dt * Xi + dXi_dt);

        // pow, not cbrt: the phase scales as v^-5 ~ 1e8 rad, so one ulp in v
        // moves it by ~5e-8 rad; pow keeps v bit-compatible with v1.
        const double v = pow(pi * s.M * freq, 1.0 / 3.0);
        tr.phi = 2.0 * (s.phi_coal - 1.0 / (32.0 * ipow<5>(v) * s.eta) * _Lambda_f_vec(v, s.v0, s.e0, s.eta));
        tr.f = freq;
        tr.fdot = fdot;
        // Newtonian (2,2) amplitude of h_lm (Eq. 79 of arXiv:0710.0614).
        tr.amp = 8.0 * std::sqrt(pi / 5.0) * s.eta * s.M / s.D * (v * v);
        return true;
    }

    GWTF_HD static void ylm(const State&, int, double cosi, cplx<double>& Yp, cplx<double>& Yn)
    {
        ylm_22(cosi, Yp, Yn);
    }
};

} // namespace models
} // namespace gwtf
