// TaylorT3, 3.5PN aligned spins, circular, (2,2) mode only.
// Parameters: M [Msun], eta, cosi, D [pc], f0 [Hz], s1, s2, phi_coal.
// Derived:    t_coal [s], delta, sigma, s.  (Port of legacy/models/taylort3spin.)
#pragma once

#include "gwtf/models/model_base.hpp"
#include "gwtf/models/taylort3spin_pn.hpp"

namespace gwtf {
namespace models {

struct TaylorT3Spin {
    static constexpr const char* name = "TaylorT3Spin";
    static constexpr int n_params = 8;
    static constexpr int n_derived = 4;
    static constexpr int n_modes = 1;
    enum { M = 0, ETA, COSI, D, F0, S1, S2, PHI_COAL, T_COAL, DELTA, SIGMA, S };

    struct State {
        double M, eta, D, phi_coal, tc, delta, sigma, s;
    };

    GWTF_HD static void compute_derived(double* p)
    {
        p[M] *= constants::MTsun;
        p[D] *= constants::pc / constants::clight;
        const double M_ = p[M], eta = p[ETA];
        // etaM_to_m1m2 (m1 >= m2)
        const double q = (1.0 + std::sqrt(1.0 - 4.0 * eta)) / (2.0 * eta) - 1.0;
        const double mu = M_ * q / ((1.0 + q) * (1.0 + q));
        const double sqrdet = std::sqrt(1.0 - 4.0 * mu / M_);
        const double m1 = M_ / 2.0 * (1.0 + sqrdet);
        const double m2 = M_ / 2.0 * (1.0 - sqrdet);
        p[DELTA] = (m1 - m2) / M_;
        p[SIGMA] = (m2 * p[S2] - m1 * p[S1]) / M_;
        p[S] = (m1 * m1 * p[S1] + m2 * m2 * p[S2]) / (M_ * M_);
        const double x0 = pow(constants::pi * M_ * p[F0], 2.0 / 3.0);
        p[T_COAL] = pn::taylort3spin::time_to_merger(x0, p[SIGMA], p[DELTA], eta, p[S]) * M_;
    }

    GWTF_HD static bool time_bounds(const double* p, double /*f_lo*/, double f_hi,
                                    double& t_start, double& t_end)
    {
        const double M_ = p[M];
        const double x = pow(constants::pi * M_ * f_hi, 2.0 / 3.0);
        const double tc_from_f = pn::taylort3spin::time_to_merger(x, p[SIGMA], p[DELTA], p[ETA], p[S]) * M_;
        t_start = 0.0;
        t_end = p[T_COAL] - tc_from_f;
        return true;
    }

    GWTF_HD static State load(const double* p)
    {
        return State{p[M], p[ETA], p[D], p[PHI_COAL], p[T_COAL], p[DELTA], p[SIGMA], p[S]};
    }

    GWTF_HD static bool track(const State& s, int /*mode*/, double t, ModeTrack& tr)
    {
        using namespace pn::taylort3spin;
        const double pi = constants::pi;
        const double tau = s.eta * (s.tc - t) / (5.0 * s.M);
        const double x = tau_to_x(tau, s.sigma, s.delta, s.eta, s.s);
        const double freq = x * std::sqrt(x) / (pi * s.M);
        tr.phi = 2.0 * (s.phi_coal - phase(x, s.sigma, s.delta, s.eta, s.s));
        tr.f = freq;
        tr.fdot = frequency_derivative(tau, s.sigma, s.delta, s.eta, s.s) / (s.M * s.M);
        const double v = std::cbrt(pi * s.M * freq);
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
