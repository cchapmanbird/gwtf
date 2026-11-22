// Fresnel time-frequency kernels: the Fourier transform, over one time
// segment of length T, of a mode with locally linear frequency evolution.
// Completing the square gives a difference of Fresnel integrals; see
// SegmentMode for the per-segment constants. Negative fdot is supported:
// the quadratic phase flips sign, which conjugates the Fresnel combination
// (C + i sgn(fdot) S) while the prefactor uses sqrt(2 |fdot|).
//
// All per-bin arithmetic runs in `Real` (float in the fast path). Anything
// involving large phases (the mode phase, f * T) is reduced in double once
// per segment in make_segment_mode(), so single precision is only called on
// O(1)-cycle quantities.
#pragma once

#include "gwtf/complex.hpp"
#include "gwtf/math.hpp"

namespace gwtf {

// Time-frequency grid. Bin k has central frequency fmin + k dF; segment n
// spans [n dT, (n + 1) dT).
struct Grid {
    int32_t nT;
    int32_t nF;
    int32_t kernel_width;  // bins either side of the mode frequency
    int32_t midpoint;      // evaluate the mode track at the segment midpoint
    double dT;
    double dF;
    double fmin;
    double tukey_alpha;    // only read by the Tukey window
};

enum class WindowKind : int32_t { Boxcar = 0, Tukey = 1 };

// Fresnel integrals S(x), C(x): rational approximation of McCormick &
// Kraemer (Coastal Eng. 2002, Sec. 3 Eq. 18), identical to v1.
template <typename T>
GWTF_HD inline void fresnel_sc(T x, T& S, T& C)
{
    T ax = std::fabs(x);
    T sinp, cosp;
    sincospi(T(0.5) * ax * ax, &sinp, &cosp);  // sin/cos(pi ax^2 / 2)
    if (ax < T(6)) {
        T f_den = T(2) + T(1.792) * ax + T(3.104) * ax * ax;
        T g_den = T(2) + T(4.142) * ax + T(3.492) * ax * ax + T(6.67) * ax * ax * ax;
        T rec = T(1) / (f_den * g_den);
        T fx = (T(1) + T(0.926) * ax) * rec * g_den;
        T gx = rec * f_den;
        S = T(0.5) - fx * cosp - gx * sinp;
        C = T(0.5) + fx * sinp - gx * cosp;
    } else {
        T r = T(1) / (T(constants::pi) * ax);
        S = T(0.5) - cosp * r;
        C = T(0.5) + sinp * r;
    }
    if (x < T(0)) {
        S = -S;
        C = -C;
    }
}

// One mode in one time segment, reduced to the constants the per-bin
// Fresnel evaluation needs. NCH channel coefficients carry the detector
// response: channel c at bin f is coef[c] * fresnel(f).
template <typename Real, int NCH>
struct SegmentMode {
    int32_t f_lo;        // first bin (absolute, clipped to the grid)
    int32_t f_hi;        // last bin, inclusive
    int32_t center;      // bin nearest the mode frequency
    Real amp_prefac;     // A / sqrt(2 |fdot|)
    Real sqrt2fdot;      // sqrt(2 |fdot|)
    Real fdot;
    Real inv_fdot;
    Real sgn;            // sign(fdot)
    Real df0;            // f0 - f_center [Hz]
    Real phase0_hc;      // constant phase / pi, reduced to [-1, 1]
    Real step_hc;        // phase / pi advance per bin from the f T term
    cplx<Real> coef[NCH];
};

// Per-mode track sample produced by models (always double precision).
struct ModeTrack {
    double phi;   // GW phase of the mode
    double f;     // frequency [Hz]
    double fdot;  // frequency derivative [Hz/s]
    double amp;   // amplitude
};

// Build the segment constants. Returns false if the mode contributes nothing
// (non-finite track, zero amplitude, non-positive frequency, or no bins on
// the grid). Bin ranges and rounding reproduce v1 exactly.
template <typename Real, int NCH>
GWTF_HD inline bool make_segment_mode(const Grid& g, const ModeTrack& tr, double amp_scale,
                                      SegmentMode<Real, NCH>& sm)
{
    if (!(is_finite(tr.phi) && is_finite(tr.f) && is_finite(tr.fdot) && is_finite(tr.amp)))
        return false;
    if (tr.amp == 0.0 || !(tr.f > 0.0))
        return false;

    // Bin nearest the mode frequency (round half to even, as Python round()).
    double c = std::rint((tr.f - g.fmin) / g.dF);
    if (!(c > -1.0e9 && c < 1.0e9))
        return false;
    const int center = int(c);

    // Extra bins swept by fdot over the segment, as v1 (using |fdot|).
    double extra_d = g.midpoint ? (std::fabs(tr.fdot) * g.dT / 2.0) / g.dF
                                : (std::fabs(tr.fdot) * g.dT) / g.dF;
    if (!(extra_d < double(g.nF)))
        extra_d = double(g.nF);
    const int extra = int(extra_d);
    int lo, hi;
    if (g.midpoint) {
        lo = center - g.kernel_width - extra;
        hi = center + g.kernel_width + extra;
    } else {
        lo = center - g.kernel_width;
        hi = center + g.kernel_width + extra;
    }
    if (lo < 0)
        lo = 0;
    if (hi > g.nF - 1)
        hi = g.nF - 1;
    if (lo > hi)
        return false;

    const double f_c = g.fmin + double(center) * g.dF;
    const double afdot = std::fabs(tr.fdot);
    const double s2f = std::sqrt(2.0 * afdot);

    sm.f_lo = lo;
    sm.f_hi = hi;
    sm.center = center;
    sm.amp_prefac = Real(tr.amp * amp_scale / s2f);
    sm.sqrt2fdot = Real(s2f);
    sm.fdot = Real(tr.fdot);
    sm.inv_fdot = Real(1.0 / tr.fdot);
    sm.sgn = tr.fdot >= 0.0 ? Real(1) : Real(-1);
    sm.df0 = Real(tr.f - f_c);

    // Constant phase in half-cycles: phi / pi, minus f_c T for the midpoint
    // convention (the -2 pi f_bin T/2 factor), reduced exactly in double.
    double ph = std::remainder(tr.phi, constants::two_pi) / constants::pi;
    if (g.midpoint) {
        ph -= std::remainder(f_c * g.dT, 2.0);
        sm.step_hc = Real(std::remainder(g.dF * g.dT, 2.0));
    } else {
        sm.step_hc = Real(0);
    }
    sm.phase0_hc = Real(std::remainder(ph, 2.0));
    return true;
}

// ---------------------------------------------------------------------------
// Precision-stable evaluation of segment Fresnel integrals.
//
// A (sub-)segment integral between offsets c_a < c_b (relative to the
// segment reference time) is
//   X = exp(-i pi |fdot| delta^2) [ (C + iS)(v_b) - (C + iS)(v_a) ],
//   v = sqrt(2|fdot|) (delta + c),   delta = (f0 - f_bin) / fdot,
// conjugated when fdot < 0. Far from the track |v| is large and the two
// phases pi |fdot| delta^2 and pi v^2 / 2 are both huge (1e5 rad is typical)
// while their difference is O(1) cycles. Using the asymptotic branch of the
// rational approximation,
//   (C + iS)(v) = sgn(v) (1 + i)/2 - i exp(i pi v^2 / 2) / (pi v),
// each endpoint is split into exp(-i pi |fdot| delta^2) P + Q where Q carries
// the combined phase q = (v^2 - v_delta^2) / 2 = 2 c sgn(fdot) (f0 - f_bin)
// + |fdot| c^2 directly. The large-phase P terms cancel exactly when both
// endpoints are asymptotic on the same side, and are otherwise only needed
// where the track crosses the bin (moderate phase). This is algebraically
// identical to v1 but keeps float32 accurate.
// ---------------------------------------------------------------------------
template <typename Real>
struct FresnelEnd {
    cplx<Real> P;  // multiplies exp(-i pi |fdot| delta^2)
    cplx<Real> Q;  // phase already combined
};

template <typename Real, int NCH>
GWTF_HD inline FresnelEnd<Real> fresnel_end(const SegmentMode<Real, NCH>& sm, Real dfj, Real delta,
                                            Real c)
{
    FresnelEnd<Real> e;
    const Real v = sm.sqrt2fdot * (delta + c);
    const Real ax = std::fabs(v);
    if (ax < Real(6)) {
        Real S, C;
        fresnel_sc(v, S, C);
        e.P = cplx<Real>(C, S);
        e.Q = cplx<Real>(Real(0), Real(0));
    } else {
        const Real h = v < Real(0) ? Real(-0.5) : Real(0.5);
        e.P = cplx<Real>(h, h);
        const Real afdot = sm.sgn * sm.fdot;
        const Real q = Real(2) * c * sm.sgn * dfj + afdot * c * c;  // half-cycles
        Real sq, cq;
        sincospi(q, &sq, &cq);
        const Real r = Real(1) / (Real(constants::pi) * v);
        e.Q = cplx<Real>(sq * r, -cq * r);  // -i exp(i pi q) / (pi v)
    }
    return e;
}

// Integral between two endpoints (b minus a), including the fdot sign.
template <typename Real, int NCH>
GWTF_HD inline cplx<Real> fresnel_between(const SegmentMode<Real, NCH>& sm, Real dfj, Real delta,
                                          const FresnelEnd<Real>& a, const FresnelEnd<Real>& b)
{
    cplx<Real> X = b.Q - a.Q;
    const cplx<Real> dP = b.P - a.P;
    if (dP.re != Real(0) || dP.im != Real(0)) {
        // |fdot| delta^2 = sgn(fdot) (f0 - f_bin) delta, in half-cycles.
        X += expipi(-sm.sgn * dfj * delta) * dP;
    }
    return sm.sgn < Real(0) ? conj(X) : X;
}

// Box-car window. j is the bin offset from sm.center.
template <typename Real, int NCH>
GWTF_HD inline cplx<Real> fresnel_boxcar(const SegmentMode<Real, NCH>& sm, int j, Real dF,
                                         Real T, bool midpoint)
{
    const Real dfj = sm.df0 - Real(j) * dF;  // f0 - f_bin
    const Real delta = dfj * sm.inv_fdot;
    const Real c0 = midpoint ? Real(-0.5) * T : Real(0);
    const Real cT = midpoint ? Real(0.5) * T : T;
    const FresnelEnd<Real> e0 = fresnel_end(sm, dfj, delta, c0);
    const FresnelEnd<Real> eT = fresnel_end(sm, dfj, delta, cT);
    const cplx<Real> X = fresnel_between(sm, dfj, delta, e0, eT);
    // Remaining phase / pi: phi/pi - f_bin T (midpoint convention).
    return expipi(sm.phase0_hc - Real(j) * sm.step_hc) * X * sm.amp_prefac;
}

// Tukey sub-integrals over the roll-on [-T/2, -b], flat [-b, b] and roll-off
// [b, T/2] regions, for the effective frequency offset dfj.
template <typename Real, int NCH>
GWTF_HD inline void tukey_sub_integrals(const SegmentMode<Real, NCH>& sm, Real dfj, Real T, Real b,
                                        cplx<Real>& F_on, cplx<Real>& F_flat, cplx<Real>& F_off)
{
    const Real delta = dfj * sm.inv_fdot;
    const FresnelEnd<Real> e0 = fresnel_end(sm, dfj, delta, Real(-0.5) * T);
    const FresnelEnd<Real> ea = fresnel_end(sm, dfj, delta, -b);
    const FresnelEnd<Real> eb = fresnel_end(sm, dfj, delta, b);
    const FresnelEnd<Real> eT = fresnel_end(sm, dfj, delta, Real(0.5) * T);
    F_on = fresnel_between(sm, dfj, delta, e0, ea);
    F_flat = fresnel_between(sm, dfj, delta, ea, eb);
    F_off = fresnel_between(sm, dfj, delta, eb, eT);
}

// Tukey window (midpoint convention only), taper fraction alpha in (0, 1].
// The cosine taper splits into exponentials at f0 +- 1/(alpha T); see v1
// fresnel/common.py for the derivation. tp = 0.25 exp(i pi (1 - alpha)/alpha).
template <typename Real, int NCH>
GWTF_HD inline cplx<Real> fresnel_tukey(const SegmentMode<Real, NCH>& sm, int j, Real dF,
                                        Real T, Real alpha, const cplx<Real>& tp)
{
    const Real dfj = sm.df0 - Real(j) * dF;
    const Real shift = Real(1) / (alpha * T);  // taper frequency offset [Hz]
    const Real b = Real(0.5) * T * (Real(1) - alpha);

    cplx<Real> F_on, F_flat, F_off, F_on_p, F_flat_p, F_off_p, F_on_m, F_flat_m, F_off_m;
    tukey_sub_integrals(sm, dfj, T, b, F_on, F_flat, F_off);
    tukey_sub_integrals(sm, dfj + shift, T, b, F_on_p, F_flat_p, F_off_p);
    tukey_sub_integrals(sm, dfj - shift, T, b, F_on_m, F_flat_m, F_off_m);

    const cplx<Real> tpc = conj(tp);
    const cplx<Real> J = F_on * Real(0.5) + tp * F_on_p + tpc * F_on_m + F_flat
                         + F_off * Real(0.5) + tpc * F_off_p + tp * F_off_m;
    return expipi(sm.phase0_hc - Real(j) * sm.step_hc) * J * sm.amp_prefac;
}

template <typename Real>
GWTF_HD inline cplx<Real> tukey_taper_phase(double alpha)
{
    double s, c;
    sincospi((1.0 - alpha) / alpha, &s, &c);
    return cplx<Real>(Real(0.25 * c), Real(0.25 * s));
}

} // namespace gwtf
