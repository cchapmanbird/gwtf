// Detector responses.
//
// Every response maps a mode (spin-weighted spherical harmonic pair
// Y_lm, Y_l-m and segment frequency f0) to per-channel complex coefficients
// coef[c], so that channel c of the time-frequency waveform is
// coef[c] * fresnel(f). The work is split by how often it changes:
//
//   Source  : per source           (sky position, polarisation)
//   Segment : per source & segment (spacecraft geometry at that time)
//   coefficients(): per mode       (needs Y_lm and the mode frequency)
//
// Response parameters are always [cosi, pol, ecliptic_long, ecliptic_lat].
#pragma once

#include "gwtf/complex.hpp"
#include "gwtf/math.hpp"

namespace gwtf {

enum class ResponseKind : int32_t { TT = 0, TDI2 = 1 };

struct ResponseParams {
    double cosi;
    double pol;
    double lam;   // ecliptic longitude
    double beta;  // ecliptic latitude
};

GWTF_HD inline ResponseParams load_response_params(const double* p)
{
    return ResponseParams{p[0], p[1], p[2], p[3]};
}

// Y_{2,2} and Y_{2,-2} at reference phase 0 (v1 Ylms(cosi, 0)).
GWTF_HD inline void ylm_22(double cosi, cplx<double>& Yp, cplx<double>& Yn)
{
    const double norm = std::sqrt(5.0 / (64.0 * constants::pi));
    Yp = cplx<double>(norm * (1.0 + 2.0 * cosi + cosi * cosi), 0.0);
    Yn = cplx<double>(norm * (1.0 - 2.0 * cosi + cosi * cosi), 0.0);
}

// ---------------------------------------------------------------------------
// Transverse-traceless polarisations (h+, hx); no detector.
//   h+ = s/2 (Y+ + conj(Y-)) h,   hx = s i/2 (Y+ - conj(Y-)) h
// with s the polarisation convention. For the (2,2) mode this reproduces v1:
//   h+ = -h N (1 + cosi^2),  hx = -h N 2i cosi.
// ---------------------------------------------------------------------------
template <typename Real>
struct TTResponse {
    static constexpr int n_channels = 2;
    struct Source {};
    struct Segment {};

    GWTF_HD Source source(const ResponseParams&) const { return {}; }
    GWTF_HD Segment segment(const Source&, int) const { return {}; }

    GWTF_HD void coefficients(const Source&, const Segment&, const cplx<double>& Yp,
                              const cplx<double>& Yn, double /*f0*/,
                              cplx<Real>* coef) const
    {
        const cplx<double> a = Yp + conj(Yn);
        const cplx<double> d = Yp - conj(Yn);
        const double h = 0.5 * pol_convention;
        coef[0] = cplx<Real>(a * h);
        coef[1] = cplx<Real>(cplx<double>(-d.im * h, d.re * h));  // i d h
    }
};

// ---------------------------------------------------------------------------
// Second-generation TDI (A, E, T), slowly varying unequal arms; a direct port
// of v1 response/transfer.py (get_AET_TFs). The polarisation tensor is
// linear in (Y+, Y-), so the arm contractions n.P.n are precomputed per
// segment without Y_lm and combined per mode:
//   n.P.n = (Y+ A + conj(Y- A)) / 2,  A = n . [s e^{-2i pol}(P+ + i Px)] . n
// Geometry arrays: orbits (nT, 3, 3) spacecraft positions [m], ltts (nT, 3)
// arm lengths [m] for links (12, 23, 31).
// ---------------------------------------------------------------------------
template <typename Real>
struct TDI2Response {
    static constexpr int n_channels = 3;
    const Real* orbits;
    const Real* ltts;

    struct Source {
        Real k[3];             // propagation direction
        cplx<Real> part[3][3]; // s e^{-2i pol} (P+ + i Px)
    };
    struct Segment {
        cplx<Real> A[3];       // n_a . part . n_a for arms n0 (2->3), n1 (1->3), n2 (1->2)
        Real kn[3];            // k . n_a
        Real kp01, kp12, kp02; // k . (p_i + p_j)
        Real L12, L23, L13;
    };

    GWTF_HD Source source(const ResponseParams& rp) const
    {
        // Angles are O(1): working precision is sufficient here.
        Source s;
        const Real lam = Real(rp.lam), beta = Real(rp.beta), pol = Real(rp.pol);
        const Real cl = std::cos(lam), sl = std::sin(lam);
        const Real c2l = std::cos(Real(2) * lam), s2l = std::sin(Real(2) * lam);
        const Real cb = std::cos(beta), sb = std::sin(beta);
        const Real cl2 = cl * cl, sl2 = sl * sl, cb2 = cb * cb, sb2 = sb * sb;
        const Real cbsl = cb * sl, cbsbsl = cbsl * sb, cbcl = cb * cl, cbclsb = cbcl * sb;
        const Real clslprod = cl * (Real(1) + sb2) * sl;

        const Real Pp[3][3] = {{cl2 * sb2 - sl2, clslprod, -cbclsb},
                               {clslprod, -cl2 + sb2 * sl2, -cbsbsl},
                               {-cbclsb, -cbsbsl, cb2}};
        const Real Pc[3][3] = {{sb * s2l, -c2l * sb, -cbsl},
                               {-c2l * sb, Real(-2) * cl * sb * sl, cbcl},
                               {-cbsl, cbcl, Real(0)}};
        const cplx<Real> e = cplx<Real>(std::cos(Real(2) * pol), -std::sin(Real(2) * pol))
                             * Real(pol_convention);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                s.part[i][j] = e * cplx<Real>(Pp[i][j], Pc[i][j]);

        s.k[0] = -cb * cl;
        s.k[1] = -cb * sl;
        s.k[2] = -sb;
        return s;
    }

    GWTF_HD static cplx<Real> contract(const Real* n, const cplx<Real> (&P)[3][3])
    {
        cplx<Real> acc;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                acc += P[i][j] * (n[i] * n[j]);
        return acc;
    }

    GWTF_HD Segment segment(const Source& s, int t_idx) const
    {
        Segment g;
        const Real* p = orbits + size_t(t_idx) * 9;  // p[sc * 3 + xyz]
        const Real* L = ltts + size_t(t_idx) * 3;
        g.L12 = L[0];
        g.L23 = L[1];
        g.L13 = L[2];
        Real n[3][3];
        for (int j = 0; j < 3; ++j) {
            n[0][j] = (p[6 + j] - p[3 + j]) / g.L23;  // 2 -> 3
            n[1][j] = (p[6 + j] - p[0 + j]) / g.L13;  // 1 -> 3
            n[2][j] = (p[3 + j] - p[0 + j]) / g.L12;  // 1 -> 2
        }
        for (int a = 0; a < 3; ++a) {
            g.A[a] = contract(n[a], s.part);
            g.kn[a] = s.k[0] * n[a][0] + s.k[1] * n[a][1] + s.k[2] * n[a][2];
        }
        g.kp01 = s.k[0] * (p[0] + p[3]) + s.k[1] * (p[1] + p[4]) + s.k[2] * (p[2] + p[5]);
        g.kp12 = s.k[0] * (p[3] + p[6]) + s.k[1] * (p[4] + p[7]) + s.k[2] * (p[5] + p[8]);
        g.kp02 = s.k[0] * (p[0] + p[6]) + s.k[1] * (p[1] + p[7]) + s.k[2] * (p[2] + p[8]);
        return g;
    }

    GWTF_HD static Real sinc(Real x) { return x == Real(0) ? Real(1) : std::sin(x) / x; }
    GWTF_HD static cplx<Real> cexpi(Real x) { return cplx<Real>(std::cos(x), std::sin(x)); }

    GWTF_HD void coefficients(const Source&, const Segment& g, const cplx<double>& Ypd,
                              const cplx<double>& Ynd, double f0d, cplx<Real>* coef) const
    {
        using C = cplx<Real>;
        const Real pi = Real(constants::pi);
        const Real c = Real(constants::clight);
        const Real f = Real(f0d);
        const C Yp(Ypd), Yn(Ynd);

        // n.P.n per arm for this mode.
        C prod[3];
        for (int a = 0; a < 3; ++a)
            prod[a] = (Yp * g.A[a] + conj(Yn * g.A[a])) * Real(0.5);

        const Real w = pi * f / c;  // omega / 2c
        const C expf_12 = cexpi(w * (g.L12 + g.kp01));
        const C expf_23 = cexpi(w * (g.L23 + g.kp12));
        const C expf_13 = cexpi(w * (g.L13 + g.kp02));

        const Real s123 = sinc(w * g.L12 * (Real(1) - g.kn[2]));
        const Real s213 = sinc(w * g.L12 * (Real(1) + g.kn[2]));
        const Real s132 = sinc(w * g.L13 * (Real(1) - g.kn[1]));
        const Real s312 = sinc(w * g.L13 * (Real(1) + g.kn[1]));
        const Real s231 = sinc(w * g.L23 * (Real(1) - g.kn[0]));
        const Real s321 = sinc(w * g.L23 * (Real(1) + g.kn[0]));

        const C iw(Real(0), w);
        const C b12 = iw * g.L12 * expf_12 * prod[2];
        const C b13 = iw * g.L13 * expf_13 * prod[1];
        const C b23 = iw * g.L23 * expf_23 * prod[0];
        const C y123 = b12 * s123, y213 = b12 * s213;
        const C y132 = b13 * s132, y312 = b13 * s312;
        const C y231 = b23 * s231, y321 = b23 * s321;

        const C z1 = cexpi(Real(2) * w * g.L12);
        const C z2 = cexpi(Real(2) * w * g.L23);
        const C z3 = cexpi(Real(2) * w * g.L13);
        const C z1s = z1 * z1, z2s = z2 * z2, z3s = z3 * z3;

        C X = (y312 + z2 * y132 - y213 - y123 * z3)
              - (y312 * z3s + y132 * z2 * z3s - y213 * z2s - y123 * z3 * z2s);
        C Y = (y123 + z3 * y213 - y321 - y231 * z1)
              - (y123 * z1s + y213 * z3 * z1s - y321 * z3s - y231 * z1 * z3s);
        C Z = (y231 + z1 * y321 - y132 - y312 * z2)
              - (y231 * z2s + y321 * z1 * z2s - y132 * z1s - y312 * z2 * z1s);
        X *= C(Real(1)) - z2s * z3s;
        Y *= C(Real(1)) - z3s * z1s;
        Z *= C(Real(1)) - z1s * z2s;
        X = conj(X);
        Y = conj(Y);
        Z = conj(Z);

        coef[0] = (Z - X) * Real(1.0 / 1.4142135623730951);               // A
        coef[1] = (X - Y * Real(2) + Z) * Real(1.0 / 2.449489742783178);  // E
        coef[2] = (X + Y + Z) * Real(1.0 / 1.7320508075688772);           // T
    }
};

} // namespace gwtf
