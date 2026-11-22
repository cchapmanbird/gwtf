// Scalar math helpers usable on host and device.
#pragma once

#include "gwtf/complex.hpp"
#include "gwtf/config.hpp"
#include "gwtf/constants.hpp"

namespace gwtf {

// x^N by repeated squaring (N >= 0, compile time).
template <int N, typename T>
GWTF_HD inline T ipow(T x)
{
    static_assert(N >= 0, "ipow requires a non-negative exponent");
    if constexpr (N == 0) {
        return T(1);
    } else if constexpr (N == 1) {
        return x;
    } else {
        T h = ipow<N / 2>(x);
        if constexpr (N % 2 == 1) {
            return h * h * x;
        } else {
            return h * h;
        }
    }
}

// sin(pi x), cos(pi x) with exact range reduction on x (precision-friendly
// for large arguments, which matters in the single-precision fast path).
GWTF_HD inline void sincospi(float x, float* s, float* c)
{
#if defined(GWTF_DEVICE_PASS)
    ::sincospif(x, s, c);
#else
    double r = std::remainder(double(x), 2.0);
    *s = float(std::sin(constants::pi * r));
    *c = float(std::cos(constants::pi * r));
#endif
}

GWTF_HD inline void sincospi(double x, double* s, double* c)
{
#if defined(GWTF_DEVICE_PASS)
    ::sincospi(x, s, c);
#else
    double r = std::remainder(x, 2.0);
    *s = std::sin(constants::pi * r);
    *c = std::cos(constants::pi * r);
#endif
}

// exp(i pi x)
template <typename T>
GWTF_HD inline cplx<T> expipi(T x)
{
    T s, c;
    sincospi(x, &s, &c);
    return cplx<T>(c, s);
}

// Reduce x to [-0.5, 0.5] (fractional cycles).
GWTF_HD inline double frac_centered(double x) { return x - std::rint(x); }

GWTF_HD inline bool is_finite(double x) { return std::isfinite(x); }

} // namespace gwtf
