// Minimal host/device complex number. Layout-compatible with C99 / numpy /
// CuPy complex types (two contiguous T), so data arrays can be reinterpreted
// directly.
#pragma once

#include "gwtf/config.hpp"

namespace gwtf {

template <typename T>
struct alignas(2 * sizeof(T)) cplx {
    T re;
    T im;

    GWTF_HD constexpr cplx() : re(0), im(0) {}
    GWTF_HD constexpr cplx(T r) : re(r), im(0) {}
    GWTF_HD constexpr cplx(T r, T i) : re(r), im(i) {}
    template <typename U>
    GWTF_HD constexpr explicit cplx(const cplx<U>& o) : re(T(o.re)), im(T(o.im)) {}

    GWTF_HD cplx& operator+=(const cplx& o) { re += o.re; im += o.im; return *this; }
    GWTF_HD cplx& operator-=(const cplx& o) { re -= o.re; im -= o.im; return *this; }
    GWTF_HD cplx& operator*=(const cplx& o)
    {
        T r = re * o.re - im * o.im;
        im = re * o.im + im * o.re;
        re = r;
        return *this;
    }
    GWTF_HD cplx& operator*=(T s) { re *= s; im *= s; return *this; }
};

template <typename T>
GWTF_HD inline cplx<T> operator+(cplx<T> a, const cplx<T>& b) { return a += b; }
template <typename T>
GWTF_HD inline cplx<T> operator-(cplx<T> a, const cplx<T>& b) { return a -= b; }
template <typename T>
GWTF_HD inline cplx<T> operator-(const cplx<T>& a) { return cplx<T>(-a.re, -a.im); }
template <typename T>
GWTF_HD inline cplx<T> operator*(cplx<T> a, const cplx<T>& b) { return a *= b; }
template <typename T>
GWTF_HD inline cplx<T> operator*(cplx<T> a, T s) { return a *= s; }
template <typename T>
GWTF_HD inline cplx<T> operator*(T s, cplx<T> a) { return a *= s; }
template <typename T>
GWTF_HD inline cplx<T> operator/(const cplx<T>& a, T s) { return cplx<T>(a.re / s, a.im / s); }
template <typename T>
GWTF_HD inline cplx<T> operator/(const cplx<T>& a, const cplx<T>& b)
{
    T d = b.re * b.re + b.im * b.im;
    return cplx<T>((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d);
}

template <typename T>
GWTF_HD inline cplx<T> conj(const cplx<T>& a) { return cplx<T>(a.re, -a.im); }
// |a|^2
template <typename T>
GWTF_HD inline T norm2(const cplx<T>& a) { return a.re * a.re + a.im * a.im; }
// conj(a) * b
template <typename T>
GWTF_HD inline cplx<T> conj_mul(const cplx<T>& a, const cplx<T>& b)
{
    return cplx<T>(a.re * b.re + a.im * b.im, a.re * b.im - a.im * b.re);
}

// exp(i x) and exp(z) in double precision (used by the per-segment response).
GWTF_HD inline cplx<double> expi(double x) { return cplx<double>(std::cos(x), std::sin(x)); }
GWTF_HD inline cplx<double> cexp(const cplx<double>& z)
{
    double m = std::exp(z.re);
    return cplx<double>(m * std::cos(z.im), m * std::sin(z.im));
}

} // namespace gwtf
