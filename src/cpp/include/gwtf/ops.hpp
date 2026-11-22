// The generic time-frequency pipeline shared by every model and backend.
//
//   source (analytic model or numerical tracks)  ->  ModeTrack per mode
//   response (TT / TDI2)                         ->  channel coefficients
//   window (box-car / Tukey Fresnel)             ->  per-bin kernel value
//   sink (waveform write / inner products)       ->  output
//
// Work is decomposed into one task per (source, time segment), or per
// (source, mode, segment) for the self-term multi-mode statistic. Each task is
// a "body" functor with a GWTF_HD operator()(int64_t idx); a backend only
// provides an executor with `launch(n, body)` (a loop on the CPU, a kernel
// launch on the GPU). Runtime options are turned into template parameters
// by the dispatch_* functions at the bottom.
#pragma once

#include <stdexcept>
#include <string>
#include <type_traits>

#include "gwtf/api.hpp"
#include "gwtf/fresnel.hpp"
#include "gwtf/models/registry.hpp"
#include "gwtf/response.hpp"

namespace gwtf {

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

// Analytic model M evaluated from its prepared parameter rows.
template <class M>
struct AnalyticSource {
    static constexpr int fused_capacity = M::n_modes;
    using State = typename M::State;
    const double* params;
    int32_t stride;

    GWTF_HD State load(int64_t s) const { return M::load(params + s * stride); }
    GWTF_HD int n_modes() const { return M::n_modes; }
    GWTF_HD bool track(const State& st, int64_t, int mode, int, double t, ModeTrack& tr) const
    {
        return M::track(st, mode, t, tr);
    }
    GWTF_HD void ylm(const State& st, int64_t, int mode, double cosi, cplx<double>& Yp,
                     cplx<double>& Yn) const
    {
        M::ylm(st, mode, cosi, Yp, Yn);
    }
};

// Model-agnostic multi-mode source: per-segment (amp, phi, f, fdot) supplied
// numerically, so any waveform model (or any array library) can drive the
// same kernels. Cap is the fused-statistic capacity (0 for waveforms).
template <int Cap>
struct NumericalSource {
    static constexpr int fused_capacity = Cap;
    struct State {};
    const double* tracks;      // (n_src, n_modes, nT, 4)
    const cplx<double>* ylms;  // (n_src, n_modes, 2)
    int32_t n_modes_;
    int32_t nT;

    GWTF_HD State load(int64_t) const { return {}; }
    GWTF_HD int n_modes() const { return n_modes_; }
    GWTF_HD bool track(const State&, int64_t s, int mode, int t_idx, double, ModeTrack& tr) const
    {
        const double* q = tracks + ((s * n_modes_ + mode) * int64_t(nT) + t_idx) * 4;
        tr.amp = q[0];
        tr.phi = q[1];
        tr.f = q[2];
        tr.fdot = q[3];
        return true;
    }
    GWTF_HD void ylm(const State&, int64_t s, int mode, double, cplx<double>& Yp,
                     cplx<double>& Yn) const
    {
        const cplx<double>* y = ylms + (s * n_modes_ + mode) * 2;
        Yp = y[0];
        Yn = y[1];
    }
};

// ---------------------------------------------------------------------------
// Per-(source, segment) evaluation
// ---------------------------------------------------------------------------

template <typename Real, class Src, class Resp, WindowKind WK>
struct SegmentEvaluator {
    static constexpr int NCH = Resp::n_channels;
    using SM = SegmentMode<Real, NCH>;

    Grid g;
    Src src;
    Resp resp;
    const double* response;  // (n_src, 4)
    const int32_t* seg_start;
    const int32_t* seg_end;
    cplx<Real> tukey_tp;     // 0.25 exp(i pi (1 - alpha) / alpha)
    double amp_scale;        // amplitude rescaling (see StatisticArgs)

    GWTF_HD bool in_band(int64_t s, int t) const { return t >= seg_start[s] && t <= seg_end[s]; }

    // Call f(SegmentMode) for every mode of source s contributing to segment t.
    template <class F>
    GWTF_HD void for_each_mode(int64_t s, int t_idx, F&& f) const
    {
        for_modes(s, t_idx, 0, src.n_modes(), f);
    }

    // As for_each_mode, restricted to modes [m_begin, m_end).
    template <class F>
    GWTF_HD void for_modes(int64_t s, int t_idx, int m_begin, int m_end, F&& f) const
    {
        const auto st = src.load(s);
        const ResponseParams rp = load_response_params(response + 4 * s);
        const auto rs = resp.source(rp);
        const auto rg = resp.segment(rs, t_idx);
        double t = g.dT * t_idx;
        if (g.midpoint)
            t += g.dT / 2;
        for (int m = m_begin; m < m_end; ++m) {
            ModeTrack tr;
            if (!src.track(st, s, m, t_idx, t, tr))
                continue;
            SM sm;
            if (!make_segment_mode<Real, NCH>(g, tr, amp_scale, sm))
                continue;
            cplx<double> Yp, Yn;
            src.ylm(st, s, m, rp.cosi, Yp, Yn);
            resp.coefficients(rs, rg, Yp, Yn, tr.f, sm.coef);
            f(sm);
        }
    }

    // Window kernel of one segment mode at absolute bin f_idx.
    GWTF_HD cplx<Real> kernel(const SM& sm, int f_idx) const
    {
        const int j = f_idx - sm.center;
        if constexpr (WK == WindowKind::Tukey) {
            return fresnel_tukey(sm, j, Real(g.dF), Real(g.dT), Real(g.tukey_alpha), tukey_tp);
        } else {
            return fresnel_boxcar(sm, j, Real(g.dF), Real(g.dT), g.midpoint != 0);
        }
    }
};

// ---------------------------------------------------------------------------
// Noise-weighted inner products at one (t, f) bin. Weights carry 4 dF.
// ---------------------------------------------------------------------------

template <typename Real, int NCH, NoiseKind NK>
struct NoiseView {
    const cplx<Real>* data;  // (nT, nF, NCH)
    const void* weights;

    // <d|h> and <h|h> contributions of the channel vector h at bin tf = t nF + f.
    GWTF_HD void accumulate(int64_t tf, const cplx<Real>* h, cplx<Real>& dh, Real& hh) const
    {
        const cplx<Real>* d = data + tf * NCH;
        if constexpr (NK == NoiseKind::Diagonal) {
            const Real* w = static_cast<const Real*>(weights) + tf * NCH;
            for (int c = 0; c < NCH; ++c) {
                dh += conj_mul(d[c], h[c]) * w[c];
                hh += norm2(h[c]) * w[c];
            }
        } else {
            const cplx<Real>* W = static_cast<const cplx<Real>*>(weights) + tf * NCH * NCH;
            for (int i = 0; i < NCH; ++i) {
                cplx<Real> Wh;
                for (int j = 0; j < NCH; ++j)
                    Wh += W[i * NCH + j] * h[j];
                dh += conj_mul(d[i], Wh);
                hh += conj_mul(h[i], Wh).re;  // real for Hermitian invC
            }
        }
    }
};

// out += v, atomically on the device (the host backend is serial).
template <typename T>
GWTF_HD inline void accumulate_out(T* out, T v)
{
#if defined(GWTF_DEVICE_PASS)
    atomicAdd(out, v);
#else
    *out += v;
#endif
}

// ---------------------------------------------------------------------------
// Task bodies: one call per (source, segment), idx = s * nT + t.
// ---------------------------------------------------------------------------

template <class Ev, typename Real>
struct WaveformBody {
    static constexpr int NCH = Ev::NCH;
    Ev ev;
    cplx<Real>* out;  // (n_src, nT, nF, NCH)

    GWTF_HD void operator()(int64_t idx) const
    {
        const int nT = ev.g.nT;
        const int64_t s = idx / nT;
        const int t = int(idx - s * nT);
        if (!ev.in_band(s, t))
            return;
        cplx<Real>* row = out + (s * nT + t) * int64_t(ev.g.nF) * NCH;
        ev.for_each_mode(s, t, [&](const typename Ev::SM& sm) {
            for (int f = sm.f_lo; f <= sm.f_hi; ++f) {
                const cplx<Real> F = ev.kernel(sm, f);
                for (int c = 0; c < NCH; ++c)
                    row[f * NCH + c] += sm.coef[c] * F;
            }
        });
    }
};

template <class Ev, typename Real, NoiseKind NK, int Cap>
struct StatisticBody {
    static_assert(Cap <= 256, "mode order is held in uint8_t");
    static constexpr int NCH = Ev::NCH;
    using SM = typename Ev::SM;
    Ev ev;
    NoiseView<Real, NCH, NK> nv;
    cplx<Real>* out;  // (n_src, nT, 2)

    GWTF_HD void self_terms(const SM& sm, int t, cplx<Real>& dh, Real& hh) const
    {
        const int64_t tf0 = int64_t(t) * ev.g.nF;
        for (int f = sm.f_lo; f <= sm.f_hi; ++f) {
            const cplx<Real> F = ev.kernel(sm, f);
            cplx<Real> h[NCH];
            for (int c = 0; c < NCH; ++c)
                h[c] = sm.coef[c] * F;
            nv.accumulate(tf0 + f, h, dh, hh);
        }
    }

    GWTF_HD void operator()(int64_t idx) const
    {
        const int nT = ev.g.nT;
        const int64_t s = idx / nT;
        const int t = int(idx - s * nT);
        cplx<Real> dh;
        Real hh = Real(0);
        if (ev.in_band(s, t)) {
            if constexpr (Cap <= 1) {
                ev.for_each_mode(s, t, [&](const SM& sm) { self_terms(sm, t, dh, hh); });
            } else {
                // Multi-mode, cross terms included: the inner products are taken of
                // the summed signal H(f) = sum_j h_j(f), since |H|^2 contains every
                // 2 Re conj(h_i) h_j. Each mode occupies a contiguous bin range, so
                // the modes are sorted by f_lo and grouped into clusters of
                // overlapping ranges; isolated modes take the self-term loop, and
                // each cluster is swept bin by bin, evaluating every kernel once.
                SM modes[Cap];
                uint8_t order[Cap];
                int n = 0;
                ev.for_each_mode(s, t, [&](const SM& sm) {
                    if (n < Cap) {
                        // Insertion by f_lo (n <= Cap is small).
                        int k = n;
                        while (k > 0 && modes[order[k - 1]].f_lo > sm.f_lo) {
                            order[k] = order[k - 1];
                            --k;
                        }
                        order[k] = uint8_t(n);
                        modes[n++] = sm;
                    }
                });
                const int64_t tf0 = int64_t(t) * ev.g.nF;
                int k0 = 0;
                while (k0 < n) {
                    int k1 = k0 + 1;
                    int hi = modes[order[k0]].f_hi;
                    while (k1 < n && modes[order[k1]].f_lo <= hi) {
                        const int h1 = modes[order[k1]].f_hi;
                        hi = h1 > hi ? h1 : hi;
                        ++k1;
                    }
                    if (k1 == k0 + 1) {
                        self_terms(modes[order[k0]], t, dh, hh);
                    } else {
                        for (int f = modes[order[k0]].f_lo; f <= hi; ++f) {
                            cplx<Real> H[NCH];
                            for (int k = k0; k < k1; ++k) {
                                const SM& sm = modes[order[k]];
                                if (f < sm.f_lo || f > sm.f_hi)
                                    continue;
                                const cplx<Real> F = ev.kernel(sm, f);
                                for (int c = 0; c < NCH; ++c)
                                    H[c] += sm.coef[c] * F;
                            }
                            nv.accumulate(tf0 + f, H, dh, hh);
                        }
                    }
                    k0 = k1;
                }
            }
        }
        out[idx * 2 + 0] = dh;
        out[idx * 2 + 1] = cplx<Real>(hh, Real(0));
    }
};

// Self-term statistic with one task per (source, mode, segment),
// idx = (s * n_modes + m) * nT + t (the layout of the tracks, so neighbouring
// threads read neighbouring track entries and add into distinct outputs).
// Modes of one (source, segment) are summed into `out` (zeroed beforehand)
// with atomics, so the result is <d|h> and sum_m <h_m|h_m>.
template <class Ev, typename Real, NoiseKind NK>
struct ModeStatisticBody {
    Ev ev;
    NoiseView<Real, Ev::NCH, NK> nv;
    cplx<Real>* out;  // (n_src, nT, 2)
    int32_t n_modes;

    GWTF_HD void operator()(int64_t idx) const
    {
        const int nT = ev.g.nT;
        const int64_t sm_idx = idx / nT;
        const int t = int(idx - sm_idx * nT);
        const int64_t s = sm_idx / n_modes;
        const int m = int(sm_idx - s * n_modes);
        if (!ev.in_band(s, t))
            return;
        cplx<Real> dh;
        Real hh = Real(0);
        bool any = false;
        ev.for_modes(s, t, m, m + 1, [&](const typename Ev::SM& sm) {
            const int64_t tf0 = int64_t(t) * ev.g.nF;
            for (int f = sm.f_lo; f <= sm.f_hi; ++f) {
                const cplx<Real> F = ev.kernel(sm, f);
                cplx<Real> h[Ev::NCH];
                for (int c = 0; c < Ev::NCH; ++c)
                    h[c] = sm.coef[c] * F;
                nv.accumulate(tf0 + f, h, dh, hh);
            }
            any = true;
        });
        if (!any)
            return;
        Real* o = reinterpret_cast<Real*>(out + (s * nT + t) * 2);
        accumulate_out(o + 0, dh.re);
        accumulate_out(o + 1, dh.im);
        accumulate_out(o + 2, hh);
    }
};

// Dense inner products of a full time-frequency waveform array.
template <typename Real, int NCH, NoiseKind NK>
struct InnerProductBody {
    NoiseView<Real, NCH, NK> nv;
    const cplx<Real>* h;  // (n_src, nT, nF, NCH)
    int32_t nT, nF;
    cplx<Real>* out;

    GWTF_HD void operator()(int64_t idx) const
    {
        const int64_t s = idx / nT;
        const int t = int(idx - s * nT);
        cplx<Real> dh;
        Real hh = Real(0);
        const cplx<Real>* row = h + idx * int64_t(nF) * NCH;
        for (int f = 0; f < nF; ++f)
            nv.accumulate(int64_t(t) * nF + f, row + f * NCH, dh, hh);
        out[idx * 2 + 0] = dh;
        out[idx * 2 + 1] = cplx<Real>(hh, Real(0));
    }
};

// Segment range from time bounds, reproducing v1's floor/clip rules
// (generator._compute_segment_indices), with NaN and overflow made explicit.
GWTF_HD inline void segment_range(const Grid& g, bool has_bounds, double t0, double t1,
                                  int32_t& s0, int32_t& s1)
{
    if (!has_bounds) {
        s0 = 0;
        s1 = g.nT - 1;
        return;
    }
    double a = std::floor(t0 / g.dT);
    if (!(a >= 0.0))
        a = 0.0;  // also NaN
    if (a > g.nT - 1)
        a = g.nT - 1;
    // A segment that only contains the f_max crossing is dropped.
    double b = g.midpoint ? std::floor(t1 / g.dT - 0.5) : std::floor(t1 / g.dT) - 1.0;
    if (!(b >= -1.0))
        b = -1.0;  // also NaN: no segments
    if (b > g.nT - 1)
        b = g.nT - 1;
    s0 = int32_t(a);
    s1 = int32_t(b);
}

template <class M>
struct PrepareBody {
    PrepareArgs a;

    GWTF_HD void operator()(int64_t s) const
    {
        double* p = a.params + s * a.stride;
        const double* r = a.raw + s * a.n_raw;
        for (int i = 0; i < a.n_raw; ++i)
            p[i] = r[i];
        for (int i = a.n_raw; i < a.stride; ++i)
            p[i] = 0.0;
        M::compute_derived(p);
        double t0 = 0.0, t1 = 0.0;
        const bool bounded = M::time_bounds(p, a.f_lo, a.f_hi, t0, t1);
        segment_range(a.grid, bounded, t0, t1, a.seg_start[s], a.seg_end[s]);
    }
};

template <class M>
struct TrackBody {
    TrackArgs a;

    GWTF_HD void operator()(int64_t idx) const
    {
        const int nT = a.grid.nT;
        const int64_t s = idx / nT;
        const int t_idx = int(idx - s * nT);
        const auto st = M::load(a.params + s * a.stride);
        double t = a.grid.dT * t_idx;
        if (a.grid.midpoint)
            t += a.grid.dT / 2;
        for (int m = 0; m < M::n_modes; ++m) {
            ModeTrack tr;
            double* o = a.out + ((s * M::n_modes + m) * int64_t(nT) + t_idx) * 4;
            if (M::track(st, m, t, tr)) {
                o[0] = tr.amp;
                o[1] = tr.phi;
                o[2] = tr.f;
                o[3] = tr.fdot;
            } else {
                o[0] = o[1] = o[2] = o[3] = 0.0;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Runtime -> compile-time dispatch
// ---------------------------------------------------------------------------

template <typename T>
struct type_tag {
    using type = T;
};
template <auto V>
struct value_tag {
    static constexpr auto value = V;
};

template <class F>
void with_precision(Precision p, F&& f)
{
    if (p == Precision::Single)
        f(type_tag<float>{});
    else if (p == Precision::Double)
        f(type_tag<double>{});
    else
        throw std::invalid_argument("pygwtf: unknown precision");
}

template <typename Real, class F>
void with_response(const DetectorArgs& d, F&& f)
{
    if (d.kind == ResponseKind::TT) {
        f(TTResponse<Real>{});
    } else if (d.kind == ResponseKind::TDI2) {
        if (!d.orbits || !d.ltts)
            throw std::invalid_argument("pygwtf: TDI response needs orbits and arm lengths");
        f(TDI2Response<Real>{static_cast<const Real*>(d.orbits), static_cast<const Real*>(d.ltts)});
    } else {
        throw std::invalid_argument("pygwtf: unknown response kind");
    }
}

template <class F>
void with_window(WindowKind w, const Grid& g, F&& f)
{
    if (w == WindowKind::Boxcar) {
        f(value_tag<WindowKind::Boxcar>{});
    } else if (w == WindowKind::Tukey) {
        if (!g.midpoint)
            throw std::invalid_argument("pygwtf: the Tukey window requires use_midpoint=True");
        if (!(g.tukey_alpha > 0.0 && g.tukey_alpha <= 1.0))
            throw std::invalid_argument("pygwtf: Tukey alpha must be in (0, 1]");
        f(value_tag<WindowKind::Tukey>{});
    } else {
        throw std::invalid_argument("pygwtf: unknown window kind");
    }
}

template <class F>
void with_noise(NoiseKind n, F&& f)
{
    if (n == NoiseKind::Diagonal)
        f(value_tag<NoiseKind::Diagonal>{});
    else if (n == NoiseKind::Full)
        f(value_tag<NoiseKind::Full>{});
    else
        throw std::invalid_argument("pygwtf: unknown noise kind");
}

// Calls f(type_tag<Model>) for an analytic model id.
template <class F>
void with_analytic_model(int32_t id, F&& f)
{
    switch (id) {
#define GWTF_CASE(ID, TYPE) \
    case ID:                \
        f(type_tag<TYPE>{}); \
        return;
        GWTF_ANALYTIC_MODELS(GWTF_CASE)
#undef GWTF_CASE
    default:
        throw std::invalid_argument("pygwtf: unknown analytic model id " + std::to_string(id));
    }
}

// Source object for a waveform call (any number of numerical modes).
template <class F>
void with_waveform_source(const SourceArgs& s, const Grid& g, F&& f)
{
    if (s.model == numerical_model_id) {
        f(NumericalSource<0>{s.tracks, static_cast<const cplx<double>*>(s.ylms), s.n_modes, g.nT});
    } else {
        with_analytic_model(s.model, [&](auto mt) {
            using M = typename decltype(mt)::type;
            f(AnalyticSource<M>{s.params, s.param_stride});
        });
    }
}

// Source object for a statistic call: numerical sources are capped at
// max_fused_modes modes (cross terms are held in local memory).
template <class F>
void with_statistic_source(const SourceArgs& s, const Grid& g, F&& f)
{
    if (s.model == numerical_model_id) {
        const auto* y = static_cast<const cplx<double>*>(s.ylms);
        if (s.n_modes <= 1)
            f(NumericalSource<1>{s.tracks, y, s.n_modes, g.nT});
        else if (s.n_modes <= max_fused_modes)
            f(NumericalSource<max_fused_modes>{s.tracks, y, s.n_modes, g.nT});
        else
            throw std::invalid_argument("pygwtf: fused statistic supports at most "
                                        + std::to_string(max_fused_modes) + " modes");
    } else {
        with_waveform_source(s, g, f);
    }
}

template <class Exec>
void dispatch_waveform(Exec& ex, const WaveformArgs& a)
{
    with_precision(a.precision, [&](auto rt) {
        using Real = typename decltype(rt)::type;
        with_response<Real>(a.det, [&](auto resp) {
            using Resp = decltype(resp);
            with_window(a.window, a.grid, [&](auto wt) {
                constexpr WindowKind WK = decltype(wt)::value;
                with_waveform_source(a.src, a.grid, [&](auto src) {
                    using Src = decltype(src);
                    using Ev = SegmentEvaluator<Real, Src, Resp, WK>;
                    Ev ev{a.grid, src, resp, a.src.response, a.src.seg_start, a.src.seg_end,
                          tukey_taper_phase<Real>(a.window == WindowKind::Tukey ? a.grid.tukey_alpha : 1.0),
                          a.amp_scale};
                    WaveformBody<Ev, Real> body{ev, static_cast<cplx<Real>*>(a.out)};
                    ex.launch(a.src.n_src * a.grid.nT, body);
                });
            });
        });
    });
}

template <class Exec>
void dispatch_statistic(Exec& ex, const StatisticArgs& a)
{
    with_precision(a.precision, [&](auto rt) {
        using Real = typename decltype(rt)::type;
        with_response<Real>(a.det, [&](auto resp) {
            using Resp = decltype(resp);
            with_window(a.window, a.grid, [&](auto wt) {
                constexpr WindowKind WK = decltype(wt)::value;
                with_noise(a.noise, [&](auto nt) {
                    constexpr NoiseKind NK = decltype(nt)::value;
                    auto* out = static_cast<cplx<Real>*>(a.out);
                    auto make_ev = [&](auto src) {
                        return SegmentEvaluator<Real, decltype(src), Resp, WK>{
                            a.grid, src, resp, a.src.response, a.src.seg_start, a.src.seg_end,
                            tukey_taper_phase<Real>(a.window == WindowKind::Tukey ? a.grid.tukey_alpha : 1.0),
                            a.amp_scale};
                    };
                    const bool self_terms_only = a.src.model == numerical_model_id
                                                 && a.src.n_modes > 1
                                                 && a.mode != StatisticMode::CrossTerms;
                    if (self_terms_only) {
                        // Capacity 0: StatisticBody takes the self-term branch.
                        const auto ev = make_ev(NumericalSource<0>{
                            a.src.tracks, static_cast<const cplx<double>*>(a.src.ylms),
                            a.src.n_modes, a.grid.nT});
                        using Ev = std::remove_const_t<decltype(ev)>;
                        NoiseView<Real, Ev::NCH, NK> nv{static_cast<const cplx<Real>*>(a.data), a.weights};
                        if (a.mode == StatisticMode::ModeParallel) {
                            ex.zero(a.out, size_t(a.src.n_src) * size_t(a.grid.nT) * 2 * sizeof(cplx<Real>));
                            ModeStatisticBody<Ev, Real, NK> body{ev, nv, out, a.src.n_modes};
                            ex.launch(a.src.n_src * a.src.n_modes * a.grid.nT, body);
                        } else if (a.mode == StatisticMode::SegmentLoop) {
                            StatisticBody<Ev, Real, NK, 0> body{ev, nv, out};
                            ex.launch(a.src.n_src * a.grid.nT, body);
                        } else {
                            throw std::invalid_argument("pygwtf: unknown statistic mode");
                        }
                        return;
                    }
                    with_statistic_source(a.src, a.grid, [&](auto src) {
                        using Src = decltype(src);
                        const auto ev = make_ev(src);
                        using Ev = std::remove_const_t<decltype(ev)>;
                        NoiseView<Real, Ev::NCH, NK> nv{static_cast<const cplx<Real>*>(a.data), a.weights};
                        StatisticBody<Ev, Real, NK, Src::fused_capacity> body{ev, nv, out};
                        ex.launch(a.src.n_src * a.grid.nT, body);
                    });
                });
            });
        });
    });
}

template <class Exec>
void dispatch_inner_product(Exec& ex, const InnerProductArgs& a)
{
    with_precision(a.precision, [&](auto rt) {
        using Real = typename decltype(rt)::type;
        with_noise(a.noise, [&](auto nt) {
            constexpr NoiseKind NK = decltype(nt)::value;
            auto go = [&](auto nch) {
                constexpr int NCH = decltype(nch)::value;
                NoiseView<Real, NCH, NK> nv{static_cast<const cplx<Real>*>(a.data), a.weights};
                InnerProductBody<Real, NCH, NK> body{nv, static_cast<const cplx<Real>*>(a.h), a.nT,
                                                     a.nF, static_cast<cplx<Real>*>(a.out)};
                ex.launch(a.n_src * a.nT, body);
            };
            switch (a.n_channels) {
            case 1: go(value_tag<1>{}); break;
            case 2: go(value_tag<2>{}); break;
            case 3: go(value_tag<3>{}); break;
            default: throw std::invalid_argument("pygwtf: inner_product supports 1-3 channels");
            }
        });
    });
}

template <class Exec>
void dispatch_prepare(Exec& ex, const PrepareArgs& a)
{
    with_analytic_model(a.model, [&](auto mt) {
        using M = typename decltype(mt)::type;
        if (a.n_raw != M::n_params || a.stride != M::n_params + M::n_derived)
            throw std::invalid_argument(std::string("pygwtf: parameter count mismatch for ") + M::name);
        ex.launch(a.n_src, PrepareBody<M>{a});
    });
}

template <class Exec>
void dispatch_tracks(Exec& ex, const TrackArgs& a)
{
    with_analytic_model(a.model, [&](auto mt) {
        using M = typename decltype(mt)::type;
        ex.launch(a.n_src * a.grid.nT, TrackBody<M>{a});
    });
}

} // namespace gwtf
