// nanobind bindings, compiled twice: into pygwtf._core_cpu (host backend)
// and, with GWTF_GPU_MODULE defined, into pygwtf._core_gpu.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <string>
#include <vector>

#include "gwtf/api.hpp"
#include "gwtf/models/registry.hpp"

namespace nb = nanobind;
using namespace gwtf;

namespace {

using In = nb::ndarray<nb::ro>;  // read-only inputs (accepts writable ones too)
using Out = nb::ndarray<>;

#if defined(GWTF_GPU_MODULE)
#if defined(GWTF_HIP)
constexpr const char* backend_name = "hip";
#else
constexpr const char* backend_name = "cuda";
#endif
bool device_ok(int32_t t)
{
#if defined(GWTF_HIP)
    return t == nb::device::rocm::value;
#else
    return t == nb::device::cuda::value || t == nb::device::cuda_managed::value;
#endif
}
#else
constexpr const char* backend_name = "cpu";
bool device_ok(int32_t t) { return t == nb::device::cpu::value; }
#endif

enum class DT { F64, F32, C128, C64, I32, Real, Complex };

nb::dlpack::dtype dtype_of(DT d, Precision p)
{
    using code = nb::dlpack::dtype_code;
    auto mk = [](code c, int bits) {
        nb::dlpack::dtype t;
        t.code = uint8_t(c);
        t.bits = uint8_t(bits);
        t.lanes = 1;
        return t;
    };
    switch (d) {
    case DT::F64: return mk(code::Float, 64);
    case DT::F32: return mk(code::Float, 32);
    case DT::C128: return mk(code::Complex, 128);
    case DT::C64: return mk(code::Complex, 64);
    case DT::I32: return mk(code::Int, 32);
    case DT::Real: return mk(code::Float, p == Precision::Single ? 32 : 64);
    case DT::Complex: return mk(code::Complex, p == Precision::Single ? 64 : 128);
    }
    return mk(code::Float, 64);
}

std::string shape_str(const std::vector<int64_t>& s)
{
    std::string r = "(";
    for (size_t i = 0; i < s.size(); ++i) {
        r += s[i] < 0 ? std::string("*") : std::to_string(s[i]);
        if (i + 1 < s.size())
            r += ", ";
    }
    return r + ")";
}

// Validate an array (device, dtype, shape with -1 wildcards, C-contiguity,
// alignment) and return its data pointer. Invalid (None) arrays are only
// allowed when `optional` is set and then yield nullptr.
template <class A>
void* check(const A& a, const char* name, DT dt, Precision prec, std::vector<int64_t> shape,
            bool optional = false)
{
    if (!a.is_valid()) {
        if (optional)
            return nullptr;
        throw nb::value_error((std::string(name) + " is required").c_str());
    }
    if (!device_ok(a.device_type()))
        throw nb::value_error((std::string(name) + ": array is on the wrong device for the "
                               + backend_name + " backend").c_str());
    const auto want = dtype_of(dt, prec);
    if (a.dtype() != want)
        throw nb::value_error((std::string(name) + ": unexpected dtype").c_str());
    if (a.ndim() != shape.size())
        throw nb::value_error((std::string(name) + ": expected shape " + shape_str(shape)).c_str());
    int64_t expected_stride = 1;
    for (int i = int(a.ndim()) - 1; i >= 0; --i) {
        if (shape[i] >= 0 && int64_t(a.shape(i)) != shape[i])
            throw nb::value_error((std::string(name) + ": expected shape " + shape_str(shape)).c_str());
        if (a.shape(i) > 1 && a.stride(i) != expected_stride)
            throw nb::value_error((std::string(name) + ": array must be C-contiguous").c_str());
        expected_stride *= int64_t(a.shape(i));
    }
    const auto align = uintptr_t(want.bits / 8);
    if (reinterpret_cast<uintptr_t>(a.data()) % align != 0)
        throw nb::value_error((std::string(name) + ": array is misaligned").c_str());
    return const_cast<void*>(static_cast<const void*>(a.data()));
}

void* stream_ptr(uintptr_t s) { return reinterpret_cast<void*>(s); }

int32_t n_channels_of(ResponseKind k) { return k == ResponseKind::TDI2 ? 3 : 2; }

struct ModelInfo {
    int32_t id;
    std::string name;
    int32_t n_params, n_derived, n_modes;
};

std::vector<ModelInfo> model_list()
{
    std::vector<ModelInfo> v;
#define GWTF_INFO(ID, TYPE) v.push_back({ID, TYPE::name, TYPE::n_params, TYPE::n_derived, TYPE::n_modes});
    GWTF_ANALYTIC_MODELS(GWTF_INFO)
#undef GWTF_INFO
    return v;
}

int32_t model_modes(int32_t id)
{
    for (const auto& m : model_list())
        if (m.id == id)
            return m.n_modes;
    throw nb::value_error(("unknown model id " + std::to_string(id)).c_str());
}

// Source block shared by waveform and statistic calls.
SourceArgs make_sources(const Grid& g, int32_t model, int64_t n_src, const In& params,
                        const In& tracks, const In& ylms, const In& seg_start, const In& seg_end,
                        const In& response)
{
    SourceArgs s{};
    s.model = model;
    s.n_src = n_src;
    if (model == numerical_model_id) {
        if (!tracks.is_valid() || tracks.ndim() != 4)
            throw nb::value_error("tracks must have shape (n_src, n_modes, nT, 4)");
        s.n_modes = int32_t(tracks.shape(1));
        s.tracks = static_cast<const double*>(
            check(tracks, "tracks", DT::F64, Precision::Double, {n_src, s.n_modes, g.nT, 4}));
        s.ylms = check(ylms, "ylms", DT::C128, Precision::Double, {n_src, s.n_modes, 2});
    } else {
        if (!params.is_valid() || params.ndim() != 2)
            throw nb::value_error("params must have shape (n_src, n_params + n_derived)");
        s.param_stride = int32_t(params.shape(1));
        s.params = static_cast<const double*>(
            check(params, "params", DT::F64, Precision::Double, {n_src, s.param_stride}));
        s.n_modes = model_modes(model);
    }
    s.seg_start = static_cast<const int32_t*>(check(seg_start, "seg_start", DT::I32, Precision::Double, {n_src}));
    s.seg_end = static_cast<const int32_t*>(check(seg_end, "seg_end", DT::I32, Precision::Double, {n_src}));
    s.response = static_cast<const double*>(check(response, "response", DT::F64, Precision::Double, {n_src, 4}));
    return s;
}

DetectorArgs make_detector(const Grid& g, int32_t kind, Precision p, const In& orbits, const In& ltts)
{
    DetectorArgs d{};
    d.kind = ResponseKind(kind);
    const bool need = d.kind == ResponseKind::TDI2;
    d.orbits = check(orbits, "orbits", DT::Real, p, {g.nT, 3, 3}, !need);
    d.ltts = check(ltts, "ltts", DT::Real, p, {g.nT, 3}, !need);
    return d;
}

// Data/noise block shared by statistic and inner_product calls. `data` and
// `weights` are each (nT, nF, nch[, nch]) shared by all sources, or carry a
// leading axis of n_data copies selected per source by `data_index`.
NoiseArgs make_noise(int32_t noise, Precision p, int64_t n_src, int64_t nT, int64_t nF, int64_t nch,
                     const In& data, const In& weights, const In& data_index, double scale)
{
    NoiseArgs n{};
    if (noise != int32_t(NoiseKind::Diagonal) && noise != int32_t(NoiseKind::Full))
        throw nb::value_error("unknown noise kind");
    n.kind = NoiseKind(noise);
    n.scale = scale;
    const bool full = n.kind == NoiseKind::Full;
    const size_t data_rank = 3, weight_rank = full ? 4 : 3;
    n.data_batched = data.is_valid() && data.ndim() == data_rank + 1;
    n.weights_batched = weights.is_valid() && weights.ndim() == weight_rank + 1;
    n.n_data = 1;
    if (n.data_batched || n.weights_batched) {
        const int64_t nd = n.data_batched ? int64_t(data.shape(0)) : int64_t(weights.shape(0));
        if (n.data_batched && n.weights_batched && int64_t(weights.shape(0)) != nd)
            throw nb::value_error("data and weights have different numbers of copies along the leading axis");
        if (nd < 1 || nd > INT32_MAX)
            throw nb::value_error("invalid number of data copies");
        n.n_data = int32_t(nd);
    }
    auto lead = [&](bool batched, std::vector<int64_t> shape) {
        if (batched)
            shape.insert(shape.begin(), n.n_data);
        return shape;
    };
    n.data = check(data, "data", DT::Complex, p, lead(n.data_batched, {nT, nF, nch}));
    if (full)
        n.weights = check(weights, "weights", DT::Complex, p, lead(n.weights_batched, {nT, nF, nch, nch}));
    else
        n.weights = check(weights, "weights", DT::Real, p, lead(n.weights_batched, {nT, nF, nch}));
    if (data_index.is_valid()) {
        n.index = static_cast<const int32_t*>(check(data_index, "data_index", DT::I32, p, {n_src}));
    } else if (n.data_batched || n.weights_batched) {
        throw nb::value_error("batched data/weights need data_index (the data copy of each source)");
    }
    return n;
}

} // namespace

#if defined(GWTF_GPU_MODULE)
NB_MODULE(_core_gpu, m)
#else
NB_MODULE(_core_cpu, m)
#endif
{
    m.doc() = "pygwtf v2 compiled backend";
    m.attr("backend") = backend_name;
    m.attr("max_fused_modes") = max_fused_modes;
    m.attr("numerical_model_id") = numerical_model_id;
    nb::dict stat_modes;
    stat_modes["cross_terms"] = int32_t(StatisticMode::CrossTerms);
    stat_modes["segment_loop"] = int32_t(StatisticMode::SegmentLoop);
    stat_modes["mode_parallel"] = int32_t(StatisticMode::ModeParallel);
    m.attr("statistic_modes") = stat_modes;
    nb::dict splits;
    splits["spread"] = int32_t(SegmentSplit::Spread);
    splits["truncate"] = int32_t(SegmentSplit::Truncate);
    m.attr("segment_splits") = splits;
    nb::dict consts;
    consts["MTsun"] = constants::MTsun;
    consts["pc"] = constants::pc;
    consts["clight"] = constants::clight;
    consts["gamma_e"] = constants::gamma_e;
    m.attr("constants") = consts;
    m.attr("pol_convention") = pol_convention;

    nb::class_<Grid>(m, "Grid")
        .def(nb::init<>())
        .def_rw("nT", &Grid::nT)
        .def_rw("nF", &Grid::nF)
        .def_rw("kernel_width", &Grid::kernel_width)
        .def_rw("midpoint", &Grid::midpoint)
        .def_rw("dT", &Grid::dT)
        .def_rw("dF", &Grid::dF)
        .def_rw("fmin", &Grid::fmin)
        .def_rw("tukey_alpha", &Grid::tukey_alpha);

    nb::class_<ModelInfo>(m, "ModelInfo")
        .def_ro("id", &ModelInfo::id)
        .def_ro("name", &ModelInfo::name)
        .def_ro("n_params", &ModelInfo::n_params)
        .def_ro("n_derived", &ModelInfo::n_derived)
        .def_ro("n_modes", &ModelInfo::n_modes);
    m.def("models", &model_list, "Compiled analytic models.");

    m.def(
        "prepare",
        [](int32_t model, const In& raw, Out params, Out seg_start, Out seg_end, const Grid& g,
           double f_lo, double f_hi, uintptr_t stream) {
            PrepareArgs a{};
            a.model = model;
            a.n_src = raw.ndim() == 2 ? int64_t(raw.shape(0)) : 0;
            a.n_raw = raw.ndim() == 2 ? int32_t(raw.shape(1)) : 0;
            a.raw = static_cast<const double*>(check(raw, "raw", DT::F64, Precision::Double, {-1, -1}));
            a.stride = params.ndim() == 2 ? int32_t(params.shape(1)) : 0;
            a.params = static_cast<double*>(check(params, "params", DT::F64, Precision::Double, {a.n_src, -1}));
            a.seg_start = static_cast<int32_t*>(check(seg_start, "seg_start", DT::I32, Precision::Double, {a.n_src}));
            a.seg_end = static_cast<int32_t*>(check(seg_end, "seg_end", DT::I32, Precision::Double, {a.n_src}));
            a.grid = g;
            a.f_lo = f_lo;
            a.f_hi = f_hi;
            a.stream = stream_ptr(stream);
            nb::gil_scoped_release release;
            backend::run_prepare(a);
        },
        nb::arg("model"), nb::arg("raw"), nb::arg("params"), nb::arg("seg_start"),
        nb::arg("seg_end"), nb::arg("grid"), nb::arg("f_lo"), nb::arg("f_hi"), nb::arg("stream") = 0,
        "Convert raw parameters to dimensionful + derived parameters and compute "
        "per-source segment ranges.");

    m.def(
        "waveform",
        [](const Grid& g, int32_t model, int64_t n_src, const In& params, const In& tracks,
           const In& ylms, const In& seg_start, const In& seg_end, const In& response,
           int32_t response_kind, const In& orbits, const In& ltts, int32_t window,
           int32_t precision, Out out, double amp_scale, uintptr_t stream) {
            WaveformArgs a{};
            a.grid = g;
            a.amp_scale = amp_scale;
            a.precision = Precision(precision);
            a.window = WindowKind(window);
            a.src = make_sources(g, model, n_src, params, tracks, ylms, seg_start, seg_end, response);
            a.det = make_detector(g, response_kind, a.precision, orbits, ltts);
            a.out = check(out, "out", DT::Complex, a.precision,
                          {n_src, g.nT, g.nF, n_channels_of(a.det.kind)});
            a.stream = stream_ptr(stream);
            nb::gil_scoped_release release;
            backend::run_waveform(a);
        },
        nb::arg("grid"), nb::arg("model"), nb::arg("n_src"), nb::arg("params").none(),
        nb::arg("tracks").none(), nb::arg("ylms").none(), nb::arg("seg_start"),
        nb::arg("seg_end"), nb::arg("response"), nb::arg("response_kind"),
        nb::arg("orbits").none(), nb::arg("ltts").none(), nb::arg("window"),
        nb::arg("precision"), nb::arg("out"), nb::arg("amp_scale") = 1.0, nb::arg("stream") = 0,
        "Accumulate time-frequency waveforms into `out` (must be zeroed).");

    m.def(
        "statistic",
        [](const Grid& g, int32_t model, int64_t n_src, const In& params, const In& tracks,
           const In& ylms, const In& seg_start, const In& seg_end, const In& response,
           int32_t response_kind, const In& orbits, const In& ltts, int32_t window,
           int32_t precision, int32_t noise, const In& data, const In& weights, Out out,
           double scale, uintptr_t stream, int32_t mode, const In& data_index) {
            StatisticArgs a{};
            a.grid = g;
            if (mode < 0 || mode > int32_t(StatisticMode::ModeParallel))
                throw nb::value_error("unknown statistic mode");
            a.mode = StatisticMode(mode);
            a.precision = Precision(precision);
            a.window = WindowKind(window);
            a.src = make_sources(g, model, n_src, params, tracks, ylms, seg_start, seg_end, response);
            a.det = make_detector(g, response_kind, a.precision, orbits, ltts);
            a.noise = make_noise(noise, a.precision, n_src, g.nT, g.nF, n_channels_of(a.det.kind),
                                 data, weights, data_index, scale);
            a.out = check(out, "out", DT::Complex, a.precision, {n_src, g.nT, 2});
            a.stream = stream_ptr(stream);
            nb::gil_scoped_release release;
            backend::run_statistic(a);
        },
        nb::arg("grid"), nb::arg("model"), nb::arg("n_src"), nb::arg("params").none(),
        nb::arg("tracks").none(), nb::arg("ylms").none(), nb::arg("seg_start"),
        nb::arg("seg_end"), nb::arg("response"), nb::arg("response_kind"),
        nb::arg("orbits").none(), nb::arg("ltts").none(), nb::arg("window"),
        nb::arg("precision"), nb::arg("noise"), nb::arg("data"), nb::arg("weights"),
        nb::arg("out"), nb::arg("scale") = 1.0, nb::arg("stream") = 0,
        nb::arg("mode") = 0, nb::arg("data_index").none() = nb::none(),
        "Per-segment <d|h> and <h|h> without materialising the waveform.\n\n"
        "data/weights may carry a leading axis of per-walker copies, selected per\n"
        "source by data_index (global-fit mode). scale: single-precision rescaling\n"
        "(weights passed / scale^2; amplitudes and data multiplied by scale).\n"
        "mode (multi-mode numerical sources only): see statistic_modes.");

    m.def(
        "inner_product",
        [](const In& h, const In& data, const In& weights, int32_t precision, int32_t noise,
           Out out, uintptr_t stream, const In& data_index, double scale) {
            InnerProductArgs a{};
            a.precision = Precision(precision);
            if (!h.is_valid() || h.ndim() != 4)
                throw nb::value_error("h must have shape (n_src, nT, nF, n_channels)");
            a.n_src = int64_t(h.shape(0));
            a.nT = int32_t(h.shape(1));
            a.nF = int32_t(h.shape(2));
            a.n_channels = int32_t(h.shape(3));
            const int64_t c = a.n_channels;
            a.h = check(h, "h", DT::Complex, a.precision, {a.n_src, a.nT, a.nF, c});
            a.noise = make_noise(noise, a.precision, a.n_src, a.nT, a.nF, c, data, weights, data_index, scale);
            a.out = check(out, "out", DT::Complex, a.precision, {a.n_src, a.nT, 2});
            a.stream = stream_ptr(stream);
            nb::gil_scoped_release release;
            backend::run_inner_product(a);
        },
        nb::arg("h"), nb::arg("data"), nb::arg("weights"), nb::arg("precision"),
        nb::arg("noise"), nb::arg("out"), nb::arg("stream") = 0,
        nb::arg("data_index").none() = nb::none(), nb::arg("scale") = 1.0,
        "Per-segment <d|h>, <h|h> of dense time-frequency waveforms (data/weights/\n"
        "data_index/scale as for statistic; h is multiplied by scale on load).");

    m.def(
        "semicoherent",
        [](const In& stat, const In& seg_start, const In& seg_end, int32_t n_seg, int32_t precision,
           int32_t split, Out out, uintptr_t stream) {
            SemiCoherentArgs a{};
            a.precision = Precision(precision);
            a.split = SegmentSplit(split);
            a.n_seg = n_seg;
            if (!stat.is_valid() || stat.ndim() != 3)
                throw nb::value_error("stat must have shape (n_src, nT, 2)");
            a.n_src = int64_t(stat.shape(0));
            a.nT = int32_t(stat.shape(1));
            a.stat = check(stat, "stat", DT::Complex, a.precision, {a.n_src, a.nT, 2});
            a.seg_start = static_cast<const int32_t*>(check(seg_start, "seg_start", DT::I32, a.precision, {a.n_src}));
            a.seg_end = static_cast<const int32_t*>(check(seg_end, "seg_end", DT::I32, a.precision, {a.n_src}));
            a.out = static_cast<double*>(check(out, "out", DT::F64, a.precision, {a.n_src}));
            a.stream = stream_ptr(stream);
            nb::gil_scoped_release release;
            backend::run_semicoherent(a);
        },
        nb::arg("stat"), nb::arg("seg_start"), nb::arg("seg_end"), nb::arg("n_seg"),
        nb::arg("precision"), nb::arg("split"), nb::arg("out"), nb::arg("stream") = 0,
        "Semi-coherent statistic sum_n |<d|h>_n|^2 / <h|h>_n over n_seg segments of each\n"
        "source's in-band tranches [seg_start, seg_end]; split: see segment_splits.");

    m.def(
        "tracks",
        [](int32_t model, const In& params, const Grid& g, Out out, uintptr_t stream) {
            TrackArgs a{};
            a.model = model;
            a.grid = g;
            a.n_src = params.ndim() == 2 ? int64_t(params.shape(0)) : 0;
            a.stride = params.ndim() == 2 ? int32_t(params.shape(1)) : 0;
            a.params = static_cast<const double*>(check(params, "params", DT::F64, Precision::Double, {-1, -1}));
            a.out = static_cast<double*>(
                check(out, "out", DT::F64, Precision::Double, {a.n_src, model_modes(model), g.nT, 4}));
            a.stream = stream_ptr(stream);
            nb::gil_scoped_release release;
            backend::run_tracks(a);
        },
        nb::arg("model"), nb::arg("params"), nb::arg("grid"), nb::arg("out"),
        nb::arg("stream") = 0,
        "Evaluate analytic mode tracks (amp, phi, f, fdot) on the segment grid.");

    m.def(
        "synchronize", [](uintptr_t stream) { backend::synchronize(stream_ptr(stream)); },
        nb::arg("stream") = 0);
}
