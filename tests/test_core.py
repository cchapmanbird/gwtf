"""Compiled core plumbing, array interop, validation and extension points."""

import numpy as np
import pytest
from conftest import (
    CONFIG,
    HAS_GPU,
    noise_arrays,
    rel_err,
    requires_gpu,
    t2_params,
)

from pygwtf import constants
from pygwtf.backend import get_backend
from pygwtf.generator import AnalyticTimeFrequencyWaveform
from pygwtf.models import AnalyticModel, TaylorT2Ecc, TaylorT3Spin

BACKENDS = ["cpu", pytest.param("gpu", marks=[requires_gpu, pytest.mark.gpu])]


@pytest.mark.parametrize("backend", BACKENDS)
def test_constants_match_lisaconstants(backend):
    c = get_backend(backend).core.constants
    assert c["MTsun"] == constants.MTsun
    assert c["pc"] == constants.pc
    assert c["clight"] == constants.clight
    assert c["gamma_e"] == constants.gamma_e


@pytest.mark.parametrize("backend", BACKENDS)
def test_model_registry(backend):
    names = {m.name: m for m in get_backend(backend).core.models()}
    for cls in (TaylorT2Ecc, TaylorT3Spin):
        model = cls(backend=backend)
        info = names[cls.kernel_name]
        assert info.n_params == model.num_parameters
        assert info.n_derived == model.num_derived_parameters
        assert model.num_modes == 1


def test_unknown_or_mismatched_model():
    class Missing(AnalyticModel):
        kernel_name = "NoSuchModel"
        parameters = ["a"]

    with pytest.raises(ValueError, match="No compiled model"):
        Missing()

    class WrongCount(AnalyticModel):
        kernel_name = "TaylorT2Ecc"
        parameters = ["M", "eta"]

    with pytest.raises(ValueError, match="declares 2 parameters"):
        WrongCount()


def test_compute_derived_parameters_matches_legacy():
    pytest.importorskip("numba")
    from pygwtf.legacy.backend import get_backend as legacy_backend
    from pygwtf.legacy.models import TaylorT3Spin as LegacyT3

    p = t2_params(4)  # reuse shapes; build T3 parameters explicitly
    raw = np.column_stack(
        [
            p["M"],
            p["eta"],
            p["cosi"],
            p["D"],
            p["f0"],
            np.full(4, 0.2),
            np.full(4, -0.1),
            p["phi_coal"],
        ]
    )
    a = np.zeros((4, 12))
    a[:, :8] = raw
    b = a.copy()
    TaylorT3Spin().compute_derived_parameters(a)
    LegacyT3(backend=legacy_backend("cpu")).compute_derived_parameters(b)
    np.testing.assert_allclose(a, b, rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_scalar_dict_returns_numpy(backend):
    g = AnalyticTimeFrequencyWaveform(
        TaylorT2Ecc, dict(CONFIG), backend=backend
    )
    h = g(t2_params())
    assert isinstance(h, np.ndarray)
    assert h.shape == (CONFIG["nT"], CONFIG["nF"], 2)


@requires_gpu
@pytest.mark.gpu
def test_cupy_in_cupy_out_and_out_argument():
    import cupy as cp

    g = AnalyticTimeFrequencyWaveform(
        TaylorT2Ecc, dict(CONFIG), backend="gpu", precision="double"
    )
    p = {k: cp.asarray(v) for k, v in t2_params(3).items()}
    h = g(p)
    assert isinstance(h, cp.ndarray)
    out = cp.empty_like(h)
    h2 = g(p, out=out)
    assert h2 is out
    assert rel_err(cp.asnumpy(h2), cp.asnumpy(h)) == 0.0
    # Array parameters (model order) work too.
    arr = cp.stack([p[k] for k in g.model.parameters], axis=1)
    assert rel_err(cp.asnumpy(g(arr)), cp.asnumpy(h)) == 0.0
    # Statistic with CuPy data and PSDs.
    psd = cp.asarray(noise_arrays(2))
    s = g(p, channels=h.sum(0), psds=psd, compute_statistic=True)
    assert isinstance(s, cp.ndarray) and s.shape == (3, CONFIG["nT"], 2)


@requires_gpu
@pytest.mark.gpu
def test_numpy_out_rejected_on_gpu():
    g = AnalyticTimeFrequencyWaveform(TaylorT2Ecc, dict(CONFIG), backend="gpu")
    with pytest.raises(TypeError, match="cupy array"):
        g(
            t2_params(2),
            out=np.zeros((2, CONFIG["nT"], CONFIG["nF"], 2), np.complex64),
        )


@pytest.mark.skipif(not HAS_GPU, reason="needs GPU")
def test_cupy_inputs_on_cpu_backend_return_cupy():
    import cupy as cp

    g = AnalyticTimeFrequencyWaveform(
        TaylorT2Ecc, dict(CONFIG), backend="cpu", precision="double"
    )
    p = {k: cp.asarray(v) for k, v in t2_params(2).items()}
    h = g(p)
    assert isinstance(h, cp.ndarray)
    assert rel_err(cp.asnumpy(h), g(t2_params(2))) == 0.0


def test_torch_roundtrip():
    torch = pytest.importorskip("torch")
    g = AnalyticTimeFrequencyWaveform(
        TaylorT2Ecc, dict(CONFIG), precision="double"
    )
    p = {k: torch.as_tensor(v) for k, v in t2_params(2).items()}
    h = g(p)
    assert isinstance(h, torch.Tensor)


def test_validation_errors():
    g = AnalyticTimeFrequencyWaveform(TaylorT2Ecc, dict(CONFIG), tdi_type=2)
    with pytest.raises(
        ValueError, match="parameters_response must be supplied"
    ):
        g(t2_params())
    with pytest.raises(
        ValueError, match="parameters_response must have shape"
    ):
        g(t2_params(3), parameters_response=np.zeros((2, 4)))
    with pytest.raises(KeyError, match="Missing parameters"):
        g({"M": 1.0})
    with pytest.raises(ValueError, match="Channels must be supplied"):
        g(t2_params(), parameters_response=np.zeros(4), compute_statistic=True)
    with pytest.raises(ValueError, match="expected"):
        g(
            t2_params(),
            parameters_response=np.zeros(4),
            compute_statistic=True,
            channels=np.zeros((3, 3, 3), complex),
            psds=np.ones((3, 3, 3)),
        )
    with pytest.raises(ValueError, match="alpha"):
        AnalyticTimeFrequencyWaveform(
            TaylorT2Ecc, dict(CONFIG), use_tukey=True
        )
    with pytest.raises(ValueError, match="not both"):
        AnalyticTimeFrequencyWaveform(
            TaylorT2Ecc, dict(CONFIG), psds=np.ones(1), invC=np.ones(1)
        )
    gt = AnalyticTimeFrequencyWaveform(
        TaylorT2Ecc, dict(CONFIG, alpha=0.5), use_tukey=True
    )
    with pytest.raises(ValueError, match="midpoint"):
        gt(t2_params(), use_midpoint=False)


def test_core_rejects_bad_arrays():
    core = get_backend("cpu").core
    g = core.Grid()
    raw = np.zeros((2, 7))
    with pytest.raises(ValueError, match="dtype"):
        core.prepare(
            0,
            raw,
            np.zeros((2, 8), np.float32),
            np.zeros(2, np.int32),
            np.zeros(2, np.int32),
            g,
            0,
            1,
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        core.prepare(
            0,
            raw,
            np.zeros((8, 2)).T,
            np.zeros(2, np.int32),
            np.zeros(2, np.int32),
            g,
            0,
            1,
        )
    with pytest.raises(ValueError, match="parameter count"):
        core.prepare(
            0,
            raw,
            np.zeros((2, 9)),
            np.zeros(2, np.int32),
            np.zeros(2, np.int32),
            g,
            0,
            1,
        )


def test_f_tranche_is_bin_centres():
    g = AnalyticTimeFrequencyWaveform(TaylorT2Ecc, dict(CONFIG))
    assert g.f_tranche[0] == g.config["fmin"] == CONFIG["dF"]
    np.testing.assert_allclose(np.diff(g.f_tranche), CONFIG["dF"])
