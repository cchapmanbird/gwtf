"""GPU vs CPU backends, and single vs double precision."""

import numpy as np
import pytest
from conftest import (
    CONFIG,
    TUKEY_CONFIG,
    noise_arrays,
    rel_err,
    requires_gpu,
    responses,
    t2_params,
    t3_params,
)

from pygwtf.generator import AnalyticTimeFrequencyWaveform
from pygwtf.models import TaylorT2Ecc, TaylorT3Spin

N_SRC = 16
MODELS = {
    "TaylorT2Ecc": (TaylorT2Ecc, t2_params),
    "TaylorT3Spin": (TaylorT3Spin, t3_params),
}


def make(
    model, backend, tdi, tukey=False, full_invC=False, precision="double"
):
    cfg = dict(TUKEY_CONFIG if tukey else CONFIG)
    return AnalyticTimeFrequencyWaveform(
        MODELS[model][0],
        cfg,
        backend=backend,
        tdi_type=tdi,
        use_tukey=tukey,
        full_invC=full_invC,
        precision=precision,
    )


def statistic_inputs(model, tdi, full_invC, scale=1e-40):
    params = MODELS[model][1](N_SRC)
    resp = responses(N_SRC) if tdi else None
    nc = 3 if tdi else 2
    g = make(model, "cpu", tdi)
    data = g(
        {k: v[:3] for k, v in params.items()},
        parameters_response=None if resp is None else resp[:3],
    ).sum(0)
    noise = noise_arrays(nc, full=full_invC, scale=scale)
    return (
        params,
        resp,
        data,
        ({"invC": noise} if full_invC else {"psds": noise}),
    )


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("tdi", [None, 2])
@pytest.mark.parametrize("tukey", [False, True])
def test_gpu_waveform_matches_cpu(model, tdi, tukey):
    params = MODELS[model][1](N_SRC)
    resp = responses(N_SRC) if tdi else None
    h_cpu = make(model, "cpu", tdi, tukey)(params, parameters_response=resp)
    h_gpu = make(model, "gpu", tdi, tukey)(params, parameters_response=resp)
    assert isinstance(h_gpu, np.ndarray)  # numpy in -> numpy out
    # PN phases are ~1e7-1e8 rad; FMA contraction and pow/log ulps on the
    # device move them by ~1e-8 rad, the same order as v1 vs v2 on the CPU.
    assert rel_err(h_gpu, h_cpu) < 1e-7


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("tdi", [None, 2])
@pytest.mark.parametrize("full_invC", [False, True])
def test_gpu_statistic_matches_cpu(model, tdi, full_invC):
    params, resp, data, noise = statistic_inputs(model, tdi, full_invC)
    kw = dict(
        channels=data,
        parameters_response=resp,
        compute_statistic=True,
        **noise,
    )
    s_cpu = make(model, "cpu", tdi, full_invC=full_invC)(params, **kw)
    s_gpu = make(model, "gpu", tdi, full_invC=full_invC)(params, **kw)
    assert rel_err(s_gpu, s_cpu) < 1e-7


@pytest.mark.parametrize(
    "backend",
    ["cpu", pytest.param("gpu", marks=[requires_gpu, pytest.mark.gpu])],
)
@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("tdi", [None, 2])
def test_single_precision_waveform(backend, model, tdi):
    params = MODELS[model][1](N_SRC)
    resp = responses(N_SRC) if tdi else None
    h64 = make(model, "cpu", tdi)(params, parameters_response=resp)
    h32 = make(model, backend, tdi, precision="single")(
        params, parameters_response=resp
    )
    assert h32.dtype == np.complex64
    assert rel_err(h32, h64) < 5e-5


@pytest.mark.parametrize(
    "backend",
    ["cpu", pytest.param("gpu", marks=[requires_gpu, pytest.mark.gpu])],
)
@pytest.mark.parametrize("tdi", [None, 2])
@pytest.mark.parametrize("full_invC", [False, True])
@pytest.mark.parametrize("psd_scale", [1.0, 1e-46])
def test_single_precision_statistic(backend, tdi, full_invC, psd_scale):
    """Single precision must survive realistic (tiny) PSD magnitudes, where the
    raw weights 4 dF / PSD would overflow float32."""
    params, resp, data, noise = statistic_inputs(
        "TaylorT3Spin", tdi, full_invC, scale=psd_scale
    )
    kw = dict(
        channels=data,
        parameters_response=resp,
        compute_statistic=True,
        **noise,
    )
    s64 = make("TaylorT3Spin", "cpu", tdi, full_invC=full_invC)(params, **kw)
    s32 = make(
        "TaylorT3Spin", backend, tdi, full_invC=full_invC, precision="single"
    )(params, **kw)
    assert s32.dtype == np.complex64
    assert np.all(np.isfinite(s32))
    # Per segment, and summed over segments as in a likelihood.
    assert rel_err(s32, s64) < 5e-5
    assert rel_err(s32.sum(1), s64.sum(1)) < 2e-5


def test_mixed_precision_override():
    g = make("TaylorT2Ecc", "cpu", None, precision="double")
    p = t2_params()
    assert g(p).dtype == np.complex128
    assert g(p, mixed_precision=True).dtype == np.complex64
    g32 = make("TaylorT2Ecc", "cpu", None, precision="single")
    assert g32(p).dtype == np.complex64
    assert g32(p, mixed_precision=False).dtype == np.complex128


def test_inner_product_matches_statistic():
    params, resp, data, noise = statistic_inputs("TaylorT2Ecc", 2, False)
    g = make("TaylorT2Ecc", "cpu", 2)
    s = g(
        params,
        channels=data,
        parameters_response=resp,
        compute_statistic=True,
        **noise,
    )
    h = g(params, parameters_response=resp)
    ip = g.inner_product(h, channels=data, **noise)
    assert rel_err(ip, s) < 1e-12
