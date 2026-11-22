"""
v2 (compiled, CPU, double precision) against the v1 numba implementation.

Each configuration triggers a numba compile of the legacy kernels, so these
tests are comparatively slow; they are marked ``legacy``.
"""

import numpy as np
import pytest
from conftest import (
    CONFIG,
    TUKEY_CONFIG,
    noise_arrays,
    rel_err,
    requires_legacy,
    responses,
    t2_params,
    t3_params,
)

from pygwtf import models as v2_models
from pygwtf.generator import AnalyticTimeFrequencyWaveform as V2

pytestmark = [pytest.mark.legacy, requires_legacy]

N_SRC = 5
# v1 evaluates cos/sin of phases ~1e7 rad directly; v2 reduces them first.
# Double-precision agreement is therefore limited to ~1e-9 relative.
TOL = 1e-7

MODELS = {
    "TaylorT2Ecc": (v2_models.TaylorT2Ecc, t2_params),
    "TaylorT3Spin": (v2_models.TaylorT3Spin, t3_params),
}

_legacy_cache = {}


def legacy_generator(model_name, tdi, tukey, full_invC):
    key = (model_name, tdi, tukey, full_invC)
    if key not in _legacy_cache:
        from pygwtf.legacy import models as lm
        from pygwtf.legacy.generator import AnalyticTimeFrequencyWaveform as V1

        cfg = dict(TUKEY_CONFIG if tukey else CONFIG)
        _legacy_cache[key] = V1(
            getattr(lm, model_name),
            cfg,
            tdi_type=tdi,
            use_tukey=tukey,
            full_invC=full_invC,
        )
    return _legacy_cache[key]


def v2_generator(model_name, tdi, tukey, full_invC):
    cfg = dict(TUKEY_CONFIG if tukey else CONFIG)
    return V2(
        MODELS[model_name][0],
        cfg,
        tdi_type=tdi,
        use_tukey=tukey,
        full_invC=full_invC,
        precision="double",
    )


def call_args(model_name, tdi):
    params = MODELS[model_name][1](N_SRC)
    resp = responses(N_SRC) if tdi else None
    if resp is not None:
        resp[:, 0] = params[
            "cosi"
        ]  # v1 TT uses the model cosi, TDI the response one
    return params, resp


@pytest.mark.parametrize("model_name", list(MODELS))
@pytest.mark.parametrize("tdi", [None, 2])
@pytest.mark.parametrize("tukey", [False, True])
def test_waveform(model_name, tdi, tukey):
    params, resp = call_args(model_name, tdi)
    h1 = legacy_generator(model_name, tdi, tukey, False)(
        params, parameters_response=resp
    )
    h2 = v2_generator(model_name, tdi, tukey, False)(
        params, parameters_response=resp
    )
    assert h1.shape == h2.shape
    assert np.array_equal(h1 != 0, h2 != 0), "different time-frequency support"
    assert rel_err(h2, h1) < TOL


@pytest.mark.parametrize("model_name", list(MODELS))
@pytest.mark.parametrize("tdi", [None, 2])
def test_waveform_start_of_segment(model_name, tdi):
    params, resp = call_args(model_name, tdi)
    h1 = legacy_generator(model_name, tdi, False, False)(
        params, parameters_response=resp, use_midpoint=False
    )
    h2 = v2_generator(model_name, tdi, False, False)(
        params, parameters_response=resp, use_midpoint=False
    )
    assert np.array_equal(h1 != 0, h2 != 0)
    assert rel_err(h2, h1) < TOL


def _data(gen1, params, resp, n_channels, seed=7):
    """Signal of the first two sources plus white-ish noise."""
    sub = {k: v[:2] for k, v in params.items()}
    sig = gen1(
        sub, parameters_response=None if resp is None else resp[:2]
    ).sum(0)
    rng = np.random.default_rng(seed)
    scale = np.abs(sig).max() * 1e-2
    noise = scale * (
        rng.normal(size=sig.shape) + 1j * rng.normal(size=sig.shape)
    )
    return sig + noise


@pytest.mark.parametrize("model_name", list(MODELS))
@pytest.mark.parametrize("tdi", [None, 2])
@pytest.mark.parametrize("tukey", [False, True])
@pytest.mark.parametrize("full_invC", [False, True])
def test_statistic(model_name, tdi, tukey, full_invC):
    params, resp = call_args(model_name, tdi)
    g1 = legacy_generator(model_name, tdi, tukey, full_invC)
    g2 = v2_generator(model_name, tdi, tukey, full_invC)
    nc = 3 if tdi else 2
    data = _data(g1, params, resp, nc)
    noise = noise_arrays(nc, full=full_invC)
    kw = {"invC": noise} if full_invC else {"psds": noise}
    s1 = g1(
        params,
        channels=data,
        parameters_response=resp,
        compute_statistic=True,
        **kw,
    )
    s2 = g2(
        params,
        channels=data,
        parameters_response=resp,
        compute_statistic=True,
        **kw,
    )
    assert s1.shape == s2.shape == (N_SRC, CONFIG["nT"], 2)
    assert rel_err(s2, s1) < TOL
    # Summed over segments, as used by likelihoods.
    assert rel_err(s2.sum(1), s1.sum(1)) < TOL


def test_segment_bounds_match_legacy():
    params, _ = call_args("TaylorT2Ecc", None)
    params["f0"][0] = 0.049  # leaves the band almost immediately
    g1 = legacy_generator("TaylorT2Ecc", None, False, False)
    g2 = v2_generator("TaylorT2Ecc", None, False, False)
    cache, _ = g1._load_parameters(params)
    s1 = g1._compute_segment_indices(cache)
    s2 = g2.segment_indices(params)
    np.testing.assert_array_equal(s1[0], s2[0])
    np.testing.assert_array_equal(s1[1], s2[1])
