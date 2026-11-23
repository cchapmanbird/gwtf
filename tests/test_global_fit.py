"""Global-fit mode: per-walker data and/or noise selected per source by data_indices."""

import numpy as np
import pytest
from conftest import (
    CONFIG,
    noise_arrays,
    rel_err,
    requires_gpu,
    requires_legacy,
    responses,
    t2_params,
)
from test_multimode import synthetic_modes

from pygwtf.generator import AnalyticTimeFrequencyWaveform
from pygwtf.models import TaylorT2Ecc
from pygwtf.multimode import MultipleHarmonicTimeFrequencyWaveform

BACKENDS = ["cpu", pytest.param("gpu", marks=[requires_gpu, pytest.mark.gpu])]
NW, NTEMP = 3, 2
N_SRC = NW * NTEMP
# Walker-fastest layout: sources w, w + NW, ... belong to walker w.
IDX = np.tile(np.arange(NW), NTEMP)
TOL = {"double": 1e-12, "single": 5e-5}


def tol(backend, precision):
    """Against a CPU double reference. GPU double is limited to ~1e-8 by
    device pow/log ulps on ~1e7 rad PN phases (as in test_backends)."""
    return 1e-7 if (backend, precision) == ("gpu", "double") else TOL[precision]


def make(backend="cpu", tdi=2, full_invC=False, precision="double", **kw):
    return AnalyticTimeFrequencyWaveform(
        TaylorT2Ecc,
        dict(CONFIG),
        backend=backend,
        tdi_type=tdi,
        full_invC=full_invC,
        precision=precision,
        **kw,
    )


def gf_inputs(tdi=2, full_invC=False):
    """Proposals around NW injections; walker w's residual and noise differ."""
    params = t2_params(N_SRC)
    resp = responses(N_SRC) if tdi else None
    nc = 3 if tdi else 2
    h = make(tdi=tdi)(
        {k: v[:NW] for k, v in params.items()},
        parameters_response=None if resp is None else resp[:NW],
    )
    signal = h.sum(0)
    # Each walker has subtracted a different fraction of its own source.
    data = np.stack([signal - (0.2 + 0.3 * w) * h[w] for w in range(NW)])
    noise = np.stack(
        [noise_arrays(nc, full=full_invC, seed=10 + w) for w in range(NW)]
    )
    return params, resp, data, noise


def noise_kw(full_invC, noise):
    return {"invC": noise} if full_invC else {"psds": noise}


def per_walker_reference(params, resp, data, noise, full_invC, tdi):
    """Shared-data path, one walker at a time (CPU, double precision)."""
    g = make(tdi=tdi, full_invC=full_invC)
    data_batched = data.ndim == 4
    noise_batched = noise.ndim == (5 if full_invC else 4)
    out = np.empty((N_SRC, CONFIG["nT"], 2), complex)
    for w in range(NW):
        src = np.flatnonzero(IDX == w)
        out[src] = g(
            {k: v[src] for k, v in params.items()},
            parameters_response=None if resp is None else resp[src],
            channels=data[w] if data_batched else data,
            compute_statistic=True,
            **noise_kw(full_invC, noise[w] if noise_batched else noise),
        )
    return out



@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("tdi", [None, 2])
@pytest.mark.parametrize("full_invC", [False, True])
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("shared", [None, "data", "noise"])
def test_matches_per_walker_shared_path(
    backend, tdi, full_invC, precision, shared
):
    params, resp, data, noise = gf_inputs(tdi, full_invC)
    if shared == "data":
        data = data[0]
    if shared == "noise":
        noise = noise[0]
    g = make(backend, tdi, full_invC, precision)
    s = g(
        params,
        parameters_response=resp,
        channels=data,
        compute_statistic=True,
        data_indices=IDX,
        **noise_kw(full_invC, noise),
    )
    ref = per_walker_reference(params, resp, data, noise, full_invC, tdi)
    assert np.all(np.isfinite(s))
    assert rel_err(s, ref) < tol(backend, precision)
    assert rel_err(s.sum(1), ref.sum(1)) < tol(backend, precision)


def test_walker_map_matters():
    params, resp, data, noise = gf_inputs()
    g = make()
    kw = dict(
        parameters_response=resp,
        channels=data,
        psds=noise,
        compute_statistic=True,
    )
    s = g(params, data_indices=IDX, **kw)
    s_rolled = g(params, data_indices=np.roll(IDX, 1), **kw)
    assert rel_err(s_rolled, s) > 1e-3


def test_stored_arrays_and_reassignment():
    params, resp, data, noise = gf_inputs()
    g = make(channels=data, psds=noise, gf_mode=True)
    kw = dict(parameters_response=resp, compute_statistic=True, data_indices=IDX)
    s = g(params, **kw)
    assert rel_err(s, g(params, channels=data, psds=noise, **kw)) < 1e-14
    # Stored noise with call-time data (the usual global-fit loop) reuses the
    # prepared weights.
    weights_before = dict(g._weight_cache)
    s_half = g(params, channels=0.5 * data, **kw)
    assert rel_err(s_half[..., 0], 0.5 * s[..., 0]) < 1e-14
    assert rel_err(s_half[..., 1], s[..., 1]) < 1e-14
    assert all(
        g._weight_cache[k][0] is v[0] for k, v in weights_before.items()
    )
    # Reassigning the noise (e.g. after a noise move) refreshes the weights.
    g.psds = 2 * noise
    s2 = g(params, **kw)
    assert rel_err(s2, 0.5 * s) < 1e-14
    # Reassigning the data refreshes the cached data copy.
    g.channels = 3 * data
    s3 = g(params, **kw)
    assert rel_err(s3[..., 0], 1.5 * s[..., 0]) < 1e-14
    # Shapes are validated on assignment.
    with pytest.raises(ValueError, match="n_walkers"):
        g.psds = noise[..., :-1]
    with pytest.raises(ValueError, match="full inverse covariance"):
        make(full_invC=True).psds = noise


def test_data_is_read_in_place():
    _, _, data, _ = gf_inputs()
    g = make(precision="single")
    d32 = np.ascontiguousarray(data, dtype=np.complex64)
    assert g._prepare_data(d32, "single") is d32
    assert g._prepare_data(data, "double") is data


def test_waveform_and_gf_mode_flag():
    params, resp, data, noise = gf_inputs()
    g = make(gf_mode=True, channels=data, psds=noise)
    # Waveforms do not use the data, so gf_mode does not affect them.
    h = g(params, parameters_response=resp)
    assert h.shape == (N_SRC, CONFIG["nT"], CONFIG["nF"], 3)
    with pytest.raises(ValueError, match="gf_mode=True"):
        g(params, parameters_response=resp, compute_statistic=True)
    with pytest.raises(ValueError, match="only applies to statistic"):
        g(params, parameters_response=resp, data_indices=IDX)


def test_invalid_inputs():
    params, resp, data, noise = gf_inputs()
    g = make()
    kw = dict(parameters_response=resp, compute_statistic=True)
    with pytest.raises(ValueError, match="pass data_indices"):
        g(params, channels=data, psds=noise, **kw)
    with pytest.raises(ValueError, match="neither channels nor the noise"):
        g(params, channels=data[0], psds=noise[0], data_indices=IDX, **kw)
    with pytest.raises(ValueError, match="2 walkers"):
        g(params, channels=data[:2], psds=noise, data_indices=IDX, **kw)
    with pytest.raises(ValueError, match=r"lie in \[0, 2\]"):
        g(params, channels=data, psds=noise, data_indices=IDX + 1, **kw)
    with pytest.raises(ValueError, match="one walker per source"):
        g(params, channels=data, psds=noise, data_indices=IDX[:-1], **kw)
    with pytest.raises(TypeError, match="integers"):
        g(params, channels=data, psds=noise, data_indices=IDX * 1.0, **kw)


def test_out_of_range_device_index_gives_nan():
    """The kernels guard the index themselves (device-side indices are not
    range-checked in Python); exercised here through the CPU core directly."""
    params, resp, data, noise = gf_inputs()
    g = make()
    h = g(params, parameters_response=resp)
    w = (4 * CONFIG["dF"]) / noise
    out = np.empty((N_SRC, CONFIG["nT"], 2), complex)
    bad = IDX.astype(np.int32)
    bad[1] = NW
    bad[4] = -1
    g.backend.core.inner_product(
        h, data, w, 1, 0, out, 0, data_index=bad, scale=1.0
    )
    assert np.all(np.isnan(out[[1, 4]].real))
    good = np.array([0, 2, 3, 5])
    ref = g.inner_product(h, channels=data, psds=noise, data_indices=IDX)
    assert rel_err(out[good], ref[good]) < 1e-14


@requires_gpu
@pytest.mark.gpu
def test_gpu_device_indices():
    import cupy as cp

    params, resp, data, noise = gf_inputs()
    kw = dict(
        parameters_response=resp,
        channels=cp.asarray(data),
        psds=cp.asarray(noise),
        compute_statistic=True,
    )
    g = make("gpu")
    s = g(params, data_indices=cp.asarray(IDX), **kw)
    ref = make()(
        params,
        parameters_response=resp,
        channels=data,
        psds=noise,
        compute_statistic=True,
        data_indices=IDX,
    )
    assert rel_err(s, ref) < tol("gpu", "double")
    bad = cp.asarray(IDX)
    bad[2] = 99
    s_bad = g(params, data_indices=bad, **kw)
    assert np.all(np.isnan(s_bad[2].real))
    assert np.all(np.isfinite(np.delete(s_bad, 2, axis=0)))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("precision", ["double", "single"])
def test_inner_product(backend, precision):
    params, resp, data, noise = gf_inputs()
    g = make(backend, precision=precision)
    kw = dict(channels=data, psds=noise, data_indices=IDX)
    s = g(params, parameters_response=resp, compute_statistic=True, **kw)
    h = g(params, parameters_response=resp)
    ip = g.inner_product(h, **kw)
    assert rel_err(ip, s) < TOL[precision]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("variant", ["cross_terms", "self_terms", "fallback"])
def test_multimode(backend, precision, variant):
    n_modes = {"cross_terms": 3, "self_terms": 3, "fallback": 19}[variant]
    tracks, ylms = synthetic_modes(n_modes, n_src=N_SRC)
    resp = responses(N_SRC)
    g = MultipleHarmonicTimeFrequencyWaveform(
        dict(CONFIG), backend=backend, tdi_type=2, precision=precision
    )
    h = np.asarray(
        MultipleHarmonicTimeFrequencyWaveform(
            dict(CONFIG), tdi_type=2, precision="double"
        )(tracks[:NW], ylms[:NW], parameters_response=resp[:NW])
    )
    data = np.stack([h.sum(0) - 0.5 * h[w] for w in range(NW)])
    noise = np.stack(
        [noise_arrays(3, seed=20 + w, scale=1e-44) for w in range(NW)]
    )
    kw = dict(
        parameters_response=resp,
        compute_statistic=True,
        cross_terms=variant != "self_terms",
    )
    s = g(tracks, ylms, channels=data, psds=noise, data_indices=IDX, **kw)
    g64 = MultipleHarmonicTimeFrequencyWaveform(
        dict(CONFIG), tdi_type=2, precision="double"
    )
    ref = np.empty_like(s, dtype=complex)
    for w in range(NW):
        src = np.flatnonzero(IDX == w)
        ref[src] = g64(
            tracks[src],
            ylms[src],
            channels=data[w],
            psds=noise[w],
            **dict(kw, parameters_response=resp[src]),
        )
    assert rel_err(s, ref) < tol(backend, precision)


@pytest.mark.legacy
@requires_legacy
@pytest.mark.parametrize("full_invC", [False, True])
def test_legacy_parity(full_invC):
    from pygwtf.legacy import models as lm
    from pygwtf.legacy.generator import AnalyticTimeFrequencyWaveform as V1

    params, resp, data, noise = gf_inputs(2, full_invC)
    kw = dict(
        parameters_response=resp,
        channels=data,
        compute_statistic=True,
        data_indices=IDX,
        **noise_kw(full_invC, noise),
    )
    s1 = V1(
        lm.TaylorT2Ecc,
        dict(CONFIG),
        tdi_type=2,
        gf_mode=True,
        full_invC=full_invC,
    )(params, **kw)
    s2 = make(full_invC=full_invC, gf_mode=True)(params, **kw)
    # As in test_legacy_parity: v1's unreduced phases limit agreement.
    assert rel_err(s2, s1) < 1e-7

