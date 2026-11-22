"""Numerical multi-mode path: equivalence with analytic models, cross terms, fallback."""

import numpy as np
import pytest
from conftest import (
    CONFIG,
    TUKEY_CONFIG,
    noise_arrays,
    rel_err,
    requires_gpu,
    responses,
    t3_params,
)

from pygwtf.generator import AnalyticTimeFrequencyWaveform
from pygwtf.models import TaylorT3Spin
from pygwtf.multimode import MultipleHarmonicTimeFrequencyWaveform

BACKENDS = ["cpu", pytest.param("gpu", marks=[requires_gpu, pytest.mark.gpu])]
N_SRC = 6


def ylm22(cosi):
    norm = np.sqrt(5 / (64 * np.pi))
    return np.stack(
        [norm * (1 + cosi) ** 2, norm * (1 - cosi) ** 2], axis=-1
    ).astype(np.complex128)


def analytic_as_tracks(gen, params):
    """(n, 1, nT, 4) tracks of an analytic model, NaN outside its in-band segments."""
    tracks = np.array(gen.tracks(params))
    s0, s1 = gen.segment_indices(params)
    t = np.arange(gen.config["nT"])
    mask = (t[None, :] < np.asarray(s0)[:, None]) | (
        t[None, :] > np.asarray(s1)[:, None]
    )
    tracks[mask[:, None, :, None].repeat(4, axis=-1)] = np.nan
    return tracks


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("tdi", [None, 2])
@pytest.mark.parametrize("tukey", [False, True])
@pytest.mark.parametrize("precision", ["double", "single"])
def test_numerical_reproduces_analytic(backend, tdi, tukey, precision):
    cfg = TUKEY_CONFIG if tukey else CONFIG
    kw = dict(
        backend=backend, tdi_type=tdi, use_tukey=tukey, precision=precision
    )
    ga = AnalyticTimeFrequencyWaveform(TaylorT3Spin, dict(cfg), **kw)
    gm = MultipleHarmonicTimeFrequencyWaveform(dict(cfg), **kw)
    params = t3_params(N_SRC)
    resp = responses(N_SRC)
    resp[:, 0] = params["cosi"]
    tracks = analytic_as_tracks(ga, params)
    ylms = ylm22(params["cosi"])[:, None, :]
    r = resp if tdi else None
    tol = 1e-12 if precision == "double" else 1e-6
    assert (
        rel_err(
            gm(tracks, ylms, parameters_response=r),
            ga(params, parameters_response=r),
        )
        < tol
    )

    nc = 3 if tdi else 2
    data = ga(
        {k: v[:2] for k, v in params.items()},
        parameters_response=None if r is None else r[:2],
    ).sum(0)
    psd = noise_arrays(nc, scale=1e-44)
    sa = ga(
        params,
        channels=data,
        psds=psd,
        parameters_response=r,
        compute_statistic=True,
    )
    sm = gm(
        tracks,
        ylms,
        channels=data,
        psds=psd,
        parameters_response=r,
        compute_statistic=True,
    )
    assert rel_err(sm, sa) < tol


def synthetic_modes(n_modes, n_src=3, overlap=True, seed=0, fdot_sign=1.0):
    """Linear-chirp harmonics; with overlap=True neighbouring modes share bins."""
    rng = np.random.default_rng(seed)
    nT, dT, dF = CONFIG["nT"], CONFIG["dT"], CONFIG["dF"]
    t = (np.arange(nT) + 0.5) * dT
    tracks = np.zeros((n_src, n_modes, nT, 4))
    for s in range(n_src):
        f0 = rng.uniform(5e-3, 1e-2)
        fdot = fdot_sign * rng.uniform(2e-12, 2e-11)
        for m in range(n_modes):
            spacing = 3 * dF if overlap else 60 * dF
            f = f0 + m * spacing + fdot * t
            tracks[s, m, :, 0] = rng.uniform(0.5, 2.0) * 1e-20
            tracks[s, m, :, 1] = (
                2 * np.pi * (f0 + m * spacing) * t
                + np.pi * fdot * t**2
                + rng.uniform(0, 6)
            )
            tracks[s, m, :, 2] = f
            tracks[s, m, :, 3] = fdot
    ylms = rng.normal(size=(n_src, n_modes, 2)) + 1j * rng.normal(
        size=(n_src, n_modes, 2)
    )
    return tracks, ylms


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("tdi", [None, 2])
@pytest.mark.parametrize("full_invC", [False, True])
@pytest.mark.parametrize("fdot_sign", [1.0, -1.0])
def test_cross_mode_terms(backend, tdi, full_invC, fdot_sign):
    """Fused multi-mode <h|h> includes cross terms: equals dense inner products of the summed waveform."""
    g = MultipleHarmonicTimeFrequencyWaveform(
        dict(CONFIG),
        backend=backend,
        tdi_type=tdi,
        full_invC=full_invC,
        precision="double",
    )
    tracks, ylms = synthetic_modes(4, fdot_sign=fdot_sign)
    resp = responses(tracks.shape[0]) if tdi else None
    nc = 3 if tdi else 2
    noise = noise_arrays(nc, full=full_invC, scale=1e-44)
    nkw = {"invC": noise} if full_invC else {"psds": noise}
    h = g(tracks, ylms, parameters_response=resp)
    data = h[0] + 0.3 * h[1]
    s = g(
        tracks,
        ylms,
        channels=data,
        parameters_response=resp,
        compute_statistic=True,
        **nkw,
    )
    s_dense = g.inner_product(h, channels=data, **nkw)
    assert rel_err(s, s_dense) < 1e-10
    # Ignoring cross terms would be badly wrong for overlapping modes.
    per_mode = sum(
        np.asarray(
            g(
                tracks[:, m : m + 1],
                ylms[:, m : m + 1],
                channels=data,
                parameters_response=resp,
                compute_statistic=True,
                **nkw,
            )
        )
        for m in range(tracks.shape[1])
    )
    assert rel_err(per_mode[..., 1], s_dense[..., 1]) > 1e-3


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("precision", ["double", "single"])
def test_many_modes_fallback(backend, precision):
    g = MultipleHarmonicTimeFrequencyWaveform(
        dict(CONFIG), backend=backend, tdi_type=2, precision=precision
    )
    n_modes = g.max_fused_modes + 3
    tracks, ylms = synthetic_modes(n_modes, n_src=2)
    resp = responses(2)
    psd = noise_arrays(3, scale=1e-44)
    h = g(tracks, ylms, parameters_response=resp, mixed_precision=False)
    data = h.sum(0)
    s = g(
        tracks,
        ylms,
        channels=data,
        psds=psd,
        parameters_response=resp,
        compute_statistic=True,
    )
    s_ref = g.inner_product(h, channels=data, psds=psd, mixed_precision=False)
    assert rel_err(s, s_ref) < (1e-12 if precision == "double" else 1e-5)
    # Fused path for exactly max_fused_modes agrees with the dense reference too.
    k = g.max_fused_modes
    hk = g(
        tracks[:, :k],
        ylms[:, :k],
        parameters_response=resp,
        mixed_precision=False,
    )
    sk = g(
        tracks[:, :k],
        ylms[:, :k],
        channels=data,
        psds=psd,
        parameters_response=resp,
        compute_statistic=True,
    )
    sk_ref = g.inner_product(
        hk, channels=data, psds=psd, mixed_precision=False
    )
    assert rel_err(sk, sk_ref) < (1e-12 if precision == "double" else 1e-5)


def test_invalid_entries_are_skipped():
    g = MultipleHarmonicTimeFrequencyWaveform(dict(CONFIG), precision="double")
    tracks, ylms = synthetic_modes(2, n_src=1, overlap=False)
    h_full = g(tracks, ylms)
    padded = tracks.copy()
    padded[:, 1, 200:] = np.nan  # mode 1 leaves after segment 199
    padded[:, 0, :50, 0] = 0.0  # mode 0 switched off for the first 50 segments
    h = g(padded, ylms)
    assert np.all(np.isfinite(h))
    ref = h_full.copy()
    only1 = g(tracks[:, 1:], ylms[:, 1:])
    only0 = g(tracks[:, :1], ylms[:, :1])
    ref[:, 200:] -= only1[:, 200:]
    ref[:, :50] -= only0[:, :50]
    assert rel_err(h, ref) < 1e-12


def test_single_source_shapes():
    g = MultipleHarmonicTimeFrequencyWaveform(dict(CONFIG), precision="double")
    tracks, ylms = synthetic_modes(2, n_src=1)
    h = g(tracks[0], ylms[0])
    assert h.shape == (CONFIG["nT"], CONFIG["nF"], 2)
    psd = noise_arrays(2)
    s = g(tracks[0], ylms[0], channels=h, psds=psd, compute_statistic=True)
    assert s.shape == (CONFIG["nT"], 2)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("tdi", [None, 2])
@pytest.mark.parametrize("full_invC", [False, True])
@pytest.mark.parametrize("kernel", ["mode_parallel", "segment_loop"])
@pytest.mark.parametrize("precision", ["double", "single"])
def test_without_cross_terms(backend, tdi, full_invC, kernel, precision):
    """cross_terms=False is the sum of single-mode statistics, any mode count."""
    g = MultipleHarmonicTimeFrequencyWaveform(
        dict(CONFIG),
        backend=backend,
        tdi_type=tdi,
        full_invC=full_invC,
        precision=precision,
    )
    g._self_term_kernel = kernel
    n_modes = g.max_fused_modes + 2  # beyond the fused limit: no fallback
    tracks, ylms = synthetic_modes(n_modes, n_src=3)
    tracks[1, 2, 100:] = np.nan  # skipped entries
    tracks[2, :, :40, 0] = 0.0
    resp = responses(3) if tdi else None
    nc = 3 if tdi else 2
    noise = noise_arrays(nc, full=full_invC, scale=1e-44)
    nkw = {"invC": noise} if full_invC else {"psds": noise}
    data = g(
        tracks[:1, :2],
        ylms[:1, :2],
        parameters_response=resp[:1] if tdi else None,
    )[0]
    kw = dict(
        channels=data, parameters_response=resp, compute_statistic=True, **nkw
    )
    s = np.asarray(g(tracks, ylms, cross_terms=False, **kw))
    ref = sum(
        np.asarray(g(tracks[:, m : m + 1], ylms[:, m : m + 1], **kw))
        for m in range(n_modes)
    )
    tol = 1e-12 if precision == "double" else 1e-5
    assert rel_err(s, ref) < tol
    # <d|h> does not depend on the cross terms.
    s_cross = np.asarray(g(tracks, ylms, **kw))
    assert rel_err(s[..., 0], s_cross[..., 0]) < tol


@requires_gpu
@pytest.mark.gpu
def test_without_cross_terms_gpu_matches_cpu():
    kw = dict(tdi_type=2, precision="double")
    gc = MultipleHarmonicTimeFrequencyWaveform(
        dict(CONFIG), backend="cpu", **kw
    )
    gg = MultipleHarmonicTimeFrequencyWaveform(
        dict(CONFIG), backend="gpu", **kw
    )
    tracks, ylms = synthetic_modes(5, n_src=4)
    resp = responses(4)
    psd = noise_arrays(3, scale=1e-44)
    data = gc(tracks[:1], ylms[:1], parameters_response=resp[:1])[0]
    args = dict(
        channels=data,
        psds=psd,
        parameters_response=resp,
        compute_statistic=True,
        cross_terms=False,
    )
    assert rel_err(gg(tracks, ylms, **args), gc(tracks, ylms, **args)) < 1e-7


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("full_invC", [False, True])
@pytest.mark.parametrize("tukey", [False, True])
def test_cross_terms_unsorted_clusters(backend, full_invC, tukey):
    """Modes given out of frequency order, mixing isolated modes and clusters."""
    cfg = TUKEY_CONFIG if tukey else CONFIG
    g = MultipleHarmonicTimeFrequencyWaveform(
        dict(cfg),
        backend=backend,
        tdi_type=2,
        full_invC=full_invC,
        use_tukey=tukey,
        precision="double",
    )
    nT, dT, dF = cfg["nT"], cfg["dT"], cfg["dF"]
    t = (np.arange(nT) + 0.5) * dT
    # Offsets in bins: a cluster of three, an isolated mode, a pair, nested
    # ranges (fdot widens mode 6), all in shuffled order.
    offsets = np.array([40.0, 0.0, 95.3, 3.2, 41.7, 6.9, 200.0, 201.5])
    fdots = np.full(offsets.size, 5e-12)
    fdots[6] = 4e-10  # wider bin range, contains mode 7
    fdots[3] = -5e-12
    n_src = 2
    tracks = np.zeros((n_src, offsets.size, nT, 4))
    rng = np.random.default_rng(3)
    for s in range(n_src):
        f0 = 4e-3 + 1e-4 * s
        for m, (off, fd) in enumerate(zip(offsets, fdots)):
            fm = f0 + off * dF
            tracks[s, m, :, 0] = rng.uniform(0.5, 2.0) * 1e-20
            tracks[s, m, :, 1] = (
                2 * np.pi * fm * t + np.pi * fd * t**2 + rng.uniform(0, 6)
            )
            tracks[s, m, :, 2] = fm + fd * t
            tracks[s, m, :, 3] = fd
    tracks[1, 2, 150:] = np.nan
    ylms = rng.normal(size=(n_src, offsets.size, 2)) + 1j * rng.normal(
        size=(n_src, offsets.size, 2)
    )
    resp = responses(n_src)
    noise = noise_arrays(3, full=full_invC, scale=1e-44)
    nkw = {"invC": noise} if full_invC else {"psds": noise}
    h = g(tracks, ylms, parameters_response=resp)
    data = h[0] + 0.3 * h[1]
    s = g(
        tracks,
        ylms,
        channels=data,
        parameters_response=resp,
        compute_statistic=True,
        **nkw,
    )
    s_dense = g.inner_product(h, channels=data, **nkw)
    assert rel_err(s, s_dense) < 1e-10
