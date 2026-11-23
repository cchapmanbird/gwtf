"""Semi-coherent statistic Upsilon = sum_n |<d|h>_n|^2 / <h|h>_n over N_seg segments."""

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
from test_multimode import analytic_as_tracks

from pygwtf.generator import AnalyticTimeFrequencyWaveform
from pygwtf.models import TaylorT2Ecc
from pygwtf.multimode import MultipleHarmonicTimeFrequencyWaveform

BACKENDS = ["cpu", pytest.param("gpu", marks=[requires_gpu, pytest.mark.gpu])]
N_SRC = 12
NT = CONFIG["nT"]


def reference(stat, s0, s1, n_seg, split="spread"):
    """Direct NumPy evaluation of the semi-coherent statistic."""
    stat = np.asarray(stat, dtype=np.complex128)
    out = np.zeros(stat.shape[0])
    for s in range(stat.shape[0]):
        L = int(s1[s]) - int(s0[s]) + 1
        if L <= 0:
            continue
        per = L // n_seg
        for n in range(n_seg):
            if split == "spread":
                b, e = s0[s] + n * L // n_seg, s0[s] + (n + 1) * L // n_seg
            else:
                b, e = s0[s] + n * per, s0[s] + (n + 1) * per
            dh = stat[s, b:e, 0].sum()
            hh = stat[s, b:e, 1].real.sum()
            if hh != 0:
                out[s] += abs(dh) ** 2 / hh
    return out


def make(backend="cpu", precision="double", **kw):
    return AnalyticTimeFrequencyWaveform(
        TaylorT2Ecc,
        dict(CONFIG),
        backend=backend,
        tdi_type=2,
        precision=precision,
        **kw,
    )


def inputs(n=N_SRC):
    params = t2_params(n, seed=5)
    # Half the sources chirp out of the band after 2 to ~300 of 365 tranches.
    params["M"][n // 2 :] = 60.0
    params["f0"][n // 2 :] = np.linspace(0.04, 0.0499, n - n // 2)
    resp = responses(n, seed=6)
    sub = {k: v[:3] for k, v in params.items()}
    data = make()(sub, parameters_response=resp[:3]).sum(0)
    rng = np.random.default_rng(8)
    data = data + 1e-2 * np.abs(data).max() * (
        rng.normal(size=data.shape) + 1j * rng.normal(size=data.shape)
    )
    return params, resp, data, noise_arrays(3)


@pytest.fixture(scope="module")
def setup():
    params, resp, data, psd = inputs()
    g = make()
    s0, s1 = (np.asarray(a) for a in g.segment_indices(params))
    # The test set must exercise sources that leave the band early.
    assert np.any(s1 < NT - 1) and np.any(s1 == NT - 1)
    return params, resp, data, psd, s0, s1


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("split", ["spread", "truncate"])
@pytest.mark.parametrize("n_seg", [1, 4, 7, 50, 1000])
def test_matches_reference(setup, backend, precision, split, n_seg):
    params, resp, data, psd, s0, s1 = setup
    g = make(backend, precision, channels=data, psds=psd)
    kw = dict(parameters_response=resp, compute_statistic=True)
    stat = g(params, **kw)
    ups = g(params, N_seg=n_seg, segment_split=split, **kw)
    assert ups.dtype == np.float64 and ups.shape == (N_SRC,)
    assert np.all(np.isfinite(ups))
    ref = reference(stat, s0, s1, n_seg, split)
    assert rel_err(ups, ref) < 1e-12
    # Same thing through the public reducer.
    ups2 = g.semi_coherent_statistic(stat, n_seg, s0, s1, segment_split=split)
    assert rel_err(ups2, ref) < 1e-12


def test_limits(setup):
    params, resp, data, psd, s0, s1 = setup
    g = make(channels=data, psds=psd)
    kw = dict(parameters_response=resp, compute_statistic=True)
    stat = g(params, **kw)
    # N_seg = 1: the coherent |<d|h>|^2 / <h|h>.
    dh, hh = stat[..., 0].sum(1), stat[..., 1].real.sum(1)
    coherent = np.where(hh != 0, np.abs(dh) ** 2 / np.where(hh != 0, hh, 1), 0)
    assert rel_err(g(params, N_seg=1, **kw), coherent) < 1e-12
    # One tranche per segment: every in-band tranche counted separately,
    # including the remainder that 'truncate' drops.
    L = s1 - s0 + 1
    per_tranche = np.abs(stat[..., 0]) ** 2 / np.where(
        stat[..., 1].real != 0, stat[..., 1].real, np.inf
    )
    k = int(L.max())
    ups = g(params, N_seg=k, **kw)
    assert rel_err(ups, per_tranche.sum(1)) < 1e-12
    # More segments than in-band tranches: the extra segments are empty and
    # add nothing (v1 gave NaN); 'truncate' then has no full segment at all.
    assert rel_err(g(params, N_seg=10 * NT, **kw), per_tranche.sum(1)) < 1e-12
    assert np.all(g(params, N_seg=10 * NT, segment_split="truncate", **kw) == 0)


def test_spread_vs_truncate_remainder(setup):
    """Truncation drops the L % N_seg final tranches of each source."""
    params, resp, data, psd, s0, s1 = setup
    g = make(channels=data, psds=psd)
    kw = dict(parameters_response=resp, compute_statistic=True)
    stat = g(params, **kw)
    L = s1 - s0 + 1
    n_seg = 7
    drop = (L > 0) & (L % n_seg != 0)
    assert drop.any()
    stat_cut = stat.copy()
    for s in np.flatnonzero(drop):
        stat_cut[s, s1[s] - L[s] % n_seg + 1 : s1[s] + 1] = 0
    s1_cut = np.where(L > 0, s1 - L % n_seg, s1)
    trunc = g(params, N_seg=n_seg, segment_split="truncate", **kw)
    assert rel_err(trunc, reference(stat_cut, s0, s1_cut, n_seg)) < 1e-12


def test_out_of_band_and_nan():
    g = make()
    stat = np.zeros((3, NT, 2), complex)
    stat[:, :, 0] = 1 + 1j
    stat[:, :, 1] = 2
    stat[2, 5, 0] = np.nan
    ups = g.semi_coherent_statistic(
        stat, 4, np.array([0, 10, 0]), np.array([-1, 5, NT - 1])
    )
    # No in-band tranches -> 0; NaN input propagates rather than vanishing.
    assert ups[0] == 0 and ups[1] == 0 and np.isnan(ups[2])
    # Ranges are clipped to the grid.
    ups = g.semi_coherent_statistic(
        stat[:1], 1, np.array([-50]), np.array([NT + 50])
    )
    assert rel_err(ups, [abs(NT * (1 + 1j)) ** 2 / (2 * NT)]) < 1e-14


def test_preallocated_output_and_errors(setup):
    params, resp, data, psd, _, _ = setup
    g = make(channels=data, psds=psd)
    kw = dict(parameters_response=resp, compute_statistic=True)
    buf = np.empty(N_SRC)
    ups = g(params, N_seg=4, search_statistic=buf, **kw)
    assert np.array_equal(ups, buf)
    single = g({k: v[0] for k, v in params.items()}, N_seg=4,
               **dict(kw, parameters_response=resp[0]))
    assert np.ndim(single) == 0 and single == pytest.approx(ups[0], rel=1e-12)
    with pytest.raises(ValueError, match="compute_statistic"):
        g(params, parameters_response=resp, N_seg=4)
    with pytest.raises(ValueError, match="positive integer"):
        g(params, N_seg=0, **kw)
    with pytest.raises(ValueError, match="segment_split"):
        g(params, N_seg=4, segment_split="even", **kw)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["double", "single"])
def test_gpu_matches_cpu(setup, precision):
    params, resp, data, psd, _, _ = setup
    kw = dict(parameters_response=resp, compute_statistic=True, N_seg=9)
    cpu = make("cpu", precision, channels=data, psds=psd)(params, **kw)
    gpu = make("gpu", precision, channels=data, psds=psd)(params, **kw)
    # Statistics differ at the device-ulp level (see test_backends).
    assert rel_err(gpu, cpu) < (1e-7 if precision == "double" else 1e-4)


def test_global_fit(setup):
    params, resp, data, psd, s0, s1 = setup
    nw = 3
    idx = np.arange(N_SRC) % nw
    datas = np.stack([(1 - 0.2 * w) * data for w in range(nw)])
    g = make()
    kw = dict(parameters_response=resp, compute_statistic=True)
    ups = g(params, channels=datas, psds=psd, data_indices=idx, N_seg=5, **kw)
    stat = g(params, channels=datas, psds=psd, data_indices=idx, **kw)
    assert rel_err(ups, reference(stat, s0, s1, 5)) < 1e-12


@pytest.mark.parametrize("backend", BACKENDS)
def test_multimode(setup, backend):
    params, resp, data, psd, s0, s1 = setup
    ga = make(channels=data, psds=psd)
    tracks = analytic_as_tracks(ga, params)
    norm = np.sqrt(5 / (64 * np.pi))
    cosi = resp[:, 0]
    ylms = np.stack([norm * (1 + cosi) ** 2, norm * (1 - cosi) ** 2], -1)
    ylms = ylms[:, None, :].astype(complex)
    gm = MultipleHarmonicTimeFrequencyWaveform(
        dict(CONFIG), backend=backend, tdi_type=2, channels=data, psds=psd,
        precision="double",
    )
    kw = dict(parameters_response=resp, compute_statistic=True)
    stat = gm(tracks, ylms, **kw)
    # With the analytic in-band ranges, multi-mode reproduces the analytic Upsilon.
    ups_a = ga(params, N_seg=6, **kw)
    ups_m = gm(tracks, ylms, N_seg=6, segment_ranges=(s0, s1), **kw)
    assert rel_err(ups_m, ups_a) < (1e-12 if backend == "cpu" else 1e-7)
    # Default ranges come from the tracks: every tranche with a usable,
    # in-band mode entry.
    m0, m1 = (np.asarray(a) for a in gm.segment_indices(tracks))
    f = tracks[:, 0, :, 2]
    band = np.isfinite(f) & (f >= CONFIG["dF"]) & (
        f <= CONFIG["dF"] * (CONFIG["nF"] + 0.5)
    )
    for s in range(N_SRC):
        on = np.flatnonzero(band[s])
        assert (m0[s], m1[s]) == ((on[0], on[-1]) if on.size else (0, -1))
    ups = gm(tracks, ylms, N_seg=6, **kw)
    assert rel_err(ups, reference(stat, m0, m1, 6)) < 1e-12


@pytest.mark.legacy
@requires_legacy
@pytest.mark.parametrize("n_seg", [1, 7, 40])
def test_legacy_parity(setup, n_seg):
    from pygwtf.legacy import models as lm
    from pygwtf.legacy.generator import AnalyticTimeFrequencyWaveform as V1

    params, resp, data, psd, s0, s1 = setup
    # v1 divides by zero for sources with fewer in-band tranches than N_seg
    # (v2 returns 0 for them, see test_limits): compare the others.
    ok = np.flatnonzero((s1 - s0 + 1) >= n_seg)
    assert ok.size >= N_SRC // 2
    kw = dict(
        parameters_response=resp[ok],
        channels=data,
        psds=psd,
        compute_statistic=True,
        N_seg=n_seg,
    )
    p = {k: v[ok] for k, v in params.items()}
    v1 = V1(lm.TaylorT2Ecc, dict(CONFIG), tdi_type=2)(p, **kw)
    v2 = make()(p, segment_split="truncate", **kw)
    assert rel_err(v2, v1) < 1e-7
