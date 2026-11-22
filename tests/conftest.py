import warnings

import numpy as np
import pytest

from pygwtf.backend import gpu_available

warnings.filterwarnings("ignore", message=".*lisaconstants.*")

HAS_GPU = gpu_available()

try:
    import numba  # noqa: F401

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

requires_gpu = pytest.mark.skipif(
    not HAS_GPU, reason="GPU backend/device not available"
)
requires_legacy = pytest.mark.skipif(
    not HAS_NUMBA, reason="numba (legacy reference) not installed"
)

DT = 86400.0 / 4
CONFIG = {
    "dT": DT,
    "nT": 365,
    "dF": 1 / DT,
    "nF": int(0.05 * DT),
    "kernel_width": 10,
}
TUKEY_CONFIG = dict(CONFIG, alpha=0.3)

RESPONSE = np.array([0.5, np.pi / 6, 5 * np.pi / 4, np.pi / 3])


def t2_params(n=None, seed=1):
    """TaylorT2Ecc parameters; n=None gives a single source."""
    if n is None:
        return {
            "M": 30.0,
            "eta": 0.25,
            "cosi": 0.5,
            "e0": 1e-3,
            "D": 1e3,
            "f0": 2e-2,
            "phi_coal": 0.1,
        }
    rng = np.random.default_rng(seed)
    return {
        "M": rng.uniform(20, 60, n),
        "eta": rng.uniform(0.15, 0.25, n),
        "cosi": rng.uniform(-1, 1, n),
        "e0": rng.uniform(1e-4, 1e-2, n),
        "D": rng.uniform(5e2, 2e3, n),
        # some sources chirp out of the band within the observation
        "f0": rng.uniform(1e-2, 4.5e-2, n),
        "phi_coal": rng.uniform(0, 2 * np.pi, n),
    }


def t3_params(n=None, seed=2):
    if n is None:
        return {
            "M": 58.13,
            "eta": 0.245,
            "cosi": 0.5,
            "D": 1e4,
            "f0": 1.5e-2,
            "s1": 0.3,
            "s2": -0.2,
            "phi_coal": 0.1,
        }
    rng = np.random.default_rng(seed)
    return {
        "M": rng.uniform(20, 80, n),
        "eta": rng.uniform(0.15, 0.25, n),
        "cosi": rng.uniform(-1, 1, n),
        "D": rng.uniform(5e3, 2e4, n),
        "f0": rng.uniform(1e-2, 4.5e-2, n),
        "s1": rng.uniform(-0.8, 0.8, n),
        "s2": rng.uniform(-0.8, 0.8, n),
        "phi_coal": rng.uniform(0, 2 * np.pi, n),
    }


def responses(n, seed=3):
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.uniform(-1, 1, n),
            rng.uniform(0, np.pi, n),
            rng.uniform(0, 2 * np.pi, n),
            np.arcsin(rng.uniform(-1, 1, n)),
        ],
        axis=1,
    )


def rel_err(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    scale = np.abs(b).max()
    return np.abs(a - b).max() / scale if scale > 0 else np.abs(a - b).max()


def noise_arrays(n_channels, full=False, seed=4, scale=1e-40):
    """Smooth random PSDs (and optionally a Hermitian positive-definite invC)."""
    rng = np.random.default_rng(seed)
    nT, nF = CONFIG["nT"], CONFIG["nF"]
    f = (np.arange(nF) + 1) / DT
    base = scale * (1 + (2e-3 / f) ** 4) * (1 + (f / 2e-2) ** 2)
    psd = base[None, :, None] * rng.uniform(0.5, 2.0, (nT, 1, n_channels))
    psd = np.broadcast_to(psd, (nT, nF, n_channels)).copy()
    if not full:
        return psd
    a = rng.normal(size=(n_channels, n_channels)) + 1j * rng.normal(
        size=(n_channels, n_channels)
    )
    a = 0.2 * a / np.abs(a).max()
    corr = np.eye(n_channels) + 0.5 * (a + a.conj().T)
    corr_inv = np.linalg.inv(corr)
    s = 1 / np.sqrt(psd)
    invC = s[..., :, None] * corr_inv[None, None] * s[..., None, :]
    return invC
