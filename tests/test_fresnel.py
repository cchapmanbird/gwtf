"""
The Fresnel kernel against direct quadrature of the segment Fourier integral,
including negative fdot (not covered by the legacy comparison).

The kernel uses the rational Fresnel approximation of McCormick & Kraemer,
whose absolute error is ~1e-3, so agreement is checked at that level.
"""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from pygwtf.multimode import MultipleHarmonicTimeFrequencyWaveform

dT = 1.0e4
CFG = {
    "dT": dT,
    "nT": 3,
    "dF": 1 / dT,
    "nF": 120,
    "kernel_width": 10,
    "alpha": 0.4,
}
# Relative to the peak bin. The Tukey kernel combines 12 Fresnel terms with
# some cancellation, so its approximation error is larger (and identical for
# +fdot and -fdot, and in single and double precision).
APPROX_TOL = {"boxcar": 5e-3, "tukey": 1e-2}


def tukey(x, T, alpha):
    w = np.ones_like(x)
    edge = alpha * T / 2
    on = x < edge
    off = x > T - edge
    w[on] = 0.5 * (1 - np.cos(2 * np.pi * x[on] / (alpha * T)))
    w[off] = 0.5 * (1 - np.cos(2 * np.pi * (T - x[off]) / (alpha * T)))
    return w


def quadrature(f_bins, amp, phi, f0, fdot, midpoint, window):
    """H(f) = int_0^T w(x) A exp(i[phi + 2 pi f0 tau + pi fdot tau^2]) exp(-2 pi i f x) dx."""
    x = np.linspace(0, dT, 200001)
    tau = x - dT / 2 if midpoint else x
    w = tukey(x, dT, CFG["alpha"]) if window == "tukey" else 1.0
    h = (
        w
        * amp
        * np.exp(1j * (phi + 2 * np.pi * f0 * tau + np.pi * fdot * tau**2))
    )
    return np.array(
        [trapezoid(h * np.exp(-2j * np.pi * f * x), x) for f in f_bins]
    )


@pytest.mark.parametrize("fdot_sign", [1.0, -1.0])
@pytest.mark.parametrize("bins_swept", [0.3, 4.0])
@pytest.mark.parametrize(
    "window,midpoint", [("boxcar", True), ("boxcar", False), ("tukey", True)]
)
@pytest.mark.parametrize("precision", ["double", "single"])
def test_against_quadrature(
    fdot_sign, bins_swept, window, midpoint, precision
):
    dF = CFG["dF"]
    fmin = dF
    f0 = fmin + 50.3 * dF
    fdot = fdot_sign * bins_swept * dF / dT
    amp, phi = 1.0, 0.7
    tracks = np.zeros((1, 1, CFG["nT"], 4))
    tracks[0, 0, 1] = [amp, phi, f0, fdot]  # only segment 1 is populated
    # TT h+ coefficient = pol_convention * (Y+ + conj(Y-)) / 2 = 1, hx = 0.
    ylms = np.array([[[-1.0, -1.0]]], dtype=np.complex128)
    g = MultipleHarmonicTimeFrequencyWaveform(
        dict(CFG), use_tukey=window == "tukey", precision=precision
    )
    h = np.asarray(g(tracks, ylms, use_midpoint=midpoint))[0, 1, :, 0]
    nz = np.nonzero(h)[0]
    f_bins = fmin + nz * dF
    ref = quadrature(f_bins, amp, phi, f0, fdot, midpoint, window)
    err = np.abs(h[nz] - ref).max() / np.abs(ref).max()
    assert err < APPROX_TOL[window]
