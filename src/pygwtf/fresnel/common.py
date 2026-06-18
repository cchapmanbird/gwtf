from cmath import exp
from math import cos, pi, sin

from numba import jit


@jit
def _fresnel(x):
    """
    Asymptotic expansion approximation for Fresnel Integrals:

    Reference: 'Polynomial approximations for Fresnel integrals in diffraction analysis'- Michael E. McCormick a,*, David R.B. Kraemer
                 https://www.sciencedirect.com/science/article/pii/S0378383901000345?via%3Dihub - Section. 3, Eq. 18.

    Parameters:
    ----------
    x: float
        The argument to the Fresnel integrals.

    Returns:
    -------
    Sx_approx: float
        Approximation to the Fresnel S integral at x.
    Cx_approx: float
        Approximation to the Fresnel C integral at x.

    """
    ax = abs(x)

    pix = pi * ax
    halfpix2 = 0.5 * pix * ax
    cospix2 = cos(halfpix2)
    sinpix2 = sin(halfpix2)

    if ax < 6:
        # Below is the clean (slow) implementation

        # fx = (1 + 0.926 * ax) / (2 + 1.792 * ax + 3.104 * ax**2)
        # gx = 1 / (2 + 4.142 * ax + 3.492 * ax**2 + 6.67 * ax**3)

        # Below is the fast (but less clean) implementation (avoids repeated float *division*).
        f_denom = 2 + 1.792 * ax + 3.104 * ax**2
        g_denom = 2 + 4.142 * ax + 3.492 * ax**2 + 6.67 * ax**3

        combined_reciprocal = 1.0 / (f_denom * g_denom)

        fx = (1 + 0.926 * ax) * combined_reciprocal * g_denom
        gx = combined_reciprocal * f_denom

        Sx_approx = 0.5 - fx * cospix2 - gx * sinpix2
        Cx_approx = 0.5 + fx * sinpix2 - gx * cospix2
    else:
        one_over_pix = 1 / pix

        Sx_approx = 0.5 - cospix2 * one_over_pix
        Cx_approx = 0.5 + sinpix2 * one_over_pix

    if x < 0:
        return -Sx_approx, -Cx_approx
    else:
        return Sx_approx, Cx_approx


@jit
def _fresnel_kernel(
    f_bin,
    amp_mode_prefactor,
    sqrt2fdot,
    phase_mode,
    f_mode,
    fdot_mode,
    one_over_fdot,
    T,
    use_midpoint,
):
    """
    Box-car window fresnel waveform kernel.

    Parameters:
    ----------
    f_bin: float
        The frequency at the centre of the bin being evaluated.
    amp_mode_prefactor: float
        The amplitude/sqrt(2 * fdot_mode) of the mode being evaluated.
    sqrt2fdot: float
        The square root of 2 times the frequency derivative of the mode being evaluated.
    phase_mode: float
        The phase of the mode being evaluated.
    f_mode: float
        The frequency of the mode being evaluated.
    fdot_mode: float
        The frequency derivative of the mode being evaluated.
    one_over_fdot: float
        The inverse of the frequency derivative of the mode being evaluated.
    T: float
        The duration of the segment being evaluated.
    use_midpoint: bool
        Whether to evaluate the mode parameters at the midpoint of the segment, or at the beginning.

    Returns:
    -------
    fct: complex
        The Fresnel waveform for the given mode, within a time-segment. Evaluated at f_bin.
    """
    # rt2fdot = sqrt(2 * fdot_mode)

    # prefac = amp_mode_prefactor

    if fdot_mode >= 0:
        sign_fdot = 1
    else:
        sign_fdot = -1

    delta_f_norm = (f_mode - f_bin) * one_over_fdot

    if use_midpoint:
        phase_fac = exp(
            1j
            * (
                phase_mode
                - pi * sign_fdot * fdot_mode * (delta_f_norm**2)
                - 2 * pi * f_bin * T / 2
            )
        )  # The extra phase factor at the end accounts for the fact that the mode parameters are evaluated at the midpoint of the segment, rather than the beginning.
        # The arguments to the Fresnel integrals are also shifted by T/2 to account for this.
        v0_C = sqrt2fdot * (-T / 2 + delta_f_norm)
        vT_C = sqrt2fdot * (T / 2 + delta_f_norm)
    else:
        phase_fac = exp(
            1j * (phase_mode - pi * sign_fdot * fdot_mode * (delta_f_norm**2))
        )
        v0_C = sqrt2fdot * delta_f_norm
        vT_C = sqrt2fdot * (T + delta_f_norm)

    S0, C0 = _fresnel(v0_C)
    ST, CT = _fresnel(vT_C)

    fct = (
        amp_mode_prefactor * phase_fac * sign_fdot * (CT - C0 + 1j * (ST - S0))
    )
    return fct
