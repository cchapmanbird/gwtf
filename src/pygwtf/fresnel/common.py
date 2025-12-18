from numba import njit
from math import cos, sin, pi, sqrt
from cmath import exp

@njit
def _fresnel(x):
    ax = abs(x)

    pix = pi * ax
    halfpix2 = 0.5 * pix * ax
    cospix2 = cos(halfpix2)
    sinpix2 = sin(halfpix2)

    if ax < 6:
        fx = (1 + 0.926 * ax) / (2 + 1.792*ax + 3.104*ax**2)
        gx = 1 / (2 + 4.142*ax + 3.492*ax**2 + 6.67*ax**3)
        Sx_approx = 0.5 - fx * cospix2 - gx * sinpix2
        Cx_approx = 0.5 + fx * sinpix2 - gx * cospix2

    else:
        Sx_approx = 0.5 - cos(halfpix2) / pix
        Cx_approx = 0.5 + sin(halfpix2) / pix

    if x < 0:
        Sx_approx = -Sx_approx
        Cx_approx = -Cx_approx
    
    return Sx_approx, Cx_approx

@njit
def _fresnel_kernel(f_bin, amp_mode, phase_mode, f_mode, fdot_mode, T):
    rt2fdot = sqrt(2 * fdot_mode)

    prefac = amp_mode / rt2fdot

    delta_f_norm = (f_mode - f_bin) / fdot_mode
    phase_fac = exp(1j*(phase_mode - pi*fdot_mode*(delta_f_norm**2)))

    v0_C = rt2fdot * delta_f_norm
    vT_C = rt2fdot * (T + delta_f_norm)

    S0, C0 = _fresnel(v0_C)
    ST, CT = _fresnel(vT_C)

    fct = prefac * phase_fac * (CT - C0 + 1j*(ST - S0))
    return fct
