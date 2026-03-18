from cmath import exp
from math import cos, pi, sin, sqrt

from numba import njit

from ..constants import clight


@njit
def Ylms(cosi, phi):
    two_cosi = cosi * 2
    cosi2 = cosi * cosi
    expn = exp(1j * 2 * phi)
    Y_22 = (1 + two_cosi + cosi2) * expn
    Y_2m2 = (1 - two_cosi + cosi2) * expn.conjugate()
    return Y_22, Y_2m2


@njit
def fill_P_lm(P_lm, parameters):
    """
    Eqn 16. from https://arxiv.org/pdf/2003.00357

    On GPU or CPU we could use arrays and a loop to make this less cumbersome
    However, I don't know how to do that in an agnostic fashion (this is a device function) so we're going to write it all out
    """
    cosi = parameters[0]
    pol = parameters[1]
    ecliptic_long = parameters[2]
    ecliptic_lat = parameters[3]

    Y_lm_22, Y_lm_2minus2 = Ylms(cosi, 0.0)

    cl = cos(ecliptic_long)
    c2l = cos(2 * ecliptic_long)
    s2l = sin(2 * ecliptic_long)
    cl2 = cl * cl
    sl = sin(ecliptic_long)
    sl2 = sl * sl

    cb = cos(ecliptic_lat)
    cb2 = cb * cb
    sb = sin(ecliptic_lat)
    sb2 = sb * sb

    cbsl = cb * sl
    cbsbsl = cbsl * sb

    cbcl = cb * cl
    cbclsb = cbcl * sb

    clslprod = cl * (1 + sb2) * sl

    Pp00 = cl2 * sb2 - sl2
    Pp01 = clslprod
    Pp02 = -cbclsb
    Pp10 = clslprod
    Pp11 = -cl2 + sb2 * sl2
    Pp12 = -cbsbsl
    Pp20 = -cbclsb
    Pp21 = -cbsbsl
    Pp22 = cb2

    Pc00 = sb * s2l
    Pc01 = -c2l * sb
    Pc02 = -cbsl
    Pc10 = -c2l * sb
    Pc11 = -2 * cl * sb * sl
    Pc12 = cbcl
    Pc20 = -cbsl
    Pc21 = cbcl
    Pc22 = 0

    expn = exp(-2j * pol)

    part = expn * (Pp00 + 1j * Pc00)
    P_lm[0, 0] = 0.5 * (Y_lm_22 * part + (Y_lm_2minus2 * part).conjugate())
    part = expn * (Pp01 + 1j * Pc01)
    P_lm[0, 1] = 0.5 * (Y_lm_22 * part + (Y_lm_2minus2 * part).conjugate())
    part = expn * (Pp02 + 1j * Pc02)
    P_lm[0, 2] = 0.5 * (Y_lm_22 * part + (Y_lm_2minus2 * part).conjugate())
    part = expn * (Pp10 + 1j * Pc10)
    P_lm[1, 0] = 0.5 * (Y_lm_22 * part + (Y_lm_2minus2 * part).conjugate())
    part = expn * (Pp11 + 1j * Pc11)
    P_lm[1, 1] = 0.5 * (Y_lm_22 * part + (Y_lm_2minus2 * part).conjugate())
    part = expn * (Pp12 + 1j * Pc12)
    P_lm[1, 2] = 0.5 * (Y_lm_22 * part + (Y_lm_2minus2 * part).conjugate())
    part = expn * (Pp20 + 1j * Pc20)
    P_lm[2, 0] = 0.5 * (Y_lm_22 * part + (Y_lm_2minus2 * part).conjugate())
    part = expn * (Pp21 + 1j * Pc21)
    P_lm[2, 1] = 0.5 * (Y_lm_22 * part + (Y_lm_2minus2 * part).conjugate())
    part = expn * (Pp22 + 1j * Pc22)
    P_lm[2, 2] = 0.5 * (Y_lm_22 * part + (Y_lm_2minus2 * part).conjugate())


@njit
def fill_k(k, parameters):
    ecliptic_long = parameters[2]
    ecliptic_lat = parameters[3]
    cos_lat = cos(ecliptic_lat)
    k[0] = -cos_lat * cos(ecliptic_long)
    k[1] = -cos_lat * sin(ecliptic_long)
    k[2] = -sin(ecliptic_lat)


@njit
def dot_three_unpack_sum(a, b, i, j):
    return (
        a[0] * (b[i, 0] + b[j, 0])
        + a[1] * (b[i, 1] + b[j, 1])
        + a[2] * (b[i, 2] + b[j, 2])
    )


@njit
def dot_R(k, p):
    return (
        k[0] * (p[0, 0] + p[1, 0] + p[2, 0])
        + k[1] * (p[0, 1] + p[1, 1] + p[2, 1])
        + k[2] * (p[0, 2] + p[1, 2] + p[2, 2])
    ) / 3


@njit
def dot_three_unpack2(a, b, k):
    return a[0] * b[k, 0] + a[1] * b[k, 1] + a[2] * b[k, 2]


@njit
def _matrix_res_pro(n_0, n_1, n_2, p):
    return (
        n_0 * p[0, 0] * n_0
        + n_0 * p[0, 1] * n_1
        + n_0 * p[0, 2] * n_2
        + n_1 * p[1, 0] * n_0
        + n_1 * p[1, 1] * n_1
        + n_1 * p[1, 2] * n_2
        + n_2 * p[2, 0] * n_0
        + n_2 * p[2, 1] * n_1
        + n_2 * p[2, 2] * n_2
    )


@njit
def sinc(x):
    if x == 0:
        return 1
    else:
        return sin(x) / x


@njit
def threevector_diff_norm(x, i, j):
    return (
        (x[i, 0] - x[j, 0]) ** 2
        + (x[i, 1] - x[j, 1]) ** 2
        + (x[i, 2] - x[j, 2]) ** 2
    ) ** 0.5


@njit
def build_ysrl(f, k, n, p, P_lm):
    """
    Builds all the y_slr

    y_sr = [[1,2],
            [2,1],
            [1,3],
            [3,1],
            [2,3],
            [3,2]]
    """

    iomega_over_2c = 1j * pi * f / (clight)

    L12 = threevector_diff_norm(p, 0, 1)
    L23 = threevector_diff_norm(p, 1, 2)
    L13 = threevector_diff_norm(p, 0, 2)

    for j in range(3):
        n[0, j] = (p[2, j] - p[1, j]) / L23  # 2->3
        n[1, j] = (p[2, j] - p[0, j]) / L13  # 1->3
        n[2, j] = (p[1, j] - p[0, j]) / L12  # 1->2

    # expf_21 = expf_12
    expf_12 = exp(
        iomega_over_2c * (L12 + dot_three_unpack_sum(k, p, 0, 1))
    )  # 1->2, 2->1
    expf_23 = exp(
        iomega_over_2c * (L23 + dot_three_unpack_sum(k, p, 1, 2))
    )  # 2->3, 3->2
    expf_13 = exp(
        iomega_over_2c * (L13 + dot_three_unpack_sum(k, p, 0, 2))
    )  # 1->3, 3->1

    k_dot_n1 = dot_three_unpack2(k, n, 0)
    k_dot_n2 = dot_three_unpack2(k, n, 1)
    k_dot_n3 = dot_three_unpack2(k, n, 2)

    sinc_factor_123 = sinc(pi * f * L12 / clight * (1 - k_dot_n3))
    sinc_factor_213 = sinc(pi * f * L12 / clight * (1 + k_dot_n3))
    sinc_factor_132 = sinc(pi * f * L13 / clight * (1 - k_dot_n2))
    sinc_factor_312 = sinc(pi * f * L13 / clight * (1 + k_dot_n2))
    sinc_factor_231 = sinc(pi * f * L23 / clight * (1 - k_dot_n1))
    sinc_factor_321 = sinc(pi * f * L23 / clight * (1 + k_dot_n1))

    prod_1 = _matrix_res_pro(n[0, 0], n[0, 1], n[0, 2], P_lm)  # 2->3, 3->2
    prod_2 = _matrix_res_pro(n[1, 0], n[1, 1], n[1, 2], P_lm)  # 1->3, 3->1
    prod_3 = _matrix_res_pro(n[2, 0], n[2, 1], n[2, 2], P_lm)  # 1->2, 2->1

    # the slr indices are those in sinc_factor
    ysrl_123 = iomega_over_2c * L12 * sinc_factor_123 * expf_12 * prod_3
    ysrl_213 = iomega_over_2c * L12 * sinc_factor_213 * expf_12 * prod_3
    ysrl_132 = iomega_over_2c * L13 * sinc_factor_132 * expf_13 * prod_2
    ysrl_312 = iomega_over_2c * L13 * sinc_factor_312 * expf_13 * prod_2
    ysrl_231 = iomega_over_2c * L23 * sinc_factor_231 * expf_23 * prod_1
    ysrl_321 = iomega_over_2c * L23 * sinc_factor_321 * expf_23 * prod_1
    return (
        ysrl_123,
        ysrl_213,
        ysrl_132,
        ysrl_312,
        ysrl_231,
        ysrl_321,
        L12,
        L23,
        L13,
    )


@njit
def get_XYZ_TFs(f, P_lm, k, p, n, tdi2):
    (
        ysrl_123,
        ysrl_213,
        ysrl_132,
        ysrl_312,
        ysrl_231,
        ysrl_321,
        L12,
        L23,
        L13,
    ) = build_ysrl(f, k, n, p, P_lm)

    x1 = 2j * pi * f * L12 / clight
    x2 = 2j * pi * f * L23 / clight
    x3 = 2j * pi * f * L13 / clight

    z1 = exp(x1)
    z2 = exp(x2)
    z3 = exp(x3)

    # X = (G31 + G13*z2 - G21 - G12*z3) - (G31*z3*z3 + G13*z2*z3*z3 - G21*z2*z2 - G12*z3*z2*z2)
    # Y = (G12 + G21*z3 - G32 - G23*z1) - (G12*z1*z1 + G21*z3*z1*z1 - G32*z3*z3 - G23*z1*z3*z3)
    # Z = (G23 + G32*z1 - G13 - G31*z2) - (G23*z2*z2 + G32*z1*z2*z2 - G13*z1*z1 - G31*z2*z1*z1)

    X = (ysrl_312 + z2 * ysrl_132 - ysrl_213 - ysrl_123 * z3) - (
        ysrl_312 * z3 * z3
        + ysrl_132 * z2 * z3 * z3
        - ysrl_213 * z2 * z2
        - ysrl_123 * z3 * z2 * z2
    )
    Y = (ysrl_123 + z3 * ysrl_213 - ysrl_321 - ysrl_231 * z1) - (
        ysrl_123 * z1 * z1
        + ysrl_213 * z3 * z1 * z1
        - ysrl_321 * z3 * z3
        - ysrl_231 * z1 * z3 * z3
    )
    Z = (ysrl_231 + z1 * ysrl_321 - ysrl_132 - ysrl_312 * z2) - (
        ysrl_231 * z2 * z2
        + ysrl_321 * z1 * z2 * z2
        - ysrl_132 * z1 * z1
        - ysrl_312 * z2 * z1 * z1
    )

    if tdi2:
        X *= 1.0 - z2 * z2 * z3 * z3
        Y *= 1.0 - z3 * z3 * z1 * z1
        Z *= 1.0 - z1 * z1 * z2 * z2

    return X.conjugate(), Y.conjugate(), Z.conjugate()


@njit
def get_AET_TFs(f, P_lm, k, p, n, tdi2):
    X, Y, Z = get_XYZ_TFs(f, P_lm, k, p, n, tdi2)

    A = (Z - X) / 2.0**0.5
    E = (X - 2.0 * Y + Z) / 6.0**0.5
    T = (X + Y + Z) / 3.0**0.5

    return A, E, T


@njit
def get_AET_TFs_equal_armlength(f, P_lm, k, p, n, tdi2):
    L = L = 2.5e9  # remove with unequal

    ysrl_123, ysrl_213, ysrl_132, ysrl_312, ysrl_231, ysrl_321, _, _, _ = (
        build_ysrl(f, k, n, p, P_lm)
    )

    x = pi * f * L / clight
    z = exp(2j * x)
    factorAE = 1j * sqrt(2) * sin(2.0 * x) * z
    factorT = 2.0 * sqrt(2) * sin(2.0 * x) * sin(x) * exp(1j * 3.0 * x)

    Araw = (
        (1.0 + z) * (ysrl_312 + ysrl_132)
        - ysrl_231
        - z * ysrl_321
        - ysrl_213
        - z * ysrl_123
    )
    Eraw = (
        1
        / sqrt(3)
        * (
            (1.0 - z) * (ysrl_132 - ysrl_312)
            + (2.0 + z) * (ysrl_123 - ysrl_321)
            + (1.0 + 2.0 * z) * (ysrl_213 - ysrl_231)
        )
    )
    Traw = (
        2
        / sqrt(6)
        * (ysrl_213 - ysrl_123 + ysrl_321 - ysrl_231 + ysrl_132 - ysrl_312)
    )

    if tdi2:
        tdi2_factor = 1.0 - exp(8j * x)
    else:
        tdi2_factor = 1.0

    A = (tdi2_factor * factorAE * Araw).conjugate()
    E = (tdi2_factor * factorAE * Eraw).conjugate()
    T = (tdi2_factor * factorT * Traw).conjugate()
    return A, E, T
