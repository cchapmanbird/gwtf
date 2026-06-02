from cmath import exp
from math import cos, pi, sin, sqrt

from numba import njit

from ..constants import clight


@njit
def Ylms(cosi, phi):
    ''''
    Normalised spherical harmonics for Y_{2,2} and Y_{2,-2}.

    Parameters:
    ----------
    cosi: float
        cosine of the inclination angle of the source (angle between the line of sight and the orbital angular momentum vector)
    phi: float
        Reference phase rotation of the source (angle between the line of sight and the major axis of the binary's orbit)
        NOTE: Usually set to zero within our code. Rotations are absorbed in the waveform phase definition itself. 
    '''
    Y_22_norm = sqrt(5 / (64 * pi))
    two_cosi = cosi * 2
    cosi2 = cosi * cosi
    expn = exp(1j * 2 * phi)
    Y_22 = Y_22_norm * (1 + two_cosi + cosi2) * expn
    Y_2m2 = Y_22_norm * (1 - two_cosi + cosi2) * expn.conjugate()
    return Y_22, Y_2m2


@njit
def fill_P_lm(P_lm, parameters):
    """
    Polarization matrices from Eqn 16. in https://arxiv.org/pdf/2003.00357

    Parameters:
    ----------
    P_lm: array (3,3)
        array to be filled with the P_lm coefficients for the source, as defined in https://arxiv.org/pdf/2003.00357
    parameters: array (4,)
                [0] cosi: cosine of the inclination angle of the source (angle between the line of sight and the orbital angular momentum vector)
                [1] pol: polarization angle of the source (angle between the line of sight and the major axis of the binary's orbit)
                [2] ecliptic_long: ecliptic longitude of the source
                [3] ecliptic_lat: ecliptic latitude of the source
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
    '''
    Wavevector pointing from the source to the centre of the SSB frame, in ecliptic coordinates.
    Parameters:
    ----------
    k : array (3,)
        array to be filled with the components of the wavevector pointing from the source to the centre of the SSB frame.
    parameters: array (4,)
            [0] cosi: cosine of the inclination angle of the source (angle between the line of sight and the orbital angular momentum vector)
            [1] pol: polarization angle of the source (angle between the line of sight and the major axis of the binary's orbit)
            [2] ecliptic_long: ecliptic longitude of the source
            [3] ecliptic_lat: ecliptic latitude of the source
    '''
    ecliptic_long = parameters[2]
    ecliptic_lat = parameters[3]
    cos_lat = cos(ecliptic_lat)
    k[0] = -cos_lat * cos(ecliptic_long)
    k[1] = -cos_lat * sin(ecliptic_long)
    k[2] = -sin(ecliptic_lat)


@njit
def dot_three_unpack_sum(a, b, i, j):
    ''''
    Computes the sum a[0] * (b[i, 0] + b[j, 0]) + a[1] * (b[i, 1] + b[j, 1]) + a[2] * (b[i, 2] + b[j, 2])

    Parameters:
    ----------
    a: array (3,)
        3-vector
    b: array (3,3)
        3x3 matrix
    i: int
        first index for b
    j: int
        second index for b

    Returns:
    -------
    result: float
        the result of the sum a[0] * (b[i, 0] + b[j, 0]) + a[1] * (b[i, 1] + b[j, 1]) + a[2] * (b[i, 2] + b[j, 2])
    '''

    return (
        a[0] * (b[i, 0] + b[j, 0])
        + a[1] * (b[i, 1] + b[j, 1])
        + a[2] * (b[i, 2] + b[j, 2])
    )


@njit
def dot_R(k, p):
    '''
    Computes the dot product of the wavevector k with the sum of the position vectors p[i] for i=0,1,2.

    Parameters:
    ----------
    k: array (3,)
        wavevector pointing from the source to the centre of the SSB frame, in ecliptic coordinates.
    p: array (3,3)
        positions of the spacecraft in the SSB frame (Ecliptic coordinates) 

    Returns:
    -------
    result: float
        the result of the dot product of the wavevector k with the sum of the position vectors
    '''
    return (
        k[0] * (p[0, 0] + p[1, 0] + p[2, 0])
        + k[1] * (p[0, 1] + p[1, 1] + p[2, 1])
        + k[2] * (p[0, 2] + p[1, 2] + p[2, 2])
    ) / 3


@njit
def dot_three_unpack2(a, b, k):
    '''
    Computes the sum a[0] * b[k, 0] + a[1] * b[k, 1] + a[2] * b[k, 2]
    
    Parameters:
    ----------
    a: array (3,)
        3-vector
    b: array (3,3)
        3x3 matrix
    k: int
        index for b

    Returns:
    -------
    result: float
        the result of the sum a[0] * b[k, 0] + a[1] * b[k, 1] + a[2] * b[k, 2]
    '''
    return a[0] * b[k, 0] + a[1] * b[k, 1] + a[2] * b[k, 2]


@njit
def _matrix_res_pro(n_0, n_1, n_2, p):
    '''
    Custom type of matrix product that appears in the transfer function calculations. Computes the sum of n[i] * p[i, j] * n[j] for i,j=0,1,2.

    Parameters:
    n_0, n_1, n_2: float
        components of the unit vector pointing along the arm of the interferometer.
    p: array (3,3)
        P_lm coefficients for the source, as defined in https://arxiv.org/pdf/2003.00357
    Returns:
    -------
    result: float
        the result of the sum of n[i] * p[i, j] * n[j]
    '''

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
    '''
    Sinc function, sin(x)/x

    Parameters:
    ----------
    x: float
        input to the sinc function
    Returns:
    -------
    result: float
        the result of the sinc function, sin(x)/x. Returns 1 if x=0 to avoid division by zero.

    '''
    if x == 0:
        return 1
    else:
        return sin(x) / x


@njit
def threevector_diff_norm(x, i, j):
    '''
    Computes the norm of the difference between two 3-vectors x[i] and x[j].

    Parameters:
    ----------
    x: array (3,3)
        array of 3-vectors
    i: int
        index of the first vector
    j: int
        index of the second vector  

    Returns:
    -------
    result: float
        the norm of the difference between x[i] and x[j], computed as sqrt((x[i, 0] - x[j, 0])^2 + (x[i, 1] - x[j, 1])^2 + (x[i, 2] - x[j, 2])^2)
    '''


    return (
        (x[i, 0] - x[j, 0]) ** 2
        + (x[i, 1] - x[j, 1]) ** 2
        + (x[i, 2] - x[j, 2]) ** 2
    ) ** 0.5


@njit
def build_ysrl(f, k, n, p, Ls, P_lm):
    """
    Builds transfer function for *single* link responses. 
    s: source index (0,1,2 for the 3 spacecraft)
    l: link index (0,1,2 for the 3 arms)
    r: response index (0,1 for the two directions of the link)
    NOTE: Usually the convention is 'slr' but here we use a slightly different convention 'srl'

    y_sr = [[1,2],
        [2,1],
        [1,3],
        [3,1],
        [2,3],
        [3,2]]

    Parameters:
    ----------
    f: float
        frequency at which to evaluate the transfer function
    k: array (3,)
        Unit vector pointing from the source to the centre of the SSB frame
    n: array (3,3) (to be filled in within the function)
        unit vectors pointing along the arms of the interferometer. 
    p: array (3,3)
        positions of the spacecraft in the SSB frame (Ecliptic coordinates)
    Ls: array (3,)
        arm lengths of the interferometer (metres)
    P_lm: array (3,3)
        P_lm coefficients for the source, as defined in https://arxiv.org/pdf/2003.00357
    
    Returns:
    -------
    ysrl_123, ysrl_213, ysrl_132, ysrl_312, ysrl_231, ysrl_321: complex
        the single link responses for the 6 possible combinations of s, r, l. The indices correspond to the following combinations of s, r, l:
        srl: 1->2, 2->1, 1->3, 3->1, 2->3, 3->2 respectively. Note that the convention here is 'srl' instead of 'slr'.

    """

    iomega_over_2c = 1j * pi * f / (clight)

    L12 = Ls[0]
    L23 = Ls[1]
    L13 = Ls[2]

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
        ysrl_321
    )


@njit
def get_XYZ_TFs(f, P_lm, k, p, Ls, n, tdi2):
    '''
    Construct the single link response functions and then combine them appropriately to get the X,Y,Z transfer functions. 
    Deals with slowly varying, unequal arm-lengths.

    In this case we are assuming 
        - Arm lengths are unequal but slowly varying
        - Delays commute. 
        - The delay factors effectively map to the (1-D^4) that is found in TDI 1.5 case to go from TDI-1 to TDI-2 (equal arm-length case)
    Under these approximations the TDI equations below reduce exactly to the equations for TDI-2. 
    Checking the Rosetta stone equations for TDI-2, simplifying them, assuming commuting delays, one recovers the TDI-equations used here I think. 

    
    X = (G31 + G13*z2 - G21 - G12*z3) - (G31*z3*z3 + G13*z2*z3*z3 - G21*z2*z2 - G12*z3*z2*z2)
    Y = (G12 + G21*z3 - G32 - G23*z1) - (G12*z1*z1 + G21*z3*z1*z1 - G32*z3*z3 - G23*z1*z3*z3)
    Z = (G23 + G32*z1 - G13 - G31*z2) - (G23*z2*z2 + G32*z1*z2*z2 - G13*z1*z1 - G31*z2*z1*z1)

    
    Parameters:
    ----------
    f: float
        frequency at which to evaluate the transfer functions
    P_lm: array (3,3)
        P_lm coefficients for the source, as defined in https://arxiv.org/pdf/2003.00357
    k: array (3,)
        Unit vector pointing from the source to the centre of the SSB frame
    p: array (3,3)
        positions of the spacecraft in the SSB frame (Ecliptic coordinates)
    Ls: array (3,)
        arm lengths of the interferometer (metres) 
    n: array (3,3) (to be filled in)
        unit vectors pointing along the arms of the interferometer.
    tdi2: bool
        whether to apply the TDI 2.0 correction factor (1 - z1^2 z2^2 z3^2) to the transfer functions.  

    '''
    (
        ysrl_123,
        ysrl_213,
        ysrl_132,
        ysrl_312,
        ysrl_231,
        ysrl_321
    ) = build_ysrl(f, k, n, p, Ls, P_lm)

    L12 = Ls[0]
    L23 = Ls[1]
    L13 = Ls[2]

    x1 = 2j * pi * f * L12 / clight
    x2 = 2j * pi * f * L23 / clight
    x3 = 2j * pi * f * L13 / clight

    z1 = exp(x1)
    z2 = exp(x2)
    z3 = exp(x3)
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
def get_AET_TFs(f, P_lm, k, p, Ls, n, tdi2):
    ''''
    'Master function' - Called directly by fresnel to get the AET transfer functions. (Default transfer function for analysis within this library)
    Calls get_XYZ_TFs to get the X,Y,Z transfer functions, and then combines them appropriately to get A,E,T.
    Deals with slowly varying, unequal arm-lengths. 

    Parameters: 
    ----------
    f: float
        frequency at which to evaluate the transfer functions
    P_lm: array (3,3) 
        P_lm coefficients for the source, as defined in https://arxiv.org/pdf/2003.00357
    k: array (3,) 
        Unit vector pointing from the source to the centre of the SSB frame
    p: array (3,3) 
        positions of the spacecraft in the SSB frame (Ecliptic coordinates)
    Ls: array (3,)
        arm lengths of the interferometer (metres)
    n: array (3,3)
        unit vectors pointing along the arms of the interferometer.
    tdi2: bool
        whether to apply the TDI 2.0 correction factor (1 - z1^2 z2^2 z3^2) to the transfer functions.
    
    Returns:
    -------
    A, E, T: complex
        The A, E, T transfer functions for the source at frequency f.
    '''


    X, Y, Z = get_XYZ_TFs(f, P_lm, k, p, Ls, n, tdi2)

    A = (Z - X) / 2.0**0.5
    E = (X - 2.0 * Y + Z) / 6.0**0.5
    T = (X + Y + Z) / 3.0**0.5

    return A, E, T


@njit
def get_AET_TFs_equal_armlength(f, P_lm, k, p, n, tdi2):
    ''''
    DEPRECATED FUNCTION. 
    Doing what is essentially the same thing as get_AET_TFs but assuming equal arm-lengths and doing some algebraic simplifications to get more compact expressions for the A,E,T transfer functions.
    
    Parameters:
    ----------
    f: float
        frequency at which to evaluate the transfer functions
    P_lm: array (3,3)
        P_lm coefficients for the source, as defined in https://arxiv.org/pdf/
    k: array (3,)
        Unit vector pointing from the source to the centre of the SSB frame
    p: array (3,3)
        positions of the spacecraft in the SSB frame (Ecliptic coordinates)
    n: array (3,3)
        unit vectors pointing along the arms of the interferometer.
    tdi2: bool
        whether to apply the TDI 2.0 correction factor (1 - D^4) to the transfer functions.
    
    '''
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
