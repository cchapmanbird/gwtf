from math import log, pi, sqrt

from numba import cuda, jit, njit

from ...constants import gamma_e


@njit
def phase(x, sigma, delta, eta, s):
    """'
    3.5PN in aligned spin effects. Circular.

    Link to PNpedia for the expression:
    https://github.com/davidtrestini/PNpedia/blob/275c95c8f6765d5628eeb2549d17250cd16e6617/Core%20post-Newtonian%20quantities/Circular%20orbits/Spinning/Nonprecessing/Without%20tidal%20effects/Waveform/phase.txt

    Parameters:
    ----------
        x (float): The PN expansion parameter, defined as (pi*M*f)^(2/3), where M is the total mass of the binary and f is the GW frequency.
        sigma (float): Reduced spin parameter (m2 * s2 - m1 * s1) / M
        delta (float): Mass difference parameter (m1 - m2) / M
        eta (float): Symmetric mass ratio m1 * m2 / M^2
        s (float): Spin parameter (m1**2 * s1 + m2**2 * s2) / (M**2)

    Returns:
    -------
        Phi_0_minus_phi (float): The value of Phi_0 - phi at the given x, sigma, delta, eta, and s.
    """

    Phi_0_minus_phi = (
        1
        + x * (3.685515873015873 + (55 * eta) / 12.0)
        + x**1.5 * (-10 * pi + (235 * s) / 6.0 + (125 * delta * sigma) / 8.0)
        + x**2
        * (
            15.051576475497606
            - 100 * s**2
            + (3085 * eta**2) / 144.0
            - 100 * s * delta * sigma
            - (405 * sigma**2) / 16.0
            + eta * (26.92956349206349 + 100 * sigma**2)
        )
        + x**3.5
        * (
            (-9018232555 * s) / 6.096384e6
            + (125925 * s**3) / 224.0
            - (170978035 * delta * sigma) / 387072.0
            + (379805 * s**2 * delta * sigma) / 448.0
            + (182755 * s * sigma**2) / 448.0
            + (1315 * delta * sigma**3) / 21.0
            + pi
            * (
                37.93888721576594
                - 200 * s**2
                - 200 * s * delta * sigma
                - (815 * sigma**2) / 16.0
            )
            + eta**2
            * (
                (-74045 * pi) / 6048.0
                + (835 * s) / 288.0
                + (7015 * delta * sigma) / 1152.0
                + (285 * s * sigma**2) / 8.0
                + (95 * delta * sigma**3) / 16.0
            )
            + eta
            * (
                (3329545 * s) / 3024.0
                - (95 * s**3) / 8.0
                + (2909765 * delta * sigma) / 5376.0
                - (285 * s**2 * delta * sigma) / 16.0
                - (385825 * s * sigma**2) / 224.0
                - (130615 * delta * sigma**3) / 448.0
                + pi * (31.292576058201057 + 200 * sigma**2)
            )
        )
        + x**3
        * (
            657.6504345051205
            - (1712 * gamma_e) / 21.0
            - (160 * pi**2) / 3.0
            + (7915 * s**2) / 63.0
            - (127825 * eta**3) / 5184.0
            + (2645 * s * delta * sigma) / 56.0
            - (1645 * sigma**2) / 128.0
            + pi * ((940 * s) / 3.0 + (745 * delta * sigma) / 6.0)
            + eta**2 * (11.003327546296296 - 120 * sigma**2)
            + eta
            * (
                -1290.7459270118156
                + (2255 * pi**2) / 48.0
                + 120 * s**2
                + 120 * s * delta * sigma
                + (5875 * sigma**2) / 112.0
            )
            - (3424 * log(2)) / 21.0
            - (856 * log(x)) / 21.0
        )
        + x**2.5
        * (
            (38645 * pi) / 1344.0
            - (555605 * s) / 2016.0
            - (15 * s**3) / 4.0
            - (41745 * delta * sigma) / 448.0
            - (45 * s**2 * delta * sigma) / 8.0
            - (45 * s * sigma**2) / 8.0
            - (15 * delta * sigma**3) / 8.0
            + eta
            * (
                (-65 * pi) / 16.0
                - (45 * s) / 8.0
                + (5 * delta * sigma) / 2.0
                + (45 * s * sigma**2) / 4.0
                + (15 * delta * sigma**3) / 8.0
            )
        )
        * log(x)
    ) / (32.0 * x**2.5 * eta)

    return Phi_0_minus_phi


@njit
def frequency(tau, sigma, delta, eta, s):
    """
    3.5PN in aligned spin effects. Circular.

    Obtained from x(tau) = F(tau).

    Link to PNpedia for the expression:
    https://github.com/davidtrestini/PNpedia/blob/275c95c8f6765d5628eeb2549d17250cd16e6617/Core%20post-Newtonian%20quantities/Circular%20orbits/Spinning/Nonprecessing/Without%20tidal%20effects/Waveform/chirp.txt


    Parameters:
    ----------
        tau (float): Defined as tau = (eta / 5) * (t-t_0), where t_0 is the initial time and t is the time variable. It is a dimensionless time parameter that measures the time to coalescence in units of the symmetric mass ratio eta.
        sigma (float): Reduced spin parameter (m2 * s2 - m1 * s1) / M
        delta (float): Mass difference parameter (m1 - m2) / M
        eta (float): Symmetric mass ratio m1 * m2 / M^2
        s (float): Spin parameter (m1**2 * s1 + m2**2 * s2) / (M**2)

    Returns:
    -------
        F (float): The value of frequency at the given tau, sigma, delta, eta, and s.
    """

    F = (
        (
            (-7729 * pi) / 172032.0
            + (110869 * s) / 258048.0
            + (8349 * delta * sigma) / 57344.0
            + eta
            * (
                (13 * pi) / 2048.0
                + (11 * s) / 1024.0
                - (3 * delta * sigma) / 1024.0
            )
        )
        / tau
        + (
            0.016046805176334857
            - (15 * s**2) / 128.0
            + (371 * eta**2) / 16384.0
            - (15 * s * delta * sigma) / 128.0
            - (243 * sigma**2) / 8192.0
            + eta * (0.027599031963045636 + (15 * sigma**2) / 128.0)
        )
        / tau**0.875
        + ((-3 * pi) / 80.0 + (47 * s) / 320.0 + (15 * delta * sigma) / 256.0)
        / tau**0.75
        + (0.03455171130952381 + (11 * eta) / 256.0) / tau**0.625
        + 1 / (8.0 * tau**0.375)
        + (
            -0.312407295555619
            + (107 * gamma_e) / 2240.0
            + (53 * pi**2) / 1600.0
            - (68381 * s**2) / 1.2288e6
            + (235925 * eta**3) / 1.4155776e7
            - (161 * pi * delta * sigma) / 2048.0
            - (132789 * sigma**2) / 7.340032e6
            + (225 * delta**2 * sigma**2) / 8192.0
            + s * ((-1269 * pi) / 6400.0 - (763 * delta * sigma) / 49152.0)
            + eta
            * (
                0.7599484976667336
                - (451 * pi**2) / 16384.0
                - (343 * s**2) / 4096.0
                - (343 * s * delta * sigma) / 4096.0
                + (53647 * sigma**2) / 786432.0
            )
            + eta**2 * (-0.002105781010219029 + (343 * sigma**2) / 4096.0)
            + (107 * log(2)) / 2240.0
            - (107 * log(tau)) / 17920.0
        )
        / tau**1.125
    ) / pi
    return F


@njit
def frequency_derivative(tau, sigma, delta, eta, s):
    """'
    Differentiated form of frequency above.

    With respect to t not tau. (Uses Chain rule to get dF/dt from dF/dtau, where tau is a function of t, using dTau/dt = -eta/5).

    Parameters:
    ----------
        tau (float): Defined as tau = (eta / 5) * (t-t_0), where t_0 is the initial time and t is the time variable. It is a dimensionless time parameter that measures the time to coalescence in units of the symmetric mass ratio eta.
        sigma (float): Reduced spin parameter (m2 * s2 - m1 * s1) / M
        delta (float): Mass difference parameter (m1 - m2) / M
        eta (float): Symmetric mass ratio m1 * m2 / M^2
        s (float): Spin parameter (m1**2 * s1 + m2**2 * s2) / (M**2)

    Returns:
    -------
        dFdt (float): The value of the derivative of frequency with respect to time at the given tau, sigma, delta, eta, and s.
    """
    dFdt = (
        eta
        * (
            (
                (-7729 * pi) / 860160.0
                + (110869 * s) / 1.29024e6
                + (8349 * delta * sigma) / 286720.0
                + eta
                * (
                    (13 * pi) / 10240.0
                    + (11 * s) / 5120.0
                    - (3 * delta * sigma) / 5120.0
                )
            )
            / tau**2
            + (
                0.0028081909058586
                - (21 * s**2) / 1024.0
                + (2597 * eta**2) / 655360.0
                - (21 * s * delta * sigma) / 1024.0
                - (1701 * sigma**2) / 327680.0
                + eta * (0.004829830593532986 + (21 * sigma**2) / 1024.0)
            )
            / tau**1.875
            + (
                (-9 * pi) / 1600.0
                + (141 * s) / 6400.0
                + (9 * delta * sigma) / 1024.0
            )
            / tau**1.75
            + (0.004318963913690476 + (11 * eta) / 2048.0) / tau**1.625
            + 3 / (320.0 * tau**1.375)
            + (
                -0.06909744507144285
                + (963 * gamma_e) / 89600.0
                + (477 * pi**2) / 64000.0
                - (205143 * s**2) / 1.6384e7
                + (47185 * eta**3) / 1.2582912e7
                - (1449 * pi * delta * sigma) / 81920.0
                - (1195101 * sigma**2) / 2.9360128e8
                + (405 * delta**2 * sigma**2) / 65536.0
                + s
                * (
                    (-11421 * pi) / 256000.0
                    - (2289 * delta * sigma) / 655360.0
                )
                + eta
                * (
                    0.17098841197501505
                    - (4059 * pi**2) / 655360.0
                    - (3087 * s**2) / 163840.0
                    - (3087 * s * delta * sigma) / 163840.0
                    + (160941 * sigma**2) / 1.048576e7
                )
                + eta**2
                * (-0.00047380072729928153 + (3087 * sigma**2) / 163840.0)
                + (963 * log(2)) / 89600.0
                - (963 * log(tau)) / 716800.0
            )
            / tau**2.125
        )
        / pi
    )

    return dFdt


@njit
def time_to_merger(x, sigma, delta, eta, s):
    """
    Series inverstion of x(tau) (used for frequency) to get t(x) in the form (NOTE: not tau, t):

    tc(x) = t_0 - t(x) (I think should be the other way around(?))

    Parameters:
    ---------
        x (float): The PN expansion parameter, defined as (pi*M*f)^(2/3), where M is the total mass of the binary and f is the GW frequency.
        sigma (float): Reduced spin parameter (m2 * s2 - m1 * s1) / M
        delta (float): Mass difference parameter (m1 - m2) / M
        eta (float): Symmetric mass ratio m1 * m2 / M^2
        s (float): Spin parameter (m1**2 * s1 + m2**2 * s2) / (M**2)

    Returns:
    -------
        tc (float): The value of time to merger at the given x, sigma, delta, eta, and s.
    """

    tc = (
        1
        / eta
        * (
            5 / (256.0 * x**4)
            + (5 * (743 + 924 * eta)) / (64512.0 * x**3)
            + (-48 * pi + 188 * s + 75 * delta * sigma) / (384.0 * x**2.5)
            + (
                5
                * (
                    -23187 * pi
                    + 221738 * s
                    + 3276 * pi * eta
                    + 5544 * s * eta
                    + 75141 * delta * sigma
                    - 1512 * delta * eta * sigma
                )
            )
            / (193536.0 * x**1.5)
            - (
                5
                * (
                    -3058673
                    + 20321280 * s**2
                    - 5472432 * eta
                    - 4353552 * eta**2
                    + 20321280 * s * delta * sigma
                    + 5143824 * sigma**2
                    - 20321280 * eta * sigma**2
                )
            )
            / (1.30056192e8 * x**2)
            + (
                -10052469856691
                + 1530761379840 * gamma_e
                + 1001432678400 * pi**2
                - 5883416985600 * pi * s
                - 2359029657600 * s**2
                + 24236159077900 * eta
                - 882121363200 * pi**2 * eta
                - 2253223526400 * s**2 * eta
                - 206607970800 * eta**2
                + 462992376000 * eta**3
                - 2331460454400 * pi * delta * sigma
                - 886871462400 * s * delta * sigma
                - 2253223526400 * s * delta * eta * sigma
                - 492159175200 * sigma**2
                + 733471200000 * delta**2 * sigma**2
                + 1948937760000 * eta * sigma**2
                + 2253223526400 * eta**2 * sigma**2
                + 658084331520 * log(2)
                + 1201719214080 * log(4)
                + 765380689920 * log(x)
            )
            / (1.20171921408e12 * x)
        )
    )

    return tc


@njit
def tau_to_x(tau, sigma, delta, eta, s):
    """
    Directly using x(tau) used for the frequency function

    PNpedia link:
    https://github.com/davidtrestini/PNpedia/blob/275c95c8f6765d5628eeb2549d17250cd16e6617/Core%20post-Newtonian%20quantities/Circular%20orbits/Spinning/Nonprecessing/Without%20tidal%20effects/Waveform/chirp.txt

    Parameters:
    ----------
        tau (float): Defined as tau = (eta / 5) * (t-t_0), where t_0 is the initial time and t is the time variable. It is a dimensionless time parameter that measures the time to coalescence in units of the symmetric mass ratio eta.
        sigma (float): Reduced spin parameter (m2 * s2 - m1 * s1) / M
        delta (float): Mass difference parameter (m1 - m2) / M
        eta (float): Symmetric mass ratio m1 * m2 / M^2
        s (float): Spin parameter (m1**2 * s1 + m2**2 * s2) / (M**2)

    Returns:
    -------
        x (float): The value of the dimensionless frequency at the given tau, sigma, delta, eta, and s.
    """

    x = (
        1
        + (
            (-113868647 * pi) / 4.3352064e8
            + (24532268147 * s) / 2.60112384e9
            + (21 * pi * s**2) / 16.0
            - (755 * s**3) / 192.0
            + (281190779 * delta * sigma) / 9.9090432e7
            + (21 * pi * s * delta * sigma) / 16.0
            - (4499 * s**2 * delta * sigma) / 768.0
            + (1711 * pi * sigma**2) / 5120.0
            - (33929 * s * sigma**2) / 20480.0
            - (325 * s * delta**2 * sigma**2) / 256.0
            - (24007 * delta * sigma**3) / 49152.0
            + eta**2
            * (
                (294941 * pi) / 3.87072e6
                + (3641 * s) / 122880.0
                - (6169 * delta * sigma) / 294912.0
            )
            + eta
            * (
                (-31821 * pi) / 143360.0
                - (33704749 * s) / 5.16096e6
                - (5756657 * delta * sigma) / 1.769472e6
                - (21 * pi * sigma**2) / 16.0
                + (1259 * s * sigma**2) / 192.0
                + (493 * delta * sigma**3) / 256.0
            )
        )
        / tau**0.875
        + (
            (-11891 * pi) / 53760.0
            + (357923 * s) / 161280.0
            + (96473 * delta * sigma) / 129024.0
            + eta
            * (
                (109 * pi) / 1920.0
                - (187 * s) / 5760.0
                - (79 * delta * sigma) / 1536.0
            )
        )
        / tau**0.625
        + (
            0.0770935689090451
            - (5 * s**2) / 8.0
            + (31 * eta**2) / 288.0
            - (5 * s * delta * sigma) / 8.0
            - (81 * sigma**2) / 512.0
            + eta * (0.12607990244708994 + (5 * sigma**2) / 8.0)
        )
        / sqrt(tau)
        + (-0.2 * pi + (47 * s) / 60.0 + (5 * delta * sigma) / 16.0)
        / tau**0.375
        + (0.18427579365079366 + (11 * eta) / 48.0) / tau**0.25
        + (
            -1.6730147506856445
            + (107 * gamma_e) / 420.0
            + pi**2 / 6.0
            - (47 * pi * s) / 48.0
            - (1583 * s**2) / 4032.0
            + (25565 * eta**3) / 331776.0
            - (149 * pi * delta * sigma) / 384.0
            - (529 * s * delta * sigma) / 3584.0
            - (671 * sigma**2) / 8192.0
            + (125 * delta**2 * sigma**2) / 1024.0
            + eta
            * (
                4.033581021911924
                - (451 * pi**2) / 3072.0
                - (3 * s**2) / 8.0
                - (3 * s * delta * sigma) / 8.0
                + (2325 * sigma**2) / 7168.0
            )
            + eta**2 * (-0.03438539858217592 + (3 * sigma**2) / 8.0)
            + (107 * log(2)) / 420.0
            - (107 * log(tau)) / 3360.0
        )
        / tau**0.75
    ) / (4.0 * tau**0.25)
    return x


@njit
def _get_amplitude(t, f, fdot, parameters):
    '''
    Newtonian Amplitude for the (2,2) mode. 

    Note: Amplitude of *h_lm* not h_plus

    See E.g. Eqn 79 of https://arxiv.org/pdf/0710.0614

    Parameters:
    ----------
        t (float): Time at which to evaluate the amplitude.
        f (float): Frequency at which to evaluate the amplitude.
        fdot (float): Time derivative of the frequency at time t.
        parameters (array-like): An array containing the parameters of the system, where:
            parameters[0] (float): Total mass M of the binary system.
            parameters[1] (float): Symmetric mass ratio eta of the binary system.
            parameters[3] (float): Distance D to the binary system. 

    Returns:
        A (float): The amplitude of the (2,2) mode at time t. 
    '''
    M = parameters[0]
    eta = parameters[1]
    D = parameters[3]
    v = (pi * M * f) ** (1 / 3)
    A = 8*sqrt(pi/5)*eta * M / D * (v) ** 2
    return A


@njit
def _get_hplus_hcross(hlm, parameters):
    '''
    Get hplus and hcross from hlm for the (2,2) mode.

    Assume the waveform amplitude is appropriately normalized such that hlm is the amplitude of the (2,2) mode. 
    Uses the normalised form of the spin-weighted spherical harmonic Y22. 

    Parameters:
    ----------
        hlm (float): The amplitude of the (2,2) mode of the waveform.
        parameters (array-like): An array containing the parameters of the system, where:
            parameters[2] (float): Cosine of the inclination angle of the binary system.

    Returns:
        hplus (float): The plus polarization of the gravitational wave.
        hcross (float): The cross polarization of the gravitational wave.        
    
    '''
    cosi = parameters[2]
    Y22_norm = sqrt(5 / (64 * pi))
    hplus = -hlm * Y22_norm * (1 + cosi**2)
    hcross = -hlm * Y22_norm * (2j * cosi)
    return hplus, hcross


@njit
def _get_time_to_coalescence(M, eta, f0, delta, sigma, s):
    ''''
    Get time to coalescence from a given starting frequency f0.
    
    Parameters:
    ----------
        M (float): Total mass of the binary system.
        eta (float): Symmetric mass ratio of the binary system.
        f0 (float): Starting frequency from which to calculate the time to coalescence.
        delta (float): Mass difference parameter (m1 - m2) / M
        sigma (float): Reduced spin parameter (m2 * s2 - m1 * s1) / M
        s (float): Spin parameter (m1**2 * s1 + m2**2 * s2) / M

    Returns:
    ---------
        tc (float): Time to coalescence from the given starting frequency f0.
    '''
    x0 = (pi * M * f0) ** (2 / 3)
    tc = time_to_merger(x0, sigma, delta, eta, s) * M
    return tc


@njit
def _get_time_to_coalescence_cpu_wrap(t_coal, parameters):
    '''
    CPU-wrapper for time-to-coalescence function. 
    Loops through each set of parameters and calculates time to merger 

    Parameters:
    ----------
        t_coal (array-like): An array to store the calculated time to coalescence (array to be filled in place).
        parameters (array-like): An array of shape (N, 12) containing the parameters for N different binary systems, where each row corresponds to a binary system and the columns correspond to the parameters in the following order:
            parameters[i, 0] (float): Total mass M of the binary system.
            parameters[i, 1] (float): Symmetric mass ratio eta of the binary system
            parameters[i, 4] (float): Starting frequency f0 from which to calculate the time to coalescence for the i-th binary system.
            parameters[i, 9] (float): Mass difference parameter delta for the i-th binary system.
            parameters[i, 10] (float): Reduced spin parameter sigma for the i-th binary system.
            parameters[i, 11] (float): Spin parameter s for the i-th binary system.
    
    '''
    for i in range(len(t_coal)):
        t_coal[i] = _get_time_to_coalescence(
            parameters[i, 0],#M
            parameters[i, 1],#eta
            parameters[i, 4],#f0
            parameters[i, 9],#delta
            parameters[i, 10],#sigma
            parameters[i, 11],#s
        )
        
@cuda.jit
def _get_time_to_coalescence_gpu_wrap(t_coal, parameters):
    '''
    GPU-wrapper for time-to-coalescence function.
    Each thread calculates the time to coalescence for a single set of parameters.

    Parameters:
    ----------
        t_coal (array-like): An array to store the calculated time to coalescence (array to be filled in place).
        parameters (array-like): An array of shape (N, 12) containing the parameters for N different binary systems, where each row corresponds to a binary system and the columns correspond to the parameters in the following order:
            parameters[i, 0] (float): Total mass M of the binary system.
            parameters[i, 1] (float): Symmetric mass ratio eta of the binary system
            parameters[i, 4] (float): Starting frequency f0 from which to calculate the time to coalescence for the i-th binary system.
            parameters[i, 9] (float): Mass difference parameter delta for the i-th binary system.
            parameters[i, 10] (float): Reduced spin parameter sigma for the i-th binary system.
            parameters[i, 11] (float): Spin parameter s for the i-th binary system.
    
    '''

    idx = cuda.grid(1) # Working out what index of the array this thread should operate on. 
    if idx < t_coal.size:
        t_coal[idx] = _get_time_to_coalescence(
            parameters[idx, 0],
            parameters[idx, 1],
            parameters[idx, 4],
            parameters[idx, 9],
            parameters[idx, 10],
            parameters[idx, 11],
        )


@njit
def _get_time_to_f(f, tc, M, eta, delta, sigma, s):
    '''
    Get time to coalescence from a given frequency f.
    Uses direct analytical inversion of the frequency function to get time as a function of frequency.

    Parameters:
    ----------
        f (float): Frequency.
        tc (float): Time to coalescence.
        M (float): Total mass.
        eta (float): Symmetric mass ratio.
        delta (float): Mass difference parameter.
        sigma (float): Reduced spin parameter.
        s (float): Spin parameter.

    Returns:
    -------
        t_to_f (float): Time to coalescence from the given frequency f.
    '''
    x = (pi * M * f) ** (2 / 3)
    tc_from_f = time_to_merger(x, sigma, delta, eta, s) * M
    return tc - tc_from_f


@njit
def _get_time_to_f_cpu_wrap(t_from_f, f, tc, parameters):
    '''
    CPU-wrapper for time-from-frequency function.
    Loops through each set of parameters and calculates time to merger from frequency f.
    Parameters:
    ----------
        t_from_f (array-like): An array to store the calculated time to coalescence
        f (float): Frequency from which to calculate the time to coalescence.
        tc (array-like): An array containing the time to coalescence for each set of
            parameters, where tc[i] corresponds to the time to coalescence for the i-th set of parameters.
        parameters (array-like): An array of shape (N, 12) containing the parameters for N different binary systems.
            parameters[i, 0] (float): Total mass M of the binary system.
            parameters[i, 1] (float): Symmetric mass ratio eta of the binary system
            parameters[i, 9] (float): Mass difference parameter delta for the i-th binary system.
            parameters[i, 10] (float): Reduced spin parameter sigma for the i-th binary system.
            parameters[i, 11] (float): Spin parameter s for the i-th binary system.

    '''
    for i in range(len(t_from_f)):
        t_from_f[i] = _get_time_to_f(
            f,
            tc[i],
            parameters[i, 0],
            parameters[i, 1],
            parameters[i, 9],
            parameters[i, 10],
            parameters[i, 11],
        )


@cuda.jit
def _get_time_to_f_gpu_wrap(t_to_f, f, tc, parameters):
    ''''
    GPU wrapper for time-from-frequency function.
    Each thread calculates the time to coalescence from frequency f for a single set of parameters
    Parameters:
    ----------
        t_to_f (array-like): An array to store the calculated time to coalescence
        f (float): Frequency from which to calculate the time to coalescence.
        tc (array-like): An array containing the time to coalescence for each set of
            parameters, where tc[i] corresponds to the time to coalescence for the i-th set of parameters.
        parameters (array-like): An array of shape (N, ) containing the parameters for N different binary systems, where each row corresponds to a binary system and the columns correspond to the parameters in the following order:
            parameters[i, 0] (float): Total mass M of the binary system.
            parameters[i, 1] (float): Symmetric mass ratio eta of the binary system
            parameters[i, 9] (float): Mass difference parameter delta for the i-th binary system.
            parameters[i, 10] (float): Reduced spin parameter sigma for the i-th binary system.
            parameters[i, 11] (float): Spin parameter s for the i-th binary system.


    '''
    idx = cuda.grid(1)
    if idx < t_to_f.size:
        t_to_f[idx] = _get_time_to_f(
            f,
            tc[idx],
            parameters[idx, 0],
            parameters[idx, 1],
            parameters[idx, 9],
            parameters[idx, 10],
            parameters[idx, 11],
        )


@njit
def _get_phi_f_fdot(t, parameters):
    '''
    Get the phase, frequency and frequency derivative at time t for a given set of parameters.

    Parameters:
    ----------
        t (float): Time at which to evaluate the phase, frequency and frequency derivative.
        parameters (array-like): An array containing the parameters of the system, where:
            parameters[0] (float): Total mass M of the binary system.
            parameters[1] (float): Symmetric mass ratio eta of the binary system.
            parameters[7] (float): Coalescence phase of the binary system.
            parameters[8] (float): Time to coalescence tc of the binary system.
            parameters[9] (float): Mass difference parameter delta of the binary system.
            parameters[10] (float): Reduced spin parameter sigma of the binary system.
            parameters[11] (float): Spin parameter s of the binary system.

    Returns: 
    --------
        phi (float): The phase of the gravitational wave at time t.
        freq (float): The frequency of the gravitational wave at time t.
        fdot (float): The time derivative of the frequency of the gravitational wave at time t.

    '''
    M = parameters[0]
    eta = parameters[1]
    coalecence_phase = parameters[7]
    tc = parameters[8]
    delta = parameters[9]
    sigma = parameters[10]
    s = parameters[11]

    tau = eta * (tc - t) / (5 * M)

    # Dimensionless frequency parameter x(tau)
    x = tau_to_x(tau, sigma, delta, eta, s)

    # dimensionfull frequency
    freq = (x ** (3 / 2)) / (pi * M)
    # GW phase
    phi = 2 * (coalecence_phase - phase(x, sigma, delta, eta, s))

    fdot = frequency_derivative(tau, sigma, delta, eta, s) / M**2

    return phi, freq, fdot
