from math import log, pi, sqrt

from numba import cuda, jit

from ...constants import gamma_e


@jit
def _calculate_T(v, v0, e0, eta):
    """
    Calculates the value of T using the (TaylorT2) Equation 6.7b from arXiv:1605.00304v2.

    Parameters:
      v (float or array of floats): (pi*M*f)^{1/3} with f as frequencies of the binary system.
      v0 (float): (pi*M*f_0)^{1/3}, with f_0 as the initial GW frequency of the binary
      e0 (float): The initial eccentricity of the binary system.
      eta (float): The symmetric mass ratio of the binary system.

    Returns:
      T (float or array of floats): The calculated value(s) of T at fs provided into v.

    """
    term1 = (
        1
        + ((743 / 252) + (11 / 3) * eta) * v**2
        - (32 / 5) * pi * v**3
        + ((3058673 / 508032) + (5429 / 504) * eta + (617 / 72) * eta**2)
        * v**4
        + (-(7729 / 252) + (13 / 3) * eta) * pi * v**5
    )

    term2 = (
        (-10052469856691 / 23471078400)
        + (6848 / 105) * gamma_e
        + (128 / 3) * pi**2
        + ((3147553127 / 3048192) - (451 / 12) * pi**2) * eta
        - (15211 / 1728) * eta**2
        + (25565 / 1296) * eta**3
        + (3424 / 105) * log(16 * v**2)
    ) * v**6

    term3 = (
        ((-15419335 / 127008) - (75703 / 756) * eta + (14809 / 378) * eta**2)
        * pi
        * v**7
    )

    term4 = (
        -(157 / 43)
        * e0**2
        * (v0 / v) ** (19 / 3)
        * (
            1
            + ((17592719 / 5855472) + (1103939 / 209124) * eta) * v**2
            + ((2833 / 1008) - (197 / 36) * eta) * v0**2
            - (2819123 / 384336) * pi * v**3
            + (377 / 72) * pi * v0**3
            + (
                (955157839 / 302766336)
                + (1419591809 / 88306848) * eta
                + (91918133 / 6307632) * eta**2
            )
            * v**4
            + (
                (49840172927 / 5902315776)
                - (42288307 / 26349624) * eta
                - (217475983 / 7528464) * eta**2
            )
            * v**2
            * v0**2
            + (
                -(1193251 / 3048192)
                - (66317 / 9072) * eta
                + (18155 / 1296) * eta**2
            )
            * v0**4
            - ((166558393 / 12462660) + (679533343 / 28486080) * eta)
            * pi
            * v**5
            + (-(7986575459 / 387410688) + (555367231 / 13836096) * eta)
            * pi
            * v**3
            * v0**2
            + ((6632455063 / 421593984) + (416185003 / 15056928) * eta)
            * pi
            * v**2
            * v0**3
            + ((764881 / 90720) - (949457 / 22680) * eta) * pi * v0**5
            + (
                -(2604595243207055311 / 16582316889600000)
                + (31576663 / 2472750) * gamma_e
                + (924853159 / 40694400) * pi**2
                + ((17598403624381 / 86141905920) - (886789 / 180864) * pi**2)
                * eta
                + (203247603823 / 5127494400) * eta**2
                + (2977215983 / 109874880) * eta**3
                + (226088539 / 7418250) * log(2)
                - (65964537 / 2198000) * log(3)
                + (31576663 / 4945500) * (log(16) + log(v**2))
            )
            * v**6
            + (
                (2705962157887 / 305188466688)
                + (14910082949515 / 534079816704) * eta
                - (99638367319 / 2119364352) * eta**2
                - (18107872201 / 227074752) * eta**3
            )
            * v**4
            * v0**2
            - (1062809371 / 27672192) * pi**2 * v**3 * v0**3
            + (
                -(20992529539469 / 17848602906624)
                - (15317632466765 / 637450103808) * eta
                + (8852040931 / 2529563904) * eta**2
                + (20042012545 / 271024704) * eta**3
            )
            * v**2
            * v0**4
            + (
                (26531900578691 / 168991764480)
                - (3317 / 126) * gamma_e
                + (122833 / 10368) * pi**2
                + ((9155185261 / 548674560) - (3977 / 1152) * pi**2) * eta
                - (5732473 / 1306368) * eta**2
                - (3090307 / 139968) * eta**3
                + (87419 / 1890) * log(2)
                - (26001 / 560) * log(3)
                - (3317 / 252) * (log(16) + log(v0**2))
            )
            * v0**6
        )
    )

    T = term1 + term2 + term3 + term4
    return T


@jit
def _Lambda_f_vec(v, v0, e0, eta):
    """
    Calculates the value of Lambda_f using the (TaylorT2) Equation 6.6a from arXiv:1605.00304v2.

    Parameters:
      v (float): (pi*M*f)^{1/3} with f as frequencies of the binary system.
      v0 (float): (pi*M*f_0)^{1/3}, with f_0 as the initial GW frequency of the binary
      e0 (float): The initial eccentricity of the binary system.
      eta (float): The symmetric mass ratio of the binary system.

    Returns:
      lambda_ (float or array of floats): The calculated value(s) of lambda_f at fs provided into v.

    """

    term1 = 1 + ((3715 / 1008) + (55 / 12) * eta) * v**2 - 10 * pi * v**3
    term2 = (
        (15293365 / 1016064) + (27145 / 1008) * eta + (3085 / 144) * eta**2
    ) * v**4
    term3 = ((38645 / 2016) - (65 / 24) * eta) * pi * log(v**3) * v**5
    term4 = (
        (12348611926451 / 18776862720)
        - (1712 / 21) * gamma_e
        - (160 / 3) * pi**2
        + ((-15737765635 / 12192768) + (2255 / 48) * pi**2) * eta
        + (76055 / 6912) * eta**2
        - (127825 / 5184) * eta**3
        - (856 / 21) * log(16 * v**2)
    ) * v**6
    term5 = (
        (
            (77096675 / 2032128)
            + (378515 / 12096) * eta
            - (74045 / 6048) * eta**2
        )
        * pi
        * v**7
    )
    term6 = (
        -(785 / 272)
        * e0**2
        * (v0 / v) ** (19 / 3)
        * (
            1
            + ((6955261 / 2215584) + (436441 / 79128) * eta) * v**2
            + ((2833 / 1008) - (197 / 36) * eta) * v0**2
            - (1114537 / 141300) * pi * v**3
            + (377 / 72) * pi * v0**3
            + (
                (377620541 / 107433216)
                + (561233971 / 31334688) * eta
                + (36339727 / 2238192) * eta**2
            )
            * v**4
            + (
                (19704254413 / 2233308672)
                - (16718633 / 9970128) * eta
                - (85978877 / 2848608) * eta**2
            )
            * v**2
            * v0**2
            + (
                -(1193251 / 3048192)
                - (66317 / 9072) * eta
                + (18155 / 1296) * eta**2
            )
            * v0**4
            - ((131697334 / 8456805) + (268652717 / 9664920) * eta) * pi * v**5
            + ((-3157483321 / 142430400) + (219563789 / 5086800) * eta)
            * pi
            * v**3
            * v0**2
            + ((2622133397 / 159522048) + (164538257 / 5697216) * eta)
            * pi
            * v**2
            * v0**3
            + ((764881 / 90720) - (949457 / 22680) * eta) * pi * v0**5
            + (
                (-204814565759250649 / 1061268280934400)
                + (12483797 / 791280) * gamma_e
                + (365639621 / 13022208) * pi**2
                + (
                    (34787542048195 / 137827049472)
                    - (8764775 / 1446912) * pi**2
                )
                * eta
                + (80353703837 / 1640798208) * eta**2
                + (5885194385 / 175799808) * eta**3
                + (89383841 / 2373840) * log(2)
                - (26079003 / 703360) * log(3)
                + (12483797 / 1582560) * log(16 * v**2)
            )
            * v**6
            + (
                (1069798992653 / 108292681728)
                + (5894683956785 / 189512193024) * eta
                - (39391912661 / 752032512) * eta**2
                - (7158926219 / 80574912) * eta**3
            )
            * v**4
            * v0**2
            - (420180449 / 10173600) * pi**2 * v**3 * v0**3
            + (
                (-8299372143511 / 6753525424128)
                - (6055808184535 / 241197336576) * eta
                + (3499644089 / 957132288) * eta**2
                + (7923586355 / 102549888) * eta**3
            )
            * v**2
            * v0**4
            + (
                (26531900578691 / 168991764480)
                - (3317 / 126) * gamma_e
                + (122833 / 10368) * pi**2
                + ((9155185261 / 548674560) - (3977 / 1152) * pi**2) * eta
                - (5732473 / 1306368) * eta**2
                - (3090307 / 139968) * eta**3
                + (87419 / 1890) * log(2)
                - (26001 / 560) * log(3)
                - (3317 / 252) * log(16 * v0**2)
            )
            * v0**6
        )
    )

    lamba_ = term1 + term2 + term3 + term4 + term5 + term6

    return lamba_


@jit
def _Xi_ecc_term(theta, theta_0, eta):
    e_terms = (
        1
        + (-7647061 / 70265664 + 209353 / 836496 * eta) * theta**2
        + (4445 / 3456 - 185 / 288 * eta) * theta_0**2
        - 256883 / 3074688 * pi * theta**3
        + 61 / 2880 * pi * theta_0**3
        + (
            -1172375466061 / 6586984406016
            + 439689491 / 8477457408 * eta
            + 91590343 / 1867059072 * eta**2
        )
        * theta**4
        + (
            1024948915 / 1170505728
            - 1026871 / 6967296 * eta
            + 14971 / 165888 * eta**2
        )
        * theta_0**4
    )
    cross_terms = (
        (
            -4855883735 / 34691162112
            + 495545305 / 1264781952 * eta
            - 1046765 / 6511104 * eta**2
        )
        * theta**2
        * theta_0**2
    )
    pi_terms = (
        (49653615200993 / 96325793464320 - 449103385817 / 1146735636480 * eta)
        * pi
        * theta**5
        + (-12410299 / 17418240 + 576391 / 1451520 * eta) * pi * theta_0**5
        + (-1141844935 / 10626121728 + 47523355 / 885510144 * eta)
        * pi
        * theta**3
        * theta_0**2
        + (-466470721 / 202365112320 + 12770533 / 2409108480 * eta)
        * pi
        * theta**2
        * theta_0**3
    )
    eta_terms = (
        (
            -744458420948735 / 3252088301027328
            + 12376149277895 / 68362216538112 * eta
            + 896664159665 / 30111928713216 * eta**2
            - 457951715 / 14532784128 * eta**3
        )
        * theta**4
        * theta_0**2
    )
    cross_pi2 = -15669863 / 8855101440 * pi**2 * theta**3 * theta_0**3
    high_order_eta = (
        (
            -7837846874888815 / 82246362193723392
            + 76760406851419 / 326374453149696 * eta
            - 60493466573 / 1295136718848 * eta**2
            + 3134223763 / 138764648448 * eta**3
        )
        * theta**2
        * theta_0**4
    )
    log_terms = (
        71857244107315089475141 / 32866417392257433600000
        - 75936937 / 158256000 * gamma_e
        - 9863961577 / 44275507200 * pi**2
        + (-672050112032567 / 87421728522240 + 1214953 / 3858432 * pi**2) * eta
        + 380720733285643 / 1129197326745600 * eta**2
        - 231474834959 / 8065695191040 * eta**3
        - 53821 / 14836500 * log(2)
        - 65964537 / 140672000 * log(3)
        - 75936937 / 158256000 * log(theta)
    ) * theta**6 + (
        -55579234653596057 / 23361421521715200
        + 15943 / 40320 * gamma_e
        + 3968617 / 16588800 * pi**2
        + (21736949245913 / 1685528248320 - 12751 / 24576 * pi**2) * eta
        - 1742350567 / 4013162496 * eta**2
        + 4790953 / 143327232 * eta**3
        + 8453 / 7560 * log(2)
        - 26001 / 35840 * log(3)
        + 15943 / 40320 * log(theta_0)
    ) * theta_0**6

    return (
        e_terms
        + cross_terms
        + pi_terms
        + eta_terms
        + cross_pi2
        + high_order_eta
        + log_terms
    )


@jit
def _Xi(theta, theta_0, e0, eta):
    term1 = (
        1
        + (743 / 2688 + 11 / 32 * eta) * theta**2
        - 3 / 10 * pi * theta**3
        + (1855099 / 14450688 + 56975 / 258048 * eta + 371 / 2048 * eta**2)
        * theta**4
        + (-7729 / 21504 + 13 / 256 * eta) * pi * theta**5
    )

    term2 = (
        -720817631400877 / 288412611379200
        + 107 / 280 * gamma_e
        + 53 / 200 * pi**2
        + (25302017977 / 4161798144 - 451 / 2048 * pi**2) * eta
        - 30913 / 1835008 * eta**2
        + 235925 / 1769472 * eta**3
        + 107 / 280 * log(2 * theta)
    ) * theta**6

    term3 = (
        (
            -188516689 / 433520640
            - 97765 / 258048 * eta
            + 141769 / 1290240 * eta**2
        )
        * pi
        * theta**7
    )

    theta_ratio = (theta_0 / theta) ** (19 / 3)

    e_term_part = _Xi_ecc_term(theta, theta_0, eta)

    eccentricity_term = -471 / 344 * e0**2 * theta_ratio * e_term_part

    # Compute the final value of Xi
    Xi = term1 + term2 + term3 + eccentricity_term

    return Xi


@jit
def _foft(t, tc, e0, M, eta):
    tau = (tc - t) * eta / (5 * M)
    theta = tau ** (-1 / 8)
    tau_0 = tc * eta / (5 * M)
    theta_0 = tau_0 ** (-1 / 8)

    Xi = _Xi(theta, theta_0, e0, eta)

    return (1 / 8 / tau ** (3 / 8)) * Xi / (pi * M)


@jit
def _dXi_dtheta(theta, theta_0, e0, eta, include_ecc_term):
    term1_dtheta = (
        2 * (743 / 2688 + 11 / 32 * eta) * theta
        - 3 / 10 * pi * 3 * theta**2
        + 4
        * (1855099 / 14450688 + 56975 / 258048 * eta + 371 / 2048 * eta**2)
        * theta**3
        + 5 * (-7729 / 21504 + 13 / 256 * eta) * pi * theta**4
    )

    term2_common = (
        -720817631400877 / 288412611379200
        + 107 / 280 * gamma_e
        + 53 / 200 * pi**2
        + (25302017977 / 4161798144 - 451 / 2048 * pi**2) * eta
        - 30913 / 1835008 * eta**2
        + 235925 / 1769472 * eta**3
    )

    term2_dtheta = (
        6 * (term2_common + 107 / 280 * log(2 * theta)) + 107 / 280
    ) * theta**5

    term3_dtheta = (
        7
        * (
            -188516689 / 433520640
            - 97765 / 258048 * eta
            + 141769 / 1290240 * eta**2
        )
        * pi
        * theta**6
    )

    if include_ecc_term:
        de_terms_dtheta = (
            2 * (-7647061 / 70265664 + 209353 / 836496 * eta) * theta
            - 3 * 256883 / 3074688 * pi * theta**2
            + 4
            * (
                -1172375466061 / 6586984406016
                + 439689491 / 8477457408 * eta
                + 91590343 / 1867059072 * eta**2
            )
            * theta**3
        )

        dcross_terms_dtheta = (
            2
            * (
                -4855883735 / 34691162112
                + 495545305 / 1264781952 * eta
                - 1046765 / 6511104 * eta**2
            )
            * theta
            * theta_0**2
        )

        dpi_terms_dtheta = (
            5
            * (
                49653615200993 / 96325793464320
                - 449103385817 / 1146735636480 * eta
            )
            * pi
            * theta**4
            + 3
            * (-1141844935 / 10626121728 + 47523355 / 885510144 * eta)
            * pi
            * theta**2
            * theta_0**2
            + 2
            * (-466470721 / 202365112320 + 12770533 / 2409108480 * eta)
            * pi
            * theta
            * theta_0**3
        )

        deta_terms_dtheta = (
            4
            * (
                -744458420948735 / 3252088301027328
                + 12376149277895 / 68362216538112 * eta
                + 896664159665 / 30111928713216 * eta**2
                - 457951715 / 14532784128 * eta**3
            )
            * theta**3
            * theta_0**2
        )

        dcross_pi2_dtheta = (
            -15669863 / 8855101440 * pi**2 * 3 * theta**2 * theta_0**3
        )

        dhigh_order_eta_dtheta = (
            2
            * (
                -7837846874888815 / 82246362193723392
                + 76760406851419 / 326374453149696 * eta
                - 60493466573 / 1295136718848 * eta**2
                + 3134223763 / 138764648448 * eta**3
            )
            * theta
            * theta_0**4
        )

        log_terms_common = (
            71857244107315089475141 / 32866417392257433600000
            - 75936937 / 158256000 * gamma_e
            - 9863961577 / 44275507200 * pi**2
            + (-672050112032567 / 87421728522240 + 1214953 / 3858432 * pi**2)
            * eta
            + 380720733285643 / 1129197326745600 * eta**2
            - 231474834959 / 8065695191040 * eta**3
            - 53821 / 14836500 * log(2)
            - 65964537 / 140672000 * log(3)
        )

        dlog_terms_dtheta = (
            6 * (log_terms_common - 75936937 / 158256000 * log(theta))
            - 75936937 / 158256000
        ) * theta**5

        d_eterm_part_dtheta = (
            de_terms_dtheta
            + dcross_terms_dtheta
            + dpi_terms_dtheta
            + deta_terms_dtheta
            + dcross_pi2_dtheta
            + dhigh_order_eta_dtheta
            + dlog_terms_dtheta
        )

        e_term_part = _Xi_ecc_term(theta, theta_0, eta)

        theta_ratio = (theta_0 / theta) ** (19 / 3)
        dtheta_ratio_dtheta = -19 / 3 * theta_ratio / theta
        deccentricity_term_dtheta = (
            -471
            / 344
            * e0**2
            * (
                dtheta_ratio_dtheta * e_term_part
                + theta_ratio * d_eterm_part_dtheta
            )
        )

    else:
        deccentricity_term_dtheta = 0

    return (
        term1_dtheta + term2_dtheta + term3_dtheta + deccentricity_term_dtheta
    )


@jit
def _d_foft_dt(t, tc, e0, M, eta, include_ecc_term=False):
    tau = (tc - t) * eta / (5 * M)
    theta = tau ** (-1 / 8)
    tau_0 = tc * eta / (5 * M)
    theta_0 = tau_0 ** (-1 / 8)

    # Compute d(tau)/dt
    dtau_dt = -eta / (5 * M)
    dtheta_dt = (-1 / 8) * tau ** (-9 / 8) * dtau_dt

    Xi = _Xi(theta, theta_0, e0, eta)

    # Compute d(Xi)/d(theta)
    dXi_dtheta = _dXi_dtheta(theta, theta_0, e0, eta, include_ecc_term)

    # Compute d(Xi)/dt = d(Xi)/d(theta) * d(theta)/dt
    dXi_dt = dXi_dtheta * dtheta_dt

    # Compute d/dt of (1 / 8 / tau**(3/8))
    d_prefactor_dt = -3 / 8 * (1 / 8) * tau ** (-11 / 8) * dtau_dt

    # Final derivative
    d_foft_dt = d_prefactor_dt * Xi / (pi * M) + (
        1 / 8 / tau ** (3 / 8)
    ) * dXi_dt / (pi * M)

    return d_foft_dt


@jit
def _get_phi_f_fdot(t, parameters):
    # M, eta, cosinc, e0, D, f0, coalescence_phase

    M = parameters[0]
    eta = parameters[1]
    e0 = parameters[3]
    f0 = parameters[5]
    phi_coal = parameters[6]
    tc = parameters[7]

    v0 = (pi * M * f0) ** (1 / 3)

    tau = (tc - t) * eta / (5 * M)
    theta = tau ** (-1 / 8)
    tau_0 = tc * eta / (5 * M)
    theta_0 = tau_0 ** (-1 / 8)

    dtau_dt = -eta / (5 * M)
    dtheta_dt = (-1 / 8) * tau ** (-9 / 8) * dtau_dt

    Xi = _Xi(theta, theta_0, e0, eta)

    tau_prefac = 1 / 8 / tau ** (3 / 8) / (pi * M)

    freq = tau_prefac * Xi

    # Compute d(Xi)/d(theta)
    dXi_dtheta = _dXi_dtheta(theta, theta_0, e0, eta, False)

    # Compute d(Xi)/dt = d(Xi)/d(theta) * d(theta)/dt
    dXi_dt = dXi_dtheta * dtheta_dt

    # Compute d/dt of (1 / 8 / tau**(3/8))
    d_prefactor_dt = -3 / 8 * dtau_dt / tau

    # Final derivative
    fdot = tau_prefac * (d_prefactor_dt * Xi + dXi_dt)

    v = (pi * M * freq) ** (1 / 3)

    phi0 = 2 * (
        phi_coal - 1 / (32 * v**5 * eta) * _Lambda_f_vec(v, v0, e0, eta)
    )
    return phi0, freq, fdot


@jit
def _get_amplitude(t, f, fdot, parameters):
    '''
    Note: Amplitude of *h_lm* not h_plus

    See E.g. Eqn 79 of https://arxiv.org/pdf/0710.0614
    '''

    M = parameters[0]
    eta = parameters[1]
    D = parameters[4]
    v = (pi * M * f) ** (1 / 3)
    A = 8*sqrt(pi/5)*eta * M / D * (v) ** 2
    return A


@jit
def _get_hplus_hcross(hlm, parameters):
    cosi = parameters[2]
    Y22_norm = sqrt(5 / (64 * pi))
    hplus = -hlm * Y22_norm * (1 + cosi**2)
    hcross = -hlm * Y22_norm * (2j * cosi)
    return hplus, hcross


@jit
def _get_time_to_coalescence(M, eta, e0, f0):
    v0 = (pi * M * f0) ** (1 / 3)
    tc = 5 / 256 * M / eta / (v0**8) * _calculate_T(v0, v0, e0, eta)
    return tc


@jit
def _get_time_to_coalescence_cpu_wrap(t_coal, parameters):
    for i in range(len(t_coal)):
        t_coal[i] = _get_time_to_coalescence(
            parameters[i, 0],
            parameters[i, 1],
            parameters[i, 3],
            parameters[i, 5],
        )


@cuda.jit
def _get_time_to_coalescence_gpu_wrap(t_coal, parameters):
    idx = cuda.grid(1)
    if idx < t_coal.size:
        t_coal[idx] = _get_time_to_coalescence(
            parameters[idx, 0],
            parameters[idx, 1],
            parameters[idx, 3],
            parameters[idx, 5],
        )


@jit
def _get_time_to_f(f, tc, M, eta, e0, f0):
    v0 = (pi * M * f0) ** (1 / 3)
    v = (pi * M * f) ** (1 / 3)
    tc_from_f = 5 / 256 * M / eta / (v**8) * _calculate_T(v, v0, e0, eta)
    return tc - tc_from_f


@jit
def _get_time_to_f_cpu_wrap(t_from_f, f, tc, parameters):
    for i in range(len(t_from_f)):
        t_from_f[i] = _get_time_to_f(
            f,
            tc[i],
            parameters[i, 0],
            parameters[i, 1],
            parameters[i, 3],
            parameters[i, 5],
        )


@cuda.jit
def _get_time_to_f_gpu_wrap(t_to_f, f, tc, parameters):
    idx = cuda.grid(1)
    if idx < t_to_f.size:
        t_to_f[idx] = _get_time_to_f(
            f,
            tc[idx],
            parameters[idx, 0],
            parameters[idx, 1],
            parameters[idx, 3],
            parameters[idx, 5],
        )
