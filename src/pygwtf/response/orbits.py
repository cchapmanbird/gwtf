import numpy as np
import scipy
from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord

from ..constants import R, f_m, clight

def Orbit(
    t, initial_ecliptic_longitude, e, n, initial_orientation_of_constellation=0
):
    """
    Function to calculate the position of the spacecrafts in the constellation as a function of time.
    https://arxiv.org/pdf/gr-qc/0311069

    Args:
        t (jax.numpy.array): Time array
        initial_ecliptic_longitude (float): Initial ecliptic longitude of the constellation
        e (float): Eccentricity of the orbit
        n (int): Spacecraft number
        initial_orientation_of_constellation (float): Initial orientation of the constellation

    Returns:
        pos (jax.numpy.array): Position of the spacecraft as a function of time

    """
    alpha = 2 * np.pi * f_m * t + initial_ecliptic_longitude
    beta = 2 * np.pi * n / 3 + initial_orientation_of_constellation

    x = (
        R * np.cos(alpha)
        + 0.5 * e * R * (np.cos(2 * alpha - beta) - 3 * np.cos(beta))
        + 1
        / 8
        * e**2
        * R
        * (
            3 * np.cos(3 * alpha - 2 * beta)
            - 10 * np.cos(alpha)
            - 5 * np.cos(alpha - 2 * beta)
        )
    )

    y = (
        R * np.sin(alpha)
        + 0.5 * e * R * (np.sin(2 * alpha - beta) - 3 * np.sin(beta))
        + 1
        / 8
        * e**2
        * R
        * (
            3 * np.sin(3 * alpha - 2 * beta)
            - 10 * np.sin(alpha)
            + 5 * np.sin(alpha - 2 * beta)
        )
    )

    z = -np.sqrt(3) * e * R * np.cos(alpha - beta) + np.sqrt(3) * e**2 * R * (
        np.cos(alpha - beta) ** 2 + 2 * np.sin(alpha - beta) ** 2
    )

    pos = np.array([x, y, z])

    return pos


def get_analytic_orbits(t_tranche):
    a = R
    e0 = 2.5e9 / (2 * np.sqrt(3) * a)
    kappa = 0  # - 1.7967674211761813 - 20/180*np.pi # GWSPACE INITIAL ECLIPTIC LONGITUDE

    pos_0 = Orbit(
        t_tranche,
        kappa,  # initial ecliptic latitude
        e0,
        n=0,
    )
    pos_1 = Orbit(t_tranche, kappa, e0, n=1)
    pos_2 = Orbit(t_tranche, kappa, e0, n=2)
    return np.array([pos_0, pos_1, pos_2]).transpose(2, 0, 1).copy()


def read_in_mojito_orbit(filepath: str):
    """
    Plugin for reading in Mojito orbit files for LISA spacecraft positions.
    NOTE: Should really be deprecated by the mojito package, but this is a quick and dirty way to get the orbits in for now.
    NOTE: Should also be reading in and building interpolants on the light-travel-times as thats available too.
    NOTE: This is *not* compatible with the
        additional trim that is needed to get rid of the low frequency noise that comes from the high pass
        filter of mojito data... But I will deal with that once it causes problems.
    NOTE: Hardcoded to the 0.4hz data for now.

    Args:
        filepath (str): Path to the Mojito orbit file
    Returns:
        times (numpy.array): Time array
        positions (list): List of numpy arrays for each spacecraft position
    """
    import h5py

    orbit_datafile = h5py.File(filepath, "r")

    # Extract positions (N_times_orbital,#Spacecraft,#{x,y,z})
    positions = orbit_datafile["tcb"]["x"][()]

    N_times_orbital = positions.shape[0]

    # Orbital data time spacing.
    orbital_dt = orbit_datafile.attrs["dt"]

    # print('Orbital dt (s):',orbital_dt)

    # Orbital time array (Covers 61171239.327664s to 197671239.32766402s)
    orbital_times = orbital_dt * np.arange(
        N_times_orbital
    )  # t0 for this is actually 61171239.327664

    # Trying Sylvains reference times: from https://gitlab.esa.int/lisa-sgs/wav/sobhb/-/blob/main/mojitolight_testing/sobhb_mojitolight_testing.ipynb?ref_type=heads

    # Orbital data is given from
    # Our data collection starts at 97729939.827664 (0.4Hz), this relative to the t0 above is 36558700.5
    # Our data collection ends at 160846189.07766402 (0.4Hz), this relative to the t0 above is 99674949.75000001

    data_collection_start = 36558700.5  # This is our new t0 (literally will be zero), relative to the start of the orbital data.
    data_collection_end = 99674949.75000001  # TBD, need to check this. :(
    filter_indices = np.where(
        (orbital_times >= data_collection_start)
        & (orbital_times <= data_collection_end)
    )[0]

    positions = positions[
        filter_indices, :, :
    ]  # Shape (N_times_orbital_filtered,#Spacecraft,#{x,y,z})
    orbital_times = orbital_times[
        filter_indices
    ]  # Shape (N_times_orbital_filtered,)

    # Shifting all times to zero otherwise will have to add time shift to times coming out of waveform.
    orbital_times_shifted = (
        orbital_times - data_collection_start
    )  # Shift to start at zero.

    pos_sc1 = positions[:, 0, :].T  # Shape (3,N_times_orbital)
    pos_sc2 = positions[:, 1, :].T
    pos_sc3 = positions[:, 2, :].T

    # Convert from ICRS (orbit file coordinates.) to ecliptic coords using astropy, for each spacecraft.
    final_postions = []
    for i in range(3):
        c_icrs = SkyCoord(
            positions[:, i, 0].T,
            positions[:, i, 1].T,
            positions[:, i, 2].T,
            frame="icrs",
            unit="m",
            representation_type="cartesian",
        )

        c_ecliptic = c_icrs.transform_to(BarycentricMeanEcliptic)
        c_ecliptic.representation_type = "cartesian"
        final_postions.append(
            np.array(
                [c_ecliptic.x.value, c_ecliptic.y.value, c_ecliptic.z.value]
            )
        )

    pos_sc1 = final_postions[0]
    pos_sc2 = final_postions[1]
    pos_sc3 = final_postions[2]

    return (pos_sc1, pos_sc2, pos_sc3, orbital_times_shifted)

def read_in_mojito_ltts(filepath: str):
    '''
    Read in mojito LTT file. 
    Should really do this in a better way, currently requires a unique LTT file which contains the LTTs for each link (assuming symmetric LTTs for now). 
    Following same time conventions as the orbit file mojito. (Assuming that the LTT times are already subset to the data collection time in preprocessing.)

    Only extracts: {12, 23, 31} links for now, but can extract full set {12, 23, 31, 13, 32, 21}

    Args:
        filepath (str): Path to the Mojito LTT file
    Returns:
        ltts (numpy.array): Array of shape (N_times, 3) containing the LTTs for the 3 links (12, 23, 31) as a function of time.
    '''
    ltt_file = np.load(filepath)

    L12 = ltt_file['ltts'][:, 0]
    L23 = ltt_file['ltts'][:, 1]
    L31 = ltt_file['ltts'][:, 2]

    LTT_times_shifted = ltt_file['ltt_times'] - ltt_file['ltt_times'][0] # Shift to start at zero, same convention as the orbit file.


    return(L12, L23, L31, LTT_times_shifted)

def generate_mojito_orbit_splines_resample(mojito_orbit_filepath: str, 
                                           mojito_ltt_filepath: str,
                                           t_sft: np.ndarray):
    """
    Generates orbit splines from a mojito orbit file. Then evaluate them on SFT time grid.

    :param mojito_orbit_filepath: Description
    :param dt: Description

    """
    sc1, sc2, sc3, mojito_orbital_times = read_in_mojito_orbit(
        mojito_orbit_filepath
    )

    p = np.array([sc1, sc2, sc3])

    cubic_temp_interpolant = scipy.interpolate.CubicSpline(
        x=mojito_orbital_times, y=p.T, axis=0, extrapolate=True
    )  # This is a cubic interpolant over the original orbital times.

    # Evaluate the cubic interpolant on the SFT time grid
    SFT_midpoint_times = (t_sft[1:] + t_sft[:-1]) / 2

    p_fine = cubic_temp_interpolant(SFT_midpoint_times).T

    L12, L23, L31, LTT_times = read_in_mojito_ltts(
        mojito_ltt_filepath
    )

    Ls = np.array([L12, L23, L31])

    cubic_temp_interpolant_LTTs = scipy.interpolate.CubicSpline(
        x=LTT_times, y=Ls.T, axis=0, extrapolate=True
    )  # This is a cubic interpolant over the original orbital times.

    LTTs_fine = cubic_temp_interpolant_LTTs(SFT_midpoint_times)


    return p_fine, LTTs_fine

def get_analytic_ltts(spacecraft_orbits):
    """
    Get the analytic light travel times for each link as a function of time, given the spacecraft orbits.
    This is used for TDI generation.

    Args:
        spacecraft_orbits (numpy.array): Array of shape (nT, 3, 3) containing the positions of the 3 spacecraft as a function of time.

    Returns:
        ltts (numpy.array): Array of shape (nT, 3) containing the LTTs for the 3 links (12, 23, 31) as a function of time.
    """
    sc1 = spacecraft_orbits[:, 0, :]
    sc2 = spacecraft_orbits[:, 1, :]
    sc3 = spacecraft_orbits[:, 2, :]

    L12 = np.linalg.norm(sc1 - sc2, axis=1)/clight
    L23 = np.linalg.norm(sc2 - sc3, axis=1)/clight
    L31 = np.linalg.norm(sc3 - sc1, axis=1)/clight

    return np.array([L12, L23, L31]).T
