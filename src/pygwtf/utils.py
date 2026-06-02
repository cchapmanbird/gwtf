from numba import jit


@jit
def complex_inner_product(h1, h2, inv_psd, df):
    ''''
    Usual noise-weighted inner product between two time-series in the frequency domain.
    Base of every statistic.

    Parameters
    ----------
    h1 : array_like or float
        First time-series in the frequency domain.
    h2 : array_like or float
        Second time-series in the frequency domain.
    inv_psd : array_like or float
        Reciprocal (1/psd) of the power spectral density of the noise. Prefolded
        outside the kernel so the inner loop multiplies instead of dividing
        (fp64 division is far slower than multiply on GPU).
    df : float or float
        Frequency resolution of the time-series.

    Returns
    -------
    array_like or float
        Noise-weighted inner product of the two time-series.

    '''
    return 4 * df * h1.conjugate() * h2 * inv_psd
