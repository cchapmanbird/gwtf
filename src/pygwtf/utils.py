from numba import njit

@njit
def complex_inner_product(h1, h2, psd, df):
    return 4 * df * h1. conjugate() * h2 / psd
