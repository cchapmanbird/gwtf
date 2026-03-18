import lisaconstants as lc
from scipy.special import digamma

# Note list of constants available in https://lisa-constants-3fcef9.io.esa.int/list.html

clight = (
    lc.C
)  # 299792458.0    # Speed of light in vacuum [m.s^-1] (CODATA 2014)
G = lc.NEWTON_CONSTANT  # 6.67408e-11         # Newtonian constant of graviation [m^3.kg^-1.s^-2] (CODATA 2014)
gamma_e = -digamma(1)  # Euler-Mascheroni constant

GMsun = lc.GM_SUN  # 1.32712442099e+20   ## GMsun in SI (http://asa.usno.navy.mil/static/files/2016/Astronomical_Constants_2016.pdf)
pc = lc.PARSEC_METER  # 3.08567758149136727e+16   # Parsec [m] (XXIX General Assembly of the International Astronomical Union, RESOLUTION B2 on recommended zero points for the absolute and apparent bolometric magnitude scales, 2015)

YRSID_SI = (
    lc.SIDEREALYEAR_J2000DAY * 24 * 60 * 60
)  # 31558149.763545600  ## siderial year [sec] (http://hpiers.obspm.fr/eop-pc/models/constants.html)

MTsun = (
    GMsun / clight**3
)  ## Solar mass in seconds [s] (http://asa.usno.navy.mil/static/files/2016/Astronomical_Constants_2016.pdf)

R = lc.au  # Astronomical unit [m]

f_m = 1 / YRSID_SI  # 1/ year

# Armlength of LISA in seconds (light travel time between spacecraft, ASSUMING CONSTANT ARMLENGTHS)
Armlength = 2.5e9 / clight
