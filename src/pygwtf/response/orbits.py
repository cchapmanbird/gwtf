from ..constants import f_m, R
import numpy as np

def Orbit(t,initial_ecliptic_longitude,e,n,initial_orientation_of_constellation=0):
    '''
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
    
    '''
    alpha = 2*np.pi*f_m*t + initial_ecliptic_longitude
    beta = 2*np.pi*n/3+initial_orientation_of_constellation

    x = (R*np.cos(alpha)+0.5*e*R*(np.cos(2*alpha-beta)-3*np.cos(beta))
         +1/8*e**2*R*(3*np.cos(3*alpha-2*beta)-10*np.cos(alpha)-5*np.cos(alpha-2*beta)))
         
    y = (R*np.sin(alpha)+0.5*e*R*(np.sin(2*alpha-beta)-3*np.sin(beta))
         +1/8*e**2*R*(3*np.sin(3*alpha-2*beta)-10*np.sin(alpha)+5*np.sin(alpha-2*beta)))

    z = (-np.sqrt(3)*e*R*np.cos(alpha-beta)+np.sqrt(3)*e**2*R*(np.cos(alpha-beta)**2+2*np.sin(alpha-beta)**2))

    pos = np.array([x,y,z])
    
    return(pos)


def get_analytic_orbits(t_tranche):
    a = R
    e0 = 2.5e9/(2*np.sqrt(3)*a)
    kappa = 0 #- 1.7967674211761813 - 20/180*np.pi # GWSPACE INITIAL ECLIPTIC LONGITUDE

    pos_0 = Orbit(t_tranche,
                kappa,# initial ecliptic latitude
                e0,
                n=0)
    pos_1 = Orbit(t_tranche,kappa,e0,n=1)
    pos_2 = Orbit(t_tranche,kappa,e0,n=2)
    return np.array([pos_0, pos_1, pos_2]).transpose(2, 0, 1).copy()
