import numpy as np
from other_repos.pyTSEB.pyTSEB import meteo_utils
from other_repos.pyTSEB.pyTSEB import resistances
from other_repos.pyTSEB.pyTSEB import net_radiation
from other_repos.pyTSEB.pyTSEB import clumping_index 
from other_repos.pyTSEB.pyTSEB import TSEB


def calc_emiss_atm(ea, t_a_k):
    '''Atmospheric emissivity

    Estimates the effective atmospheric emissivity for clear sky.

    Parameters
    ----------
    ea : float
        atmospheric vapour pressure (mb).
    t_a_k : float
        air temperature (Kelvin).

    Returns
    -------
    emiss_air : float
        effective atmospheric emissivity.

    References
    ----------
    .. [Brutsaert1975] Brutsaert, W. (1975) On a derivable formula for long-wave radiation
        from clear skies, Water Resour. Res., 11(5), 742-744,
        htpp://dx.doi.org/10.1029/WR011i005p00742.'''

    emiss_air = 1.24 * (ea / t_a_k)**(1. / 7.)  # Eq. 11 in [Brutsaert1975]_

    return np.asarray(emiss_air)


# calculate windspeed and downscale
def calc_wind_speed(u, v):
    ws = np.hypot(u, v)
    ws = np.maximum(ws, 1.0)
    return ws

# estimate canopy height from estimated LAI
def hc_from_lai(lai, hc_max, lai_max, hc_min=0):
    """
    Estimates canopy height from LAI
    Parameters
    ----------
    lai : array
            Actual Leaf Area Index
    hc_max : float or array
            Maximum Canopy Height at lai_max (m)
    lai_max : float or array
            LAI at which maximum height is achieved
    hc_min : float or array, optional
            Canopy height (m) at LAI=0
    
    Returns
    -------
    h_c : array
            Canopy height (m)
    """
    h_c = hc_min + lai * (hc_max - hc_min) / lai_max
    h_c = np.clip(h_c, hc_min, hc_max)
    return h_c


# estimate long wave irradiance
def calc_longwave_irradiance(ea, t_a_k, p, z_T, h_C):
        '''Longwave irradiance

        Estimates longwave atmospheric irradiance from clear sky.
        By default there is no lapse rate correction unless air temperature
        measurement height is considerably different than canopy height, (e.g. when
        using NWP gridded meteo data at blending height)

        Parameters
        ----------
        ea : float
                atmospheric vapour pressure (mb).
        t_a_k : float
                air temperature (K).
        p : float
                air pressure (mb)
        z_T: float
                air temperature measurement height (m), default 2 m.
        h_C: float
                canopy height (m), default 2 m,

        Returns
        -------
        L_dn : float
                Longwave atmospheric irradiance (W m-2) above the canopy
        '''
        meteo_utils.calc_stephan_boltzmann
        lapse_rate = meteo_utils.calc_lapse_rate_moist(t_a_k, ea, p)
        t_a_surface = t_a_k - lapse_rate * (h_C - z_T)
        emisAtm = calc_emiss_atm(ea, t_a_surface)
        L_dn = emisAtm * meteo_utils.calc_stephan_boltzmann(t_a_surface)
        return np.asarray(L_dn)

def calc_fg_gutman(ndvi, ndvi_min, ndvi_max):
        num = ndvi - ndvi_min
        denum = ndvi_max - ndvi_min

        fg = num/denum
        fg[fg<0] = 0
        fg[fg>1] = 1

        return fg