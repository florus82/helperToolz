import numpy as np

# calculate windspeed and downscale
def calc_wind_speed(u, v):
    ws = (u ** 2 + v ** 2) ** 0.5
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

        lapse_rate = TSEB.met.calc_lapse_rate_moist(t_a_k, ea, p)
        t_a_surface = t_a_k - lapse_rate * (h_C - z_T)
        emisAtm = calc_emiss_atm(ea, t_a_surface)
        L_dn = emisAtm * TSEB.met.calc_stephan_boltzmann(t_a_surface)
        return np.asarray(L_dn)

def calc_fg_gutman(ndvi, ndvi_min, ndvi_max):
        num = ndvi - ndvi_min
        denum = ndvi_max - ndvi_min

        fg = num/denum
        fg[fg<0] = 0
        fg[fg>1] = 1

        return fg