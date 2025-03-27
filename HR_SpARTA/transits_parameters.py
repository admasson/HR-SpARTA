################################################################################
## This is were we define all parameters used throughout the reduction
## Coding : utf-8
## Author : Adrien Masson (amasson@cab.inta-csic.es)
## Date   : March 2025
################################################################################

#-------------------------------------------------------------------------------
# Here are some notes and usefull links
#-------------------------------------------------------------------------------
'''
Note on ephemerids & midpoint: check the ExoClock database : https://www.exoclock.space/database/planets (always use BJD-TBD temporal values for SPIRou !)

Note on argument of periapsis: in papers, it often refers to the stellar argument instead of the planet one. In such case, one has to add/remove 180° to get the planet one

Tools : https://exoplanetarchive.ipac.caltech.edu/overview/ ; https://www.exoclock.space/database/planets ; http://exoplanet.eu/catalog/
'''

#-------------------------------------------------------------------------------
# Import usefull libraries and modules
#-------------------------------------------------------------------------------

from astropy.time import Time
from astropy import constants as const
import copy
import datetime
import numpy as np

print(f'{datetime.datetime.now().time()} : transits_info.py has been loaded')


#-------------------------------------------------------------------------------
# Define the exoplanet system's parameters in a dictionnary: WASP-76 b 
#-------------------------------------------------------------------------------

WASP76_2020 = {
    # file structure
    'data_dir'      : './data/WASP-76/2020-10-31/',                              # data directory
    'SNR_key'       : 'SPEMSNR',                                                # SNR key in SPIRou header
    'SNR_hdu_index' : 0,                                                        # index of the header's hdu containing the SNR

    # transit properties
    'midpoint'      : 2459153.8848966,                                          # transit midpoint in BJD-TBD (Computed using T0 from ExoClock - Kokori et al 2022)
    'U_midpoint'    : 0.00012,                                                  # incertainty midpoint in BJD-TBD (days) (From ExoClock - Kokori et al 2022)

    # stellar properties
    'Ms'            : 1.458 * const.M_sun.value,                                # Host star mass [kg] (Ehrenreich2020)
    'Rs'            : 1.756 * const.R_sun.value,                                # Stellar radius (Ehrenreich2020)
    'Ks'            : 116.02,                                                   # Host star RV amplitude [m/s] (Ehrenreich2020)
    'c'             : [0.190, 0.305],                                           # Limb darkening coefficients (EXOFAST)
    'Teff'          : 6329,                                                     # Effective Temperature (K) (Ehrenreich2020)
    'log(g)'        : 4.196,                                                    # Decimal logarithm of the surface gravity (cm/s**2) (Ehrenreich2020)
    'M/H'           : 0.366,                                                    # Metallicity index relative to the Sun (Ehrenreich2020)
    'vsini'         : 1.48e3,                                                   # stellar rotational velocity [m/s] (Ehrenreich2020)
    'Sys_Age'       : 1.816,                                                    # (Gyr) Age of the system (Ehrenreich2020)
    'Vs'            : -0.79e3,                                                  # GaiaDR3, [m/s] Vs < 0 means star is moving toward us. This Vsys contains the gravitational redshift & convective blueshift

    # planet properties
    'Mp'            : 0.894 * const.M_jup.value,                                # Planet mass [kg] (Ehrenreich2020)
    'Rp'            : 1.854 * const.R_jup.value,                                # Planet radius (Ehrenreich2020)
    'Porb'          : 1.8098806,                                                # Planet orbital period [days] (Ehrenreich2020)
    'Teq'           : 2160,                                                     # Equilibrium (or Effective) Temperature (K) of the Planet (West2016)

    # orbital properties
    'a'             : 0.0330 * const.au.value,                                  # Semi Major axis (Ehrenreich2020)
    'i'             : 89.623,                                                   # inclination (Ehrenreich2020)
    'e'             : 0,                                                        # excentricity (fixed)
    'w'             : 0,                                                        # longitude of periapse (deg) (fixed) -> BY DEFAULT, LITTERATURE GIVES w AS THE STELLAR ONE -> ADD 180° OR PI IF YOU NEED THE PLANET'S ONE !! 
    'lbda'          : 61.28,                                                    # spin-orbit angle (deg) (Ehrenreich2020)
}

# Manually compute Kp & logg
WASP76_2020['Kp'] =  WASP76_2020['Ks'] * WASP76_2020['Ms'] / WASP76_2020['Mp']  # Planet RV amplitude [m/s] manually computed
WASP76_2020['planet_logg'] = np.log10(const.G.cgs.value*(WASP76_2020['Mp']*1e3/(WASP76_2020['Rp']*100)**2)) # planet log gravity in cgs