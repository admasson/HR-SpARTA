import numpy as np
from scipy.special import voigt_profile, wofz
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize, leastsq
# import voigts #fast compiled voigt algorithms
# from utils import stop
import time

try:
	fvoigt = voigts.voigts.voigthumlicek
	# print("***WARNING: using compiled voigts module for line profile!***")
except:
	# print("***WARNING: using scipy voigts module for line profile!***")
	def fvoigt(v,a):
		return wofz(v+a*1j)

elec_ch= 1.6021766208e-19   # A . s or C
elec_ch_cgs = 4.8032e-10 #cm^3/2 g^1/2 s^-1

c_light_m=299792458.    # m / s
c_light_cm=29979245800.    # cm / s

m_e = 9.10938356e-31    # kg
m_e_cgs = 9.10938356e-28    # g

Eps_0 = 8.854187817620e-12  # A^2 s^4 kg−1 m−3
Eps_0_cgs = 1

JOULE_TO_CM1=5.040963080525957e+22



# (pi elec_ch**2)/ (4 pi Epsilon_0 m_e c)
factor = elec_ch**2/(4*Eps_0*m_e* c_light_m)    # m**2 / s
#factor = np.pi * elec_ch**2  /(m_e * c) # C**2 . s / (kg . m)
print(factor)

factor_cgs = np.pi**0.5 * elec_ch_cgs**2 / (m_e_cgs * c_light_cm) # cm^2 * s^-1

#print(factor_cgs*np.pi**0.5)
#stop()

amu=1.660531e-27    # kg
amu_cgs=1.660531e-24    # g

m_he = 4.002602 * amu
m_he_cgs = 4.002602 * amu_cgs

m_si = 28.0855 * amu

k_boltz=1.3806488e-23   # m2 kg s-2 K-1
k_boltz_cgs=1.3806488e-16 # cm^2 g s^-2 K^-1

h_planck=6.62606957e-34  # kg m2 s-1
h_planck_cgs =6.62606957e-27 # g cm^2 s^-1

Aki_He = 1.0216e+07 # natural line broadening # s^-1
Aki_Si = 1.97e+07 # s-1

f_osc_si = 3.47e-1
f_oscHe = {'He1' : 5.9902e-02,'He2': 1.7974e-01,'He3': 2.9958e-01}



ref_wav_He_vacuum={'He1':10832.057472, 'He2':10833.216751, 'He3':10833.306444} # angstrom

ref_wav_He_air={'He1':10829.09114, 'He2':10830.25010, 'He3':10830.33977}

ref_wav_Si_air=10827.088

ref_wav_Si_vacuum=10830.057


x0_he={ 'He1':-32.08124686382154, 'He2':0, 'He3': 2.4821145512899165} # km/s

x0_si = -87.44120428220987


coeffs = [0.59238671, 0.15456145]

def lambdatokms(l,r):
    c=299792458.
    return c*(l/r-1.)/1000.

def quad_law(mu_value):

    return 1.-coeffs[0]*(1.-mu_value)-coeffs[1]*(1.-mu_value)*(1.-mu_value)


############################################
### -----  to fit on velocity axis ----- ###
############################################

def abs_line_vel(x,f_osc,ref_wav, T, n_col, m, x0,delta_d, Aki):

    '''

    Function to build an absorption line.
    Either use Gaussian profile or Voigt profile.
    Decided with the value of d, d = 0 for a Gaussian.
    d != 0 for a Voigt.

    Arguments:
    - x = wavelength vector [e-10 m]

    - f_osc is the oscillator stength

    - T for temperature [K]

    - n_col is column density

    - m mass of the element

    - lambda0 is the transition wavelength [e-10 m]

    - d is the damping factor = damping / width

    '''

#    broad_NT = (c/R_pow)**2 + (v_sini*1e3)**2

    vt = np.sqrt(2.*k_boltz * T / m)

    width = np.sqrt(vt**2)# + broad_NT)

    d = delta_d * ref_wav * 1e-10 * Aki/(4*np.pi * width)

    if d !=0:
    # Voigt

        # With d  = damping / width, avec damping dans l'unité de width
        # we specify what fraction of width the damping is with d

        z =1e3*(x -x0)/(width) + 1j*d

        line_profile = wofz(z).real

    else :
    # Gaussian

        line_profile = np.exp( - ( x*1e3 - x0*1e3 )**2 / (width**2))

    return np.exp(- factor * f_osc * ref_wav * 1e-10 / (np.sqrt(np.pi) * width) * n_col * line_profile)



##################################################################



###############################################################
### ----- to fit on wavelength axis Voigt or Gaussian ----- ###
###############################################################


# See Rutten 2003
def abs_line_wav(x,f_osc, T, n_col, m, lambda0,delta_d, Aki):

    '''

    Function to build an absorption line.
    Either use Gaussian profile or Voigt profile.
    Decided with the value of d, d = 0 for a Gaussian.
    d != 0 for a Voigt.

    Arguments:
    - x = wavelength vector [e-10 m]

    - f_osc is the oscillator stength

    - T for temperature [K]

    - n_col is column density

    - m mass of the element

    - lambda0 is the transition wavelength [e-10 m]

    - d is the damping factor = delta_d * lambda0 * 1e-10 * Aki / (4.0 * np.pi * width)

    - Aki in s-1 Einstein coefficient

    '''

    #broad_NT = (c_light_m/R_pow)**2 + (v_sini*1e3)**2

    vt = np.sqrt(2.*k_boltz_cgs * T / m)
#    print(vt/1e5)
#    stop()

    width = np.sqrt(vt**2)# + broad_NT)

    d = delta_d * lambda0 * 1e-10 * Aki/(4*np.pi * width)

    if d !=0:
    # Voigt

        # With d  = damping / width, avec damping dans l'unité de width
        # we specify what fraction of width the dmaping is with d

        z = c_light_cm/ lambda0 * (x - lambda0) / (width) + 1j*d

        line_profile = wofz(z).real

    else :
    # Gaussian

        line_profile = np.exp( - ( c_light_cm/lambda0 * (x - lambda0) / width )**2)

    return np.exp(- factor_cgs * f_osc * lambda0 * 1e-8 / width * n_col * line_profile)
    #return np.exp(- factor * f_osc * lambda0 * 1e-10 / (np.sqrt(np.pi) * width) * n_col * line_profile)

#########################################################################


def He_triplet_line_wav(x, T,n_col,delta_d, choice):

    return_value = 1.

    if choice[0]:

        return_value *= abs_line_wav(x,f_oscHe['He1'],T, n_col, m_he_cgs, ref_wav_He_vacuum['He1'],delta_d, Aki_He)
        #return_value *= abs_line_wav(x,f_oscHe['He1'],T, n_col, m_he, ref_wav_He_air['He1'],delta_d, Aki_He)

    if choice[1] :

        return_value *= abs_line_wav(x,f_oscHe['He2'],T, n_col, m_he_cgs, ref_wav_He_vacuum['He2'],delta_d, Aki_He)
        #return_value *= abs_line_wav(x,f_oscHe['He2'],T, n_col, m_he, ref_wav_He_air['He2'],delta_d, Aki_He)

    if choice[2] :

        return_value *= abs_line_wav(x,f_oscHe['He3'],T, n_col, m_he_cgs, ref_wav_He_vacuum['He3'],delta_d, Aki_He)
        #return_value *= abs_line_wav(x,f_oscHe['He3'],T, n_col, m_he, ref_wav_He_air['He3'],delta_d, Aki_He)

    return return_value
