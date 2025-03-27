################################################################################
## Module holding the main reduction process tools through the DataSet class
## Coding : utf-8
## Author : Adrien Masson (amasson@cab.inta-csic.es)
## Date   : March 2025
################################################################################

#-------------------------------------------------------------------------------
# Import usefull libraries and modules
#-------------------------------------------------------------------------------
import os
import datetime
import pickle
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import batman
import radvel
import time
import threading

from matplotlib.widgets import *
from warnings import warn

from astropy.io import fits
from astropy.modeling import models, fitting, polynomial
from astropy.time import Time
from astropy import constants as const
from astropy.convolution import convolve

from PyAstronomy import pyasl
from multiprocessing import Pool

from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator
from scipy import optimize

from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA

from kneed import DataGenerator, KneeLocator
from contextlib import contextmanager

# import William's code to compute He stellar line
from HR_SpARTA.He_line_model import He_triplet_line_wav

# custom plot functions
from HR_SpARTA.plots import *

# Safety check when module has been imported for debugging purpose
print(f'{datetime.datetime.now().time()} : reduction.py has been loaded')

#-------------------------------------------------------------------------------
# Define some global values
#-------------------------------------------------------------------------------
# speed of light in vacuum, m/s
c = const.c.value 

# theoritical position of the 3 He lines in the Rest Frame in air [nm]
He_theo_wave_air = np.array([1082.909,1083.025,1083.034]) 

# theoritical position of the seven stellar lines around the He lines in the Rest Frame in air [nm]
Stellar_theo_wave_air = {'Mg I':1081.1084,'Fe Ia':1081.8276,'Si Ia':1082.7091,'Ca I':1083.8970,'Si Ib':1084.3854,
                    'Fe Ib':1084.9467,'Fe Ic':1086.3520,'He Ia':1082.909,'He Ib':1083.025,'He Ic':1083.034}

# air refractive index at 1 micron (from https://refractiveindex.info/)
n = 1.00027394

# theoritical position of the 3 He lines in the Rest Frame in Vacuum
He_theo_wave = He_theo_wave_air * n

# rename lowess function for easier access
lowess    = sm.nonparametric.lowess

# Define a Memory lock for safe access when doing parallel computing
lock = threading.Lock() 

# Thread_counter for threading monitoring
thread_counter = 0 

# Counting convergence fails in some threaded functions
fail_counter = 0 

#-------------------------------------------------------------------------------
# Define utility functions
#-------------------------------------------------------------------------------

# Context manager to measure execution time of blocks. 
@contextmanager
def measure_time(label="Execution",display=True):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    if display: print(f"{label}: {end - start:.6f} seconds")


# convolve at SPIRou resolution
def convolve_SPIRou(wave,spectrum):

    pixel_size = 2*2.28e3 # SPIRou element resolution in m/s
    nb_points = 11 # size, in pixel, of the door function used for the convolution

    half_size = pixel_size / 2
    pixel = np.linspace(-half_size,half_size,nb_points)

    convolved_spec = np.zeros(spectrum.size)

    f = interp1d(wave,spectrum,fill_value=np.nan)

    for v in pixel:
        # mask wavelength shifted outside the interpolation domain
        mask_down = (wave / (1 + v/const.c.value)) < wave.min()
        mask_up   = (wave / (1 + v/const.c.value)) > wave.max()
        mask = np.logical_or(mask_down,mask_up) # contains True where shifted wavelength are outside the valid interpolation domain
        convolved_spec[~mask] += f(wave[~mask] / (1 + v/const.c.value))
        # replace values outside range by nan
        convolved_spec[mask] = np.nan

    # normalise
    convolved_spec /= len(pixel)

    # cut invalid values and store
    mask = np.isfinite(spectrum)
    wave_masked = wave[mask]
    spectrum_convolved = convolved_spec[mask]

    return(wave_masked,spectrum_convolved)

# convolve at an instrumental resolution or for a given velocity broadening
def convolve_velocity(wave,spectrum,velocity_bin,nb_points=11):

    # pixel_size : # instrumental element resolution in m/s (R = v/c, so taking 2 bins for sampling we should have pixel_size = dv = c / (2R) )
    # nb_points : # size, in pixel, of the door function used for the convolution

    half_size = velocity_bin / 2
    pixel = np.linspace(-half_size,half_size,nb_points)

    convolved_spec = np.zeros(spectrum.size)

    f = interp1d(wave,spectrum,fill_value=np.nan)

    for v in pixel:
        # mask wavelength shifted outside the interpolation domain
        mask_down = (wave / (1 + v/const.c.value)) < wave.min()
        mask_up   = (wave / (1 + v/const.c.value)) > wave.max()
        mask = np.logical_or(mask_down,mask_up) # contains True where shifted wavelength are outside the valid interpolation domain
        convolved_spec[~mask] += f(wave[~mask] / (1 + v/const.c.value))
        # replace values outside range by nan
        convolved_spec[mask] = np.nan

    # normalise
    convolved_spec /= len(pixel)

    # cut invalid values and store
    mask = np.isfinite(spectrum)
    wave_masked = wave[mask]
    spectrum_convolved = convolved_spec[mask]

    return(wave_masked,spectrum_convolved)

# build a DataSet class from a transit dictionnary
def load_transit(transit,plot_all=False):
    '''
    take a transit dictionnary such that in transts_info.py and apply reduction
    & doppler shifting to align in exoplanet Rest Frame and find the He triplet
    signature from the planet.
    '''
    ## retrieve the transiting exoplanet parameters
    data_dir      = transit['data_dir']       # data directory
    SNR_key       = transit['SNR_key']        # SNR key
    SNR_hdu_index = transit['SNR_hdu_index']  # index of the header's hdu containing the SNR
    midpoint      = transit['midpoint']       # transit midpoint, BJD-TBD derived from UT hours (Bonomo et al. 2017)
    U_midpoint    = transit['U_midpoint']     # incertainty midpoint in DAYS (Bonomo et al. 2017)
    Ms            = transit['Ms']             # Host star mass [kg] (de Kok et al 2013)
    Mp            = transit['Mp']             # Planet mass [kg] (de Kok et al 2103)
    Ks            = transit['Ks']             # Host star RV amplitude [m/s] (Triaud et al 2009)
    Porb          = transit['Porb']           # Planet orbital period [days] (Baluev et al 2015)
    Vs            = transit['Vs']             # With Gaia (cf CDS / SIMBAD) -> Vs < 0 means star is moving toward us
    Kp            = transit['Kp']
    
    ## load transit in a data_set
    if SNR_hdu_index == None:
        SNR_hdu_index = 1
    file_format = 't.fits'
    if file_format=='t.fits': hdu = {'data':1, 'wavelengths':2, 'blaze':3, 'tellurics':4,'OHLine':5}
    if file_format=='e.fits': hdu = {'data':1, 'wavelengths':5, 'blaze':9}
    data_set = DataSet()
    data_set.load_data(transit_dic=transit,file_format=file_format,hdu=hdu)   
    
    # plot
    if plot_all or False: plot_grid(data_set.wave[0],data_set.data[0],lw='0.2')
    if plot_all or False: plot_grid(data_set.wave[0],data_set.tellurics[0],lw='0.2')
    
    return(data_set)

# a simple PCA using svd on an array X
def PCA(X, plot=False):
    '''
    Compute the PCA on the given data array. 

    /!\ THE DATA MUST ALREADY HAVE BEEN CENTERED AND REDUCED, WITH NaN REPLACED BY 0 /!\
        
    takes:
        - X: 2D array corresponding to the centered and reduced data, shape must be n x p with n the number of samples and p the number of variables/features/characteristic (pick your favorite). 
        - plot: bool, do we plot the "scree plot" (variance per PCs) ?
        
    return:
        - U: contains the left singular vector matrix as columns (n x n)
        - S: 1D array of size (p) containing the singular values (real, non-negative).
        - VT: transpose of the V matrix (p x p) containing the right singular vectors as rows. The columns of V are the principal directions/eigenvectors
        - var: the variance corresponding to each Principal Component, 1D array of size (p)
    
    notes:
        - If the PCA is performed along the "time axis", n is the number of spectral bins and p the number of observations/transit phases   
        - U*S yield the principal components ("scores", i.e. the projection of the data on the new principal axes/transformed variables in the new PCA coordinate system)
        - The variance var_i of each PC (i.e. the eigenvalues of the covariance matrix) is computed from the eigenvalues 's_i' of the S matrix with: var_i = s_i²/(n-1)
    '''
    # do SVD: X must have wavelength in first axis and time (features) in second axis
    U, S, VT = np.linalg.svd(X,full_matrices=False,compute_uv=True)
    # compute variance
    var      = (S ** 2) / (X.shape[0]-1) # variance per Principal Component
    var_norm = var/var.sum()
    # plot the variance per PC -> "scree plot"
    if plot:
        plt.figure()
        ax1 = plt.gca()
        ax2 = plt.twinx()
        ax1.plot(var_norm,'k+')
        ax2.plot(var,'k+')
        ax1.set_ylabel('Normalised variance')
        ax2.set_ylabel('Variance (eigenvalue)')
        plt.title('Scree plot')
        ax1.set_xlabel('Principal component n°')
    # return results
    return(U,S,VT,var)

# transform data for PCA: center and reduce
def transform_data(X,data_set=None):
    '''
    Small utility function for PCA: center and reduce data, transpose them, and remove NaN for direct use with PCA
    Return the data centered and reduced, along with a dictionnary containing the NaN mask, mean, and std values to revert the transformation
    Example usage:
        $ X, transfo_params = transform_data(data_set.data[:,order]) # -> the array will be transposed in the function call, no need to transpose before !!
    '''
    # transpose so that spectra are order in rows (axis = 0) and transit phase in columns (axis = 1)
    X = np.array(X.T) # we also use np.array to ensure we're not working with a masked array or anything SVD won't handle
    # reduce and center -> apply SNR² weighting? I think we should, check explanation at the end of this subsection... In which case the centering should be done before transpose with the DataSet.weighted_mean / weighted_std functions...
    if data_set==None:
        mean = np.nanmean(X,axis=1)
        std  = np.nanstd(X,axis=1)
    else: # if a data_set is provided, use it to weight the mean & std with SNR²
        mean = data_set.weighted_mean(X.T)
        std  = data_set.weighted_std(X.T)

    X = (X - mean[...,None])/std[...,None]
    # replace NaN values with 0.
    mask = ~np.isfinite(X)
    X[mask] = 0
    # return array along with the transformation parameters for revert transform
    return(X,(mean,std,mask))

# set the invert transform 
def invert_transform(X,mean,std,mask):
    '''
    Revert the transformation to get the data back to their original mean & scale, and also re-introduce the original NaN where they were
    Example usage:
        $ X, transfo_params = transform_data(data_set.data[:,order])
        $ data_set.data[:,order] = invert_transform(X,*transfo_params)
    '''
    # replace NaN
    X[mask] = np.nan
    # recenter/rescale
    X = (X * std[...,None])+mean[...,None]
    # transpose
    X = X.T
    # return array
    return(X)

# implementation of the SysREM algorithm (Tamuz+2005): find the best u,w vector such that a 2D matrix X is ~ equal to u.w^T
def sysrem(X, sigma, num_iters=1000, tol=1e-6):
    """
    Perform SysREM with weighted least squares.
    
    Parameters:
        X (numpy.ndarray): Input matrix of size (N, M).
        sigma (numpy.ndarray): Uncertainty matrix (same shape as X).
        num_iters (int): Number of iterations.
        tol (float): Convergence tolerance.
    
    Returns:
        u (numpy.ndarray): Left singular vector (size N).
        w (numpy.ndarray): Right singular vector (size M).
    """
    N, M = X.shape
    
    # Initialize u and w
    u = np.ones(N)
    w = np.ones(M)
    
    # Iteratively find best u & w based on equations (2) & (3) from Tamuz et al. 2005
    prev_u_norm, prev_w_norm = 0, 0

    for _ in range(num_iters):
        # update u
        u = np.nansum(X * w / sigma**2, axis=1) / np.nansum((w**2)/sigma**2, axis=1) # One element per wavelength. Don't forget axis=1 in denum since sigma is 2D
        # Update w
        w = np.nansum(X.T * u / sigma.T**2, axis=1) / np.nansum((u**2)/sigma.T**2, axis=1) # One element per obs. Don't forget axis=1 in denum since sigma is 2D
        # Compute norms to check for convergence
        u_norm, w_norm = np.linalg.norm(u), np.linalg.norm(w)
        # print(f"Iteration {_}: ||u|| = {u_norm}, ||w|| = {w_norm}")      
        if abs(u_norm - prev_u_norm) < tol and abs(w_norm - prev_w_norm) < tol:
            # print(f"Done in {_} iterations")
            break
        prev_u_norm, prev_w_norm = u_norm, w_norm

    Converged = (_+1 < num_iters) # True if succesfully converged

    # also compute reconstruction and its variance following u & w
    X_recon = np.outer(u,w)
    var = np.std(X_recon)**2

    return u, w, var, X_recon, Converged


#-------------------------------------------------------------------------------
# The DataSet class is the core of the reduction process : we use it to import
# all SPIRou data, apply the different reduction steps on them to remove most
# of the telluric and stellar signal, and save the reduced data in a file.
#-------------------------------------------------------------------------------
class DataSet:
    '''
    ####### DataSet CLASS #######
    The DataSet class allows to proceed the different reduction steps directly implement as internal method in this class.

    Start using this class by creating a DataSet object :
    $ from hrsparta_modules import *
    $ my_data_set = DataSet(save_history = True)

    Then load your data (see DataSet.load_data() documentation for more informations on parameters) :
    $ my_data_set.load_data(path = "path/to/data", file_format = "e.fits", SNR_key = 'SPEMSNR', hdu = {"data":1,"wavelengths":5,"blaze":9}, print_info = True)

    Note that the dictionnary used in hdu_dic parameter must strictly respect the format {"data":int,"wavelengths":int,"blaze":int} as it allows the class to know in which hdu are stored the data on which computing the reduction steps.

    Some reduction steps can require to work with 'off-transit' or 'on-transit' spectra only. In this case, you will need to call the following method for the DataSet object to find 'off' and 'on' transit spectra:
    $ my_data_set.find_off_transit(duration, U_duration, midpoint, U_midpoint, print_info = True)

    ########### EXAMPLE ###########
    You can then apply the different reduction steps in any order, for example :
    $ my_data_set.norm_with_median()
    $ my_data_set.div_by_weighted_average()
    $ my_data_set.div_by_moving_average(area=120)
    $ my_data_set.airmass_correction(model_deg = 2, iterations = 3, n_sigma = 4, spectra_used_for_model = 'off-transit', spectra_used_for_division = 'none', plot_sigma = True)
    $ my_data_set.load_tellurics(path = "path/to/data", file_format = 't.fits', hdu_index = 4)
    $ my_data_set.telluric_transmission_clipping(transmission_threshold = 0.4)
    $ my_data_set.sigma_clipping(model_deg = 4, n_sigma = 4, iterations = 3, transmission_correction = True)
    $ my_data_set.apply_pca()

    Each step modify the DataSet.data attribute, which is a data cube containing the hdu loaded with the index stored in hdu_dic["data"].
    Thus at the end of all above lines, the final reduced data are stored as a numpy.ndarray in my_data_set.data

    ###### VIEW STEPS HISTORY ######
    If "save_history" has been set to True when creating the DataSet object, then each called step will by default save its produced data along with the parameters used for this step (if any) in a "history" dictionnary in the form {'stepID' : {'data' : produced_data , 'parameter' : parameter_value , ...}.
    The saving behaviour can be set for a specifical step with the boolean parameter 'save' :
        - if "save_history" has been set to True when initializing class, then each called step will save its produced data state along with its parameters in "history" except if it was called with 'save = False'
        - if "save_history" has been set to False when initializing class, then only reduction step methods called with 'save = True' will write their data in "history", other will only write their parameters

    You can thus recall the order in which you have called the different reduction steps by calling "my_data_set.history.keys()", and get the corresponding state of data after a specific reduction step with "my_data_set.history['stepID']['data']",
    which returns the data as they were after this step and before the following ones. This can be usefull for example to plot your data before and after a specifical reduction step on a same plot to make sure the step is doing what you expect.
    Calling "my_data_set.history['stepID']['parameters']" will display the different parameters used when this step has been called (if none were given, then the dictionnary is empty).

    For example, after having run the above reduction steps, "my_data_set.history" should be in the form :

    {'load_data'                        : {'data' : np.ndarray , 'parameters' : {'path' : "path/to/data" , 'file_format' : "e.fits" , ...}} ,   # the raw data right after having been loaded and before any reduction steps, along with the parameters used for loading them
     'norm_with_median'                 : {'data' : np.ndarray , 'parameters' : {}} ,                                                           # data after normalization with median
     'div_by_weighted_average'          : {'data' : np.ndarray , 'parameters' : {}} ,
     'div_by_moving_average'            : {'data' : np.ndarray , 'parameters' : {}} ,
     'airmass_correction'               : {'data' : np.ndarray , 'parameters' : {'model_deg' : 4 , 'iterations' : 3 , ...}} ,
     'load_tellurics'                   : {'data' : np.ndarray , 'parameters' : {'path' : "path/to/data", file_format : 't.fits', ...}} ,
     'telluric_transmission_clipping'   : {'data' : np.ndarray , 'parameters' : {'transmission_threshold' : 0.4}} ,
     'sigma_clipping'                   : {'data' : np.ndarray , 'parameters' : {'model_deg' : 4 , 'n_sigma' : 4 , ... }} ,
     'apply_pca'                        : {'data' : np.ndarray , 'parameters' : {}}
     }

     ######## MEMORY ISSUES #######
     Please note that this will keep store in memory as many data numpy array as steps for which saving has been allow. When working with huge data set on a computer poor in capacity, memory issues can arise.
     If you run into memory issue, please consider reducing the number of saving steps by setting "save" to False for steps for which you don't need to monitor the direct effect on data.

     TO DO : You can also monitor the memory consumption of your saved steps by calling "my_data_set.memory_consumption()", which will display informations on the memory used by your "my_data_set" object.

     ######## COMPRENSIVE LIST OF CLASS & METHODS #######
     TO DO !
     ####################################################
    '''

    def __init__(self,save_history=True):
        self.save_history   = save_history              # set the default saving behaviour for each reduction steps
        self.history        = {}                        # the dictionnary containing the history of called steps with save = True
        self.files          = []                        # the list of loaded files
        self.transit_dic    = {}                        # the transit dictionnary with all target info
        self.shape          = None                      # the shape of data
        self.data           = None                      # a data cube storing spectra from all loaded files in one three dimensionnal numpy.ndarray on which are applied the reduction steps
        self.headers        = {}                        # a dictionnary containing the header of each loaded data file with its dictionnary key corresponding to its index in self.data cube
        self.wave           = None                      # an array the same shape of self.data containing the corresponding wavelengths
        self.blaze          = None                      # an array the same shape of self.data containing the corresponding blaze function
        self.SNR            = None                      # an array containing the SNR values corresponding to each loaded file using a given SNR keyword
        self.airmass        = None                      # an array containing the airmass corresponding to each loaded files
        self.BERV           = None                      # an array containing the BERV [m/s] for each observation
        self.tellurics      = None                      # telluric spectra (stored in t.fits for example) used for clipping using theoritacl telluric lines spectra (see self.load_tellurics())
        self.midpoint       = None                      # transit midpoint (in BJD-TBD)
        self.svd_reconstructed_data = None              # If a PCA using SVD is performed : will contain the reconstructed data using only the K first PCs (K being given by user)
        self.SVD_V          = None                      # If a PCA using SVD is performed : will contain a list of matrix V (1 per order) whose columns are the eigenvectors computed during SVD
        self.ke             = None                      # Contains the KeplerEllipse class computed with pyasl by the compute_kepler_orbit() method
        # mask generated by methods
        self.airmass_corr_mask = None                   # a mask containing True where elements have been clipped during the airmass_correction step
        self.off_transit_mask  = None                   # a mask containing only the index of files corresponding to 'off-transit' observations
        self.on_transit_mask   = None                   # a mask containing only the index of files corresponding to 'on-transit' observations
        self.clipping_mask     = None                   # a mask containing True where elements have been clipped during the sigma_clipping step

    def load_data(self, transit_dic, file_format, hdu, print_info = True, save = None, overwrite = False):
        # load the data, must have a correct transit dictionnary provided

        # prevent overwriting if method is launched twice
        if not self.data is None:
            if not overwrite:
                print("Data have already been loaded through this method. If you want to overwrite them, please call this method again adding the following parameter : 'overwrite = True' (You will lose previous reduction steps !).")
                return()
            else:
                print("Data have already been loaded through this method : overwriting them")

        # Current step ID
        step_ID = 'load_data'

        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history

        # set what is the expected hdu dictionnary depending on the file type
        if file_format == 't.fits':
            expected_hdu_keys = ['data','wavelengths','blaze','tellurics', 'OHLine']
        elif file_format == 'e.fits':
            expected_hdu_keys = ['data','wavelengths','blaze']
        else:
            raise NameError(f'Unknown file format "{file_format}", please enter either "t.fits" or "e.fits".')

        # Verifying that the hdu parameter format is valid
        if expected_hdu_keys != list(hdu.keys()):
            a = ' : int'
            raise NameError(f"'hdu' parameter is in wrong format : provided keys are {list(hdu.keys())} instead of {expected_hdu_keys}.\nPlease provide a dictionnary strictly respecting the following format : hdu = {{ {str([el+a for el in expected_hdu_keys])[1:-1]} }}\nwith 'int' being the index in primary header for the corresponding hdu (for example 'data' : 1 to use 'fluxAB' hdu from e.fits as data. See https://www.cfht.hawaii.edu/Instruments/SPIRou/FileStructure.php)")

        ## Retrieve current step parameters
        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')

        # extract all files from directory matching with file_format
        self.transit_dic = transit_dic
        self.path = transit_dic['data_dir']
        if print_info : print(f"Loading files with format '{file_format}' in '{self.path}' :")
        self.files = [file for file in os.listdir(self.path) if file_format in file]
        if print_info : print(f"{len(self.files)} out of {len(os.listdir(self.path))} files selected !")

        # also using its header to make sure we are on the right set of data and have selected the right channel
        if print_info : print('\nThe following hdu will be extracted with the following roles. Please make sure this is what you want :')

        # opening the first fits to get its shape
        # /！\ WE SUPPOSE THAT ALL FITS HAVE THE SAME SHAPE /!\
        f = fits.open(self.path+self.files[0])

        for hdu_role, hdu_index in hdu.items():
            try:
                if print_info : print(f"Selected HDU '{f[hdu_index].header['EXTNAME']}' at index {hdu_index} as '{hdu_role}'.")
            except Exception as e:
                print(f"Failed to load HDU {hdu_index} as '{hdu_role}'. Error is :{e}")

        # create a big cubic numpy array : each given fits file will have its 2D data stored along the first axis
        self.data  = np.zeros((len(self.files),*f[hdu['data']].data.shape))
        self.shape = self.data.shape
        self.blaze = np.zeros(self.shape)
        self.wave  = np.zeros(self.shape)
        self.SNR   = []
        self.airmass = []
        self.BERV = []
        if file_format == 't.fits':
            # test if there is a OH line HDU, as it may not be always the case ?
            try:
                f[hdu['OHLine']].header
                has_OH = True
            except:
                print(f"No OH hdu found for index #{hdu['OHLine']}, filling with zeros.")
                has_OH = False
            self.tellurics  = np.zeros((len(self.files),*f[hdu['tellurics']].data.shape))
            self.OH_lines   = np.zeros((len(self.files),*f[hdu['tellurics']].data.shape))

        if print_info : print(f"\nBuilding data cube of shape : {self.shape}")
        if print_info : print('Filling the data cube with selected files...')

        # orders files in observation time order
        obs_time_list = [fits.open(self.path + el)[1].header['BJD'] for el in self.files] # starting time of observation in Barycentric Julian Date
        sorting_index = np.argsort(obs_time_list)
        self.files = [self.files[k] for k in sorting_index]
        f.close()

        # then fill the cube with each data fits:
        for k,el in enumerate(self.files):
            # print(el)
            f = fits.open(self.path + el)
            self.data[k]        = f[hdu['data']].data
            self.wave[k]        = f[hdu['wavelengths']].data
            self.blaze[k]       = f[hdu['blaze']].data
            if file_format == 't.fits':
                self.tellurics[k]   = f[hdu['tellurics']].data
                if has_OH: self.OH_lines[k]    = f[hdu['OHLine']].data
            # also save headers and retrieve SNR & airmass
            self.headers[k]     = {'Primary header':f[0].header, 'data header':f[hdu['data']].header, 'wave header':f[hdu['wavelengths']].header, 'blaze header':f[hdu['blaze']].header}
            self.SNR.append(f[self.transit_dic['SNR_hdu_index']].header[self.transit_dic['SNR_key']])
            self.airmass.append(f[0].header['AIRMASS'])
            self.BERV.append(f[hdu['data']].header['BERV']*1e3)
            f.close()

        self.SNR = np.array(self.SNR)
        self.airmass = np.array(self.airmass)
        self.BERV = np.array(self.BERV)

        # give a default name for the data_set to ease & homogeneize naming format
        self.name = '_'.join(self.transit_dic['data_dir'].replace('//','/').split('/')[-4:-1])

        ## compute off & on transit mask
        self.find_off_transit(print_info)

        ## compute orbital parameters & transit window weight
        self.compute_kepler_orbit(show=False)
        self.compute_transit_window(show=False)

        ## compute velocities
        Vp, Vtot = self.compute_RV(self.time_vector) # planet's RV in Stellar & Earth RF respectively. The Kepler Ellipse is aligned in time because the time of periastron has been computed from the absolute time of mid transit (in BJD-TDB), so here we feed with the absolute time vector
        
        # Star RV around barycenter
        Vrv_star = -1 * self.transit_dic['Mp']/self.transit_dic['Ms'] * Vp
        
        # BERV & systemic velocity
        BERV = self.BERV
        Vs = self.transit_dic['Vs']

        # Relative velocity between observer on Earth and the star system in earth Rest Frame [m/s] : Vs > 0 means earth moves toward the star
        Vd = self.BERV - self.transit_dic['Vs'] - Vrv_star

        # save velocities in the class
        self.Vtot = Vtot
        self.Vd = Vd
        self.Vrv_star = Vrv_star
        self.Vp = Vp

        if print_info: print(f"\n{len(self.files)} files with '{file_format}' have been succesfully loaded from '{self.path}'\nData, wavelengths, blaze and corresponding headers have all been loaded from these files in data cubes of shape {self.shape}")

        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

    def load_espresso_data(self, transit_dic, print_info = True, save = None, overwrite = False):
        # load the data, must have a correct transit dictionnary provided

        # prevent overwriting if method is launched twice
        if not self.data is None:
            if not overwrite:
                print("Data have already been loaded through this method. If you want to overwrite them, please call this method again adding the following parameter : 'overwrite = True' (You will lose previous reduction steps !).")
                return()
            else:
                print("Data have already been loaded through this method : overwriting them")

        # Current step ID
        step_ID = 'load_data'

        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        
        ## Retrieve current step parameters
        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')

        # extract all files from directory matching with file_format
        self.transit_dic = transit_dic
        self.path = transit_dic['data_dir']
        if print_info : print(f"Loading files in '{self.path}' :")
        self.files = [file for file in os.listdir(self.path) if 'S2DA' in file and '.fits' in file]
        if print_info : print(f"{len(self.files)} out of {len(os.listdir(self.path))} files selected !")

        # opening the first fits to get its shape
        # /！\ WE SUPPOSE THAT ALL FITS HAVE THE SAME SHAPE /!\
        f = fits.open(self.path+self.files[0])     

        # create a big cubic numpy array : each given fits file will have its 2D data stored along the first axis
        self.data  = np.zeros((len(self.files),*f[1].data.shape))
        self.shape = self.data.shape
        self.blaze = np.ones(self.shape)
        self.wave  = np.zeros(self.shape)
        self.SNR   = []
        self.airmass = []
        self.BERV = []

        if print_info : print(f"\nBuilding data cube of shape : {self.shape}")
        if print_info : print('Filling the data cube with selected files...')

        # orders files in observation time order
        obs_time_list = np.array([fits.open(self.path + el)[0].header['HIERARCH ESO QC BJD'] for el in self.files]) # starting time of observation in Barycentric Julian Date
        sorting_index = np.argsort(obs_time_list)
        self.files = [self.files[k] for k in sorting_index]
        f.close()

        # then fill the cube with each data fits:
        for k,el in enumerate(self.files):
            # print(el)
            f = fits.open(self.path + el)
            self.data[k]        = f[1].data
            self.wave[k]        = f[4].data / 10 # convert Angtroms to nm
            # also save headers and retrieve SNR & airmass
            prim_hdr = f[0].header
            data_hdr = f[1].header
            # add BJD to the data header for compatibility with the find_off_transit function
            data_hdr['BJD'] = prim_hdr['HIERARCH ESO QC BJD']
            data_hdr['BERV'] = prim_hdr['HIERARCH ESO QC BERV']
            self.headers[k]     = {'Primary header':prim_hdr, 'data header':data_hdr}

            self.SNR.append(f[0].header['HIERARCH ESO QC ORDER30 SNR'])
            self.airmass.append( (f[0].header['HIERARCH ESO TEL1 AIRM START']+f[0].header['HIERARCH ESO TEL1 AIRM END'])/2) # avg airmass btw start & end
            self.BERV.append(f[0].header['HIERARCH ESO QC BERV']*1e3) # m/s
            f.close()

        print('Note that we arbitrary take the SNR from header corresponding to order 30, as SNR has the same relative behavior along time in all orders')

        # load telluric correction from Molecfit generated template (check HowTo_Molecfit.txt & read_molecfit_output_espresso.ipynb for more details)
        molecfit_files = [files for files in os.listdir(self.path) if 'Telluric_Correction_Molecfit' in files]
        if len(molecfit_files) == 0:
            print(f'No molecfit telluric correction files found in {self.path}: CAN\'T APPLY TELLURIC CORRECTION !!! RUN MOLECFIT FIRST ON S1D FILES TO GENERATE THE TELLURIC TEMPLATE, THEN STORE THE FILES IN WITH "Telluric_Correction_Molecfit" IN NAME IN SAME FOLDER AS DATA !!')
        elif len(molecfit_files) != self.shape[0]:
            raise NameError(f'Found {len(molecfit_files)} "Telluric_Correction_Molecfit" files but {self.shape[0]} S2D data files. Please make sure you have a telluric template for each data file !')
        else:
            print(f'Found {len(molecfit_files)} molecfit telluric correction files in {self.path}')
            telluric_template = np.array([fits.open(self.path+'/'+files)[3].data['mtrans'] for files in molecfit_files]) # telluric template built from s1d files
            telluric_wave = np.array([fits.open(self.path+'/'+files)[1].data['WAVE'][0] for files in molecfit_files]) / 10 # wavelength of s1d in nm
            # order by bjd so that each telluric files match the correct s2d file
            s1d_bjd = np.array([fits.open(self.path+'/'+files)[0].header['HIERARCH ESO QC BJD'] for files in molecfit_files])
            tellu_sort_index = np.argsort(s1d_bjd)
            # check that sorted BJD matches btw telluric templates & data
            if not np.all(s1d_bjd[tellu_sort_index] == obs_time_list[sorting_index]): 
                print(f'Found BJD mismatch between telluric template and data files: {s1d_bjd[tellu_sort_index]} vs {obs_time_list[sorting_index]}')
                raise NameError('Mismatch between sorted BJD of telluric files & S2D data: please check for duplicated or missing files, otherwise telluric templates would be associated to data that doesn\'t correspond...')
            # sort telluric templates by BJD
            telluric_template = telluric_template[tellu_sort_index]            
            # interpolate the telluric template to the data wave
            telluric_interp = np.nan * np.ones_like(self.data)
            for obs in range(telluric_template.shape[0]):
                telluric_interp[obs] = interp1d(telluric_wave[obs], telluric_template[obs], fill_value=np.nan, bounds_error=False)(self.wave[obs])
            # store as attribute
            self.tellurics = telluric_interp
            print('Tellurics have been saved as attribute. Run data_set.apply_telluric_correction() to apply correction !')

        # store some attributes
        self.SNR = np.array(self.SNR)
        self.airmass = np.array(self.airmass)
        self.BERV = np.array(self.BERV)

        # give a default name for the data_set to ease & homogeneize naming format
        self.name = '_'.join(self.transit_dic['data_dir'].replace('//','/').split('/')[-4:-1])

        ## compute off & on transit mask
        self.find_off_transit(print_info)

        ## compute orbital parameters & transit window weight
        self.compute_kepler_orbit(show=False)
        self.compute_transit_window(show=False)

        ## compute velocities
        # Get position vector (same units as sma) and true anomaly (radians) at given times from mid-transit
        r, f = self.ke.xyzPos(self.time_from_mid, getTA=True)
        # grab other orbital parameters
        Kp = self.transit_dic['Kp'] # m/s
        i_rad = np.radians(self.transit_dic['i'])
        w_rad = np.radians(self.transit_dic['w'])
        e = self.transit_dic['e']
        # Planet instant velocity on it's orbit projected on line of sight [m/s] : Add -1 factor to get negative velocity when the planet is moving toward us (blueshift)
        Vp = -1 * Kp * np.sin(i_rad) * (np.cos(w_rad + f) + e * np.cos(w_rad)) # same units as Kp
        Vrv_star = -1 * self.transit_dic['Mp']/self.transit_dic['Ms'] * Vp
        # BERV & systemic velocity
        BERV = self.BERV
        Vs = self.transit_dic['Vs']

        # In case of circular orbit, at beginning of transit the planet moves toward us so the star moves backward, so Vrv_star must be > 0
        # Relative velocity between observer on Earth and the star system in earth Rest Frame [m/s] : Vs > 0 means earth moves toward the star
        Vd = BERV - Vs - Vrv_star
        # Total relative velocity between observer on Earth and the Planet Rest Frame [m/s] : Vtot > 0 means we're moving toward the planet, Vtot < 0 means we're moving backward wrt the planet
        Vtot = BERV - Vs - Vp
        # save velocities in the class
        self.Vtot = Vtot
        self.Vd = Vd
        self.Vrv_star = Vrv_star
        self.Vp = Vp

        # Remove empty or null columns: any column full of 0 or NaN is removed
        mask = np.all( (self.data == 0) + np.isnan(self.data), axis=0)
        self.data[:,mask] = np.nan
        print('Removed empty or null columns from data cube')

        if print_info: print(f"\n{len(self.files)} files have been succesfully loaded from '{self.path}'\nData, wavelengths, blaze and corresponding headers have all been loaded from these files in data cubes of shape {self.shape}")

        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

    def load_tellurics(self, tellu_hdu_index, OH_hdu_index, print_info = True, overwrite = False):
        # load telluric Reconstructed model from "t files". Must simply provide the hdu index for the "Recon" & "OH Line" data in the FITS
        file_format = 't.fits'
        path = self.transit_dic['data_dir']
        # prevent overwriting if method is launched twice
        if self.tellurics is not None:
            if not overwrite:
                print("Data have already been loaded through this method. If you want to overwrite them, please call this method again adding the following parameter : 'overwrite = True' (You will lose previous reduction steps !).")
                return()
            else:
                print("Data have already been loaded through this method : overwriting them")

        # extract all files from directory matching with file_format
        if print_info : print(f"Loading tellurics spectra in files with format '{file_format}' in '{path}' :")
        self.tellu_files = [file for file in os.listdir(path) if file_format in file]
        if print_info : print(f"{len(self.tellu_files)} out of {len(os.listdir(path))} files selected !")

        # also using its header to make sure we are on the right set of data and have selected the right channel
        if print_info : print('\nThe following hdu will be extracted with the following roles. Please make sure this is what you want :')

        # orders files in observation time order
        obs_time_list = [fits.open(path + el)[1].header['BJD'] for el in self.tellu_files] # starting time of observation in Barycentric Julian Date
        sorting_index = np.argsort(obs_time_list)
        self.tellu_files = [self.tellu_files[k] for k in sorting_index]

        # opening the first fits to get its shape
        # /！\ WE SUPPOSE ALL FITS HAVE THE SAME SHAPE /!\
        f = fits.open(path+self.tellu_files[0])
        if print_info : print(f"Selected HDU '{f[tellu_hdu_index].header['EXTNAME']}' at index {tellu_hdu_index} as 'tellurics'.")
        # test if there is a OH line HDU, as it may not be always the case ?
        try:
            f[OH_hdu_index].header
            has_OH = True
        except:
            print(f"No OH hdu found for index #{OH_hdu_index}")
            has_OH = False

        if print_info and has_OH: print(f"Selected HDU '{f[OH_hdu_index].header['EXTNAME']}' at index {OH_hdu_index} as 'OH lines'.")

        # create a big cubic numpy array : each given fits file will have its 2D data stored along the first axis
        self.tellurics  = np.zeros((len(self.tellu_files),*f[tellu_hdu_index].data.shape))
        self.OH_lines   = np.zeros((len(self.tellu_files),*f[tellu_hdu_index].data.shape))

        # compute the kepler Elliptical orbit
        self.compute_kepler_orbit(show=False)

        if print_info : print(f"\nBuilding data cube of shape : {self.tellurics.shape}")
        if print_info : print('Filling the data cube with selected files...')

        # then fill the cube with each fits data:
        for k,el in enumerate(self.tellu_files):
            f = fits.open(path + el)
            self.tellurics[k] = f[tellu_hdu_index].data
            if has_OH: self.OH_lines[k] = f[OH_hdu_index].data
            f.close()

        if print_info: print(f"\n{len(self.tellu_files)} files with '{file_format}' have been succesfully loaded from '{path}'\nTellurics spectra and corresponding headers have all been loaded from these files in data cubes of shape {self.tellurics.shape}")
        if not has_OH: print("WARNING: OH lines HDU was not found: OH lines has been filled with 0., proceed with cautions.")

    def plot_before_after(self, obs, std = False, **kwargs):
        '''
        Plot a grid of each order of spectra for a given observation before and after last step using DataSet history by default
        Adding "std_axis = int" allows to plot the std of each spectra along given axis.
        '''

        last_step_ID = list(self.history.keys())[-1]
        previous_step_ID = list(self.history.keys())[-2]

        # Handle the 'Xlabel' argument to prevent duplicata from kwargs
        if 'Xlabel' in kwargs:
            Xlabel = kwargs['Xlabel']
            kwargs.pop('Xlabel')
        else:
            Xlabel = f'$\lambda$ [nm]'

        if not std:
            plot_grid(self.wave[obs],[self.history[previous_step_ID]['data'][obs],self.data[obs]],Xlabel=Xlabel, title=f'Spectra before (blue) & after (orange) {last_step_ID} for obs {obs}',**kwargs)
        else:
            plot_grid(self.wave[obs],[self.weighted_std(self.history[previous_step_ID]['data']),self.weighted_std(self.data)],Xlabel=Xlabel, title=f'Std along observation before (blue) & after (orange) {last_step_ID} for obs {obs}',**kwargs)

    def plot_all_before_after(self,obs,order,std=False,**kwargs):

        last_key = list(self.history.keys())[0]
        for key in list(self.history.keys())[1:]:
            print(f"Plotting {last_key} Vs {key}...")
            # Handle the 'Xlabel' argument to prevent duplicata from kwargs
            if 'Xlabel' in kwargs:
                Xlabel = kwargs['Xlabel']
                kwargs.pop('Xlabel')
            else:
                Xlabel = f'$\lambda$ [nm]'

            if isinstance(obs,int) and isinstance(order,int):
                if not std:
                    plt.figure()
                    plt.plot(self.wave[obs,order],self.history[last_key]['data'][obs,order],label=last_key,**kwargs)
                    plt.plot(self.wave[obs,order],self.history[key]['data'][obs,order],label=key,**kwargs)
                    plt.title(f'Spectra before (blue) & after (orange) {key} for obs {obs}')
                    plt.xlabel(Xlabel)
                    plt.ylabel('Rel Flux')
                    plt.legend()
                else:
                    plt.figure()
                    plt.plot(self.wave[obs,order],self.weighted_std(self.history[last_key]['data'][obs,order]),label=last_key,**kwargs)
                    plt.plot(self.wave[obs,order],self.weighted_std(self.history[key]['data'][obs,order]),label=key,**kwargs)
                    plt.title(f'Spectra before (blue) & after (orange) {key} for obs {obs}')
                    plt.xlabel(Xlabel)
                    plt.ylabel('Rel Flux')
                    plt.legend()

            else:
                if not std:
                    plot_grid(self.wave[obs][order],[self.history[last_key]['data'][obs][order],self.history[key]['data'][obs][order]],Xlabel=Xlabel, title=f'Spectra before (blue) & after (orange) {key} for obs {obs}',**kwargs)
                else:
                    plot_grid(self.wave[obs][order],[self.weighted_std(self.history[last_key]['data'][order]),self.weighted_std(self.history[key]['data'][order])],Xlabel=Xlabel, title=f'Std along axis {std_axis} before (blue) & after (orange) {last_step_ID} for obs {obs}',**kwargs)

            last_key = key

    def plot_2d_step(self,order,contact_point=False,vmax='None',vmin='None',**kwargs):
        '''
        plot data in 2d (time Vs wavelength) for each reduction steps
        '''
        for key, value in self.history.items():
            data = self.history[key]['data'][:,order]
            # show data before stellar lines correction in earth Rest Frame
            fig,ax = plt.subplots(1,1,figsize=(12,8),sharex='all')
            # 2D plot

            # 2D plot without 1D slice in Star RoF
            if vmax=='None' and vmin=='None':
                imshow_obj = ax.pcolormesh(self.wave[0,order],self.time_from_mid,data,shading='auto')
            else:
                imshow_obj = ax.pcolormesh(self.wave[0,order],self.time_from_mid,data,shading='auto',vmin=vmin,vmax=vmax)

            xmin = self.wave[0,order].min()
            xmax = self.wave[0,order].max()
            # show transit start/stop
            if contact_point:
                # /!\ T1-T4 contact points are already defined for a midpoint at T=0. T1-T4 are in days
                ax.hlines(self.T1,xmin,xmax,ls='--',color='k',label='transit start/stop')
                ax.hlines(self.T4,xmin,xmax,ls='--',color='k')
                ax.hlines(self.T2,xmin,xmax,ls='-.',color='k',label='ingress/egress')
                ax.hlines(self.T3,xmin,xmax,ls='-.',color='k')

            fig.colorbar(imshow_obj,ax=ax)
            ax.set_ylabel('Time from mid [days]')

            ax.legend(loc='upper center', bbox_to_anchor=(1.3, 1))

            ax.set_title(f'{self.path} after {key}')
            ax.ticklabel_format(useOffset=False)
            ax.set_xlabel('$\lambda$ [nm]')

    def plot_2d_order(self,order,contact_point=False,vmax='None',vmin='None',**kwargs):
        '''
        plot data in 2d (time Vs wavelength) for each reduction steps
        '''
        data = self.data[:,order]
        # show data before stellar lines correction in earth Rest Frame
        fig,ax = plt.subplots(1,1,figsize=(12,8),sharex='all')
        # 2D plot without 1D slice in Star RoF
        if vmax=='None' and vmin=='None':
            imshow_obj = ax.pcolormesh(self.wave[0,order],self.time_from_mid,data,shading='auto')
        else:
            imshow_obj = ax.pcolormesh(self.wave[0,order],self.time_from_mid,data,shading='auto',vmin=vmin,vmax=vmax)

        xmin = self.wave[0,order].min()
        xmax = self.wave[0,order].max()
        # show transit start/stop
        if contact_point:
            # /!\ T1-T4 contact points are already defined for a midpoint at T=0. T1-T4 are in days
            ax.hlines(self.T1,xmin,xmax,ls='--',color='k',label='transit start/stop')
            ax.hlines(self.T4,xmin,xmax,ls='--',color='k')
            ax.hlines(self.T2,xmin,xmax,ls='-.',color='k',label='ingress/egress')
            ax.hlines(self.T3,xmin,xmax,ls='-.',color='k')

        fig.colorbar(imshow_obj,ax=ax)
        ax.set_ylabel('Time from mid [days]')

        ax.legend(loc='upper center', bbox_to_anchor=(1.3, 1))

        ax.set_title(f'{self.path} - Order {order}')
        ax.ticklabel_format(useOffset=False)
        ax.set_xlabel('$\lambda$ [nm]')

    
    def plot_2d_obs(self,obs,vmax='None',vmin='None',**kwargs):
        '''
        plot data in 2d (time Vs wavelength) for each reduction steps
        '''
        data = self.data[obs,:]
        
        fig,ax = plt.subplots(1,1,figsize=(12,8),sharex='all')
        if vmax=='None' and vmin=='None':
            plt.imshow(data,origin='lower',aspect='auto')
        else:
            plt.imshow(data,origin='lower',aspect='auto',vmin=vmin,vmax=vmax)

        ax.set_ylabel('Spectral order')
        ax.set_title(f'{self.path} - Observation {obs}')
        ax.ticklabel_format(useOffset=False)
        ax.set_xlabel('Pixel n°')

    def weighted_mean(self,data,transit_mask='full',custom_mask=None):
        '''
        Compute mean along observation weighted by SNR**2 for given data cube
        transit_mask can take 3 values:
        - 'full' : all observations are used to compute the weighted mean
        - 'off'  : only off-transit observations are used to compute the eighted mean
        - 'on'   : only on-transit observations btw T1 & T4 are used to compute the weighted mean. No transit window weight used in the mean
        - 't2t3' : only on-transit observations btw T2 & T3 are used to compute the weighted mean. No transit window weight used in the mean
        - 'tlc'  : use observations weighted by the transit window weight. Equivalent to 'on' but weighted with the transit window weight computed with batman
        - 'custom' : use a custom map of weights. Note that in this case the SNR² will not be included in the weights: user has to manually include it in the weights map. The map must have the same dimension as the data (3 axis: obs, order, wavelength)
        '''
        weights = self.SNR*self.SNR
        if transit_mask == "full":
            pass
        elif transit_mask == "off":
            # if self.off_transit_mask==None: raise NameError('You must run DataSet.find_off_transit() first !')
            weights[self.on_transit_mask] = 0. # set on transit observation's weights to 0
        elif transit_mask == "on":
            weights[self.off_transit_mask] = 0. # set off transit observation's weights to 0
        elif transit_mask == "t2t3":
            weights[self.time_from_mid<self.T2] = 0. # set weights of obs before T2 to 0
            weights[self.time_from_mid>self.T3] = 0. # set weights of obs after T3 to 0
        elif transit_mask =='tlc':
            weights *= self.transit_weight # is 0 for off-transit observations
        elif transit_mask == 'custom':
            if custom_mask is None: raise NameError('transit_mask is "custom", you must provide a map of weights as "custom_mask"')
            weights = custom_mask
        else:
            raise NameError('transit_mask must be "full", "on", "off", "t2t3", "tlc" or "custom".')

        masked_data = np.ma.masked_invalid(data) # data with mask corresponding to inf or nan elements
        data_avg = np.ma.average(masked_data,axis=0,weights=weights)
        return data_avg

    def median(self,data,transit_mask='full'):
        '''
        Compute median along observation weighted by SNR**2 for given data cube
        transit_mask can take 3 values:
        - 'full' : all observations are used to compute the weighted mean
        - 'off'  : only off-transit observations are used to compute the eighted mean
        - 'on'   : only on-transit observations btw T1 & T4 are used to compute the weighted mean. No transit window weight used in the mean
        - 't2t3' : only on-transit observations btw T2 & T3 are used to compute the weighted mean. No transit window weight used in the mean
        - 'tlc'  : use observations weighted by the transit window weight. Equivalent to 'on' but weighted with the transit window weight computed with batman
        '''
        weights = self.SNR*self.SNR
        if transit_mask == "full":
            pass
        elif transit_mask == "off":
            # if self.off_transit_mask==None: raise NameError('You must run DataSet.find_off_transit() first !')
            weights[self.on_transit_mask] = 0. # set on transit observation's weights to 0
        elif transit_mask == "on":
            weights[self.off_transit_mask] = 0. # set off transit observation's weights to 0
        elif transit_mask == "t2t3":
            weights[self.time_from_mid<self.T2] = 0. # set weights of obs before T2 to 0
            weights[self.time_from_mid>self.T3] = 0. # set weights of obs after T3 to 0
        elif transit_mask =='tlc':
            weights *= self.transit_weight # is 0 for off-transit observations
        else:
            raise NameError('transit_mask must be "full", "on", "off", "t2t3" or "tlc".')

        masked_data = np.ma.masked_invalid(data) # data with mask corresponding to inf or nan elements
        data_avg = np.ma.median(masked_data,axis=0,weights=weights)
        return data_avg

    def weighted_std(self,data,transit_mask='full'):
        '''
        Compute std along observations weighted by SNR**2 for given data cube
        transit_mask can take 3 values:
        - 'full' : all observations are used to compute the weighted mean
        - 'off'  : only off-transit observations are used to compute the eighted mean
        - 'on'   : only on-transit observations btw T1 & T4 are used to compute the weighted mean. No transit window weight used in the mean
        - 't2t3' : only on-transit observations btw T2 & T3 are used to compute the weighted mean. No transit window weight used in the mean
        - 'tlc'  : use observations weighted by the transit window weight. Equivalent to 'on' but weighted with the transit window weight computed with batman
        '''
        weights = self.SNR*self.SNR
        if transit_mask == "full":
            pass
        elif transit_mask == "off":
            # if self.off_transit_mask==None: raise NameError('You must run DataSet.find_off_transit() first !')
            weights[self.on_transit_mask] = 0. # set on transit observation's weights to 0
        elif transit_mask == "on":
            weights[self.off_transit_mask] = 0. # set off transit observation's weights to 0
        elif transit_mask == "t2t3":
            weights[self.time_from_mid<self.T2] = 0. # set weights of obs before T2 to 0
            weights[self.time_from_mid>self.T3] = 0. # set weights of obs after T3 to 0
        elif transit_mask =='tlc':
            weights *= self.transit_weight # is 0 for off-transit observations
        else:
            raise NameError('transit_mask must be "full", "on", "off", "t2t3" or "tlc".')

        masked_data = np.ma.masked_invalid(data) # data with mask corresponding to inf or nan elements
        N = np.sum(weights>0) # nb of non zero weights. See https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
        data_avg = self.weighted_mean(data,transit_mask)
        std = np.sqrt( (N / (N-1)) * np.ma.average( (masked_data-data_avg)**2, weights=weights, axis=0) )

        # also remove values with extremely low std (outliers, constant/hot pixels...)
        thresh = 1e-15
        std = np.ma.masked_where(std < thresh, std)

        return std

    def find_off_transit_old(self, duration, U_duration, midpoint, U_midpoint, print_info = True, overwrite = False, save = None):
        '''
        duration : in days
        U_duration : in days
        midpoint : in BJD-TBD (Barycentric Julian Days in Barycentric Dynamical Time standard)
        U_midpoint : in BJD-TBD (Barycentric Julian Days in Barycentric Dynamical Time standard)
        '''
        # Current step ID
        step_ID = 'find_off_transit'
        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        # Retrieve current step parameters

        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')
        # Warn if step already found in history
        if(step_ID in self.history.keys()):
            warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            k = 0
            former_step_ID = step_ID
            while step_ID in self.history.keys():
                k+=1
                step_ID = former_step_ID+'_'+str(k)
        # prevent overwriting if method is launched twice
        if self.off_transit_mask != None:
            if not overwrite:
                print("A mask has already been built for off transit spectra. If you want to overwrite it, please call this method again adding the following parameter : 'overwrite = True'")
                return()
            else:
                print("A mask has already been built for off transit spectra : overwriting it")

        self.off_transit_mask = []
        self.on_transit_mask = []
        # add midpoint in attribute
        self.midpoint = midpoint # in BJD-TBD
        # Compute transit start and stop with incertainties
        transit_start = midpoint - U_midpoint - (duration - U_duration) / 2.
        transit_stop = midpoint + U_midpoint + (duration + U_duration) / 2.

        # Selecting only files whose observations stopped before transit start and started before transit end
        n = len(self.files)
        if print_info : print('\nBuilding the off-transit spectra masked :\n')
        for index,file in enumerate(self.files):
            if print_info : print(f'\r{index+1} / {n}',end='',flush=True)
            f = fits.open(self.path+file)
            obs_mid = f[1].header['BJD']
            if (( obs_mid >= transit_start) and ( obs_mid <= transit_stop)):
                self.on_transit_mask.append(index)
            else:
                self.off_transit_mask.append(index)
            index += 1
            f.close()

        # Show selected files index
        if print_info:
            print()
            print(f'Found {len(self.off_transit_mask)} off-transit spectra among {n} spectra')
            # print('Index list of off_transit spectra : ',self.off_transit_mask)
            # print('Corresponding off transit spectra names : ',[f'...{self.files[k][4:]}' for k in self.off_transit_mask])

        if print_info : print('\nFind off transit spectra done !')

        # Save in history
        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

        else :
            self.history[step_ID] = {'parameters' : parameters}

    def find_off_transit(self, print_info = True, overwrite = False, save = None):
        '''
        Find off and on-transit observations by compairing each observation time with the 4 contact points of transit,
        computed with PyAstronomy.pyasl module
        Also compute the time vector
        - 'a'           : planet semi-major axis in m
        - 'Rs'          : stellar radii in m
        - 'Rp'          : planet radii in m
        - 'Porb'        : orbital period in days
        - 'i'           : orbital inclination in degree
        - 'e'           : excentricity
        - 'w'           : argument of periastron
        - 'midpoint'    : midpoint time of the transit, in BJD-TBD
        '''
        # Current step ID
        step_ID = 'find_off_transit'
        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        # Retrieve current step parameters

        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')
        # Warn if step already found in history
        if(step_ID in self.history.keys()):
            warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            k = 0
            former_step_ID = step_ID
            while step_ID in self.history.keys():
                k+=1
                step_ID = former_step_ID+'_'+str(k)
        # prevent overwriting if method is launched twice
        if self.off_transit_mask != None:
            if not overwrite:
                print("A mask has already been built for off transit spectra. If you want to overwrite it, please call this method again adding the following parameter : 'overwrite = True'")
                return()
            else:
                print("A mask has already been built for off transit spectra : overwriting it")

        self.off_transit_mask = []
        self.on_transit_mask = []
        # also build ingress & egress mask
        self.ingress_mask = []
        self.egress_mask = []
        # add midpoint in attribute
        self.midpoint = self.transit_dic['midpoint'] # in BJD-TBD
        # Compute transit's four contact points
        # SMA in stellar radii
        sma = self.transit_dic['a'] / self.transit_dic['Rs']
        # Rp/Rs
        rprs = self.transit_dic['Rp'] / self.transit_dic['Rs']
        # Orbital inclination
        inc = self.transit_dic['i']
        # Orbital period (time units are arbitrary but must be consistent)
        p = self.transit_dic['Porb']
        # Eccentricity
        e = self.transit_dic['e']
        # Argument of periastron (planetary orbit)
        w = self.transit_dic['w']
        # Time of periastron passage : we set the origin at midpoint
        tau = 0.
        # compute contact points for primary transit ("p")
        self.contact_points = np.array(pyasl.transit_T1_T4_ell(sma, rprs, inc, p, tau, e, w, transit="p"))
        # sometimes transit won't have t2 & t3 points because the planet doesn't fully cross in front of the star, we must handle the case when t2 & t3 are thus None values :
        if self.contact_points[1] is None : self.contact_points[1] = self.contact_points[0]
        if self.contact_points[2] is None : self.contact_points[2] = self.contact_points[3]
        self.contact_points -= ((self.contact_points[0]+self.contact_points[3])/2) # center contact points around 0. (which is the midpoint value)
        self.T1,self.T2,self.T3,self.T4 = self.contact_points
        # Selecting only files whose observations stopped before transit start and started before transit end
        n = len(self.files)
        if print_info : print('\nBuilding the off-transit spectra masked :\n')
        for index,hdr in self.headers.items():
            if print_info : print(f'\r{index+1} / {n}',end='',flush=True)
            obs_mid = hdr['data header']['BJD']
            if (( (obs_mid-self.midpoint) < self.T1) or ( (obs_mid-self.midpoint) > self.T4)):
                self.off_transit_mask.append(index)
            else:
                self.on_transit_mask.append(index)
            # also build ingress & egress mask
            if (( (obs_mid-self.midpoint) <= self.T2) and ( (obs_mid-self.midpoint) >= self.T1)):
                self.ingress_mask.append(index)
            if (( (obs_mid-self.midpoint) <= self.T4) and ( (obs_mid-self.midpoint) >= self.T3)):
                self.egress_mask.append(index)


        self.time_vector = np.array([HDU['data header']['BJD'] for HDU in self.headers.values()]) # vector containing the date of each observation in BJDTBD
        self.time_from_mid = (self.time_vector - self.midpoint) # Time from mid-transit in days (BJDTBD)

        # Show selected files index
        if print_info:
            print()
            print(f'Found {len(self.off_transit_mask)} off-transit spectra among {n} spectra')
            # print('Index list of off_transit spectra : ',self.off_transit_mask)
            # print('Corresponding off transit spectra names : ',[f'...{self.files[k][4:]}' for k in self.off_transit_mask])

        if print_info : print('\nFind off transit spectra done !')

        # Save in history
        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

        else :
            self.history[step_ID] = {'parameters' : parameters}

    def div_by_blaze(self, save = None, print_info = True):
        # Current step ID
        step_ID = 'div_by_blaze'
        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        ## Retrieve current step parameters
        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')
        # Warn if step already found in history
        if(step_ID in self.history.keys()):
            warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            k = 0
            former_step_ID = step_ID
            while step_ID in self.history.keys():
                k+=1
                step_ID = former_step_ID+'_'+str(k)
        ## DIVISION BY BLAZE FUNCTION ##
        self.data /= self.blaze
        ################################

        if print_info : print('\nDivision by blaze function done !')

        # Save in history
        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

        else :
            self.history[step_ID] = {'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

    def norm_with_median(self, save = None, print_info = True, compute_median_after_div_by_avg = True):
        # Current step ID
        step_ID = 'norm_with_median'
        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        ## Retrieve current step parameters
        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')
        # Warn if step already found in history
        if(step_ID in self.history.keys()):
            warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            k = 0
            former_step_ID = step_ID
            while step_ID in self.history.keys():
                k+=1
                step_ID = former_step_ID+'_'+str(k)
        ## NORMALIZATION WITH MEDIAN ALONG OBSERVATIONS ##
        if compute_median_after_div_by_avg:
            # compute median on a copy of data from which we removed the avg signal
            data_copy = np.copy(self.data)
            if type(data_copy) == np.ma.masked_array: data_copy = data_copy.data
            data_copy /= self.weighted_mean(data_copy, 'full')
            self.data /= np.nanmedian(data_copy,axis=2)[...,None] 

        elif type(self.data) == np.ma.masked_array:
            self.data /= np.nanmedian(self.data.data,axis=2)[...,None] # if data are masked
        
        else:
            self.data /= np.nanmedian(self.data,axis=2)[...,None] # by adding [...,None]  numpy will automatically duplicate the 2D computed median array to match with self.data 3D shape

        ##################################################

        if print_info : print('\nNormalization with median along observations done !')

        # Save in history
        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

        else :
            self.history[step_ID] = {'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

    def div_by_weighted_average(self, transit_mask="full", save = None, print_info = True):
        # Current step ID
        step_ID = 'div_by_weighted_average'
        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        ## Retrieve current step parameters
        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')
        # Warn if step already found in history
        if(step_ID in self.history.keys()):
            warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            k = 0
            former_step_ID = step_ID
            while step_ID in self.history.keys():
                k+=1
                step_ID = former_step_ID+'_'+str(k)
        ## Division by SNR**2 weighted average spectra along observations ##
        data_avg = self.weighted_mean(self.data,transit_mask)
        self.data /= data_avg
        ####################################################################

        if print_info : print('\nDivision by SNR**2 weighted average spectra along observations done !')

        # Save in history
        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

        else :
            self.history[step_ID] = {'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

    def div_by_median(self, save = None, transit_mask="full", do_fit = False, print_info = True):
        # Current step ID
        step_ID = 'div_by_weighted_median'
        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        ## Retrieve current step parameters
        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')
        # Warn if step already found in history
        if(step_ID in self.history.keys()):
            warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            k = 0
            former_step_ID = step_ID
            while step_ID in self.history.keys():
                k+=1
                step_ID = former_step_ID+'_'+str(k)
        ## Division by SNR**2 best fit of weighted median spectra along observations ##
        data_median = self.weighted_median(self.data,transit_mask)

        if do_fit:
            self.data /= data_median
        else:
            self.data /= data_median

        ####################################################################

        if print_info : print('\nDivision by SNR**2 weighted average spectra along observations done !')

        # Save in history
        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

        else :
            self.history[step_ID] = {'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

    def div_by_moving_average(self, area, save = None, print_info = True):
        '''
        area : area size, in pixels, on which is computed the moving average
        '''
        # Current step ID
        step_ID = 'div_by_moving_average'
        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        ## Retrieve current step parameters
        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')
        # Warn if step already found in history
        if(step_ID in self.history.keys()):
            warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            k = 0
            former_step_ID = step_ID
            while step_ID in self.history.keys():
                k+=1
                step_ID = former_step_ID+'_'+str(k)

        ## Division by moving average computed by convoluing with a door function ##
        # We use Anne Boucher technique: we compute the moving average on a version of the data that has been divided by a simple master out to estimate the modal noise, & we apply this continuum correction on the data before division by master out
        master_out = self.weighted_mean(self.data,transit_mask='full')
        # create a matrix corresponding to a "door" function to convolve with our data
        kernel = np.ones(area)
        # normalize
        kernel /= kernel.sum()
        # computing the convolution of each spectra with this door function
        data_convolved = np.zeros(self.shape) * np.nan
        if print_info : print(f'\nComputing moving average by convoluing each spectra with a normalized door function of size {area} pixels :\n')
        for obs in range(self.shape[0]):
            for order in range(self.shape[1]):
                if print_info: print(f'\r{order + obs*self.shape[1] +1}/{self.shape[0]*self.shape[1]}',end='',flush=True)
                # convolve data (normalized and divided by mean along observations) with our kernel.
                # with 'valid' mode, convolution product is only given for points where the signals overlap completely. Values outside the signal boundary have no effect.
                # data_convolved[obs,order,half_area:self.shape[2]-half_area+1] = np.convolve(self.data[obs,order,:],kernel,'valid')
                data_convolved[obs,order] = convolve(self.data[obs,order,:]/master_out[order,:],kernel,boundary='fill',fill_value=1,nan_treatment='fill',preserve_nan=True,normalize_kernel=True)

        # plot_grid(self.wave[0],data_convolved[0],lw=0.2)
        self.data /= data_convolved
        ############################################################################

        if print_info : print(f'\nDivision by moving average with area size {area} done !')

        # Save in history
        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

        else :
            self.history[step_ID] = {'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

    def airmass_correction(self, model_deg, iterations, n_sigma, spectra_used_for_model = 'all', use_log = False, spectra_used_for_division = 'none', sigma_clip = True, plot_sigma = False, save = None, print_info = True):
        '''
        spectra_used_for_model : 'all', 'off-transit','on-transit'
        spectra_used_for_division : 'all', 'off-transit', 'on-transit', 'none'
        '''

        ## Warnings and saving preparation
        # Current step ID
        step_ID = 'airmass_correction'
        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        ## Retrieve current step parameters
        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')
        # Warn if step already found in history
        if(step_ID in self.history.keys()):
            warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            k = 0
            former_step_ID = step_ID
            while step_ID in self.history.keys():
                k+=1
                step_ID = former_step_ID+'_'+str(k)        # Error if self.off_transit_mask doesn't exist yet is needed for the step
        if ((spectra_used_for_division in ['off-transit','on-transit']) or (spectra_used_for_model in ['off-transit','on-transit'])) and (self.off_transit_mask==None): raise NameError(f"'spectra_used_for_division' has been set to '{spectra_used_for_division}', but you don't have launch the find_off_transit() method yet.\nPlease use the find_off_transit() method before airmass_correction() or change 'spectra_used_for_division' to 'all' or 'none' !")
        # Warn if spectra already div by mean previously and is asked to be divided again by mean here
        if(spectra_used_for_division in ['all','on-transit','off-transit']) and ('div_by_weighted_average' in self.history.keys()) : warn(f"Warning : spectra_used_for_division has been set to {spectra_used_for_division}, meaning spectra will be divided by its mean along observations during this step, but division by weighted average has already been done before, thus spectra will be divided twice !")


        ## Airmass correction : fitting each pixel variations along the observations and apply sigma clipping ##
        # For plotting the different I_th and sigma values along the iterations
        I_th_list = []
        sigma_list = []
        # number of observations
        N = self.shape[0]
        # degree of freedom
        dof = model_deg + 1
        # copy the data arrays & move to log space if required
        data_copy = np.copy(self.data)
        if use_log: 
            print('working on log-space')
            data_copy = np.log(data_copy)
        # Array to build model
        I = np.copy(data_copy)
        I[~ np.isfinite(self.data)] = 1. # set nan to 1 to minimize their effect on the model (otherwise model was badly estimated due to nan)
        # reshape the array to work with 2D array of shape (N_observations, len(orders)*len(wavelengths))
        I = I.reshape(self.shape[0],self.shape[1]*self.shape[2])
        # original array reshaped for final returning step
        original_spectra = np.ma.masked_invalid(data_copy).reshape(I.shape)

        # plot sigma clipping effect on pixels if True
        if(plot_sigma):
            fig1, ax1 = plt.subplots(figsize=(10,10))
            img = np.zeros(shape=I.shape, dtype=int)

        # mask for removed values : True where an element is masked (filled with False by default)
        mask = ~np.ones(I.shape).astype('bool')

        ## Model fit with Least Square
        # define the mask used for modelling
        if spectra_used_for_model=='all' :
            model_mask = np.ones(self.SNR.shape,dtype='bool')
        elif spectra_used_for_model=='off-transit':
            model_mask = self.off_transit_mask
        elif spectra_used_for_model=='on-transit':
            model_mask = self.on_transit_mask
        else:
            raise NameError(f"'{spectra_used_for_model}' is not a valid argument for spectra_used_for_model. Accepted arguments are : 'all', 'off-transit' or 'on-transit'.")

        # define our masked jacobian and weights matrix to compute the model on 'all', 'off-transit' or 'on-transit' observations
        W = np.diag(self.SNR[model_mask])**2
        J_masked = []
        for k in range(model_deg+1):
            J_masked.append((self.airmass[model_mask])**k)
        J_masked = np.array(J_masked).T

        # Build a new Jacobian matrix corresponding to all observations in order to apply our masked model on it
        J = []
        for k in range(model_deg+1):
            J.append((self.airmass)**k)
        J = np.array(J).T

        # intermediary matrix used later for computing model with data
        X = np.linalg.inv(J_masked.T @ W @ J_masked) @ J_masked.T @ W

        ## loop over iterations, each time clipping more outliers
        if print_info : print(f'\nApplying airmass correction, looping over iterations :\n')
        for i in range(iterations):
            if print_info: print(f'\r{i+1}/{iterations}',end='',flush=True)
            best_param = np.ma.dot(X, I[model_mask])
            # compute the modeled spectra on all observations using the best parameters found above with masked model
            I_th = np.ma.dot(J, best_param)
            I_th_list.append(I_th)

            if(sigma_clip):
                ## sigma clipping
                # compute the variance of each pixel with respect to the computed model along the observations (ie the weighted quadratic sum of each pixel's deviation from the model)
                A = np.ma.dot(np.diag(self.SNR) , (I_th - I))**2 / np.sum(self.SNR*self.SNR) # weighted quadratic distance from model
                variance = (N / (N - dof)) * np.sum( A , axis = 0 )
                # repeat the variance vector into a 2D array with the same shape as I_best and Y :
                variance_array = np.tile(variance,(N,1))
                # compute the mask whose elements are True when corresponding to a pixel whose deviation from the model is higher than n*sigma (ie variance higher than (n*sigma)²)
                temp_mask = (N*A >= (n_sigma*n_sigma) * variance_array)
                # save current sigma value for further plotting
                sigma_list.append(np.sqrt(variance_array*np.sum(self.SNR*self.SNR) / (N*np.power(self.SNR,2)[...,None])))

                # plot sigma clipping effect on pixels if True
                if(plot_sigma):
                    img[temp_mask] = i+1

                # update mask
                mask = np.logical_or(mask,temp_mask)
                # apply mask (bad values here are given their corresponding theoritical value. This is the best compromise found yet to use matricial computation of models that prevent from excluding values or giving them nan values, though it is not rigours to proceed this way)
                I[mask] = I_th[mask]

            # in case sigma clipping = False
            else:
                print('no sigma clipping, exiting loop')
                break

        def onclick(event):
            '''
            On click with mouse third button on 2d imshow : show the fit plot corresponding to the selected pixel
            '''
            nonlocal I, I_th
            if(event.button == 2):
                fig2, ax2 = plt.subplots(figsize=(15,15))
                index = int(event.xdata)
                wavelength = self.wave.reshape(self.shape[0],self.shape[1]*self.shape[2])[0,index]
                iteration_colors = []

                # plot model lines and +- sigma lines
                # for k,theo in enumerate(I_th_list):
                for k,theo in enumerate(I_th_list[:2]):
                    sigma = sigma_list[k]
                    # line, = ax2.plot(self.airmass,theo[:,index],label=f'Model @ iteration {k+1}')
                    line, = ax2.plot(self.airmass,theo[:,index],label=f'Fit at iteration {k+1}')
                    iteration_colors.append(line.get_color())

                    # ax2.plot(self.airmass,theo[:,index]+n_sigma*sigma[:,index],'_',c=iteration_colors[k],label=f'$\pm$ {n_sigma}$\sigma$ @ iteration {k+1}')
                    # ax2.plot(self.airmass,theo[:,index]-n_sigma*sigma[:,index],'_',c=iteration_colors[k])

                    ax2.plot(self.airmass,theo[:,index]+n_sigma*sigma[:,index],'--',c=iteration_colors[k])
                    ax2.plot(self.airmass,theo[:,index]-n_sigma*sigma[:,index],'--',c=iteration_colors[k])

                # # plot data with clipping colors
                # for pixel_color in np.unique(img[:,index]):
                #     color_mask = np.where(img[:,index]==pixel_color)[0]
                #     if(pixel_color==0):
                #         label='Unclipped'
                #         color = 'r'
                #     else:
                #         label=f'Clipped @ iteration {pixel_color}'
                #         color = iteration_colors[pixel_color-1]

                #     ax2.plot(self.airmass[color_mask],original_spectra[color_mask,index],'o',c=color,label=label)

                # plot data with clipping colors
                for pixel_color in np.unique(img[:,index]):
                    color_mask = np.where(img[:,index]==pixel_color)[0]
                    if(pixel_color==0):
                        # label='Unclipped'
                        color = 'k'
                    else:
                        label=f'Excluded from fit'
                        color = 'r'

                    ax2.plot(self.airmass[color_mask],original_spectra[color_mask,index],'+',c=color)

                ax2.set_title(f'Airmass fit at spectral bin $\lambda_i$ = {wavelength:.2f} nm')
                # ax2.set_title(f'Airmass fit for pixel @ $\lambda$ = {wavelength:.2f} $\AA$\npixel index = {index}')
                ax2.legend()

                ax2.set_xlabel(f'Airmass $\mu$')
                ax2.set_ylabel(r'Normalized intensity I$_{\lambda_i}$')

                plt.show()

        if(plot_sigma):
            # create custom colorbar with same iteration colors as fit plot
            custom_colors = ['r']+plt.rcParams['axes.prop_cycle'].by_key()['color']
            nb_colors = len(np.unique(img))
            cmap = colors.ListedColormap(custom_colors[:nb_colors])
            bounds = np.arange(nb_colors)
            # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            
            # find a good example for manuscript
            print('Here are some index corresponding to iteration 2:')
            iteration_mask = img==2
            no_nan_mask = np.isfinite(original_spectra)
            print(np.where(iteration_mask*no_nan_mask))
            print(self.wave.mean(axis=0).flatten()[np.where(iteration_mask*no_nan_mask)[1]])

            # cursor widget
            axim1 = ax1.imshow(img,aspect='auto',cmap='rainbow',interpolation='nearest')
            ax1.set_xlabel(f'$\lambda$ [nm]')
            ax1.set_ylabel('Observation N°')
            ax1.set_title(f'Removed pixels by sigma clipping after {iterations} iterations')

            cursor = Cursor(ax1, useblit=True, color='k', linewidth=1)

            cbar = fig1.colorbar(axim1, ax=ax1,label='Iteration n°', ticks=range(nb_colors))
            cbar.ax.set_yticklabels(np.arange(nb_colors))

            # cb = plt.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm,spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
            connection_id = fig1.canvas.mpl_connect('button_press_event', onclick)

            # use wavelengths as xticks
            waveticks = np.around(self.wave.mean(axis=0).flatten(), 2)
            ax1.set_xticks([px for px in range(0,waveticks.size,1000)])
            ax1.set_xticklabels(waveticks[::1000],rotation=45,fontsize=10)


            # upper axis
            ax3 = ax1.twiny()
            new_tick_locations = np.array([order*self.shape[2] for order in range(self.shape[1])])
            ax3.set_xlim(ax1.get_xlim())
            ax3.set_xticks(new_tick_locations)
            ax3.set_xticklabels([order for order in range(self.shape[1])],rotation=45,fontsize=10)
            ax3.set_xlabel(r"Orders")

            # vertical lines
            for x in new_tick_locations:
                ax1.axvline(x,c='k')

            plt.show()

        # weights for weighted average computation
        weights = self.SNR*self.SNR
        # reshape to match data
        weights_array = np.tile(weights,(self.shape[1],1)).T

        # compute the average theoric spectra along chosen observation
        if spectra_used_for_division=='all' or spectra_used_for_division=='none':
            avg_mask = np.ones(weights.shape,dtype='bool')
            I_th_avg = (I_th.T * weights).T.sum(axis=0) / weights.sum()
            I_th_avg = np.tile(I_th_avg,(*self.shape[:1],1))

        elif spectra_used_for_division=='off-transit':
            avg_mask = self.off_transit_mask
            mu_avg = (weights[avg_mask] * self.airmass[avg_mask]).sum() / (weights[avg_mask]).sum()
            mu_avg_2 = (weights[avg_mask] * self.airmass[avg_mask]**2).sum() / (weights[avg_mask]).sum()
            I0, I1, I2 = best_param
            I_th_avg =  I0 + I1 * mu_avg + I2 * mu_avg_2
            I_th_avg = np.tile(I_th_avg,(*self.shape[:1],1))

        elif spectra_used_for_division=='on-transit':
            avg_mask = [index for index in np.arange(weights.size) if not index in self.off_transit_mask]
            mu_avg = (weights[avg_mask] * self.airmass[avg_mask]).sum() / (weights[avg_mask]).sum()
            mu_avg_2 = (weights[avg_mask] * self.airmass[avg_mask]**2).sum() / (weights[avg_mask]).sum()
            I0, I1, I2 = best_param
            I_th_avg =  I0 + I1 * mu_avg + I2 * mu_avg_2
            I_th_avg = np.tile(I_th_avg,(*self.shape[:1],1))

        else:
            raise NameError(f"'{spectra_used_for_division}' is not a valid argument for spectra_used_for_division. Accepted arguments are : 'all', 'none', 'off-transit' or 'on-transit'.")

        # compute the corrected spectra (better use correction on spectra than I, which has been modified during the clipping process)
        I_corr = original_spectra - I_th + I_th_avg

        # compute the average spectra
        I_avg = (I[avg_mask].T * weights[avg_mask]).T.sum(axis=0) / weights[avg_mask].sum(axis=0)
        I_avg = np.tile(I_avg,(*self.shape[:1],1))

        # verify that in this case, I_avg = <I_th>
        if print_info:
            print()
            print(f'I_avg / I_th_avg =',np.ma.average(I_avg/I_th_avg))

        # Compute I_corr divided by I_avg (if demanded)
        if not spectra_used_for_division=='none':
            I_corr /= I_avg

        # return to linear space
        if use_log: I_corr = np.exp(I_corr)

        # Directly return I_corr as we have already divided by weighted mean before airmass correction:
        self.data = np.reshape(I_corr,self.shape)
        # Store the sigma clipping mask created in a class attribut
        self.airmass_corr_mask = mask
        #########################################################################################################

        if print_info : print('\nAirmass correction done !')
        # Save in history
        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

        else :
            self.history[step_ID] = {'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

    # <f
    # def new_airmass_correction(self, model_deg, iterations, n_sigma, spectra_used_for_model = 'all', spectra_used_for_division = 'none', sigma_clip = True, plot_sigma = False, save = None, print_info = True):
    #     '''
    #     Same as the old function, but this time loopoing on every wavelengths to fit each of them without their nan values for improved accuracy
    #     spectra_used_for_model : 'all', 'off-transit','on-transit'
    #     spectra_used_for_division : 'all', 'off-transit', 'on-transit', 'none'
    #     '''
    #
    #     ## Warnings and saving preparation
    #     # Current step ID
    #     step_ID = 'airmass_correction'
    #     # Default behaviour if no save parameter has been set
    #     if(save==None): save = self.save_history
    #     ## Retrieve current step parameters
    #     # retrieve currently defined local variables in current scope : parameters and step_ID
    #     parameters = locals()
    #     # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
    #     parameters.pop('self')
    #     # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
    #     parameters.pop('step_ID')
    #     # Warn if step already found in history
    #     if(step_ID in self.history.keys()):
    #         warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
    #         # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
    #         k = 0
    #         former_step_ID = step_ID
    #         while step_ID in self.history.keys():
    #             k+=1
    #             step_ID = former_step_ID+'_'+str(k)        # Error if self.off_transit_mask doesn't exist yet is needed for the step
    #     if ((spectra_used_for_division in ['off-transit','on-transit']) or (spectra_used_for_model in ['off-transit','on-transit'])) and (self.off_transit_mask==None): raise NameError(f"'spectra_used_for_division' has been set to '{spectra_used_for_division}', but you don't have launch the find_off_transit() method yet.\nPlease use the find_off_transit() method before airmass_correction() or change 'spectra_used_for_division' to 'all' or 'none' !")
    #     # Warn if spectra already div by mean previously and is asked to be divided again by mean here
    #     if(spectra_used_for_division in ['all','on-transit','off-transit']) and ('div_by_weighted_average' in self.history.keys()) : warn(f"Warning : spectra_used_for_division has been set to {spectra_used_for_division}, meaning spectra will be divided by its mean along observations during this step, but division by weighted average has already been done before, thus spectra will be divided twice !")
    #
    #
    #     ## Airmass correction : fitting each pixel variations along the observations and apply sigma clipping ##
    #     # number of observations
    #     N = self.shape[0]
    #     # degree of freedom
    #     dof = model_deg + 1
    #     # copy the data arrays
    #     I = np.copy(self.data)
    #     # reshape the array to work with 2D array of shape (N_observations, len(orders)*len(wavelengths))
    #     I = I.reshape(self.shape[0],self.shape[1]*self.shape[2])
    #     I -= 1
    #     I_th = np.zeros(I.shape)
    #     # original array reshaped for final returning step
    #     original_spectra = np.ma.masked_invalid(self.data).reshape(I.shape)
    #
    #     # For plotting the different I_th and sigma values along the iterations
    #     I_th_list = []
    #     sigma_list = []
    #
    #     # plot sigma clipping effect on pixels if True
    #     if(plot_sigma):
    #         fig1, ax1 = plt.subplots(figsize=(10,10))
    #         img = np.zeros(shape=I.shape, dtype=int)
    #
    #     # mask for removed values : True where an element is masked (filled with False by default)
    #     mask = ~np.ones(I.shape).astype('bool')
    #
    #     ## Model fit with Least Square
    #     # define the mask used for modelling
    #     if spectra_used_for_model=='all' :
    #         model_mask = np.ones(self.SNR.shape,dtype='bool')
    #     elif spectra_used_for_model=='off-transit':
    #         model_mask = self.off_transit_mask
    #     elif spectra_used_for_model=='on-transit':
    #         model_mask = self.on_transit_mask
    #     else:
    #         raise NameError(f"'{spectra_used_for_model}' is not a valid argument for spectra_used_for_model. Accepted arguments are : 'all', 'off-transit' or 'on-transit'.")
    #
    #     # define our masked jacobian and weights matrix to compute the model on 'all', 'off-transit' or 'on-transit' observations
    #     W = self.SNR[model_mask]**2
    #     x = self.airmass[model_mask]
    #     if print_info : print(f'\nApplying airmass correction, looping over iterations :\n')
    #
    #     # looping on all wavelengths and fit the model without the nan
    #     for l in range(I.shape[1]):
    #         if print_info and l%1000==0: print(f'\r{100*(l+1)/I.shape[1]:.0f}%',end='',flush=True)
    #         # mask containing false where values have been sigma clipped
    #         sigma_mask = np.ones(self.shape[0])
    #         y_th_list = [] # for plotting
    #         std_list = [] # for plotting
    #         y_th = np.zeros(self.airmass.shape)
    #         ## loop over iterations, each time clipping more outliers
    #         for i in range(iterations):
    #             y = I[model_mask,l]
    #             # ignore invalid values & clipped values for model estimation
    #             valid_mask = np.logical_and(np.isfinite(y),sigma_mask)
    #             # if no valid values : break
    #             if not np.sum(valid_mask)==0:
    #                 # fit polynomial model
    #                 best_param = np.polyfit(x[valid_mask],y[valid_mask],model_deg,w=W[valid_mask])
    #                 # compute the modeled spectra on all observations using the best parameters found above with masked model
    #                 y_th = np.sum([best_param[k]*self.airmass**k for k in range(model_deg)],axis=0)
    #                 y_th_list.append(y_th)
    #
    #                 if(sigma_clip):
    #                     ## sigma clipping
    #                     # compute the variance of each pixel with respect to the computed model along the observations (ie the weighted quadratic sum of each pixel's deviation from the model)
    #                     std = np.sqrt( (N / (N-dof)) * np.ma.average((y-y_th)**2, weights=W))
    #                     # compute the mask whose elements are False when corresponding to a pixel whose deviation from the model is higher than n*sigma (ie variance higher than (n*sigma)²)
    #                     threshold_max = np.ma.masked_array(y > y_th + n_sigma*std/np.sqrt(N))
    #                     threshold_min = np.ma.masked_array(y < y_th - n_sigma*std/np.sqrt(N))
    #                     new_mask = np.logical_or(threshold_max,threshold_min)
    #                     # if new_mask doesn't contain any new values to mask: stop
    #                     if(new_mask.sum()==0): break
    #                     sigma_mask = ~new_mask # contains true where values are not clipped
    #                     # save current sigma value for further plotting
    #                     std_list.append(std)
    #                     # plot sigma clipping effect on pixels if True
    #                     if(plot_sigma):
    #                         img[sigma_mask,l] = i+1
    #
    #                 # in case sigma clipping = False
    #                 else:
    #                     print('no sigma clipping, exiting loop')
    #                     break
    #
    #             I_th[:,l] = y_th + 1
    #             sigma_list.append(std_list)
    #             I_th_list.append(y_th_list)
    #
    #     def onclick(event):
    #         '''
    #         On click with mouse third button on 2d imshow : show the fit plot corresponding to the selected pixel
    #         '''
    #         nonlocal I
    #         if(event.button == 2):
    #             fig2, ax2 = plt.subplots(figsize=(15,15))
    #             index = int(event.xdata)
    #             wavelength = self.wave.reshape(self.shape[0],self.shape[1]*self.shape[2])[0,index]
    #             iteration_colors = []
    #
    #             # plot model lines and +- sigma lines
    #             for k,theo in enumerate(I_th_list[index]):
    #                 sigma = sigma_list[index][k]
    #                 print(sigma)
    #                 line, = ax2.plot(self.airmass,theo,label=f'Model @ iteration {k+1}')
    #                 iteration_colors.append(line.get_color())
    #
    #                 ax2.plot(self.airmass,theo+n_sigma*sigma,'_',c=iteration_colors[k],label=f'$\pm$ {n_sigma}$\sigma$ @ iteration {k+1}')
    #                 ax2.plot(self.airmass,theo-n_sigma*sigma,'_',c=iteration_colors[k])
    #
    #             # plot data with clipping colors
    #             for pixel_color in np.unique(img[:,index]):
    #                 color_mask = np.where(img[:,index]==pixel_color)[0]
    #                 if(pixel_color==0):
    #                     label='Unclipped'
    #                     color = 'r'
    #                 else:
    #                     label=f'Clipped @ iteration {pixel_color}'
    #                     color = iteration_colors[pixel_color-1]
    #
    #                 ax2.plot(self.airmass[color_mask],original_spectra[color_mask,index],'o',c=color,label=label)
    #
    #             ax2.set_title(f'Airmass fit for pixel @ $\lambda$ = {wavelength:.2f} $\AA$\npixel index = {index}')
    #             ax2.legend()
    #
    #             ax2.set_xlabel('airmass')
    #             ax2.set_ylabel('normalized flux')
    #
    #             plt.show()
    #
    #     if(plot_sigma):
    #         # create custom colorbar with same iteration colors as fit plot
    #         custom_colors = ['r']+plt.rcParams['axes.prop_cycle'].by_key()['color']
    #         nb_colors = len(np.unique(img))
    #         cmap = colors.ListedColormap(custom_colors[:nb_colors])
    #         bounds = np.arange(nb_colors)
    #         # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    #
    #         # cursor widget
    #         axim1 = ax1.imshow(img,aspect='auto',cmap='rainbow',interpolation='nearest')
    #         ax1.set_xlabel(f'$\lambda$ [nm]')
    #         ax1.set_ylabel('Observation N°')
    #         ax1.set_title(f'Removed pixels by sigma clipping after {iterations} iterations')
    #
    #         cursor = Cursor(ax1, useblit=True, color='k', linewidth=1)
    #
    #         cbar = fig1.colorbar(axim1, ax=ax1,label='Iteration n°', ticks=range(nb_colors))
    #         cbar.ax.set_yticklabels(np.arange(nb_colors))
    #
    #         # cb = plt.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm,spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
    #         connection_id = fig1.canvas.mpl_connect('button_press_event', onclick)
    #
    #         # use wavelengths as xticks
    #         waveticks = np.around(self.wave.mean(axis=0).flatten(), 2)
    #         ax1.set_xticks([px for px in range(0,waveticks.size,1000)])
    #         ax1.set_xticklabels(waveticks[::1000],rotation=45,fontsize=10)
    #
    #
    #         # upper axis
    #         ax3 = ax1.twiny()
    #         new_tick_locations = np.array([order*self.shape[2] for order in range(self.shape[1])])
    #         ax3.set_xlim(ax1.get_xlim())
    #         ax3.set_xticks(new_tick_locations)
    #         ax3.set_xticklabels([order for order in range(self.shape[1])],rotation=45,fontsize=10)
    #         ax3.set_xlabel(r"Orders")
    #
    #         # vertical lines
    #         for x in new_tick_locations:
    #             ax1.axvline(x,c='k')
    #
    #         plt.show()
    #
    #     # weights for weighted average computation
    #     weights = self.SNR*self.SNR
    #     # reshape to match data
    #     weights_array = np.tile(weights,(self.shape[1],1)).T
    #
    #     # compute the average theoric spectra along chosen observation
    #     if spectra_used_for_division=='all' or spectra_used_for_division=='none':
    #         avg_mask = np.ones(weights.shape,dtype='bool')
    #         I_th_avg = (I_th.T * weights).T.sum(axis=0) / weights.sum()
    #         I_th_avg = np.tile(I_th_avg,(*self.shape[:1],1))
    #
    #     elif spectra_used_for_division=='off-transit':
    #         avg_mask = self.off_transit_mask
    #         mu_avg = (weights[avg_mask] * self.airmass[avg_mask]).sum() / (weights[avg_mask]).sum()
    #         mu_avg_2 = (weights[avg_mask] * self.airmass[avg_mask]**2).sum() / (weights[avg_mask]).sum()
    #         I0, I1, I2 = best_param
    #         I_th_avg =  I0 + I1 * mu_avg + I2 * mu_avg_2
    #         I_th_avg = np.tile(I_th_avg,(*self.shape[:1],1))
    #
    #     elif spectra_used_for_division=='on-transit':
    #         avg_mask = [index for index in np.arange(weights.size) if not index in self.off_transit_mask]
    #         mu_avg = (weights[avg_mask] * self.airmass[avg_mask]).sum() / (weights[avg_mask]).sum()
    #         mu_avg_2 = (weights[avg_mask] * self.airmass[avg_mask]**2).sum() / (weights[avg_mask]).sum()
    #         I0, I1, I2 = best_param
    #         I_th_avg =  I0 + I1 * mu_avg + I2 * mu_avg_2
    #         I_th_avg = np.tile(I_th_avg,(*self.shape[:1],1))
    #
    #     else:
    #         raise NameError(f"'{spectra_used_for_division}' is not a valid argument for spectra_used_for_division. Accepted arguments are : 'all', 'none', 'off-transit' or 'on-transit'.")
    #
    #     # compute the corrected spectra (better use correction on spectra than I, which has been modified during the clipping process)
    #     I_corr = original_spectra - I_th + I_th_avg
    #
    #     # compute the average spectra
    #     I_avg = (I[avg_mask].T * weights[avg_mask]).T.sum(axis=0) / weights[avg_mask].sum(axis=0)
    #     I_avg = np.tile(I_avg,(*self.shape[:1],1))
    #
    #     # verify that in this case, I_avg = <I_th>
    #     if print_info:
    #         print()
    #         print(f'I_avg / I_th_avg =',np.ma.average(I_avg/I_th_avg))
    #
    #     # Compute I_corr divided by I_avg (if demanded)
    #     if not spectra_used_for_division=='none':
    #         I_corr /= I_avg
    #
    #     # Directly return I_corr as we have already divided by weighted mean before airmass correction:
    #     self.data = np.reshape(I_corr,self.shape)
    #     # Store the sigma clipping mask created in a class attribut
    #     self.airmass_corr_mask = mask
    #     #########################################################################################################
    #
    #     if print_info : print('\nAirmass correction done !')
    #     # Save in history
    #     if save:
    #         self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
    #         if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")
    #
    #     else :
    #         self.history[step_ID] = {'parameters' : parameters}
    #         if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")
    # f> new_airmass_correction -> not working ?

    def telluric_transmission_clipping(self, min_threshold = 0.4, remove_entire_spec_line = False, max_threshold = 0.97, save = None, print_info = True, plot = False):
        # Current step ID
        step_ID = 'telluric_transmission_clipping'
        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        ## Retrieve current step parameters
        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')
        # Warn if step already found in history
        if(step_ID in self.history.keys()):
            warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            k = 0
            former_step_ID = step_ID
            while step_ID in self.history.keys():
                k+=1
                step_ID = former_step_ID+'_'+str(k)        # Raise error if tellurics have not been loaded yet
        if self.tellurics is None: raise NameError(f"You must launch the load_tellurics() method before using telluric_transmission_clipping() !")

        ## Clip all pixels in data that correspond to a nan or a value in the corresponding telluric spectra under the transmission_threshold set ##
        if not remove_entire_spec_line:
            if print_info : print(f'\nClipping all pixels corresponding to a transmission under {min_threshold} or a nan value at corresponding wavelenght in telluric spectra :')
            telluric_clipping_mask = self.tellurics < min_threshold
            telluric_clipping_mask = np.logical_or(telluric_clipping_mask,~np.isfinite(self.tellurics))
            if print_info : print(f'Fraction of pixels that will be removed : {telluric_clipping_mask.sum()/telluric_clipping_mask.size:.3f}')
            self.data[telluric_clipping_mask] = np.nan
            if print_info : print(f'\nPixels clipping using tellurics spectra done with transmission threshold values : min={min_threshold} and max={max_threshold}!')

        else:
            # remove the "entire" (< max_threshold) telluric spectra lines whose center falls below the min_threshold
            # work on flatten data, easier
            if print_info : print(f'\nClipping all data points corresponding to wavelengths of spectral lines below {max_threshold} transmittance and whose center falls below {min_threshold}.')
            tellu_flat = np.mean(self.tellurics,axis=0).flatten() # tale the mean telluric profile along observations
            thresh_min_index = np.where(tellu_flat < min_threshold)[0] # index of tellurics data points below the threshold

            new_data = np.copy(self.data).reshape(self.shape[0],self.shape[1]*self.shape[2]) # flatten observations & wavelengths
            
            last_region = []
            mask_list = [] # for plotting
            print()
            for k,center_index in enumerate(thresh_min_index):
                print(f'\r{k+1}/{thresh_min_index.size}',end='',flush=True)

                if center_index in last_region:
                    # print(f'{center_index} already in last found region, skipping')
                    continue

                region_to_mask = []
                i = center_index
                region_to_mask.append(i)

                # exploring to the right
                while (i+1 > 0) : # prevent segmentation fault
                    if  i+1>=self.shape[1]*self.shape[2]: # stop if we reach the right limit of the spectrum
                        break
                    if (tellu_flat[i]>tellu_flat[i+1]) or (tellu_flat[i+1]>max_threshold) : # region stop when reaching another spectral line or exceeding the maximum threshold
                        break
                    i+=1
                    region_to_mask.append(i)

                # exploring to the left
                i = center_index
                while (i-1 > 0) : # prevent segmentation fault
                    if (tellu_flat[i]>tellu_flat[i-1]) or (tellu_flat[i-1]>max_threshold) : # region stop when reaching another spectral line or exceeding the maximum threshold
                        break
                    i-=1
                    region_to_mask = [i] + region_to_mask # append to left so the index are in ascending order

                last_region = region_to_mask
                mask_list.append(region_to_mask)
                # remove the spectral line found
                new_data[:,region_to_mask]= np.nan

            if plot:
                fig,ax = plt.subplots(2,1,sharex=True)
                # print(np.sum(region_to_mask)/len(region_to_mask))
                ax[0].plot(self.wave[0].flatten(),tellu_flat,'k',label='Telluric transmission model') # telluric transmission spectrum
                for k,region_to_mask in enumerate(mask_list):
                    if k==0: ax[0].plot(self.wave[0].flatten()[region_to_mask],tellu_flat[region_to_mask],'r',label='Telluric clipping mask')
                    else: ax[0].plot(self.wave[0].flatten()[region_to_mask],tellu_flat[region_to_mask],'r')
                ax[0].legend()
                ax[1].plot(self.wave[0].flatten(),self.data[0].flatten(),'r',label='Masked data')
                ax[1].plot(self.wave[0].flatten(),new_data[0].flatten(),'k')
                ax[1].legend(loc='lower right')                
                ax[1].set_xlabel('Wavelength [nm]')
                ax[0].set_ylabel('Transmission [%]')
                ax[1].set_ylabel('Normalized intensity')

            new_data = new_data.reshape(self.shape)
            self.data = np.copy(new_data)
            if print_info : print(f'\nTelluric clipping with telluric spectral line removing between {min_threshold} & {max_threshold} transmittance done !')
        ############################################################################################################################################

        # Save in history
        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

        else :
            self.history[step_ID] = {'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

    def sigma_clipping_on_obs(self, model_deg, iterations, n_sigma, plot_sigma=False, save = None, print_info = True):
        '''
        For each wavelength (of all orders), take the evolution of the intensity at this wavelength along the observations,
        draw a polynomial model of this evolution, and remove any pixel whose intensity is distant from the model more than n_sigma.
        '''
        # Current step ID
        step_ID = 'sigma_clipping_obs'
        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        # Retrieve current step parameters

        # print('TO DO: WE COULD REMOVE ONLY THE MOST DEVIANT VALUE FROM THE MODEL AT EACH ITERATION, THEN RECOMPUTE MODEL AGAIN, INSTEAD OF REMOVING ALL DEVIANT VALUES FROM MODEL AT EACH ITERATION...')

        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')
        # Warn if step already found in history
        if(step_ID in self.history.keys()):
            warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            k = 0
            former_step_ID = step_ID
            while step_ID in self.history.keys():
                k+=1
                step_ID = former_step_ID+'_'+str(k)
        ## DO STUFF HERE ##
        # For plotting the different I_th and sigma values along the iterations
        I_th_list = []
        sigma_list = []
        observation_indexs = np.arange(self.shape[0])
        # number of observations
        N = self.shape[0]
        # degree of freedom
        dof = model_deg + 1
        # copy the data arrays
        I = np.copy(self.data)
        # reshape the array to work with 2D array of shape (N_observations, len(orders)*len(wavelengths))
        I = I.reshape(self.shape[0],self.shape[1]*self.shape[2])
        # mask for removed values : True where an element is masked (filled with False by default)
        mask = ~np.ones(I.shape).astype('bool')
        # original array reshaped for final returning step
        original_spectra = np.copy(self.data.reshape(I.shape))

        # plot sigma clipping effect on pixels if True
        if(plot_sigma):
            fig1, ax1 = plt.subplots(figsize=(10,10))
            img = np.zeros(shape=I.shape, dtype=int)

        ## Model fit with Least Square
        # define our masked jacobian and weights matrix to compute the model on 'all', 'off-transit' or 'on-transit' observations
        W = np.diag(self.SNR)**2
        # Build a new Jacobian matrix corresponding to all observations in order to apply our masked model on it
        J = []
        for k in range(model_deg+1):
            J.append(observation_indexs**k)
        J = np.array(J).T

        # intermediary matrix used later for computing model with data
        X = np.linalg.inv(J.T @ W @ J) @ J.T @ W

        ## loop over iterations, each time clipping more outliers
        if print_info : print(f'\nApplying sigma clipping on obs, looping over iterations :\n')
        
        for i in range(iterations):
            if print_info: print(f'\r{i+1}/{iterations}',end='',flush=True)
            best_param = np.ma.dot(X, I)
            # compute the modeled spectra on all observations using the best parameters found above with masked model
            I_th = np.ma.dot(J, best_param)
            I_th_list.append(I_th)

            ## sigma clipping
            # compute the variance of each pixel with respect to the computed model along the observations (ie the weighted quadratic sum of each pixel's deviation from the model)
            A = np.ma.dot(np.diag(self.SNR), (I_th - I))**2 / np.sum(self.SNR*self.SNR) # weighted quadratic distance from model
            variance = (N / (N - dof)) * np.sum( A , axis = 0 )
            # repeat the variance vector into a 2D array with the same shape as I_best and Y :
            variance_array = np.tile(variance,(N,1))
            # compute the mask whose elements are True when corresponding to a pixel whose deviation from the model is higher than n*sigma (ie variance higher than (n*sigma)²)
            temp_mask = (N*A >= (n_sigma*n_sigma) * variance_array)
            # save current sigma value for further plotting
            sigma_list.append(np.sqrt(variance_array*np.sum(self.SNR*self.SNR) / (N*np.power(self.SNR,2)[...,None])))

            # plot sigma clipping effect on pixels if True
            if(plot_sigma):
                img[temp_mask] = i+1

            # update mask
            mask = np.logical_or(mask,temp_mask)
            # apply mask (bad values here are given their corresponding theoritical value. This is the best compromise found yet to use matricial computation of models that prevent from excluding values or giving them nan values, though it is not rigours to proceed this way)
            I[mask] = I_th[mask]

        # apply clipping mask on data
        self.data[mask.reshape(self.shape)] = np.nan

        def onclick(event):
            '''
            On click with mouse third button on 2d imshow : show the fit plot corresponding to the selected pixel
            '''
            nonlocal I, I_th
            if(event.button == 3):
                fig2, ax2 = plt.subplots(figsize=(15,15))
                index = int(event.xdata)
                wavelength = self.wave.reshape(self.shape[0],self.shape[1]*self.shape[2])[0,index]
                iteration_colors = []

                # plot model lines and +- sigma lines
                for k,theo in enumerate(I_th_list):
                    sigma = sigma_list[k]
                    line, = ax2.plot(observation_indexs,theo[:,index],label=f'Model @ iteration {k+1}')
                    iteration_colors.append(line.get_color())

                    ax2.plot(observation_indexs,theo[:,index]+n_sigma*sigma[:,index],'_',c=iteration_colors[k],label=f'$\pm$ {n_sigma}$\sigma$ @ iteration {k+1}')
                    ax2.plot(observation_indexs,theo[:,index]-n_sigma*sigma[:,index],'_',c=iteration_colors[k])

                # plot data with clipping colors
                for pixel_color in np.unique(img[:,index]):
                    color_mask = np.where(img[:,index]==pixel_color)[0]
                    if(pixel_color==0):
                        label='Unclipped'
                        color = 'r'
                    else:
                        label=f'Clipped @ iteration {pixel_color}'
                        color = iteration_colors[pixel_color-1]

                    ax2.plot(observation_indexs[color_mask],original_spectra[color_mask,index],'o',c=color,label=label)

                ax2.set_title(f'Sigma clipping for pixel @ $\lambda$ = {wavelength:.2f} $\AA$\npixel index = {index}')
                ax2.legend()

                ax2.set_xlabel('Obs n°')
                ax2.set_ylabel('normalized flux')

                plt.show()

        if(plot_sigma):
            # create custom colorbar with same iteration colors as fit plot
            custom_colors = ['r']+plt.rcParams['axes.prop_cycle'].by_key()['color']
            nb_colors = len(np.unique(img))
            cmap = colors.ListedColormap(custom_colors[:nb_colors])
            bounds = np.arange(nb_colors)
            # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

            # cursor widget
            axim1 = ax1.imshow(img,aspect='auto',cmap='rainbow',interpolation='nearest')
            ax1.set_xlabel(f'$\lambda$ [nm]')
            ax1.set_ylabel('Observation N°')
            ax1.set_title(f'Removed pixels by sigma clipping after {iterations} iterations')

            cursor = Cursor(ax1, useblit=True, color='k', linewidth=1)

            cbar = fig1.colorbar(axim1, ax=ax1,label='Iteration n°', ticks=range(nb_colors))
            cbar.ax.set_yticklabels(np.arange(nb_colors))

            # cb = plt.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm,spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
            connection_id = fig1.canvas.mpl_connect('button_press_event', onclick)

            # use wavelengths as xticks
            waveticks = np.around(self.wave.mean(axis=0).flatten(), 2)
            ax1.set_xticks([px for px in range(0,waveticks.size,1000)])
            ax1.set_xticklabels(waveticks[::1000],rotation=45,fontsize=10)


            # upper axis
            ax3 = ax1.twiny()
            new_tick_locations = np.array([order*self.shape[2] for order in range(self.shape[1])])
            ax3.set_xlim(ax1.get_xlim())
            ax3.set_xticks(new_tick_locations)
            ax3.set_xticklabels([order for order in range(self.shape[1])],rotation=45,fontsize=10)
            ax3.set_xlabel(r"Orders")

            # vertical lines
            for x in new_tick_locations:
                ax1.axvline(x,c='k')


            ### also plot a single example with 3 iterations for manuscript
            # find a spectral bin for which 3 iterations have been done
            iteration_to_plot = 1 # find a bin with that many iterations
            index = 171149 # np.where(img==iteration_to_plot)[0][0]
            print(index)
            print(np.any(img==5))
            wavelength = self.wave.reshape(self.shape[0],self.shape[1]*self.shape[2])[0,index]
            iteration_colors = []
            fig_aux,ax_aux = plt.subplots()
            # plot model lines and +- sigma lines
            for k,theo in enumerate(I_th_list):
                if k>iteration_to_plot: continue 
                sigma = sigma_list[k]
                line, = ax_aux.plot(self.time_from_mid,theo[:,index],label=f'Iteration {k+1}')
                iteration_colors.append(line.get_color())

                ax_aux.plot(self.time_from_mid,theo[:,index]+n_sigma*sigma[:,index],'--',c=iteration_colors[k])
                ax_aux.plot(self.time_from_mid,theo[:,index]-n_sigma*sigma[:,index],'--',c=iteration_colors[k])

            # plot data with clipping colors
            for pixel_color in np.unique(img[:,index]):
                color_mask = np.where(img[:,index]==pixel_color)[0]
                if(pixel_color==0):
                    # label='Unclipped'
                    color = 'k'
                else:
                    # label=f'Clipped @ iteration {pixel_color}'
                    # color = iteration_colors[pixel_color-1]
                    color = 'r'

                ax_aux.plot(self.time_from_mid[color_mask],original_spectra[color_mask,index],'+',c=color)

            ax_aux.set_title(f'Intensity at $\lambda$ = {wavelength:.2f} nm')
            ax_aux.legend()

            ax_aux.set_xlabel(r'Time from mid-transit [BJD$_{TDB}$]')
            ax_aux.set_ylabel('Normalized intensity')

            plt.show()


        ###################

        if print_info : print('\nSigma clipping along observation done !')

        # Save in history
        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

        else :
            self.history[step_ID] = {'parameters' : parameters}

    def sigma_clipping_on_std(self, n_sigma, frac, max_iterations=5, transmission_correction = True, save = None, print_info = True, orders_to_plot = []):
        '''
        Apply a lowess model to each order's temporal std, and remove wavelength whose std is distant from the model than a number of sigma (being the dispersion of std around the model)
        -> this will completely remove all observation of any wavelength whose std is distant from the model
        '''
        # Current step ID
        step_ID = 'sigma_clipping_std'
        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        ## Retrieve current step parameters
        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')
        # Warn if step already found in history
        if(step_ID in self.history.keys()):
            warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            k = 0
            former_step_ID = step_ID
            while step_ID in self.history.keys():
                k+=1
                step_ID = former_step_ID+'_'+str(k)
        # Error if transmission_correction is used but tellurics have not been loaded:
        if transmission_correction and (self.tellurics is None): raise NameError(f"You must launch the load_tellurics() method before using telluric_transmission_clipping() !")

        ## Remove pixels whose std along the observation deviate from the polynomial model more than a certain threshold ##
        # number of points used for std computation
        N = self.shape[0]
        std = self.weighted_std(self.data)
        wavelengths = self.wave.mean(axis=0)

        ## model each spectra std along the observation with a polynomial fit and remove pixels whose distance from the model is higher than n_sigma*sigma
        # loop over each spectra:
        if print_info: print(f'\nApplying sigma clipping on each orders\' std :\n')

        # store the clipping mask in class attribute
        self.clipping_mask = np.zeros(self.shape,dtype='bool')

        for order in range(self.shape[1]):
            if print_info: print(f'\r{order+1}/{self.shape[1]}',end='',flush=True)
            # for better model fitting, we bring wavelengths to values around 1 (otherwise, our n degree model will have to compute with numbers ranging from 1 to the power n of our values, thus losing in precision if our values are around 1000)
            median_wave = np.median(wavelengths[order])
            wave = wavelengths[order] / median_wave
            rms = std[order]

            if(order in orders_to_plot):
                # manuscript
                # don't forget to multiply wavelength back to their original order of size
                fig,ax = plt.subplots(2,1,sharex=True,figsize=(18,18))
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                ax[1].set_ylabel(r'Standard-deviation $\sigma_{\lambda_i}$')
                ax[1].set_xlabel(f'Wavelength [nm]')
                # ax[1].set_title(f'Sigma clipping')
                ax[1].plot(self.wave[0,order],rms,'g+',zorder=1,label='Clipped data')
                
                # Perform curve fitting
                x = self.wave[0,order,rms>0]
                y = rms[rms>0]
                coefficients = np.polyfit(x, y, 2)
                polynomial = np.poly1d(coefficients)
                ax[1].plot(x,polynomial(x),label='2nd order polynomial',lw=2)
                
                # ax[1].legend()

                # also plot the corresponding spectra in transparent
                ax[0].plot(wavelengths[order],self.tellurics[0,order])
                ax[0].set_ylabel(r"Earth transmission $T$")

                # fig,ax2 = plt.subplots(1,1,sharex=True,figsize=(10,10))
                # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                # ax2.set_ylabel(r'Standard-deviation $\sigma_{\lambda_i}$')
                # ax2.set_xlabel(f'Wavelength [nm]')
                # ax2.plot(self.wave[0,order],rms,'+',color='r',zorder=1)

                # # don't forget to multiply wavelength back to their original order of size
                # fig,ax = plt.subplots(2,1,sharex=True)
                # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                # ax[1].set_ylabel('Std')
                # ax[1].set_xlabel(f'$\lambda$ [nm]')
                # ax[1].set_title(f'Sigma clipping')
                # ax[1].plot(self.wave[0,order],rms,'.',mec='k',mfc='k',label=f'Data before sigma clipping',zorder=1)
                # ax[1].legend(fontsize=5)

                # # also plot the corresponding spectra in transparent
                # ax[0].plot(wavelengths[order],self.tellurics[0,order],linewidth=0.5)
                # ax[0].set_ylabel("Transmission")
                # ax[0].set_title(f'spectra before clipping : order {order}')

                # fig,ax2 = plt.subplots(1,1,sharex=True,figsize=(10,10))
                # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                # ax2.set_ylabel('Std')
                # ax2.set_xlabel(f'$\lambda$ [nm]')
                # ax2.set_title(f'Sigma clipping')
                # ax2.plot(self.wave[0,order],rms,'.',mec='k',mfc='k',label=f'Data before sigma clipping',zorder=1)

            # looping over the given number of iteration, each time clipping pixels and computing a new model with remaining points
            # if convergence is reached before max_iterations is reached, then loop is stopped
            for i in range(max_iterations):
                # use lowess to define model, thus avoiding extrem values to affect the model
                model = lowess(rms.filled(np.nan),wave,frac=frac,return_sorted=False) # lowess won't understand np.ma.array type, so here we have to fill masked values manually
                threshold_max = np.ma.masked_array(model * (1 + n_sigma/np.sqrt(2*N)))
                threshold_min = np.ma.masked_array(model * (1 - n_sigma/np.sqrt(2*N)))
                # Using transmission_correction add a term in the threshold definition taking into account the influence of telluric lines transmission to prevent pixels with high std due to low transmission to be removed
                if(transmission_correction):
                    tellurics = np.ma.masked_invalid(self.tellurics.mean(axis=0)[order])
                    tellurics[tellurics<=0] = np.ma.masked
                    threshold_max /= np.sqrt(tellurics)
                    threshold_min /= np.sqrt(tellurics)
                # update the clipping mask
                mask_up = rms > threshold_max
                mask_down = rms < threshold_min
                new_mask = np.logical_or(mask_up,mask_down)
                # if new_mask doesn't contain any new values to mask: stop
                if(new_mask.sum()==0): break
                # mask elements
                rms[new_mask] = np.ma.masked

                if(order in orders_to_plot):
                    # rms clipped plot -> manuscript
                    ax[1].plot(self.wave[0,order],model,"-",c='r',label=f"LOWESS",zorder=3, linewidth=2)
                    ax[1].plot(self.wave[0,order],threshold_max,"--",c='r',zorder=2)
                    ax[1].plot(self.wave[0,order],threshold_min,"--",c='r',zorder=2)
                    ax[1].plot(self.wave[0,order],rms,'k+',zorder=1)
                    ax[1].legend()
                    plt.show()
                    # ax2.plot(self.wave[0,order],model,"-",c=colors[i],label=f"frac = {frac} lowess model @ iteration {i}",zorder=3, linewidth=2)
                    # ax2.plot(self.wave[0,order],threshold_max,"--",c=colors[i],label=f"Threshold @ iteration {i}",zorder=2)
                    # ax2.plot(self.wave[0,order],threshold_min,"--",c=colors[i],zorder=2)
                    # ax2.plot(self.wave[0,order],rms,'.',mec='k',mfc=colors[i],label=f'Data after sigma clipping @ iteration {i}',zorder=1)
                    # ax2.set_title(f'Sigma clipping with LOWESS, order {order} & frac = {frac}')
                    # ax2.legend()#bbox_to_anchor=(1.05, 1), loc='upper left')

                    # # rms clipped plot
                    # ax[1].plot(self.wave[0,order],model,"-",c=colors[i],label=f"Lowess model @ iteration {i}",zorder=3, linewidth=2)
                    # ax[1].plot(self.wave[0,order],threshold_max,"--",c=colors[i],label=f"Threshold @ iteration {i}",zorder=2)
                    # ax[1].plot(self.wave[0,order],threshold_min,"--",c=colors[i],zorder=2)
                    # ax[1].plot(self.wave[0,order],rms,'.',mec='k',mfc=colors[i],label=f'Data after sigma clipping @ iteration {i}',zorder=1)
                    # ax[1].legend(fontsize=5)

                    # ax2.plot(self.wave[0,order],model,"-",c=colors[i],label=f"frac = {frac} lowess model @ iteration {i}",zorder=3, linewidth=2)
                    # ax2.plot(self.wave[0,order],threshold_max,"--",c=colors[i],label=f"Threshold @ iteration {i}",zorder=2)
                    # ax2.plot(self.wave[0,order],threshold_min,"--",c=colors[i],zorder=2)
                    # ax2.plot(self.wave[0,order],rms,'.',mec='k',mfc=colors[i],label=f'Data after sigma clipping @ iteration {i}',zorder=1)
                    # ax2.set_title(f'Sigma clipping with LOWESS, order {order} & frac = {frac}')
                    # ax2.legend()#bbox_to_anchor=(1.05, 1), loc='upper left')

                # store in class attribute
                self.clipping_mask[:,order] = rms.mask

        if(len(orders_to_plot) > 0) : plt.show()

        # set masked pixels to nan in the clipped spectra
        self.data[self.clipping_mask] = np.nan
        ###################

        if print_info : print('\nSigma clipping on temporal standard deviation done !')

        # Save in history
        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

        else :
            self.history[step_ID] = {'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

    # def apply_pca(self, nb_PC_to_remove, reduce = False, plot_random_clouds = [], plot_eigen_vectors = [], plot_cum_var = False, plot_eigen_vals = False,  nb_eigen_vals_to_plot = None, save = None, print_info = True):
    #     '''
    #     nb_PC_to_remove : can be INT or LIST
    #         -> if INT : remove this nb of PC to all orders
    #         -> if LIST : must contain 1 nb per orders, and will remove the indicated nb of PC to each corresponding orders
    #     '''
    #     # Current step ID
    #     step_ID = 'apply_pca'
    #     # Default behaviour if no save parameter has been set
    #     if(save==None): save = self.save_history
    #     ## Retrieve current step parameters
    #     # retrieve currently defined local variables in current scope : parameters and step_ID
    #     parameters = locals()
    #     # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
    #     parameters.pop('self')
    #     # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
    #     parameters.pop('step_ID')
    #     # Warn if step already found in history
    #     if(step_ID in self.history.keys()):
    #         warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
    #         # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
    #         k = 0
    #         former_step_ID = step_ID
    #         while step_ID in self.history.keys():
    #             k+=1
    #             step_ID = former_step_ID+'_'+str(k)
    #     ## Center (and reduce if asked) data, compute PCA on it and remove the first principal components from data ##
    #     # For asked orders, plot a projection of the data cloud along 2 randomly picked observations before & after PCA
    #     # obsA, obsB = [], []
    #     # for k,order in enumerate(plot_random_clouds):
    #     #     while True:
    #     #         A, B = np.random.randint(0,self.shape[0]-1,2) # pick 2 random observation index
    #     #         if(A!=B): break # prevent the same observation index to be picked twice
    #     #     obsA.append(A)
    #     #     obsB.append(B)
    #     #
    #     #     fig = plt.figure()
    #     #     ax = fig.add_subplot(1,1,1)
    #     #     # plotting obsA Vs obsB before PCA
    #     #     plt.scatter(self.data[obsA[k],order],self.data[obsB[k],order],c=self.wave[0,order])
    #     #     plt.xlabel(f'obs{obsA[k]} normalized flux')
    #     #     plt.ylabel(f'obs{obsB[k]} normalized flux')
    #     #     plt.title(f'Order {order} : obs{obsA[k]} Vs obs{obsB[k]}\n(before PCA)')
    #     #     c=plt.colorbar()
    #     #     c.set_label(f'$\lambda$ [nm]')
    #     #     ax.set_aspect('equal') # important to prevent image deformation due to unequal axes

    #     # Center by removing mean
    #     data_mean = self.data.mean(axis = 0)
    #     self.data -= data_mean[None,...]
    #     # If reduce = True : reduce data by dividing them by their std
    #     if reduce :
    #         # data_std = np.nanstd(self.data,axis = 0)
    #         data_std = self.weighted_std(self.data)
    #         data_std = np.nanstd(data_std,axis = 1)
    #         self.data /= data_std[...,None]

    #     # Compute PCA and remove first Principal Components
    #     eigen_values_array = np.zeros(self.shape[:2])
    #     cumulative_variance = np.zeros(self.shape[:2])
    #     eigen_vectors_array = []
    #     nb_obs = self.shape[0]

    #     if not isinstance(nb_PC_to_remove,list):
    #         nb_PC_to_remove = [nb_PC_to_remove for o in range(self.shape[1])]

    #     for order in range(self.shape[1]):
    #         if print_info : print(f'\rPerforming PCA on order {order+1}/{self.shape[1]}',end='',flush=True)
    #         # build the PCA object using sklearn. the number of component correspond to the number of axis on which project our data
    #         pca = PCA(n_components = nb_obs)
    #         # apply PCA to our data
    #         data = self.data[:,order,:]
    #         # Our data contain NaN, which are not handled by sklearn implementation of PCA.
    #         # To correct this : if a pixel has a NaN value somewhere along the observations, the entire pixel column along the obs is set to 0, thus leading to a 0 variance of the pixel that should be neutral to the PCA
    #         nan_mask = np.any(~np.isfinite(data),axis=0)
    #         data[:,nan_mask] = 0
    #         # Fit the PCA model on our data (thus computing pca axis : eigen vectors and values)
    #         # and apply dimensionality reduction on data (they are now describe in terms of eigen vectors found in the pca (??))
    #         data_projected =  pca.fit_transform(data)
    #         # get all eigen vectors and values computed by the pca after projection
    #         eigen_vect = pca.components_
    #         eigen_vals = pca.singular_values_
    #         eigen_values_array[:,order] = eigen_vals
    #         eigen_vectors_array.append(eigen_vect)
    #         # we then remove the N first eigen vectors (by setting them to 0), project our data on this new PCA base, and get back to the initial space
    #         eigen_vect_corr = np.copy(eigen_vect)
    #         eigen_vect_corr[:nb_PC_to_remove[order],:] = 0
    #         # project back on initial basis without using the first eigen vectors
    #         data_corr = data_projected @ eigen_vect_corr
    #         # also previously values found to be nan and set to 0 before must now be reset to nan
    #         data_corr[:,nan_mask] = np.nan
    #         # store pca treated spectra in a new data cube
    #         self.data[:,order,:] = data_corr
    #         cumulative_variance[:,order] = np.cumsum(pca.explained_variance_ratio_)

    #     # plot eigen vectors for orders in given list
    #     for k,order in enumerate(plot_eigen_vectors):
    #         plot_grid(self.wave[:,order],eigen_vectors_array[order],linewidth=0.2,title=f'Eigen vectors found by PCA for order {order}\nSorted in decreasing associated eigen value',
    #                 Ylabel='Flux',Xlabel=f'$\lambda$ [nm]', show = False)

    #     # Once PCA is done, recenter our data around mean and get back to original std
    #     if reduce : self.data *= data_std[...,None]
    #     self.data += data_mean[None,...]

    #     # Plot clouds
    #     for k,order in enumerate(plot_random_clouds):
    #         fig = plt.figure()
    #         ax = fig.add_subplot(1,1,1)
    #         # plotting obsA Vs obsB before PCA
    #         plt.scatter(self.data[obsA[k],order],self.data[obsB[k],order],c=self.wave[0,order])
    #         plt.xlabel(f'obs{obsA[k]} normalized flux')
    #         plt.ylabel(f'obs{obsB[k]} normalized flux')
    #         plt.title(f'Order {order} : obs{obsA[k]} Vs obs{obsB[k]}\n(After PCA)')
    #         c=plt.colorbar()
    #         c.set_label(f'$\lambda$ [nm]')
    #         ax.set_aspect('equal') # important to prevent image deformation due to unequal axes

    #     # plot eigen values in grid
    #     if plot_eigen_vals:
    #         index = np.array([np.arange(self.shape[0]) for k in range(self.shape[1])])
    #         if nb_eigen_vals_to_plot == None:
    #             plot_grid(index,eigen_values_array.T,title=f'Eigen values used by PCA for each spectra',xlims=[-1,index.max()],
    #                   Ylabel='Eigen value',Xlabel=f'Eigen value n° in decreasing value order',ls='-',marker='.',c='k',mfc='r',mec='r', show = False)
    #         else :
    #             plot_grid(index[:,:nb_eigen_vals_to_plot],eigen_values_array.T[:,:nb_eigen_vals_to_plot],title=f'{nb_eigen_vals_to_plot} first eigen values used by PCA for each spectra',xlims=[-1,index[:,:nb_eigen_vals_to_plot].max()],
    #                   Ylabel='Eigen value',Xlabel=f'Eigen value n° in decreasing value order',ls='-',marker='.',c='k',mfc='r',mec='r', show = False)


    #     # plot cumulative variance in grid
    #     if plot_cum_var:
    #         index = np.array([np.arange(self.shape[0]) for k in range(self.shape[1])])
    #         plot_grid(index,cumulative_variance.T,title=f'Cumulative variance',xlims=[-1,index.max()],
    #                   Ylabel='Cum. Var',Xlabel=f'$\lambda$ [nm]',ls='-',marker='.',c='k',mfc='r',mec='r', show = False)

    #     if(len(plot_random_clouds)>0) or (len(plot_eigen_vectors)>0) or plot_cum_var or plot_eigen_vals: plt.show()
    #     ##############################################################################################################

    #     if print_info : print('\nPCA done !')

    #     # Save in history
    #     if save:
    #         self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
    #         if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

    #     else :
    #         self.history[step_ID] = {'parameters' : parameters}
    #         if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

    # def apply_PCA_svd_old(self, K, use_log = False, show_var = False, save = None, print_info = True, transit_mask="full"):
    #     '''
    #     Replace the PCA step by using the svd method from numpy
    #     Has an option for computing the svd in log space
    #     Use svd to find the first Principal Components (PCs) of the data cloud, reconstruct the signal using only the N first PCs, and remove this reconstructed signal from the original one.
    #     - K : int, the N first PCs to be "removed" from the data
    #     - use_log : bool, default False, whether to compute svd in log space or not
    #     - show_var : bool, default False, whether to show the variance plots or not
    #     '''
    #     # Current step ID
    #     step_ID = 'apply_svd'
    #     # Default behaviour if no save parameter has been set
    #     if(save==None): save = self.save_history
    #     # Retrieve current step parameters

    #     # retrieve currently defined local variables in current scope : parameters and step_ID
    #     parameters = locals()
    #     # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
    #     parameters.pop('self')
    #     # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
    #     parameters.pop('step_ID')
    #     # Warn if step already found in history
    #     if(step_ID in self.history  .keys()):
    #         warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
    #         # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
    #         k = 0
    #         former_step_ID = step_ID
    #         while step_ID in self.history.keys():
    #             k+=1
    #             step_ID = former_step_ID+'_'+str(k)
    #     ## Apply PCA using SVD ##
    #     # center and reduce the data
    #     if use_log:
    #         X = np.log(np.copy(self.data))
    #     else:
    #         X = np.copy(self.data)
    #     if transit_mask=="off":
    #         X = X[self.off_transit_mask]
    #     mean = np.mean(X,axis=2) # spectral mean (si PCA a 1 feature par obs, alors on enleve a chaque obs sa valeur moyenne)
    #     std  = np.std(X,axis=2) # spectral std
    #     X = (X - mean[:,:,None]) / std[:,:,None]
    #     # used to store the reconstructed data from svd : perform a CCF on it to check if no planetary signal is inside it
    #     self.svd_reconstructed_data = np.zeros(self.shape)
    #     self.SVD_V                  = []
    #     self.SVD_components         = [] # used to store the principal components
    #     # Prepare grid of plots to show variance for each orders
    #     if show_var:
    #         n_cols = int(np.sqrt(self.shape[1]))
    #         n_rows = n_cols
    #         if n_rows*n_cols < self.shape[1] : n_rows+=1
    #         fig,ax = plt.subplots(n_rows,n_cols,figsize=(15,15))
    #     # use SVD to decompose the data, order per order :
    #     print(f'Performing PCA using SVD to remove {K} first PCs :')
    #     for order in range(self.shape[1]):
    #         print(f'\r{order+1}/{self.shape[1]}',end='',flush=True)
    #         ## NaN MANAGEMENT : PROBABLY NOT OPTIMAL !
    #         # if a sample is full of NaN in X[:,order,:] : replace it by a null vector : the point will thus be centered on the mean value (=0.) and should (?) have no effect on the PCA
    #         nan_mask = ~np.isfinite(X[:,order,:])
    #         nan_index = np.all(nan_mask,axis=0)
    #         X[:,order,nan_index] = 0
    #         ##
    #         U, S, VT = np.linalg.svd(X[:,order,:].T,full_matrices=False,compute_uv=True)
    #         self.SVD_V.append(np.copy(VT.T))
    #         # we now have U,S & VT defined by : X = U.S.VT with U & VT being unitary matrix (V.VT = I, U.UT = I)
    #         # S is a diagonal rectangular matrix of same shape than data, with diagonal elements being "singular values"
    #         # The K first PCs are in the K first columns of U & K*K upper left corner of S (see https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8, https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca)
    #         X_k = U[:,:K] * S[:K] # the K first PCs (np.linalg.svd already return the singular values sorted in descending order in S)
    #         # The principal axis are the columns of V (or lines of VT) : project back X_k on the initial basis
    #         X_svd = X_k @ VT[:K]
    #         X_svd = X_svd.T
    #         # Now remove the X_svd data (reconstructed using K first PCs) from the original data : hoping this will remove most of the tellurics residuals
    #         data_reconstructed = (X_svd * std[order]) + mean[order]
    #         if use_log: data_reconstructed = np.exp(data_reconstructed)
    #         self.svd_reconstructed_data[:,order,:] = data_reconstructed
    #         # Compute and show the stored variance of each PCs for current order
    #         if show_var:
    #             var      = (S ** 2) / (X.shape[0]-1)
    #             var     /= var.sum() # normalise such that total variance is 1
    #             sub_ax = ax.flatten()[order]
    #             sub_ax.plot(var,'-.')
    #             sub_ax.set_xlabel('PCs N°')
    #             sub_ax.set_ylabel('Variance')
    #             sub_ax.text(0.8,0.85,f'Order n° {order}',transform = sub_ax.transAxes,fontsize = 12)
    #     # remove the reconstructed signal from the initial one
    #     self.data -= self.svd_reconstructed_data
    #     #########################

    #     if print_info : print('\nSVD done !')

    #     # Save in history
    #     if save:
    #         self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
    #         if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

    #     else :
    #         self.history[step_ID] = {'parameters' : parameters}

    #     if show_var:
    #         plt.tight_layout()
    #         plt.show()

    # def apply_PCA_svd(self, K, use_log = False, show_var = False, save = None, print_info = True, transit_mask="full"):
    #     '''
    #     Replace the PCA step by using the svd method from numpy
    #     Has an option for computing the svd in log space
    #     Use svd to find the first Principal Components (PCs) of the data cloud, reconstruct the signal using only the N first PCs, and remove this reconstructed signal from the original one.
    #     - K : int, the N first PCs to be "removed" from the data
    #     - use_log : bool, default False, whether to compute svd in log space or not
    #     - show_var : bool, default False, whether to show the variance plots or not
    #     '''
    #     # Current step ID
    #     step_ID = 'apply_svd'
    #     # Default behaviour if no save parameter has been set
    #     if(save==None): save = self.save_history
    #     # Retrieve current step parameters
    #     # retrieve currently defined local variables in current scope : parameters and step_ID
    #     parameters = locals()
    #     # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
    #     parameters.pop('self')
    #     # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
    #     parameters.pop('step_ID')
    #     # Warn if step already found in history
    #     if(step_ID in self.history  .keys()):
    #         warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
    #         # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
    #         k = 0
    #         former_step_ID = step_ID
    #         while step_ID in self.history.keys():
    #             k+=1
    #             step_ID = former_step_ID+'_'+str(k)

    #     # used to store the reconstructed data from svd : perform a CCF on it to check if no planetary signal is inside it
    #     self.svd_reconstructed_data = np.zeros(self.shape)
    #     self.SVD_V                  = []
    #     self.SVD_components         = [] # used to store the principal components
    #     self.K_list = [] # list of nb of PCs removed per order (used when degrading the model accordingly)

    #     # Prepare grid of plots to show variance for each orders
    #     if show_var:
    #         n_cols = int(np.sqrt(self.shape[1]))
    #         n_rows = n_cols
    #         if n_rows*n_cols < self.shape[1] : n_rows+=1
    #         fig,ax = plt.subplots(n_rows,n_cols,figsize=(30,15))

    #     # use SVD to decompose the data, order per order :
    #     print(f'Performing PCA with SVD, using Kneedle to detect and remove the first PCs :')
    #     for order in range(self.shape[1]):  
    #         print(f'\r{order+1}/{self.shape[1]}',end='',flush=True)
    #         # convert to log space if specified
    #         if use_log:
    #             data = np.log(np.copy(self.data[:,order]))
    #         else:
    #             data = np.copy(self.data[:,order])
    #         # transform data (center/reduce)
    #         X,transfo_params = transform_data(data,self)
    #         # fit PCA
    #         U,S,VT,var = PCA(X)
    #         # store the V array for applying the same PCA degradation on the model later on
    #         self.SVD_V.append(np.copy(VT.T))
    #         self.K_list.append(K)
    #         # reconstruct using K first PCs
    #         X_k = U[:,:K] * S[:K]
    #         # the principal axis are the columns of V (or lines of VT) : project back X_k on the initial basis
    #         X_svd = np.dot(X_k,VT[:K])
    #         # revert the center/reduce/invalid mask transformation: this has the same shape as a data_set.data[:,order] matrix (axis=0 are observations)
    #         X_recon = invert_transform(X_svd,*transfo_params)
    #         if use_log: X_recon = np.exp(X_recon)
    #         # Now remove the X_svd data (reconstructed using K first PCs) from the original data : hoping this will remove most of the tellurics residuals
    #         self.svd_reconstructed_data[:,order,:] = X_recon
    #         # Compute and show the stored variance of each PCs for current order
    #         if show_var:
    #             var      = (S ** 2) / (X.shape[0]-1)
    #             var     /= var.sum() # normalise such that total variance is 1
    #             sub_ax = ax.flatten()[order]
    #             sub_ax.plot(var,'k+')
    #             sub_ax.plot(var[:K],'r+') # show selected points
    #             sub_ax.vlines(elbow_point,var.min(),var.max(),color='b',ls='--',lw=0.5) # show elbow position
    #             sub_ax.set_xlabel('PCs N°')
    #             sub_ax.set_ylabel('Variance')
    #             sub_ax.text(0.8,0.85,f'Order n° {order}',transform = sub_ax.transAxes,fontsize = 12)
    #     # remove the reconstructed signal from the initial one
    #     self.data -= self.svd_reconstructed_data

    #     # also let's show how many PCs were removed per order
    #     if show_var:
    #         plt.figure()
    #         plt.plot(self.K_list,'k+')
    #         plt.xlabel('Order n°')
    #         plt.ylabel('Nb PCs removed')
    #     #########################

    def PCA(self, Kneedle = True, K = None, use_log = False, show_var = False, save = None, print_info = True):
        '''
        Replace the PCA step by using the svd method from numpy
        Has an option for computing the svd in log space
        Use svd to find the first Principal Components (PCs) of the data cloud, reconstruct the signal using only the N first PCs, and remove this reconstructed signal from the original one.
        - K : int, the N first PCs to be "removed" from the data
        - Kneedle: bool, if true let Kneedle determine the nb of PCs to remove per order
        - use_log : bool, default False, whether to compute svd in log space or not
        - show_var : bool, default False, whether to show the variance plots or not
        '''
        # Current step ID
        step_ID = 'PCA'
        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        # Retrieve current step parameters

        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')
        # Warn if step already found in history
        if(step_ID in self.history  .keys()):
            warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            k = 0
            former_step_ID = step_ID
            while step_ID in self.history.keys():
                k+=1
                step_ID = former_step_ID+'_'+str(k)

        # used to store the reconstructed data from svd : perform a CCF on it to check if no planetary signal is inside it
        self.svd_reconstructed_data = np.zeros(self.shape)
        self.SVD_V                  = [] # used to store the eigenvectors
        self.SVD_UUT                = [] # used to store the matrix corresponding to the first terms of eq 7 in Gibson 2022 (U times its pseudo inverse, sometimes with a term accounting for uncertainty), later used to degrade the model accordingly
        self.K_list = [] # list of nb of PCs removed per order (used when degrading the model accordingly)

        if not Kneedle and K is None:
            raise NameError('You must either activate the auto-determination of removed PCs by setting Kneedle to "True" or manually set the number of PCs to remove with K = your_number')

        # Prepare grid of plots to show variance for each orders
        if show_var:
            n_cols = int(np.sqrt(self.shape[1]))
            n_rows = n_cols
            if n_rows*n_cols < self.shape[1] : n_rows+=1
            fig,ax = plt.subplots(n_rows,n_cols,figsize=(30,15))

        # use SVD to decompose the data, order per order :
        print(f'Performing PCA with SVD, using Kneedle to detect and remove the first PCs :')
        for order in range(self.shape[1]):  
            print(f'\r{order+1}/{self.shape[1]}',end='',flush=True)
            # convert to log space if specified
            if use_log:
                data = np.log(np.copy(self.data[:,order]))
            else:
                data = np.copy(self.data[:,order])
            # transform data (center/reduce)
            X,transfo_params = transform_data(data,self)
            # fit PCA
            U,S,VT,var = PCA(X)
            # store the V array for applying the same PCA degradation on the model later on
            self.SVD_V.append(np.copy(VT.T))
            
            if Kneedle:
                # find the elbow position in variance plot: S -> sensitivity parameter, explained in section IV of the paper. The code has 2 behavior: online/offline, online -> previously detected knee/elbow are updated when new points are added (?), this is by default and in the paper a value of S = 1 is advised in this case (and S=0 in the offline case)
                x = np.arange(len(var))
                kneedle = KneeLocator(x, var, S=1.0, curve="convex", direction="decreasing") # find elbow position (convex and decreasing curve)
                elbow_point = kneedle.elbow # get the position of the elbow in x-axis
                K = int(elbow_point) # select all PCs located before the elbow point

            self.K_list.append(K)
            # reconstruct using K first PCs
            U_K = U[:,:K]
            X_k = U_K * S[:K]

            # the principal axis are the columns of V (or lines of VT) : project back X_k on the initial basis
            X_svd = np.dot(X_k,VT[:K])
            # revert the center/reduce/invalid mask transformation: this has the same shape as a data_set.data[:,order] matrix (axis=0 are observations)
            X_recon = invert_transform(X_svd,*transfo_params)
            if use_log: X_recon = np.exp(X_recon)
            # Now remove the X_svd data (reconstructed using K first PCs) from the original data : hoping this will remove most of the tellurics residuals
            self.svd_reconstructed_data[:,order,:] = X_recon

            # compute the prefactor to model "M" in eq 7 of Gibson 2022, can then be used to degrade the synthetic
            UUT = np.dot(U_K,np.linalg.pinv(U_K))  # U times its pseudo-inverse. THis is where we could add the 1/sigma uncertainty weight in Gibson's equation
            # store for latter use
            self.SVD_UUT.append(UUT)

            # Compute and show the stored variance of each PCs for current order
            if show_var:
                var      = (S ** 2) / (X.shape[0]-1)
                var     /= var.sum() # normalise such that total variance is 1
                sub_ax = ax.flatten()[order]
                sub_ax.plot(var,'k+')
                sub_ax.plot(var[:K],'r+') # show selected points
                if Kneedle: sub_ax.vlines(elbow_point,var.min(),var.max(),color='b',ls='--',lw=0.5) # show elbow position
                sub_ax.set_xlabel('PCs N°')
                sub_ax.set_ylabel('Variance')
                sub_ax.text(0.8,0.85,f'Order n° {order}',transform = sub_ax.transAxes,fontsize = 12)
        # remove the reconstructed signal from the initial one
        self.data -= self.svd_reconstructed_data

        # recenter around 1
        self.data += 1

        # also let's show how many PCs were removed per order
        if show_var:
            plt.figure()
            plt.plot(self.K_list,'k+')
            plt.xlabel('Order n°')
            plt.ylabel('Nb PCs removed')
        #########################

        if print_info : print('\nSVD done !')

        # Save in history
        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

        else :
            self.history[step_ID] = {'parameters' : parameters}

        if show_var:
            plt.tight_layout()
            plt.show()

    def SysREM_Kneedle(self, use_log = False, show_var = False, Nb_threads = 6, sigma_weights = True, save = None, print_info = True):
        '''
        Apply the SysREM algorithm (Tamuz et al. 2005) to remove residual systematics from the data set.
        For each spectral order X, the algorithm iteratively find the best 1D vector u & w such that X can be described as their outer product,
        which is equivalent of finding the most dominant linear trend along time for each spectral bin. This trend is then removed from the data
        before jumping to next iteration.

        SysREM can be seen as an iterative equivalent to PCA which takes into account the uncertainty in each spectral bin. This result in a longer
        but potentially more efficient removal of stellar and telluric residuals.

        Just like PCA, this SysREM implementation relies on Kneedle to determine the number of systematic removal iteration (equivalent to the number of Principal Component removed by PCA).
        A first run compute the variance captured for each iteration of SysREM (one per observation), and the nb of iteration is found by identifying the elbow in the variance distribution with Kneedle.

        Arguments are the same as for PCA:
        - use_log (bool): perform SysREM in log-space, maybe better since this the components should be linear in this space
        - show_var (bool): show the variance plot for each order along with the Kneedle marker used to determine the nb of iterations/nb of systematic removed
        With an additional argument for parallelization:
        - Nb_threads: 0 for no parallelization, or a number to set how much threads are used. We found 6 to be the optimum value for SPIRou data on laptop
        - sigma_weights (bool) : True if using the 1/sigma**2 weight (default) where sigma is the uncertainty computed as the SNR-weighted std of each spectral bin along time
        '''
        # Current step ID
        step_ID = 'SysREM_Kneedle'
        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        # Retrieve current step parameters
        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')
        # Warn if step already found in history
        if(step_ID in self.history.keys()):
            warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            k = 0
            former_step_ID = step_ID
            while step_ID in self.history.keys():
                k+=1
                step_ID = former_step_ID+'_'+str(k)   

        ## DO STUFF HERE ##
        # reset thread counter
        global thread_counter, fail_counter
        thread_counter = 0
        fail_counter = 0
        # get shapes
        N_obs, N_order,  N_wave = self.shape
        # convert to log space if specified
        if use_log:
            data = np.log(np.copy(self.data))
        else:
            data = np.copy(self.data)

        data_std = self.weighted_std(self.data)

        # define worker fucntion for threading
        def sysrem_worker(U_ARRAY,VAR_ARRAY,X_SYSREM,order_list,Nb_iter_list):
            '''
            Apply sysrem on the given order using Threading for parallelisation
            Fill the provided matrix with sysrem result, such that for a given order N:
            - the Nth order of U_ARRAY is the u vector of order N
            - the Nth column of VAR_ARRAY is the var vector of order N
            - the Nth order of X_SYSREM with the data after removal of systematics
            - order_list: list of order to be treated by thread
            - Nb_iter_list: contain the nb of iteration to apply for the corresponding order
            '''
            global thread_counter, lock, fail_counter
            for order in order_list:
                # set data
                X = np.copy(data[:,order]).T # (wavelength,obs)
                # center at 0. each spectral bin
                X -= np.nanmean(X)

                # define the uncertainty for each wavelength channel: give it the same shape as data for easy matrix multiplication
                if sigma_weights: 
                    # # Using SNR-weighted std along time for each spectral bin
                    # sigma = data_std[order][:,None] * np.ones_like(X)
                    # # normalize weight to avoid computational instabilities
                    # sigma += 1 # avoid null values
                    # sigma /= np.nanmax(sigma) # normalize by max value

                    # computed unweighted std along time axis for each spectral bin, then add a time dependence with SNR² weight
                    # Using SNR**2 for each spectral bin
                    weights = self.SNR**2 / np.sum(self.SNR**2)
                    sigma = np.outer(np.nanstd(X, axis=1),weights) # same shape as data: std along time weighted by SNR², so one error per spectral bin & per time
                    sigma = np.ma.masked_invalid(sigma) # avoid NaN
                    # # normalize weight to avoid computational instabilities
                    sigma += 1 # avoid null values
                    sigma /= np.ma.max(sigma) # normalize by max value

                else: sigma = np.ones_like(X) # 1 error per spectral bin, constant along time, for comparison with PCA

                # compute sysrem, iterate for each obs
                X_final_sysrem = np.copy(X)
                var_sysrem = []
                u_list = []

                # get nb of iterations
                Nb_iter = Nb_iter_list[order]

                # Skip if no iteration to perform
                if Nb_iter == 0: 
                    with lock: thread_counter += 1
                    print(f'\r{thread_counter}/{N_order}\tNb fails: {fail_counter}',end='',flush=True)
                    continue

                fails = 0 # fails of this run
                for k in range(Nb_iter):
                    # print(f'\r{k}/{Nb_iter}',end='',flush=True)
                    # get the best u & w for this order & iteration
                    u, w, var, X_recon, Converged = sysrem(X_final_sysrem,sigma,tol=1e-3)
                    var_sysrem.append(var)
                    u_list.append(u)
                    # remove systematic
                    X_final_sysrem -= X_recon
                    if not Converged: fails += 1

                # fill memory with lock to ensure safe access
                with lock: # only fill first columns, rest is 0.
                    U_ARRAY[:Nb_iter,order] = np.array(u_list) # same shape as X: (obs,order,wavelength). U_ARRAY[:,order,:].T will provide the original U array obtained from sysrem
                    VAR_ARRAY[order,:Nb_iter] = np.array(var_sysrem) # variance for each sysrem component for this order
                    X_SYSREM[:,order] = X_final_sysrem.T
                    thread_counter += 1 # update counter
                    fail_counter += fails
                print(f'\r{thread_counter}/{N_order}\tNb fails: {fail_counter}',end='',flush=True)

        ### FIRST RUN: ITERATE FOR ALL OBS AND MEASURE VARIANCE ###
        print(f'First round: running {N_obs} iterations of SysREM per order to measure variance and find optimum nb of iterations with Kneedle')
        print()

        # Array to store results: we only need somewhere to store the variance for first run
        _ = np.nan * np.zeros(self.shape) # will store final data after SysREM removal: unused for first round
        __ = np.nan * np.zeros(self.shape) # will store the vector found by SysREM: unused for first round
        VAR_ARRAY = np.nan * np.zeros((N_order,N_obs))
        iteration_list = [N_obs for el in range(N_order)] # perform N_obs iteration for each order

        # if no parallelisation:
        if Nb_threads == 0:
            # fill in the variance array for each order and iteration
            for order in range(N_order):
                sysrem_worker(_,VAR_ARRAY,__,[order],iteration_list)
        else:
            # Compute sysrem order-wise with paralellisation
            threads = []          
            # Divide the orders array in equal parts (one per thread)
            index_sub_arrays = np.array_split(np.arange(N_order), Nb_threads)
            for thread_idx in range(Nb_threads):
                order_list = index_sub_arrays[thread_idx] # list of order the thread will work with
                thread = threading.Thread(target=sysrem_worker, args=(_,VAR_ARRAY,__,order_list,iteration_list))
                threads.append(thread)
                thread.start()
            # Wait for all threads to finish
            for thread in threads:
                thread.join()
        
        # Find the elbow position in variance plot: S -> sensitivity parameter, explained in section IV of the paper. The code has 2 behavior: online/offline, online -> previously detected knee/elbow are updated when new points are added (?), this is by default and in the paper a value of S = 1 is advised in this case (and S=0 in the offline case)
        K_list = [] # list containing the optimum nb of SysREM iteration for each order
        print()
        print('Running Kneedle to find optimum nb of SysREM iterations per order: an arbitrary maximum of 5 removed components has been set to prevent Kneedle from removing to much components.')
        if show_var:
            n_cols = int(np.sqrt(self.shape[1]))
            n_rows = n_cols
            if n_rows*n_cols < self.shape[1] : n_rows+=1
            fig,ax = plt.subplots(n_rows,n_cols,figsize=(16,16))
        for order in range(N_order):
            var = VAR_ARRAY[order]
            x = np.arange(len(var))
            kneedle = KneeLocator(x, var, S=1.0, curve="convex", direction="decreasing") # find elbow position (convex and decreasing curve)
            elbow_point = kneedle.elbow # get the position of the elbow in x-axis
            K = min(int(elbow_point),5) # select all components located before the elbow point with an upper threshold at 5 components
            K_list.append(K)
            # Also plot variance if prompted
            if show_var:
                var     /= var.sum() # normalise such that total variance is 1
                sub_ax = ax.flatten()[order]
                sub_ax.plot(var,'k+')
                sub_ax.plot(var[:K],'r+') # show selected points
                sub_ax.vlines(elbow_point,var.min(),var.max(),color='b',ls='--',lw=0.5) # show elbow position
                sub_ax.set_xlabel('SysREM iteration N°')
                sub_ax.set_ylabel('Variance')
                sub_ax.text(0.8,0.85,f'Order n° {order}',transform = sub_ax.transAxes,fontsize = 12)

        ### SECOND RUN: APPLY SysREM USING THE OPTIMUM NB OF ITERATIONS FROM KNEEDLE ###
        thread_counter = 0
        fail_counter = 0
        # Array to store results: we only need somewhere to store the variance for first run
        U_ARRAY = np.nan * np.zeros(self.shape) # will store final data after SysREM removal
        X_SYSREM = np.nan * np.zeros(self.shape) # will store the data after applying SysREM
        _ = np.nan * np.zeros((N_order,N_obs)) # variance: unused for second round
        iteration_list = K_list

        # if no parallelisation:
        if Nb_threads == 0:
            # fill in the variance array for each order and iteration
            for order in range(N_order):
                sysrem_worker(U_ARRAY,_,X_SYSREM,[order],iteration_list)
        else:
            # Compute sysrem order-wise with paralellisation
            threads = []          
            # Divide the orders array in equal parts (one per thread)
            index_sub_arrays = np.array_split(np.arange(N_order), Nb_threads)
            for thread_idx in range(Nb_threads):
                order_list = index_sub_arrays[thread_idx] # list of order the thread will work with
                thread = threading.Thread(target=sysrem_worker, args=(U_ARRAY,_,X_SYSREM,order_list,iteration_list))
                threads.append(thread)
                thread.start()
            # Wait for all threads to finish
            for thread in threads:
                thread.join()

        # compute the prefactor to model "M" in eq 7 of Gibson 2022, can then be used to degrade the synthetic
        self.SVD_UUT = [] # used to store the matrix corresponding to the first terms of eq 7 in Gibson 2022 (U times its pseudo inverse, sometimes with a term accounting for uncertainty), later used to degrade the model accordingly
        for order in range(N_order):
            K = K_list[order]
            # grab U matrix corresponding to order, only grabing non-zeros columns
            U_K = U_ARRAY[:K,order].T
            # compute pre-factor for Gibson correction
            UUT = np.dot(U_K,np.linalg.pinv(U_K))  # U times its pseudo-inverse. THis is where we could add the 1/sigma uncertainty weight in Gibson's equation
            # store for latter use
            self.SVD_UUT.append(UUT)
        
        # Update data after optimal SysREM application
        if use_log: X_SYSREM = np.exp(X_SYSREM)
        
        self.data = X_SYSREM

        if print_info : 
            print()
            print('SysREM FUNCTION done !')
    
        # Save in history
        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")
        else :
            self.history[step_ID] = {'parameters' : parameters}

    # def testing_PCA(self, use_log = False, show_var = [], save = None, print_info = True, transit_mask="full"):
    #     '''
    #     Replace the PCA step by using the svd method from numpy
    #     Has an option for computing the svd in log space
    #     Use svd to find the first Principal Components (PCs) of the data cloud, reconstruct the signal using only the N first PCs, and remove this reconstructed signal from the original one.
    #     - This time we determine the nb of K component to remove per order based on the evolution of variance with respect to white noise
    #     - use_log : bool, default False, whether to compute svd in log space or not
    #     - show_var : bool, default False, whether to show the variance plots or not
    #     '''
    #     # Current step ID
    #     step_ID = 'apply_svd_auto_K'
    #     # Default behaviour if no save parameter has been set
    #     if(save==None): save = self.save_history
    #     # Retrieve current step parameters

    #     # retrieve currently defined local variables in current scope : parameters and step_ID
    #     parameters = locals()
    #     # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
    #     parameters.pop('self')
    #     # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
    #     parameters.pop('step_ID')
    #     # Warn if step already found in history
    #     if(step_ID in self.history  .keys()):
    #         warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
    #         # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
    #         k = 0
    #         former_step_ID = step_ID
    #         while step_ID in self.history.keys():
    #             k+=1
    #             step_ID = former_step_ID+'_'+str(k)
        
    #     ## Apply PCA using SVD ##
    #     # copy data
    #     if use_log:
    #         X = np.log(np.copy(self.data))
    #     else:
    #         X = np.copy(self.data)
    #     if transit_mask=="off":
    #         X = X[self.off_transit_mask]

    #     # remove NaN
    #     nan_mask = ~np.isfinite(X)
    #     X[nan_mask] = 0.

    #     # setup white noise matrix
    #     X_noise = np.random.normal(0,1,X.shape)

    #     # used to store the reconstructed data from svd : perform a CCF on it to check if no planetary signal is inside it
    #     self.svd_reconstructed_data = np.zeros(self.shape)
    #     self.SVD_V                  = []
    #     self.SVD_components         = [] # used to store the principal components
        
    #     # use SVD to decompose the data, order per order :
    #     print(f'Performing PCA using SVD and remove first PCs based on distance from white noise variance trend:')
    #     for order in range(self.shape[1]):
            
    #         X_order = X[:,order,:].T # transpose: 1 column per feature (i.e. observation)

    #         # do pca
    #         pca = PCA()
    #         pca.fit(X_order)
           
    #         # save the U S V matrix
    #         U = pca.transform(X_order)
    #         S = pca.singular_values_
    #         V = pca.components_

    #         # remove the first components
    #         X_transformed = pca.fit_transform(X_order)
    #         X_transformed[:, :2] = 0
    #         X_reconstructed = pca.inverse_transform(X_transformed)

    #         var = pca.explained_variance_ratio_

    #         # we now have U,S & VT defined by : X = U.S.VT with U & VT being unitary matrix (V.VT = I, U.UT = I)
    #         # S is a diagonal rectangular matrix of same shape than data, with diagonal elements being "singular values"

    #         # let's find the nb of K to remove with respect to a white noise
    #         U_n, S_n, VT_n = np.linalg.svd(X_noise[:,order,:].T,full_matrices=False,compute_uv=True)
    #         var_noise = (S_n**2)/(X.shape[0]-1)
    #         var_noise /= var_noise.sum()
    #         # fit var_noise with a linear fit
    #         slope, intercept = np.polyfit(np.arange(var.size), var_noise, 1)
    #         var_noise_fit = slope*np.arange(var.size)+intercept

    #         # Compute and show the stored variance of each PCs for current order
    #         if order in show_var:
    #             # sub_ax = ax.flatten()[order]
    #             fig, sub_ax = plt.subplots(1,1,figsize=(15,5))
    #             sub_ax.plot(var,'k',marker='.',ls='-',label='Data')
    #             # sub_ax.plot(var[removal_mask],marker='.',ls='',color='grey')
    #             # sub_ax.plot(var[removed_components],'grey',marker='.',ls='',label='Removed')
    #             # sub_ax.plot(y_fit,'r',marker='',ls='-',label='Linear fit')
    #             sub_ax.plot(var_noise,color='r',marker='.',ls='-',label='White noise')
    #             # sub_ax.plot(var_noise_fit,'r-')
    #             # sub_ax.plot(thresh,'r--',label='Rejection threshold')
    #             sub_ax.set_xlabel('Principal component N°')
    #             sub_ax.set_ylabel(f'Normalized variance')
    #             # sub_ax.text(0.8,0.85,f'Order n° {order}',transform = sub_ax.transAxes,fontsize = 12)
    #             sub_ax.legend()
    #             sub_ax.set_title(f'Spectral order n°{order}')

    #     # remove the reconstructed signal from the initial one
    #     # self.data -= self.svd_reconstructed_data
    #     #########################

    #     if print_info : print('\nSVD done !')

    #     # Save in history
    #     if save:
    #         self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
    #         if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

    #     else :
    #         self.history[step_ID] = {'parameters' : parameters}

    #     if show_var:
    #         plt.tight_layout()
    #         plt.show()

    # def apply_PCA_svd_auto_K(self, use_log = False, show_var = [], save = None, print_info = True, transit_mask="full"):
    #     '''
    #     Replace the PCA step by using the svd method from numpy
    #     Has an option for computing the svd in log space
    #     Use svd to find the first Principal Components (PCs) of the data cloud, reconstruct the signal using only the N first PCs, and remove this reconstructed signal from the original one.
    #     - This time we determine the nb of K component to remove per order based on the evolution of variance with respect to white noise
    #     - use_log : bool, default False, whether to compute svd in log space or not
    #     - show_var : bool, default False, whether to show the variance plots or not
    #     '''
    #     # Current step ID
    #     step_ID = 'apply_svd_auto_K'
    #     # Default behaviour if no save parameter has been set
    #     if(save==None): save = self.save_history
    #     # Retrieve current step parameters

    #     # retrieve currently defined local variables in current scope : parameters and step_ID
    #     parameters = locals()
    #     # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
    #     parameters.pop('self')
    #     # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
    #     parameters.pop('step_ID')
    #     # Warn if step already found in history
    #     if(step_ID in self.history  .keys()):
    #         warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
    #         # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
    #         k = 0
    #         former_step_ID = step_ID
    #         while step_ID in self.history.keys():
    #             k+=1
    #             step_ID = former_step_ID+'_'+str(k)
    #     ## Apply PCA using SVD ##
    #     # center and reduce the data
    #     if use_log:
    #         X = np.log(np.copy(self.data))
    #     else:
    #         X = np.copy(self.data)
    #     if transit_mask=="off":
    #         X = X[self.off_transit_mask]

    #     # # we must center data by substracting the mean of each feature: here features are the observations (1 PC per obs), so we substract the spectral mean to each obs
    #     # # X[~np.isfinite(X)] = 0 # remove NaN to avoid errors
    #     # mean = np.nanmean(X,axis=2)
    #     # std = np.nanstd(X,axis=2)
    #     nan_mask = ~np.isfinite(X)
    #     X[nan_mask] = 0.

    #     # print(mean.shape)
    #     # print(std.shape)
    #     # print(X.shape)

    #     # X = (X - mean[...,None]) / std[...,None]

    #     # center & reduce data with sklearn
    #     scaler = StandardScaler()
        
    #     # mean = np.nanmean(X,axis=2) # spectral mean (si PCA a 1 feature par obs, alors on enleve a chaque obs sa valeur moyenne)
    #     # std  = np.nanstd(X,axis=2) # spectral std
    #     # X = (X - mean[:,:,None]) / std[:,:,None]
    #     # X[np.isnan(X)] = 0
    #     # print(X)
    #     # print(mean)
    #     # print(std)
    #     # print(X.shape)
    #     # since we have center and reduce data, we will compare them with normal noise (std=1, mean=0)
    #     X_noise = np.random.normal(0,1,X.shape)

    #     # used to store the reconstructed data from svd : perform a CCF on it to check if no planetary signal is inside it
    #     self.svd_reconstructed_data = np.zeros(self.shape)
    #     self.SVD_V                  = []
    #     self.SVD_components         = [] # used to store the principal components
    #     # Prepare grid of plots to show variance for each orders
    #     # if show_var:
    #     #     n_cols = int(np.sqrt(self.shape[1]))
    #     #     n_rows = n_cols
    #     #     if n_rows*n_cols < self.shape[1] : n_rows+=1
    #     #     fig,ax = plt.subplots(n_rows,n_cols,figsize=(15,15))
    #     # use SVD to decompose the data, order per order :
    #     print(f'Performing PCA using SVD and remove first PCs based on distance from white noise variance trend:')
    #     for order in range(self.shape[1]):
    #         # X_scaled = scaler.fit_transform(X[:,order,:].T) # take the transpose so that columns are observations (features) and rows are spectra (data)
    #         X_scaled = np.copy(X[:,order,:].T)
    #         print(X_scaled.shape)
    #         print(np.nanmean(X_scaled,axis=-1),np.nanstd(X_scaled,axis=-1))

    #         # print(f'\r{order+1}/{self.shape[1]}',end='',flush=True)
    #         ## NaN MANAGEMENT : PROBABLY NOT OPTIMAL !
    #         # if a sample is full of NaN in X[:,order,:] : replace it by a null vector : the point will thus be centered on the mean value (=0.) and should (?) have no effect on the PCA
    #         # nan_mask = ~np.isfinite(X_scaled)
    #         # nan_index = np.all(nan_mask,axis=0)
    #         # X_scaled[:,nan_index] = 0
    #         ##
    #         U, S, VT = np.linalg.svd(X_scaled,full_matrices=False,compute_uv=True)
    #         self.SVD_V.append(np.copy(VT.T))
    #         var      = (S ** 2) / (X.shape[0]-1) # variance per Principal Component
    #         var     /= var.sum() # normalise such that total variance is 1
    #         # # mask component with null variance
    #         # null_mask = var>1e-7
    #         # var = var[null_mask]

    #         # we now have U,S & VT defined by : X = U.S.VT with U & VT being unitary matrix (V.VT = I, U.UT = I)
    #         # S is a diagonal rectangular matrix of same shape than data, with diagonal elements being "singular values"

    #         # let's find the nb of K to remove with respect to a white noise
    #         U_n, S_n, VT_n = np.linalg.svd(X_noise[:,order,:].T,full_matrices=False,compute_uv=True)
    #         var_noise = (S_n**2)/(X.shape[0]-1)
    #         var_noise /= var_noise.sum()
    #         # fit var_noise with a linear fit
    #         slope, intercept = np.polyfit(np.arange(var.size), var_noise, 1)
    #         var_noise_fit = slope*np.arange(var.size)+intercept


    #         # var_noise = var_noise[null_mask]

    #         # # threshold to remove a component: a factor time a linear interpolation of the white noise tendency
    #         # slope, intercept = np.polyfit(np.arange(var.size), var_noise, 1)
    #         # var_noise_fit = slope*np.arange(var.size)+intercept
    #         # var_thresh = 1*np.std(var_noise_fit-var) # set threshold at 3 std around white noise

    #         # removal_mask = (var-var_thresh-var_noise_fit)>0 # remove component above the threshold
    #         # K = np.sum(removal_mask)
    #         # print(f'\nOrder: {order}',K,' removed')

    #         # # Or how about we iteratively do a linear fit on the variance from data, rejecting value outside +-3 sigma range and stopping once convergence?
    #         # # the variance doesn't follow exactly a white noise trend but we clearly see a change from non linear to linear trend...
    #         # x = np.arange(X.shape[0])
    #         # y = np.copy(var)
    #         # removed_components = []
    #         # # We do a first linear fit, then we loop over each point and remove it if the linear fit on the remaining points is better (from the std of residuals)
    #         # slope, intercept = np.polyfit(x, y, 1)
    #         # y_fit = slope * x + intercept
    #         # std_res = np.std(y - y_fit)
    #         # for k in x:
    #         #     # new fit without this point
    #         #     x_new = x[x!=x[k]]
    #         #     y_new = y[y!=y[k]]
    #         #     slope, intercept = np.polyfit(x_new, y_new, 1)
    #         #     y_fit_new = slope * x_new + intercept
    #         #     std_res_new = np.std(y_new - y_fit_new)
    #         #     print(std_res-std_res_new)
    #         #     # if the difference in residual std btw both fit is higher than n sigma, then remove the point and adopt new fit
    #         #     if (std_res - std_res_new)/std_res>0.2: # delta std higher than 3 btw two fits
    #         #         removed_components.append(k)
    #         #         x = x_new
    #         #         y = y_new
    #         #         std_res = std_res_new
 
    #         # # try to find the linear tendency in the plot with a Huber loss (https://stackoverflow.com/questions/61143998/numpy-best-fit-line-with-outliers)
    #         # x = np.arange(X.shape[0])
    #         # y = np.copy(var)
    #         # # Reshape x for sklearn (as it expects a 2D array)
    #         # x_reshaped = x.reshape(-1, 1)
    #         # # Initialize the Huber Regressor
    #         # huber = HuberRegressor()
    #         # # Fit the model
    #         # huber.fit(x_reshaped, y)
    #         # # Predicted values using the Huber Regressor
    #         # y_fit = huber.predict(x_reshaped)
    #         # # define a threshold: fit + std of white noise dispersion ?
    #         # thresh = y_fit + 3*np.std(var_noise-var_noise_fit)
    #         # # reject values
    #         # removed_components = np.where(var>thresh)

    #         # # Iteratively do a linear fit on data with Huber loss (more robust agaisnt outliers)
    #         # # we have a distribution in sampled variance, we estimate the standard error on variance (SE = var/(2(N-1)), since SE(sigma) = sigma/sqrt(2(N-1)) )
    #         # # removing any value farther than 3*fit/N and fitting again on the remaining value until convergence
    #         # max_iter = 5
    #         # x = np.arange(X.shape[0])
    #         # y = np.copy(var)
    #         # # Reshape x for sklearn (as it expects a 2D array)
    #         # x_reshaped = x.reshape(-1, 1)
    #         # # Initialize the Huber Regressor
    #         # huber = HuberRegressor()
    #         # mask = np.ones_like(y) # mask of valid data
    #         # for iter in range(max_iter):
    #         #     # Fit the model
    #         #     huber.fit(x_reshaped[mask], y[mask])
    #         #     # Predicted values using the Huber Regressor
    #         #     y_fit = huber.predict(x_reshaped[mask])
    #         #     # define threshold
    #         #     thresh = 3 * y_fit/(2*(y_fit.size-2)) # -2 since linear fit has 2 free params
    #         #     # stop if converge
    #         #     if not np.any(y[mask]>thresh): break
    #         #     else: mask = y > thresh # else, update mask
    #         # removed_components = np.where(mask)




    #         K = 2

    #         # # The K first PCs are in the K first columns of U & K*K upper left corner of S (see https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8, https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca)
    #         # X_k = U[:,:K] * S[:K] # the K first PCs (np.linalg.svd already return the singular values sorted in descending order in S)
    #         # # The principal axis are the columns of V (or lines of VT) : project back X_k on the initial basis
    #         # X_svd = X_k @ VT[:K]
    #         # X_svd = X_svd.T
    #         # # Now remove the X_svd data (reconstructed using K first PCs) from the original data : hoping this will remove most of the tellurics residuals
    #         ## data_reconstructed = (X_svd * std[order]) + mean[order]
    #         # data_reconstructed = scaler.inverse_transform(X_svd)
    #         # data_reconstructed[nan_mask] = np.nan
    #         # if use_log: data_reconstructed = np.exp(data_reconstructed)
    #         # self.svd_reconstructed_data[:,order,:] = data_reconstructed

    #         # Compute and show the stored variance of each PCs for current order
    #         if order in show_var:
    #             # sub_ax = ax.flatten()[order]
    #             fig, sub_ax = plt.subplots(1,1,figsize=(15,5))
    #             sub_ax.plot(var,'k',marker='.',ls='-',label='Data')
    #             # sub_ax.plot(var[removal_mask],marker='.',ls='',color='grey')
    #             # sub_ax.plot(var[removed_components],'grey',marker='.',ls='',label='Removed')
    #             # sub_ax.plot(y_fit,'r',marker='',ls='-',label='Linear fit')
    #             sub_ax.plot(var_noise,color='r',marker='.',ls='-',label='White noise')
    #             # sub_ax.plot(var_noise_fit,'r-')
    #             # sub_ax.plot(thresh,'r--',label='Rejection threshold')
    #             sub_ax.set_xlabel('Principal component N°')
    #             sub_ax.set_ylabel(f'Normalized variance')
    #             # sub_ax.text(0.8,0.85,f'Order n° {order}',transform = sub_ax.transAxes,fontsize = 12)
    #             sub_ax.legend()
    #             sub_ax.set_title(f'Spectral order n°{order}')

    #     # remove the reconstructed signal from the initial one
    #     # self.data -= self.svd_reconstructed_data
    #     #########################

    #     if print_info : print('\nSVD done !')

    #     # Save in history
    #     if save:
    #         self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
    #         if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

    #     else :
    #         self.history[step_ID] = {'parameters' : parameters}

    #     if show_var:
    #         plt.tight_layout()
    #         plt.show()

    def div_by_lowess(self, frac, save = None, print_info = True):
        # Current step ID
        step_ID = 'div_by_lowess'
        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        ## Retrieve current step parameters
        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')
        # Warn if step already found in history
        if(step_ID in self.history.keys()):
            warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            k = 0
            former_step_ID = step_ID
            while step_ID in self.history.keys():
                k+=1
                step_ID = former_step_ID+'_'+str(k)
        lowess_line = np.zeros(self.shape)

        ## div by lowess ##
        for obs in range(self.shape[0]):
            for order in range(self.shape[1]):
                if print_info: print(f'\r{obs*self.shape[1]+order+1}/{self.shape[0]*self.shape[1]}',end='',flush=True)
                valid_mask = np.isfinite(self.data[obs,order]) # a mask to compute model only where Y is not nan or inf
                Y = self.data[obs,order,valid_mask]
                X = self.wave[obs,order,valid_mask]
                Y_lowess = lowess(Y,X,frac=frac,return_sorted=False)

                lowess_line[obs,order,valid_mask]  = Y_lowess
                lowess_line[obs,order,~valid_mask] = np.nan # put nan where model could not be computed due to nan in data

        self.data /= lowess_line
        ###################

        if print_info : print('\nDivision by Lowess done !')

        # Save in history
        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

        else :
            self.history[step_ID] = {'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

    def save(self, path, name, print_info=True):
        '''
        Save the current data state in a numpy file at the given path, along with a metadata file in format 'path_metadata' containing the stored history of the data
        '''
        # check if name already exist
        while name+'.pkl' in os.listdir(path):
            do_overwrite = input(f'{name} already exist in {path}, overwrite it ? (y/n) :')
            if do_overwrite=='y':
                break
            elif do_overwrite=='n':
                name=input(f'Please rename your save:')
            else:
                print('Please enter "y" or "n"')

        # save class with pickle
        with open(path+name+'.pkl','wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

        # also save its history in txt
        with open(path+name+'_history.txt','w') as output:
            for step in self.history.keys():
                output.write(step+' :\n')
                for key,value in self.history[step]['parameters'].items():
                    output.write(f'----- {key} : {value}\n')
                output.write('\n')

        if print_info: print(f'DataSet succesfully saved in {path+name}.pkl !')

    # def apply_svd_projection(self, K, use_log = False, show_var = False, save = None, print_info = True):
    #     '''
    #     Same as apply_PCA_svd, but here we project the data on a basis lacking the K first PCs instead of substracting to the
    #     data the reconstructed signal using only K first PCs.
    #     '''
    #     # Current step ID
    #     step_ID = 'apply_svd_projection'
    #     # Default behaviour if no save parameter has been set
    #     if(save==None): save = self.save_history
    #     # Retrieve current step parameters

    #     # retrieve currently defined local variables in current scope : parameters and step_ID
    #     parameters = locals()
    #     # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
    #     parameters.pop('self')
    #     # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
    #     parameters.pop('step_ID')
    #     # Warn if step already found in history
    #     if(step_ID in self.history.keys()):
    #         warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
    #         # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
    #         k = 0
    #         former_step_ID = step_ID
    #         while step_ID in self.history.keys():
    #             k+=1
    #             step_ID = former_step_ID+'_'+str(k)
    #     self.SVD_V = []
    #     ## Apply PCA using SVD ##
    #     # center and reduce the data
    #     if use_log:
    #         # PB MAY ARISE WHEN USING LOG ON DATA THAT CONTAINS ZEROS !
    #         X = np.log(np.copy(self.data))
    #     else:
    #         X = np.copy(self.data)
    #     mean = np.mean(X,axis=2) # spectral mean (si PCA a 1 feature par obs, alors on enleve a chaque obs sa valeur moyenne)
    #     std  = np.std(X,axis=2) # spectral std
    #     X = (X - mean[:,:,None]) / std[:,:,None]
    #     # used to store the reconstructed data from svd : perform a CCF on it to check if no planetary signal is inside it
    #     result = np.zeros(self.shape)
    #     # Prepare grid of plots to show variance for each orders
    #     if show_var:
    #         n_cols = int(np.sqrt(self.shape[1]))
    #         n_rows = n_cols
    #         if n_rows*n_cols < self.shape[1] : n_rows+=1
    #         fig,ax = plt.subplots(n_rows,n_cols,figsize=(15,15))
    #     # use SVD to decompose the data, order per order :
    #     print(f'Performing PCA using SVD to remove {K} first PCs :')
    #     for order in range(self.shape[1]):
    #         print(f'\r{order+1}/{self.shape[1]}',end='',flush=True)
    #         ## NaN MANAGEMENT : PROBABLY NOT OPTIMAL !
    #         # if a sample is full of NaN in X[:,order,:] : replace it by a null vector : the point will thus be centered on the mean value (=0.) and should (?) have no effect on the PCA
    #         nan_mask = ~np.isfinite(X[:,order,:])
    #         nan_index = np.all(nan_mask,axis=0)
    #         X[:,order,nan_index] = 0
    #         ##
    #         U, S, VT = np.linalg.svd(X[:,order,:].T,full_matrices=False,compute_uv=True) # /!\ SVD must be perform on the transpose of data (we must have 1 column per sample, 1 line per variable)
    #         self.SVD_V.append(np.copy(VT.T))
    #         # we now have U,S & VT defined by : X = U.S.VT with U & VT being unitary matrix (V.VT = I, U.UT = I)
    #         # S is a diagonal rectangular matrix of same shape than data, with diagonal elements being "singular values"
    #         # The K first PCs are in the K first columns of U & K*K upper left corner of S (see https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8, https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca)
    #         X_svd_basis = S * U # Here we reconstruct the entire data in the svd basis
    #         VT[:K] = 0.
    #         X_svd = X_svd_basis @ VT # then reproject it back on the original basis but from which the K first axis have been removed
    #         # retranspose to fit the original shape
    #         X_svd = X_svd.T
    #         # Now remove the X_svd data (reconstructed using K first PCs) from the original data : hoping this will remove most of the tellurics residuals
    #         X_svd = (X_svd * std[order]) + mean[order]
    #         if use_log: X_svd = np.exp(X_svd)
    #         # self.svd_reconstructed_data[:,order,:] = data_reconstructed
    #         # Compute and show the stored variance of each PCs for current order
    #         if show_var:
    #             var      = (S ** 2) / (self.shape[0]-1)
    #             var     /= var.sum() # normalise such that total variance is 1
    #             sub_ax = ax.flatten()[order]
    #             sub_ax.plot(var,'-.')
    #             sub_ax.set_xlabel('PCs N°')
    #             sub_ax.set_ylabel('Variance')
    #             sub_ax.text(0.8,0.85,f'Order n° {order}',transform = sub_ax.transAxes,fontsize = 12)

    #         # update data
    #         self.data[:,order,:] = X_svd
    #     #########################

    #     if print_info : print('\nSVD done !')

    #     # Save in history
    #     if save:
    #         self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
    #         if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

    #     else :
    #         self.history[step_ID] = {'parameters' : parameters}

    #     if show_var:
    #         plt.tight_layout()
    #         plt.show()

    def apply_pca_on_synth_projection(self,synth):
        '''
        Given a synthetic spectra for a given observation, apply the effect of the SVD computed for this data_set on this observation on the given synth.
        To do so, we use the already known V matrix containing the eigenvectors (in columns) of the SVD decomposition already computed,
        and project the synthetic spectra onto the PCA base using this matrix V, then reproject on the original basis using V.T from which the same first K PCs as in data have been set to 0.
        '''
        # recall the parameters used for svd
        if 'apply_svd' in self.history.keys():
            K = self.history['apply_svd']['parameters']['K']
            use_log = self.history['apply_svd']['parameters']['use_log']

        elif 'apply_svd_projection' in self.history.keys():
            K = self.history['apply_svd_projection']['parameters']['K']
            use_log = self.history['apply_svd_projection']['parameters']['use_log']
        
        elif 'PCA_Kneedle' in self.history.keys():
            K  = self.K_list
            # print('Detecting PCA Kneedle in data history')
            use_log = self.history['PCA_Kneedle']['parameters']['use_log']

        else:
            raise NameError('You must apply an svd step on this DataSet before using this function on a synthetic spectra')

        # used to store the reconstructed data from svd : perform a CCF on it to check if no planetary signal is inside it
        result = np.zeros(synth.shape)
        recon_data = np.zeros(synth.shape)

        for order in range(self.shape[1]):
            # convert to log space if specified
            if use_log:
                data = np.log(np.copy(synth[:,order]))
            else:
                data = np.copy(synth[:,order])
            # # transform data (center/reduce)
            # X,transfo_params = transform_data(data) # we could pass a data_set object ("self") as second argument here to weight the mean & std used to center the data, but it makes this operation 3 times longer. Since we're going to call it a high number of times (for CCF or )

            # do we need to center/reduce data since we're only projecting them on existing axis?
            # Try without centering/reducing data
            X = data.T

            # extract the V (eigenvectors) matrix from the previous SVD on data
            V = self.SVD_V[order]
            # remove first K PCs from the VT matrix
            if type(K)==list:
                k = K[order]
            else:
                k = K
            # project the synthetic in the PCA basis using only K first PCs
            X_svd = np.dot(X,V[:,:k])
            # project back to original basis and remove the reconstructed data
            X_recon = np.dot(X_svd,V.T[:k])
            
            # invert transform & store result
            # X_recon = invert_transform(X_recon,*transfo_params)
            X_recon = X_recon.T

            if use_log: X_recon = np.exp(X_recon)

            # PCA introduce signal in off-transit obs: this is also the case for the planetary signal in the real data, but we may lose overall SNR when including these off-transit observation... Manually set to 1 off-transit obs (data are centered around 1: 1 = no signal)
            X_recon[self.off_transit_mask] = 1

            recon_data[:,order,:] = X_recon
        # return result center at 1
        result = synth - recon_data + 1
        return(result)

    def apply_pca_on_synth(self,synth):
        '''
        Given a synthetic spectra for a given observation, apply the effect of the SVD computed for this data_set on this observation on the given synth.
        To do so, we use equation 7 from Gibson et al 2022 which involves computing the U.U^dagger (U times its pseudo inverse), with U containing the basis vector from the PCA/SVD decomposition obtained during the reduction. 
        '''
        # recall the parameters used for svd
        if 'apply_svd' in self.history.keys():
            use_log = self.history['apply_svd']['parameters']['use_log']

        elif 'apply_svd_projection' in self.history.keys():
            use_log = self.history['apply_svd_projection']['parameters']['use_log']
        
        elif 'PCA_Kneedle' in self.history.keys():
            use_log = self.history['PCA_Kneedle']['parameters']['use_log']

        elif 'PCA' in self.history.keys():
            use_log = self.history['PCA']['parameters']['use_log']

        elif 'SysREM_Kneedle' in self.history.keys():
            use_log = self.history['SysREM_Kneedle']['parameters']['use_log']
        else:
            raise NameError('You must apply an svd step on this DataSet before using this function on a synthetic spectra')

        # used to store the reconstructed data from svd : perform a CCF on it to check if no planetary signal is inside it
        result = np.zeros(synth.shape)
        recon_data = np.zeros(synth.shape)

        for order in range(self.shape[1]):
            X = np.copy(synth[:,order]).T

            # extract the U.U^T (U times its pseudo invert) matrix from the previous PCA performed on data. This matrix only contains the K first PCs for the given order
            UUT = self.SVD_UUT[order]
            # UUT is a masked array: convert to array to speed-up matrix multiplication
            UUT = UUT.data
            
            if use_log: 
                X_log = np.log(X)
                prod = np.dot(UUT, X_log)
                X_recon = np.exp(X_log - prod)            
            else:
                prod = np.dot(UUT, X)
                X_recon = X - prod

            # retranspose
            X_recon = X_recon.T

            # PCA introduce signal in off-transit obs: this is also the case for the planetary signal in the real data, but we may lose overall SNR when including these off-transit observation... Manually set to 1 off-transit obs (data are centered around 1: 1 = no signal)
            # X_recon[self.off_transit_mask] = 1

            recon_data[:,order,:] = X_recon # data reconstructed using only K first PCs following the same eigenvectors as in the data set

        # degrade model following the SVD on data, center at 1, and return result (degraded model)
        result = recon_data
        return(result)

    def undo_last_step(self,print_info=True):
        # undo the last reduction step
        last_step = list(self.history.keys())[-1]
        penultimate_step = list(self.history.keys())[-2]
        # bring data back to their penultimate state
        if print_info: print(f'Bringing back data to their value after "{penultimate_step}"')
        self.data = self.history[penultimate_step]['data']
        # erase last step from history
        if print_info: print(f'Removing "{last_step}" from DataSet history')
        del self.history[last_step]

    def doppler_shift(self,V = None, prebuilt = None,print_info=True,save=None):
        '''
        Doppler shift the data using the given velocity vector or prebuilt option

        - V : radial velocity vector [m/s] of the rest of frame toward which data are shifted.
        V > 0 means data are blueshifted -> c'est l'inverse non ?
        V < 0 means data are redshifted
        The number of element in the vector must be equal to the number of observations (one velocity per transit phase in the data)

        - prebuilt : can be "earth_to_star", "earth_to_planet" or "star_to_planet", in which case the data are shifted
        from the eart/earth/stellar RF to the stellar/planet/planet RF respectively using the precomputed velocities from
        the intern oribtal model
        '''
        if (V is None) and (prebuilt is None):
            raise NameError('You must at least provide a velocity or a prebuilt mode')
        if (V is None) and (prebuilt not in ["earth_to_star", "earth_to_planet","star_to_planet"]):
            raise NameError('prebuilt must be one of "earth_to_star", "earth_to_planet" or "star_to_planet"')
        if (prebuilt in ["earth_to_star", "earth_to_planet"]) and ("doppler_shift" in self.history.keys()):
            raise NameError('Trying to shift data from Earth RF to stellar/planet RF, but data are not in Earth RF (doppler_shift found in data history)')
        if (prebuilt == "star_to_planet") and not ("doppler_shift" in self.history.keys()):
            raise NameError('Trying to shift data from stellar RF to stellar/planet RF, but data are still in Earth RF (doppler_shift not found in data history)')
        if not (V is None) and not (prebuilt is None):
            raise NameError('You must only provide a velocity or a prebuilt mode, not both')

        # Current step ID
        step_ID = 'doppler_shift'
        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        # Retrieve current step parameters

        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')
        # Warn if step already found in history
        if(step_ID in self.history.keys()):
            warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            k = 0
            former_step_ID = step_ID
            while step_ID in self.history.keys():
                k+=1
                step_ID = former_step_ID+'_'+str(k)

        ## DO STUFF HERE ##
        if prebuilt   == "earth_to_star":
            V = self.Vd
        elif prebuilt == "earth_to_planet":
            V = self.Vtot
        elif prebuilt == "star_to_planet":
            V = self.Vrv_star-self.Vp

        # wave_shifted = self.wave / (1 - (V[...,None,None] / const.c.value)) # V0 > 0 -> redshift, V0 < 0 -> blueshift
        shifted_spec = np.zeros(self.shape)
        for obs in range(self.shape[0]):
            for order in range(self.shape[1]):
                # oversample to reduce interpolation error during shifting
                wave_over = np.linspace(self.wave[obs,order,0],self.wave[obs,order,-1],self.wave[obs,order].size*1000) # with x1000 oversampling, the square interpolation error on a given order is ~e-11
                spec_over = interp1d(self.wave[obs,order],self.data[obs,order])(wave_over)
                # then shift
                doppler_interp = interp1d(wave_over/(1-(V[obs]/const.c.value)),spec_over,bounds_error=False)
                shifted_spec[obs,order] = doppler_interp(self.wave[obs,order])
        self.data = shifted_spec
        ###################

        if print_info : print(f'\nDoppler shifting done !')

        # Save in history
        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

        else :
            self.history[step_ID] = {'parameters' : parameters}

    def StellarCorr_DivByWeightedAvg(self, radius=25, plots = False, save = None, print_info = True):
        '''
        Apply the stellar correction with a PYSME stellar grid for the div by weighted avg step
        stellar_interpoler_path (str) : the path to the pickle object containing the Stellar Interpoler to compute the stellar grid (see PYSME_StellarGrid notebook)
        transit (dic) : a dictionnary containing the informations of the transit (see transit_info.py)
        radius (int) : radius, in number of cells, used to compute the stellar grid
        plots (bool) : whether to plots the stellar grid and spectra or not

        -> Now only used to show the PYSME grid but no longer used for the actuel correction
        '''
        # Current step ID
        step_ID = 'StellarCorrDivByWeightedAvg'
        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')
        # Warn if step already found in history
        if(step_ID in self.history.keys()):
            warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            k = 0
            former_step_ID = step_ID
            while step_ID in self.history.keys():
                k+=1
                step_ID = former_step_ID+'_'+str(k)

        ## DO STUFF HERE ##
        # load stellar interpoler
        print('Loading the Stellar Interpoler object :')
        with open(self.transit_dic['stellar_interpoler_path'],'rb') as file_in:
            MyInterp = pickle.load(file_in)
            total_interp = MyInterp.Interpoler
        print(f'Interpoler loaded, continuum normalised = {MyInterp.NormalizedByContinuum}')

        def limb_dark(mu):
            # see if the limb dark law is uniform or not
            if len(self.transit_dic['c'])==4:
                c1,c2,c3,c4 = self.transit_dic['c']
                limb_avg = 1-c1-c2-c3-c4+0.8*c1+(2./3.)*c2+(4./7.)*c3+0.5*c4
                limb = limb_avg * (1-c1*(1-np.sqrt(mu))-c2*(1-mu)-c3*(1-np.power(mu,3/2))-c4*(1-mu**2))
            elif len(self.transit_dic['c']==0):
                limb = 1
            else:
                raise NameError(f'Limb dark function with coeff {self.transit_dic["c"]} have not been implemented in this function : need to update using formula in http://lkreidberg.github.io/batman/docs/html/tutorial.html#limb-darkening-options')

            return(limb)

        vsini = MyInterp.vsini # rotational velocity v.sin(i) of the stellar surface [m/s] (ref https://iopscience.iop.org/article/10.3847/1538-3881/153/1/21 for HD189733)

        # build the stellar surface grid with a resolution of 0.01 Rstar * 0.01 Rstar
        N_cell = radius*2+1 # total nb of cell along an axis (+1 to account for the 0)
        Vmap = np.zeros((N_cell,N_cell))

        print(f'Computing stellar spectra on a grid of {N_cell} x {N_cell} cells :')
        stellar_surface_grid = np.zeros((N_cell,N_cell,MyInterp.synth_wave.size)) * np.nan

        k=1
        index_i = 0
        for i in range(-radius,radius+1):
            index_j = 0
            for j in range(-radius,radius+1):

                print(f'\r{k} / {(N_cell)**2}',end='',flush=True)

                # get mu corresponding to current cell
                r = np.sqrt( (i**2) + (j**2))
                r /= radius
                # if outside the stellar surface
                if r > 1:
                    stellar_surface_grid[index_j,index_i] = np.nan

                else:
                    # limb dark angle
                    alpha = np.arcsin(r)
                    mu = np.cos(alpha)

                    # spherical coordinate for RME
                    theta = np.arcsin(j / radius)
                    # inclination (0 = North Pole, pi/2 = equator)
                    phi = (np.pi/2) - np.arcsin(i / radius)

                    # doppler shift the spectra to account for stellar rotation (RME)
                    # see https://dynref.engr.illinois.edu/rvs.html for spherical coordinates/
                    Vrot = -1*vsini * np.sin(phi) * np.sin(theta)
                    Vmap[index_i,index_j] = Vrot

                    # compute the stellar spectra of this cell
                    cell_spectra = total_interp([-1*Vrot,mu])[0,:] # need to invert the sign of Vrot to get the blue & redshift on correct side of the star

                    # if the PYSME grid was generated with continuum fixed at 1 : apply a non-linear limb dark law on the continuum
                    if MyInterp.NormalizedByContinuum:
                        stellar_surface_grid[index_i,index_j] = limb_dark(mu)*cell_spectra
                    else:
                        stellar_surface_grid[index_i,index_j] = cell_spectra

                index_j+=1
                k+=1
            index_i += 1

        # Compute the cells crossed by the planet during transit using pyasL.KeplerEllipse model
        pos = self.ke.xyzPos(self.time_from_mid)
        # x & y coordinate of the planet trajectory during transit, normalized in stellar radii
        a = pos[:,0]/self.transit_dic['Rs']
        y_a = pos[:,1]/self.transit_dic['Rs']

        # # Compute the cells crossed by the planet during transit
        # b = (self.transit_dic['a']/self.transit_dic['Rs']) * np.cos(self.transit_dic['i']*np.pi/180.) # impact parameter
        # lbda = np.pi * (self.transit_dic['lbda']/180.) # spin orbit angle [rad]
        #
        # # trajectory --> Verify with https://iopscience.iop.org/article/10.1086/428344/pdf for geometry and formula
        # x = np.linspace(-1,1,100)
        # y = -1*np.tan(lbda)*x + (b / np.cos(lbda))*np.ones(x.shape) # found using the tangent equation of a circle of radius b (impact parameter) & center on (0,0)
        #
        # # convert time from mid transit to coordinate on stellar surface
        # a = (self.time_from_mid / (self.T4 - self.T1))
        # y_a = -1*np.tan(lbda)*a + (b / np.cos(lbda))*np.ones(a.shape)

        if plots :
            # plot the mean surface grid
            plt.figure(figsize=(10,10))

            lmin, lmax = 1083.3,1083.4
            index = np.where( (self.wave[0].flatten()>lmin) * (self.wave[0].flatten()<lmax))[0][0]

            # index = 4088//2
            # print(index)
            half_cell = (1/radius)/2 # the size of half a cell in the plot
            # we plot intensity map normalised by the integrated flux of the stellar surface at the given wavelength
            plt.imshow(stellar_surface_grid[:,:,index]/np.nansum(stellar_surface_grid[:,:,index]),origin='lower',interpolation='None',extent=[-1-half_cell,1+half_cell,-1-half_cell,1+half_cell])
            # if we set the extent to [-1,1,-1,1], the left/bottom side of the first cell correspond to -1 & the right/up side of the last cell correspond to +1,
            # which does not fit with our grid : we want the center of the first cell to be -1, and the center of the last cell to be +1

            plt.plot(a,y_a,label='planet trajectory',color='r')
            # plt.title(f'$\lambda$ = {self.wave[0].flatten()[index]:.2f} nm')
            plt.title(f'Stellar surface intensity grid at $\lambda$ = {self.wave[0].flatten()[index]:.2f} nm')
            c = plt.colorbar(format='%.1e')
            c.set_label('Relative Intensity')
            plt.ylabel('Stellar Radius')
            plt.xlabel('Stellar Radius')
            plt.xlim(-1.1,1.1)
            plt.ylim(-1.1,1.1)
            # plt.xticks([i/radius for i in range(-radius,radius+1)],rotation=90)
            # plt.yticks([i/radius for i in range(-radius,radius+1)])
            # plt.grid()

            ###
            # get the masked cell for a given on_transit obs
            obs_index = 10 # chooose a on-transit obs
            obs_index = self.on_transit_mask[obs_index]
            masked_cell = np.zeros((N_cell,N_cell),dtype='bool')
            i_index = 0
            for i in range(-radius,radius+1):
                j_index = 0
                for j in range(-radius,radius+1):
                    # a cell is masked if it center lies within the radius of the planet over the stellar surface. /!\ i is index for rows, j for column ! a correspond to x coordinate, and _ya to y : we thus compare j (columns) to a and i (rows) to y_a !
                    if np.sqrt( ((j/radius) - a[obs_index])**2 + ((i/radius) - y_a[obs_index])**2 ) <= (self.transit_dic['Rp']/self.transit_dic['Rs']):
                        masked_cell[i_index,j_index] = True
                    j_index += 1
                i_index += 1
            plt.imshow(masked_cell,origin='lower',interpolation='None',extent=[-1-half_cell,1+half_cell,-1-half_cell,1+half_cell],alpha=0.2,cmap='Greys')
            # zoom on the masked cells
            # plt.xlim(a[obs_index]-(10/radius),a[obs_index]+(10/radius))
            # plt.ylim(y_a[obs_index]-(10/radius),y_a[obs_index]+(10/radius))
            ###


            # draw the planet position over the stellar surface
            for k,el in enumerate(a):
                if k==obs_index:
                    plt.gca().add_patch(plt.Circle((el,y_a[k]),self.transit_dic['Rp']/self.transit_dic['Rs'],color='k',alpha=0.5,label='planet surface'))

            # Or only show the center of the planet position
            plt.plot(a,y_a,'.k',ms=5,label='Planet center during transit')

            plt.legend()

            # ### plot the neighborhood of the plotted wvl in the spectra
            # plt.figure()
            # zoom = 100
            # ax=plt.gca()
            #
            # steps = 5
            #
            # # /!\ : first index of stellar_surface_grid is rows, second is columns !
            #
            # ax.set_prop_cycle('color',plt.cm.turbo(np.linspace(0,1,1+stellar_surface_grid.shape[0]//steps)))
            # for j in range(0,stellar_surface_grid.shape[0],steps):
            #
            #     if j > (stellar_surface_grid.shape[0]//2):
            #         ls='--'
            #     else:
            #         ls='-'
            #
            #     # plot spectra with respect to theta, fixing phi at the center of the stellar surface
            #     temp = stellar_surface_grid[radius,j,index-zoom:index+zoom]
            #
            #     mu = np.sqrt(1-( (j-radius)/radius)**2)
            #     theta = np.arcsin( (j-radius) / radius)
            #
            #     plt.plot(self.wave[0].flatten()[index-zoom:index+zoom],temp,label=theta,ls=ls)
            #
            # plt.vlines(self.wave[0].flatten()[index],stellar_surface_grid[radius,radius,index-zoom:index+zoom].min(),stellar_surface_grid[radius,radius,index-zoom:index+zoom].max(),'r')
            # plt.legend()
            #
            # ### compare the data weighted off transit mean with the pysme off transit means
            # plt.figure()
            # plt.plot(self.wave[0].flatten(),self.weighted_mean(self.data,transit_mask="off").flatten(),label='Data weighted off transit mean')
            # mean_synth = np.nanmean(stellar_surface_grid,axis=(0,1))
            # plt.plot(MyInterp.synth_wave,mean_synth/np.nanmedian(mean_synth))
            # plt.legend()
            # plt.xlabel(f'$\lambda$')
            # plt.ylabel('Rel Intensity')
            # plt.title('Off-transit : Data mean Vs Synthetic avg on all disc')


        # print('\nNow applying the correction on each observations :')
        # # divide each data observations by a mean spectra corrected from CLV & RME using the grid
        # # first compute the off transit spectra
        # off_transit_stellar_spec = np.nanmean(stellar_surface_grid,axis=(0,1))
        # stellar_correction_sampled = np.ones(self.shape)
        #
        # for obs in range(self.shape[0]):
        #     # get the masked cells for a given obs
        #
        #     if obs in self.on_transit_mask:
        #         print(f'\r{obs-self.on_transit_mask[0]+1}/{len(self.on_transit_mask)}',end='',flush=True) # only show counter for on transit computation because the off-transit  is way faster
        #         # build the mask
        #         masked_cell = np.ones((N_cell,N_cell),dtype='bool')
        #         nan_mask = np.ones((N_cell,N_cell),dtype='bool')
        #
        #         i_index = 0
        #         for i in range(-radius,radius+1):
        #             j_index = 0
        #             for j in range(-radius,radius+1):
        #                 # a cell is masked if it center lies within the radius of the planet over the stellar surface. /!\ i is index for rows, j for column ! a correspond to x coordinate, and _ya to y : we thus compare j (columns) to a and i (rows) to y_a !
        #                 if np.all(np.isnan(stellar_surface_grid[i_index,j_index])) :
        #                     masked_cell[i_index,j_index] = False
        #                     nan_mask[i_index,j_index] = False
        #                 elif (np.sqrt( ((j/radius) - a[obs])**2 + ((i/radius) - y_a[obs])**2 ) <= (self.transit_dic['Rp']/self.transit_dic['Rs'])):
        #                     masked_cell[i_index,j_index] = False
        #                 else:
        #                     pass
        #                 j_index += 1
        #             i_index += 1
        #
        #         # Now build the on transit stellar spectra : the mask contain False where a cell is masked
        #         on_transit_stellar_spec = np.mean(stellar_surface_grid,where=masked_cell[:,:,None],axis=(0,1)) # using a mask with the "where" option specify which data not to include in the mean (add the [:,:,None] so that the mask is duplicated along the missing dimension)
        #         # on_transit_stellar_spec = np.nanmean(stellar_surface_grid,axis=(0,1)) # using a mask with the "where" option specify which data not to include in the mean (add the [:,:,None] so that the mask is duplicated along the missing dimension)
        #
        #         # Then compute the correcting factor by dividing the on_transit by the off_transit
        #         stellar_correction = on_transit_stellar_spec / off_transit_stellar_spec
        #
        #         # interpolate the correction on SPIRou sampling wavelength
        #         interpoler = interp1d(MyInterp.synth_wave,stellar_correction,bounds_error=False,fill_value=np.nan)
        #         stellar_correction_sampled[obs] = interpoler(self.wave[0].flatten()).reshape(self.shape[1],-1)
        #
        #         # apply the correction on data
        #         # mean_data = self.weighted_mean(self.data,off_transit_only=True)*stellar_correction_sampled
        #
        #     else:
        #         # mean_data = self.weighted_mean(self.data,off_transit_only=True)
        #         pass
        #
        # mean_data = stellar_correction_sampled * self.weighted_mean(self.data,transit_mask="off")[None,...]
        # self.data /= mean_data

        ###################

        if print_info : print('\nStellarCorr_DivByWeightedAvg done !')

        # Save in history
        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

        else :
            self.history[step_ID] = {'parameters' : parameters}

    #<f
    # def StellarCorr_DivByWeightedAvg_lessMemory(self, radius=25, plots = False, save = None, print_info = True):
    #     '''
    #     Apply the stellar correction with a PYSME stellar grid for the div by weighted avg step
    #     stellar_interpoler_path (str) : the path to the pickle object containing the Stellar Interpoler to compute the stellar grid (see PYSME_StellarGrid notebook)
    #     transit (dic) : a dictionnary containing the informations of the transit (see transit_info.py)
    #     radius (int) : radius, in number of cells, used to compute the stellar grid
    #     plots (bool) : whether to plots the stellar grid and spectra or not
    #
    #     -> This version use a less memory consumption way to compute mean on/off-transit spectra in the correction (allows for bigger grids, but may be slower ?)
    #     '''
    #     # Current step ID
    #     step_ID = 'StellarCorr_DivByWeightedAvg_lessMemory'
    #     # Default behaviour if no save parameter has been set
    #     if(save==None): save = self.save_history
    #     # retrieve currently defined local variables in current scope : parameters and step_ID
    #     parameters = locals()
    #     # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
    #     parameters.pop('self')
    #     # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
    #     parameters.pop('step_ID')
    #     # Warn if step already found in history
    #     if(step_ID in self.history.keys()):
    #         warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
    #         # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
    #         k = 0
    #         former_step_ID = step_ID
    #         while step_ID in self.history.keys():
    #             k+=1
    #             step_ID = former_step_ID+'_'+str(k)
    #
    #     ## DO STUFF HERE ##
    #     # load stellar interpoler
    #     print('Loading the Stellar Interpoler object :')
    #     with open(self.transit_dic['stellar_interpoler_path'],'rb') as file_in:
    #         MyInterp = pickle.load(file_in)
    #         total_interp = MyInterp.Interpoler
    #     print(f'Interpoler loaded, continuum normalised = {MyInterp.NormalizedByContinuum}')
    #
    #     def limb_dark(mu):
    #         # see if the limb dark law is uniform or not
    #         if len(self.transit_dic['c'])==4:
    #             c1,c2,c3,c4 = self.transit_dic['c']
    #             limb_avg = 1-c1-c2-c3-c4+0.8*c1+(2./3.)*c2+(4./7.)*c3+0.5*c4
    #             limb = limb_avg * (1-c1*(1-np.sqrt(mu))-c2*(1-mu)-c3*(1-np.power(mu,3/2))-c4*(1-mu**2))
    #         elif len(self.transit_dic['c']==0):
    #             limb = 1
    #         else:
    #             raise NameError(f'Limb dark function with coeff {self.transit_dic["c"]} have not been implemented in this function : need to update using formula in http://lkreidberg.github.io/batman/docs/html/tutorial.html#limb-darkening-options')
    #
    #         return(limb)
    #
    #     vsini = MyInterp.vsini # rotational velocity v.sin(i) of the stellar surface [m/s] (ref https://iopscience.iop.org/article/10.3847/1538-3881/153/1/21 for HD189733)
    #
    #     # Instead of building the full grid, compute the off-transit spectra by summing up each cell spectra and immediatly free the memory of this cell after
    #     N_cell = radius*2+1 # total nb of cell along an axis (+1 to account for the 0)
    #     Vmap = np.zeros((N_cell,N_cell))
    #
    #     print(f'Computing off-transit stellar spectra on a grid of {N_cell} x {N_cell} cells :')
    #
    #     off_transit_cells_sum = np.zeros(MyInterp.synth_wave.size)
    #     Nb_valid_cell = 0 # count the nb of cells that are not nan. Used for the mean
    #     cell_spectra = total_interp([0,0])[0,:] # a dummy cell just to get the right size of a cell
    #     Nb_NaN_array = np.zeros(cell_spectra.shape) # a vector that count the nb of nan at each wavelength in all cells spectra
    #
    #     k=1
    #     index_i = 0
    #     for i in range(-radius,radius+1):
    #         index_j = 0
    #         for j in range(-radius,radius+1):
    #
    #             print(f'\r{k} / {(N_cell)**2}',end='',flush=True)
    #
    #             # get mu corresponding to current cell
    #             r = np.sqrt( (i**2) + (j**2))
    #             r /= radius
    #             # if outside the stellar surface
    #             if r > 1:
    #                 continue
    #
    #             else:
    #                 # limb dark angle
    #                 alpha = np.arcsin(r)
    #                 mu = np.cos(alpha)
    #
    #                 # spherical coordinate for RME
    #                 theta = np.arcsin(j / radius)
    #                 # inclination (0 = North Pole, pi/2 = equator)
    #                 phi = (np.pi/2) - np.arcsin(i / radius)
    #
    #                 # doppler shift the spectra to account for stellar rotation (RME)
    #                 # see https://dynref.engr.illinois.edu/rvs.html for spherical coordinates/
    #                 Vrot = -1*vsini * np.sin(phi) * np.sin(theta)
    #                 Vmap[index_i,index_j] = Vrot
    #
    #                 # compute the stellar spectra of this cell
    #                 cell_spectra = total_interp([-1*Vrot,mu])[0,:] # need to invert the sign of Vrot to get the blue & redshift on correct side of the star
    #                 # incremente the nb of invalid values for each wavelength of the spectra
    #                 invalid_mask = ~np.isfinite(cell_spectra) # contains a 1 at each wavelength storing an invalid number
    #                 Nb_NaN_array += invalid_mask
    #                 # replace invalid values in cell_spectra by 0, so that they will not be taken into account in the mean
    #                 cell_spectra[invalid_mask] = 0
    #
    #                 # if the PYSME grid was generated with continuum fixed at 1 : apply a non-linear limb dark law on the continuum
    #                 if MyInterp.NormalizedByContinuum:
    #                     off_transit_cells_sum += limb_dark(mu)*cell_spectra
    #                 else:
    #                     off_transit_cells_sum += cell_spectra
    #
    #                 Nb_valid_cell +=1
    #             index_j+=1
    #             k+=1
    #         index_i += 1
    #
    #     # compute the mean by dividing each sum up wavelength by the nb of cells minus the nb of invalid values encounter at this specific wavelength
    #     off_transit_stellar_spec = off_transit_cells_sum / (Nb_valid_cell - Nb_NaN_array)
    #
    #     # Compute the cells crossed by the planet during transit
    #     b = (self.transit_dic['a']/self.transit_dic['Rs']) * np.cos(self.transit_dic['i']*np.pi/180.) # impact parameter
    #     lbda = np.pi * (self.transit_dic['lbda']/180.) # spin orbit angle [rad]
    #
    #     # trajectory --> Verify with https://iopscience.iop.org/article/10.1086/428344/pdf for geometry and formula
    #     x = np.linspace(-1,1,100)
    #     y = -1*np.tan(lbda)*x + (b / np.cos(lbda))*np.ones(x.shape) # found using the tangent equation of a circle of radius b (impact parameter) & center on (0,0)
    #
    #     # convert time from mid transit to coordinate on stellar surface
    #     a = (self.time_from_mid / (self.T4 - self.T1))
    #     y_a = -1*np.tan(lbda)*a + (b / np.cos(lbda))*np.ones(a.shape)
    #
    #     #<f
    #     ## With this method though we can't show the stellar surface
    #     # if plots :
    #     #     # plot the mean surface grid
    #     #     plt.figure(figsize=(10,10))
    #     #
    #     #     lmin, lmax = 963.9,963.94
    #     #     index = np.where( (self.wave[0].flatten()>lmin) * (self.wave[0].flatten()<lmax))[0][0]
    #     #
    #     #     # index = 10001
    #     #     # print(index)
    #     #     half_cell = (1/radius)/2 # the size of half a cell in the plot
    #     #     plt.imshow(stellar_surface_grid[:,:,index],origin='lower',interpolation='None',extent=[-1-half_cell,1+half_cell,-1-half_cell,1+half_cell])
    #     #     # if we set the extent to [-1,1,-1,1], the left/bottom side of the first cell correspond to -1 & the right/up side of the last cell correspond to +1,
    #     #     # which does not fit with our grid : we want the center of the first cell to be -1, and the center of the last cell to be +1
    #     #
    #     #     plt.plot(x,y,label='planet trajectory',color='r')
    #     #     plt.title(f'$\lambda$ = {self.wave[0].flatten()[index]:.2f} nm')
    #     #     c = plt.colorbar()
    #     #     c.set_label('Relative Intensity')
    #     #     plt.ylabel('Stellar Radius')
    #     #     plt.xlabel('Stellar Radius')
    #     #     plt.xlim(-1.1,1.1)
    #     #     plt.ylim(-1.1,1.1)
    #     #     plt.xticks([i/radius for i in range(-radius,radius+1)],rotation=90)
    #     #     plt.yticks([i/radius for i in range(-radius,radius+1)])
    #     #     plt.grid()
    #     #
    #     #     ###
    #     #     # get the masked cell for a given on_transit obs
    #     #     obs_index = 10 # chooose a on-transit obs
    #     #     obs = self.on_transit_mask[obs_index]
    #     #     masked_cell = np.zeros((N_cell,N_cell),dtype='bool')
    #     #     i_index = 0
    #     #     for i in range(-radius,radius+1):
    #     #         j_index = 0
    #     #         for j in range(-radius,radius+1):
    #     #             # a cell is masked if it center lies within the radius of the planet over the stellar surface. /!\ i is index for rows, j for column ! a correspond to x coordinate, and _ya to y : we thus compare j (columns) to a and i (rows) to y_a !
    #     #             if np.sqrt( ((j/radius) - a[obs_index])**2 + ((i/radius) - y_a[obs_index])**2 ) <= (self.transit_dic['Rp']/self.transit_dic['Rs']):
    #     #                 masked_cell[i_index,j_index] = True
    #     #             j_index += 1
    #     #         i_index += 1
    #     #     plt.imshow(masked_cell,origin='lower',interpolation='None',extent=[-1-half_cell,1+half_cell,-1-half_cell,1+half_cell],alpha=0.2,cmap='Greys')
    #     #     # zoom on the masked cells
    #     #     # plt.xlim(a[obs_index]-(10/radius),a[obs_index]+(10/radius))
    #     #     # plt.ylim(y_a[obs_index]-(10/radius),y_a[obs_index]+(10/radius))
    #     #     ###
    #
    #         #
    #         # # draw the planet position over the stellar surface
    #         # for k,el in enumerate(a):
    #         #     if k==obs_index:
    #         #         plt.gca().add_patch(plt.Circle((el,y_a[k]),self.transit_dic['Rp']/self.transit_dic['Rs'],color='k',alpha=0.5,label='planet surface'))
    #         #
    #         # # Or only show the center of the planet position
    #         # plt.plot(a,y_a,'.k',ms=5,label='Planet center during transit')
    #         #
    #         # plt.legend()
    #         #
    #         # ### plot the neighborhood of the plotted wvl in the spectra
    #         # plt.figure()
    #         # zoom = 100
    #         # ax=plt.gca()
    #         #
    #         # steps = 5
    #         #
    #         # # /!\ : first index of stellar_surface_grid is rows, second is columns !
    #         #
    #         # ax.set_prop_cycle('color',plt.cm.turbo(np.linspace(0,1,1+stellar_surface_grid.shape[0]//steps)))
    #         # for j in range(0,stellar_surface_grid.shape[0],steps):
    #         #
    #         #     if j > (stellar_surface_grid.shape[0]//2):
    #         #         ls='--'
    #         #     else:
    #         #         ls='-'
    #         #
    #         #     # plot spectra with respect to theta, fixing phi at the center of the stellar surface
    #         #     temp = stellar_surface_grid[radius,j,index-zoom:index+zoom]
    #         #
    #         #     mu = np.sqrt(1-( (j-radius)/radius)**2)
    #         #     theta = np.arcsin( (j-radius) / radius)
    #         #
    #         #     plt.plot(self.wave[0].flatten()[index-zoom:index+zoom],temp,label=theta,ls=ls)
    #         #
    #         # plt.vlines(self.wave[0].flatten()[index],stellar_surface_grid[radius,radius,index-zoom:index+zoom].min(),stellar_surface_grid[radius,radius,index-zoom:index+zoom].max(),'r')
    #         # plt.legend()
    #         #
    #         # ### compare the data weighted off transit mean with the pysme off transit means
    #         # plt.figure()
    #         # plt.plot(self.wave[0].flatten(),self.weighted_mean(self.data,off_transit_only=True).flatten(),label='Data weighted off transit mean')
    #         # mean_synth = np.nanmean(stellar_surface_grid,axis=(0,1))
    #         # plt.plot(MyInterp.synth_wave,mean_synth/np.nanmedian(mean_synth))
    #         # plt.legend()
    #         # plt.xlabel(f'$\lambda$')
    #         # plt.ylabel('Rel Intensity')
    #         # plt.title('Off-transit : Data mean Vs Synthetic avg on all disc')
    #     #f>
    #
    #     print('\nNow applying the correction on each observations :')
    #     # divide each data observations by a mean spectra corrected from CLV & RME using the grid
    #     # first compute the off transit spectra
    #     stellar_correction_sampled = np.ones(self.shape)
    #
    #     for obs in self.on_transit_mask:
    #         # get the masked cells for a given obs
    #
    #         print(f'\r{obs-self.on_transit_mask[0]+1}/{len(self.on_transit_mask)}',end='',flush=True) # only show counter for on transit computation because the off-transit  is way faster
    #         # build the mask
    #         masked_cell = np.ones((N_cell,N_cell),dtype='bool')
    #         nan_mask = np.ones((N_cell,N_cell),dtype='bool')
    #
    #         # for each masked cell : remove it from the sum of off-transit cells
    #         on_transit_stellar_spec = np.copy(off_transit_cells_sum)
    #         nb_masked_cell = 0
    #
    #         # a vector that count the nb of nan at each wavelength in all cells spectra
    #         Nb_NaN_array = np.zeros(cell_spectra.shape)
    #
    #         i_index = 0
    #         for i in range(-radius,radius+1):
    #             j_index = 0
    #             for j in range(-radius,radius+1):
    #                 r = np.sqrt( (i**2) + (j**2))
    #                 r /= radius
    #                 # if outside the stellar surface
    #                 if r > 1:
    #                     continue
    #                 # if on a masked cell : remove it from spectra
    #                 # a cell is masked if it center lies within the radius of the planet over the stellar surface. /!\ i is index for rows, j for column ! a correspond to x coordinate, and _ya to y : we thus compare j (columns) to a and i (rows) to y_a !
    #                 elif (np.sqrt( ((j/radius) - a[obs])**2 + ((i/radius) - y_a[obs])**2 ) <= (self.transit_dic['Rp']/self.transit_dic['Rs'])):
    #                     # re-compute the corresponding cell to remove it from the mean spectra
    #                     # limb dark angle
    #                     alpha = np.arcsin(r)
    #                     mu = np.cos(alpha)
    #
    #                     # spherical coordinate for RME
    #                     theta = np.arcsin(j / radius)
    #                     # inclination (0 = North Pole, pi/2 = equator)
    #                     phi = (np.pi/2) - np.arcsin(i / radius)
    #
    #                     # doppler shift the spectra to account for stellar rotation (RME)
    #                     # see https://dynref.engr.illinois.edu/rvs.html for spherical coordinates/
    #                     Vrot = -1*vsini * np.sin(phi) * np.sin(theta)
    #
    #                     # compute the stellar spectra of this cell
    #                     cell_spectra = total_interp([-1*Vrot,mu])[0,:] # need to invert the sign of Vrot to get the blue & redshift on correct side of the star
    #
    #                     # incremente the nb of invalid values for each wavelength of the spectra
    #                     invalid_mask = ~np.isfinite(cell_spectra) # contains a 1 at each wavelength storing an invalid number
    #                     Nb_NaN_array += invalid_mask
    #                     # replace invalid values in cell_spectra by 0, so that they will not be taken into account in the mean
    #                     cell_spectra[invalid_mask] = 0
    #
    #                     # if the PYSME grid was generated with continuum fixed at 1 : apply a non-linear limb dark law on the continuum
    #                     if MyInterp.NormalizedByContinuum:
    #                         masked_spectra = limb_dark(mu)*cell_spectra
    #                     else:
    #                         masked_spectra = cell_spectra
    #
    #                     on_transit_stellar_spec -= masked_spectra
    #                     nb_masked_cell += 1
    #                 # if on stellar surface but not a masked cell : nothing happen
    #                 else:
    #                     pass
    #                 j_index += 1
    #             i_index += 1
    #
    #         # here, invalid values were not removed from the off-transit spectra (because were set to 0), so they should not count in the soustraction of masked_cells in the mean)
    #         on_transit_stellar_spec /= (Nb_valid_cell - (nb_masked_cell-Nb_NaN_array))
    #
    #         # Then compute the correcting factor by dividing the on_transit by the off_transit
    #         stellar_correction = on_transit_stellar_spec / off_transit_stellar_spec
    #         stellar_correction[~np.isfinite(stellar_correction)] = 1. # set to 1. all pixels that are not defined
    #
    #         # interpolate the correction on SPIRou sampling wavelength
    #         interpoler = interp1d(MyInterp.synth_wave,stellar_correction,bounds_error=False,fill_value=np.nan)
    #         stellar_correction_sampled[obs] = interpoler(self.wave[0].flatten()).reshape(self.shape[1],-1)
    #
    #     mean_data = stellar_correction_sampled * self.weighted_mean(self.data,transit_mask="off")[None,...]
    #     self.data /= mean_data
    #
    #     ###################
    #
    #     if print_info : print('\nStellarCorr_DivByWeightedAvg done !')
    #
    #     # Save in history
    #     if save:
    #         self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
    #         if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")
    #
    #     else :
    #         self.history[step_ID] = {'parameters' : parameters}
    #f>

    def StellarCorr_KeplerEll(self, restframe, transit_mask='off', radius=25, plots = False, save = None, print_info = True):
        '''
        IMPORTANT: data must be in Earth RF when calling this function. We can specify in which restframe is the master spectrum computed (Earth or stellar):
        this will compute the average of the data in the corresponding rest frame to compute the master, using a copy of the data for the computation to avoid
        adding interpolation error in the real data. Plus, we can define a transit_mask to choose if the master spectrum is based on the average of off-transit
        data only (default) or all data (on+off).
        '''

        # restframe: in which restframe will be the master spectrum computed. If stellar: a copy of the data is shifted in stellar restframe for the computation, and the resulting master spectrum is shifted back to earth rest frame
        if restframe not in ['Earth','Stellar']: raise NameError(restframe+' must be "Earth" or "Stellar"')
        if 'doppler_shift' in self.history.keys(): raise NameError('Doppler shift detected in data history. This step must be done on data in Earth RF')
        # Current step ID
        step_ID = 'StellarCorr_KeplerEll'
        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')
        # Warn if step already found in history
        if(step_ID in self.history.keys()):
            warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            k = 0
            former_step_ID = step_ID
            while step_ID in self.history.keys():
                k+=1
                step_ID = former_step_ID+'_'+str(k)

        # check if compute_kepler_orbit() has been initialized
        if self.ke is None: raise NameError('You must initialize the KeplerEllipse orbit before using this method: DataSet.compute_kepler_orbit().')

        ## DO STUFF HERE ##
        # get the RME/CLV deformation map in stellar RF: it's already convolved at SPIRou resolution and sampled on SPIRou channels
        stellar_correction = self.Compute_StellarCorr_KeplerEll(radius,plots)
        # Shift the correction in the Earth RF
        stellar_correction_shift = np.ones(stellar_correction.shape)
        for order in range(self.shape[1]):
            for obs in self.on_transit_mask:
                stellar_correction_shift[obs,order] = interp1d(self.wave[obs,order]/(1+self.Vd[obs]/const.c.value), stellar_correction[obs,order], bounds_error=False)(self.wave[obs,order])

        if plots:
            plt.figure()
            for obs in self.on_transit_mask[::4]:
                plt.plot(self.wave[0,0,:],stellar_correction_shift[obs,0],label=f'{self.time_from_mid[obs]*24:.2f}')
            plt.plot(self.wave[0,0,:],(self.weighted_mean(self.data,transit_mask='off')[0]-1)/1e2+1,'-k',label='data/100')
            plt.legend()

        # compute master out
        if restframe=='Earth':
            # compute master spectrum directly in Earth RF
            master_spectrum = stellar_correction_shift * self.weighted_mean(self.data,transit_mask=transit_mask)[None,...]
        elif restframe=='Stellar':
            # shift a copy of the data in stellar RF
            copy_data = copy.deepcopy(self.data)
            copy_data_shift = np.nan*np.zeros(self.shape)
            for order in range(self.shape[1]):
                for obs in range(self.shape[0]):
                    copy_data_shift[obs,order] = interp1d(self.wave[obs,order]/(1-self.Vd[obs]/const.c.value), copy_data[obs,order], bounds_error=False)(self.wave[obs,order])

            # compute master spectrum + correction in stellar RF
            master_spectrum_stellarRF = stellar_correction * self.weighted_mean(copy_data_shift,transit_mask=transit_mask)[None,...]

            # shift master spectrum back in Earth RF
            master_spectrum = np.nan*np.ones(self.shape)
            for order in range(self.shape[1]):
                for obs in range(self.shape[0]):
                    master_spectrum[obs,order] = interp1d(self.wave[obs,order]/(1+self.Vd[obs]/const.c.value), master_spectrum_stellarRF[obs,order], bounds_error=False)(self.wave[obs,order])

        # apply correction on data in Earth RF
        self.data /= master_spectrum
        ###################

        if print_info : print('StellarCorr_KeplerEll done !')

        # Save in history
        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
        if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

        else :
            self.history[step_ID] = {'parameters' : parameters}

    def Compute_StellarCorr_KeplerEll(self, radius=25, plots = False):
        '''
        Apply the stellar correction with a PYSME stellar grid for the div by weighted avg step
        This new version takes into account the orbital parameters such as excentricity. It uses the kepler ellipse model from pyasl

        stellar_interpoler_path (str) : the path to the pickle object containing the Stellar Interpoler to compute the stellar grid (see PYSME_StellarGrid notebook)
        transit (dic) : a dictionnary containing the informations of the transit (see transit_info.py)
        radius (int) : radius, in number of cells, used to compute the stellar grid
        plots (bool) : whether to plots the stellar grid and spectra or not

        -> This version use a less memory consumption way to compute mean on/off-transit spectra in the correction (allows for bigger grids, but may be slower ?)
        '''

        # load stellar interpoler
        print('Loading the Stellar Interpoler object :')
        with open(self.transit_dic['stellar_interpoler_path'],'rb') as file_in:
            MyInterp = pickle.load(file_in)
            total_interp = MyInterp.Interpoler
        print(f'Interpoler loaded, continuum normalised = {MyInterp.NormalizedByContinuum}')

        def limb_dark(mu):
            # see if the limb dark law is uniform or not
            if len(self.transit_dic['c'])==4:
                c1,c2,c3,c4 = self.transit_dic['c']
                limb_avg = 1-c1-c2-c3-c4+0.8*c1+(2./3.)*c2+(4./7.)*c3+0.5*c4
                limb = limb_avg * (1-c1*(1-np.sqrt(mu))-c2*(1-mu)-c3*(1-np.power(mu,3/2))-c4*(1-mu**2))
            elif len(self.transit_dic['c']==0):
                limb = 1
            else:
                raise NameError(f'Limb dark function with coeff {self.transit_dic["c"]} have not been implemented in this function : need to update using formula in http://lkreidberg.github.io/batman/docs/html/tutorial.html#limb-darkening-options')

            return(limb)

        vsini = MyInterp.vsini # rotational velocity v.sin(i) of the stellar surface [m/s] (ref https://iopscience.iop.org/article/10.3847/1538-3881/153/1/21 for HD189733)

        # Instead of building the full grid, compute the off-transit spectra by summing up each cell spectra and immediatly free the memory of this cell after
        N_cell = radius*2+1 # total nb of cell along an axis (+1 to account for the 0)
        Vmap = np.zeros((N_cell,N_cell))

        print(f'Computing off-transit stellar spectra on a grid of {N_cell} x {N_cell} cells :')

        off_transit_cells_sum = np.zeros(MyInterp.synth_wave.size)
        Nb_valid_cell = 0 # count the nb of cells that are not nan. Used for the mean
        cell_spectra = total_interp([0,0])[0,:] # a dummy cell just to get the right size of a cell
        Nb_NaN_array = np.zeros(cell_spectra.shape) # a vector that count the nb of nan at each wavelength in all cells spectra

        # plot for verification
        if plots:
            plt.figure()

        k=0
        index_i = 0
        for i in range(-radius,radius+1):
            index_j = 0
            for j in range(-radius,radius+1):
                k+=1
                print(f'\r{k} / {(N_cell)**2}',end='',flush=True)
                # get mu corresponding to current cell
                r = np.sqrt( (i**2) + (j**2))
                r /= radius
                # if outside the stellar surface
                if r > 1:
                    index_j+=1
                    continue
                else:
                    # limb dark angle
                    alpha = np.arcsin(r)
                    mu = np.cos(alpha)

                    # spherical coordinate for RME
                    theta = np.arcsin(j / radius)
                    # inclination (0 = North Pole, pi/2 = equator)
                    phi = (np.pi/2) - np.arcsin(i / radius)

                    # doppler shift the spectra to account for stellar rotation (RME)
                    # see https://dynref.engr.illinois.edu/rvs.html for spherical coordinates/
                    Vrot = -1*vsini * np.sin(phi) * np.sin(theta)
                    Vmap[index_i,index_j] = Vrot

                    # compute the stellar spectra of this cell
                    cell_spectra = total_interp([-1*Vrot,mu])[0,:] # need to invert the sign of Vrot to get the blue & redshift on correct side of the star
                    # incremente the nb of invalid values for each wavelength of the spectra
                    invalid_mask = ~np.isfinite(cell_spectra) # contains a 1 at each wavelength storing an invalid number
                    Nb_NaN_array += invalid_mask
                    # replace invalid values in cell_spectra by 0, so that they will not be taken into account in the mean
                    cell_spectra[invalid_mask] = 0

                    # if the PYSME grid was generated with continuum fixed at 1 : apply a non-linear limb dark law on the continuum
                    if MyInterp.NormalizedByContinuum:
                        off_transit_cells_sum += limb_dark(mu)*cell_spectra
                    else:
                        off_transit_cells_sum += cell_spectra

                    Nb_valid_cell +=1
                    index_j+=1

                # plot for verification
                if plots:
                    if index_i==25 and index_j%5==0: plt.plot(MyInterp.synth_wave,cell_spectra,label=f'theta={theta:.2f}')

            index_i += 1

        if plots:
            plt.figure()
            plt.imshow(Vmap)

        # compute the mean by dividing each sum up wavelength by the nb of cells minus the nb of invalid values encounter at this specific wavelength
        off_transit_stellar_spec = off_transit_cells_sum / (Nb_valid_cell - Nb_NaN_array)
        # convolve to SPIRou resolution #
        wave_masked,off_transit_stellar_spec_convolved = convolve_SPIRou(MyInterp.synth_wave,off_transit_stellar_spec)
        # interpolate on SPIRou sampling wavelength
        interpoler = interp1d(wave_masked,off_transit_stellar_spec_convolved,bounds_error=False,fill_value=np.nan)
        final_disk_avg_off_spec = interpoler(self.wave)

        print('\nNow applying the correction on each observations :')

        # Compute the cells crossed by the planet during transit using pyasL.KeplerEllipse model
        pos = self.ke.xyzPos(self.time_from_mid)
        # x & y coordinate of the planet trajectory during transit, normalized in stellar radii
        a = pos[:,0]/self.transit_dic['Rs']
        y_a = pos[:,1]/self.transit_dic['Rs']

        final_disk_avg_on_spec = np.ones(self.shape)

        for obs in self.on_transit_mask:
            # get the masked cells for a given obs

            print(f'\r{obs-self.on_transit_mask[0]+1}/{len(self.on_transit_mask)}',end='',flush=True) # only show counter for on transit computation because the off-transit  is way faster
            # build the mask
            masked_cell = np.ones((N_cell,N_cell),dtype='bool')
            nan_mask = np.ones((N_cell,N_cell),dtype='bool')

            # for each masked cell : remove it from the sum of off-transit cells
            on_transit_stellar_spec = np.copy(off_transit_cells_sum)
            nb_masked_cell = 0

            # a vector that count the nb of nan at each wavelength in all cells spectra
            # Because the masked cell are computed exactly the same way as the cells used to build the off transit spectra, the invalid values are the same at a given cell and wavelength. Thus we can keep the same NaN_array for the computation of the mean on transit spectra
            # Nb_NaN_array = np.zeros(cell_spectra.shape)

            i_index = 0
            for i in range(-radius,radius+1):
                j_index = 0
                for j in range(-radius,radius+1):
                    r = np.sqrt( (i**2) + (j**2))
                    r /= radius
                    # if outside the stellar surface
                    if r > 1:
                        j_index += 1
                        continue
                    # if on a masked cell : remove it from spectra
                    # a cell is masked if it center lies within the radius of the planet over the stellar surface. /!\ i is index for rows, j for column ! a correspond to x coordinate, and _ya to y : we thus compare j (columns) to a and i (rows) to y_a !
                    elif (np.sqrt( ((j/radius) - a[obs])**2 + ((i/radius) - y_a[obs])**2 ) <= (self.transit_dic['Rp']/self.transit_dic['Rs'])):
                        # re-compute the corresponding cell to remove it from the mean spectra
                        # limb dark angle
                        alpha = np.arcsin(r)
                        mu = np.cos(alpha)

                        # spherical coordinate for RME
                        theta = np.arcsin(j / radius)
                        # inclination (0 = North Pole, pi/2 = equator)
                        phi = (np.pi/2) - np.arcsin(i / radius)

                        # doppler shift the spectra to account for stellar rotation (RME)
                        # see https://dynref.engr.illinois.edu/rvs.html for spherical coordinates/
                        Vrot = -1*vsini * np.sin(phi) * np.sin(theta)

                        # compute the stellar spectra of this cell
                        cell_spectra = total_interp([-1*Vrot,mu])[0,:] # need to invert the sign of Vrot to get the blue & redshift on correct side of the star

                        # incremente the nb of invalid values for each wavelength of the spectra
                        invalid_mask = ~np.isfinite(cell_spectra) # contains a 1 at each wavelength storing an invalid number
                        # Nb_NaN_array += invalid_mask
                        # replace invalid values in cell_spectra by 0, so that they will not be taken into account in the mean
                        cell_spectra[invalid_mask] = 0

                        # if the PYSME grid was generated with continuum fixed at 1 : apply a non-linear limb dark law on the continuum
                        if MyInterp.NormalizedByContinuum:
                            masked_spectra = limb_dark(mu)*cell_spectra
                        else:
                            masked_spectra = cell_spectra

                        on_transit_stellar_spec -= masked_spectra
                        nb_masked_cell += 1
                        j_index += 1
                    # if on stellar surface but not a masked cell : nothing happen
                    else:
                        j_index += 1
                        pass
                i_index += 1

            # here, invalid values were not removed from the off-transit spectra (because were set to 0), so they should not count in the soustraction of masked_cells in the mean)
            on_transit_stellar_spec /= (Nb_valid_cell - Nb_NaN_array - nb_masked_cell)
            # convolve to SPIRou resolution #
            wave_masked,on_transit_stellar_spec_convolved = convolve_SPIRou(MyInterp.synth_wave,on_transit_stellar_spec)
            # interpolate on data wavelengths
            final_disk_avg_on_spec[obs] = interp1d(wave_masked,on_transit_stellar_spec_convolved,bounds_error=False,fill_value=np.nan)(self.wave[obs]) # this is what we'll use to compute the deformation map

            # plot for verification
            if plots:
                if obs%5==0:
                    plt.figure()
                    plt.title(obs)
                    plt.plot(self.wave[obs,0],self.data[obs,0],label=f'data (earth RF ? Stellar RF ?)')
                    plt.plot(self.wave[obs,0],final_disk_avg_on_spec[obs,0]/np.median(final_disk_avg_on_spec[obs,0]),label=f'pysme in stellar RF')
                    plt.legend()

        # Then compute the correcting factor by dividing the on_transit by the off_transit
        stellar_correction = final_disk_avg_on_spec / final_disk_avg_off_spec
        stellar_correction[~np.isfinite(stellar_correction)] = 1. # set to 1. all pixels that are not defined
        stellar_correction[self.off_transit_mask] = 1. # set to 1. all off transit observations

        # Plots the transit trajectory with on/off transit observations²
        if plots:
            plt.figure(figsize=(10,10))
            plt.title('Planet transit trajectory')
            # star
            circle1 = plt.Circle((0, 0), 1, color='yellow')
            plt.gca().add_patch(circle1)
            # velocity map of stellar surface
            # plt.imshow(Vmap)
            # Pyasl trajectory
            # only show the orbital path at negativ z i.e on the visible side of the LOS
            plt.plot(a,y_a,'k+',label='Planet center')
            # add circles to show planet radius
            for obs in range(self.shape[0]):
                if obs in self.on_transit_mask:
                    plt.gca().add_patch(plt.Circle((a[obs], y_a[obs]), self.transit_dic['Rp']/self.transit_dic['Rs'], color='green',alpha=0.5,fill=False))
                else:
                    plt.gca().add_patch(plt.Circle((a[obs], y_a[obs]), self.transit_dic['Rp']/self.transit_dic['Rs'], color='red',alpha=0.5,fill=False))
            # arrow to show direction
            dx = a[self.on_transit_mask[1]] - a[self.on_transit_mask[0]]
            dy = y_a[self.on_transit_mask[1]] - y_a[self.on_transit_mask[0]]
            plt.arrow(a[self.on_transit_mask[0]],y_a[self.on_transit_mask[0]],dx,dy,width=0.005,color='black')
            plt.xlim(-1.1,1.1)
            plt.ylim(-1.1,1.1)
            plt.xlabel('x (stellar radii)')
            plt.ylabel('y (stellar radii)')
            plt.hlines(0,-1,1,'k',ls='--')
            plt.vlines(0,-1,1,'k',ls='--')
            plt.legend()

        return stellar_correction

    def StellarCorr_KeplerEll_Helium(self, radius=25, He_mask=True, plots = False, save = None, print_info = True):
        '''
        Divide by master spectrum corrected from RME and CLV effect using PYSME + W.Dethier code for the chromospheric He
        The master spectrum is here computed using all observations (on+off transit) for all wavelength, except for the He
        for which only the off transit observation are averaged together. This enhance the correction of the stellar lines,
        without removing the excess of He due to the planetary transit
        '''
        # We make sure that data have not been doppler shifted, as we need to do the fit in earth RF
        if 'doppler_shift' in self.history.keys(): raise NameError('Doppler shift detected in data history. This step must be done on data in Earth RF')

        # Current step ID
        step_ID = 'StellarCorr_KeplerEll_Helium'
        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')
        # Warn if step already found in history
        if(step_ID in self.history.keys()):
            warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            k = 0
            former_step_ID = step_ID
            while step_ID in self.history.keys():
                k+=1
                step_ID = former_step_ID+'_'+str(k)


        # check if compute_kepler_orbit() has been initialized
        if self.ke is None: raise NameError('You must initialize the KeplerEllipse orbit before using this method: DataSet.compute_kepler_orbit().')

        ## DO STUFF HERE ##
        # get the RME/CLV deformation map in stellar RF: it's already convolved at SPIRou resolution and sampled on SPIRou channels
        stellar_correction = self.Compute_StellarCorr_KeplerEll_Helium(radius,plots)
        # Shift the correction in the RF set as parameter ('Earth' or 'Stellar')
        stellar_correction_shift = np.ones(stellar_correction.shape)
        for order in range(self.shape[1]):
            for obs in self.on_transit_mask:
                stellar_correction_shift[obs,order] = interp1d(self.wave[obs,order]/(1+self.Vd[obs]/const.c.value), stellar_correction[obs,order], bounds_error=False)(self.wave[obs,order])

        if plots:
            plt.figure()
            for obs in self.on_transit_mask[::4]:
                plt.plot(self.wave[0,0,:],stellar_correction_shift[obs,0],label=f'{self.time_from_mid[obs]*24:.2f}')
            plt.plot(self.wave[0,0,:],(self.weighted_mean(self.data,transit_mask='off')[0]-1)/1e2+1,'-k',label='data/100')
            plt.legend()

        if He_mask:
            # compute master spectrum: in the case of He, we average all (on+off) transit observation except in a 0.1nm bin around the He line
            TLC = np.copy(self.transit_weight)
            He_pos = He_theo_wave[1] / (1+(self.Vtot/const.c.value)) # planetary He position shift in earth RF
            window_size = 0.1 # size, in nm, of the mask window around the theoritical telluric, stellar and He position
            He_mask = ~np.bool_(TLC*(np.abs(self.wave[:,0] - He_pos[:,None])<window_size).T).T # Mask is True every where except in the He bin
            # create the map of weights for the master spectrum computation, including the SNR² weight
            weights = self.SNR*self.SNR
            weights = weights[:,None] * He_mask # adding the wavelength axis [:,None] and multiply SNR with the boolean He mask
            weights = weights[:,None] # adding the order axis to match the shape of the data in case of He studies
            # compute master spectrum: mean of full transit spectra excluding the planetary He bin
            mean_data = self.weighted_mean(self.data,transit_mask="custom",custom_mask=weights)
        else:
            mean_data = self.weighted_mean(self.data,transit_mask="off")
        # apply the stellar RME/CLV correction to the master spectrum
        mean_data = stellar_correction_shift * mean_data[None,...]
        # correct data
        self.data /= mean_data
        ###################

        if print_info : print('\StellarCorr_KeplerEll done !')

        # Save in history
        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
        if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")

        else :
            self.history[step_ID] = {'parameters' : parameters}

    def Compute_StellarCorr_KeplerEll_Helium(self, radius=25, plots = False):
        '''
        Same as Compute_StellarCorr_KeplerEll but also adding the chromospheric helium
        Code for computing He line profile is from W.Dethier
        We fit the He line profile on the master off-transit spectrum from the data, with 2 free parameters (temperature & column density)
        Once the best parameters are found, we multiply the He line profile to the PYSME spectra (so we use the same relative intensity for the He than for the photospheric lines, for exemple for the limb darkening)
        '''
        # load stellar interpoler
        print('Loading the Stellar Interpoler object :')
        with open(self.transit_dic['stellar_interpoler_path'],'rb') as file_in:
            MyInterp = pickle.load(file_in)
            total_interp = MyInterp.Interpoler
        print(f'Interpoler loaded, continuum normalised = {MyInterp.NormalizedByContinuum}')

        def limb_dark(mu):
            # see if the limb dark law is uniform or not
            if len(self.transit_dic['c'])==4:
                c1,c2,c3,c4 = self.transit_dic['c']
                limb_avg = 1-c1-c2-c3-c4+0.8*c1+(2./3.)*c2+(4./7.)*c3+0.5*c4
                limb = limb_avg * (1-c1*(1-np.sqrt(mu))-c2*(1-mu)-c3*(1-np.power(mu,3/2))-c4*(1-mu**2))
            elif len(self.transit_dic['c']==0):
                limb = 1
            else:
                raise NameError(f'Limb dark function with coeff {self.transit_dic["c"]} have not been implemented in this function : need to update using formula in http://lkreidberg.github.io/batman/docs/html/tutorial.html#limb-darkening-options')

            return(limb)

        ###################################################################
        ### FITTING STELLAR CHROMOSPHERIC HE LINE PROFILE ON MASTER-OUT ###
        ###################################################################

        delta_d = 1000 # He damping factor: don't know what to do with it, as William set it to 1 or 1000 depending on the stars... its 1000 for HD189733
        # From testing, there is no difference btw delta_d = 1 & 1000
        # Setting it to 0 will use Gaussian line profile

        choice = [True,True,True] # set all True to include all lines of the He triplet

        # prepare some figures for plotting
        fig1,ax1 = plt.subplots(1,1)
        colors1 = ['b','g','r']
        # fig2,ax2 = plt.subplots(1,1)
        # fig3,ax3 = plt.subplots(1,1)
        # fig4,ax4 = plt.subplots(1,1)

        # Now, we write a function that takes the colonne density n and temperature T as parameters, and return the corresponding disk-averaged stellar spectrum
        # after convolution to SPIRou resolution & normalisation in the He region
        # We will call this function multiple time to fit the He model to the data
        def disk_avg_stellar_spectrum(T,n,radius):
            # n: cm-2
            # T: K
            # radius: radius in cells of the stellar grid. Set a small number (e.g ~10) for faster fitting (n,T), then use a bigger nb (e.g 25) for the final spectrum
            vsini = MyInterp.vsini # rotational velocity v.sin(i) of the stellar surface [m/s] (ref https://iopscience.iop.org/article/10.3847/1538-3881/153/1/21 for HD189733)
            # Instead of building the full grid, compute the off-transit spectra by summing up each cell spectra and immediatly free the memory of this cell after
            N_cell = radius*2+1 # total nb of cell along an axis (+1 to account for the center cell)
            Vmap = np.zeros((N_cell,N_cell))

            off_transit_cells_sum = np.zeros(MyInterp.synth_wave.size)

            Nb_valid_cell = 0 # count the nb of cells that are not nan. Used for the mean
            cell_spectra = total_interp([0,0])[0,:] # a dummy cell just to get the right size of a cell
            Nb_NaN_array = np.zeros(cell_spectra.shape) # a vector that count the nb of nan at each wavelength in all cells spectra

            wave = MyInterp.synth_wave*10 # e-10m

            He_stellar_flux = He_triplet_line_wav(wave,T,n,delta_d,choice)

            #### Compute the disk-averaged spectrum ###
            k=0
            index_i = 0
            ax1_counter = 0 # for plots
            for i in range(-radius,radius+1):
                index_j = 0
                for j in range(-radius,radius+1):
                    k+=1
                    print(f'\r{k} / {(N_cell)**2}',end='',flush=True)
                    # get mu corresponding to current cell
                    r = np.sqrt( (i**2) + (j**2))
                    r /= radius
                    # if outside the stellar surface
                    if r > 1:
                        index_j+=1
                        continue
                    else:
                        # limb dark angle
                        alpha = np.arcsin(r)
                        mu = np.cos(alpha)

                        # spherical coordinate for RME
                        theta = np.arcsin(j / radius)
                        # inclination (0 = North Pole, pi/2 = equator)
                        phi = (np.pi/2) - np.arcsin(i / radius)

                        # doppler shift the spectra to account for stellar rotation (RME)
                        # see https://dynref.engr.illinois.edu/rvs.html for spherical coordinates/
                        Vrot = -1*vsini * np.sin(phi) * np.sin(theta)
                        Vmap[index_i,index_j] = Vrot

                        # compute the stellar spectra of this cell
                        cell_spectra = total_interp([-1*Vrot,mu])[0,:] # need to invert the sign of Vrot to get the blue & redshift on correct side of the star

                        # incremente the nb of invalid values for each wavelength of the spectra
                        invalid_mask = ~np.isfinite(cell_spectra) # contains a 1 at each wavelength storing an invalid number
                        Nb_NaN_array += invalid_mask
                        # replace invalid values in cell_spectra by 0, so that they will not be taken into account in the mean
                        cell_spectra[invalid_mask] = 0

                        # plot a few individual local spectra before adding He
                        if plots:
                            if (i==0) and (j in [-radius//2,0,radius//2]):
                                ax1.plot(MyInterp.synth_wave,cell_spectra,ls='--',label=fr'No He,    $\theta$ = {theta}',c=colors1[ax1_counter])

                        # --------
                        # Add the Helium triplet: we shift the line by Vrot and multiply to get the same relative flux with limb darkening
                        # shift: we set a "+" on velocity to have the same shift as for the rest of the spectrum
                        He_stellar_flux_shift = interp1d(MyInterp.synth_wave/(1+Vrot/const.c.value),He_stellar_flux,fill_value=1,bounds_error=False)(MyInterp.synth_wave)
                        # multiply
                        cell_spectra*=He_stellar_flux_shift
                        # --------

                        # plot a few individual local spectra after adding He
                        if plots:
                            if (i==0) and (j in [-radius//2,0,radius//2]):
                                ax1.plot(MyInterp.synth_wave,cell_spectra,ls='--',label=fr'With He, $\theta$ = {theta}',c=colors1[ax1_counter])
                                ax1_counter += 1

                        # if the PYSME grid was generated with continuum fixed at 1 : apply a non-linear limb dark law on the continuum
                        if MyInterp.NormalizedByContinuum:
                            off_transit_cells_sum += limb_dark(mu)*cell_spectra
                        else:
                            off_transit_cells_sum += cell_spectra

                        Nb_valid_cell +=1
                        index_j+=1

                index_i += 1

            # compute the mean by dividing each sum up wavelength by the nb of cells minus the nb of invalid values encounter at this specific wavelength
            off_transit_stellar_spec = off_transit_cells_sum / (Nb_valid_cell - Nb_NaN_array)

            ### Shift in Earth RF ###
            # the disk-integrated spectrum computed with data is averaged on off-transits obs. BERV and RV of star have thus an influence on the master spectrum computed with data,
            # that we reproduce in the PYSME spectrum by averaging PYSME spectrum shifted accordingly for each off transit observation in the data
            off_transit_stellar_spec_shift = []
            for obs in self.off_transit_mask:
                pysme_shifted_spec = interp1d(MyInterp.synth_wave/(1+self.Vd[obs,None]/const.c.value), off_transit_stellar_spec, bounds_error=False)(MyInterp.synth_wave)
                off_transit_stellar_spec_shift.append(pysme_shifted_spec)
            off_transit_stellar_spec_shift = np.array(off_transit_stellar_spec_shift) # 2D matrix, with 1 shifted PYSME spectra in Earth RF per off-transit data observation
            off_transit_stellar_spec_shift_mean = np.nanmean(off_transit_stellar_spec_shift,axis=0) # average the shifted PYSME spectra to reproduce the off-transit average of observations in Earth RF on data side

            ### convolve to SPIRou resolution ###
            wave_masked,off_transit_stellar_spec_convolved = convolve_SPIRou(MyInterp.synth_wave,off_transit_stellar_spec_shift_mean)
            print()
            return(wave_masked,off_transit_stellar_spec_convolved)

        # This is the function we call for the fit: it calls the computation of the disk averaged off-transit spectrum, and prepare the data for the fit (normalisation and )
        def fitting_function(wave,T,n,mult_const,offset,radius):
            # just calling the above function with correct radius, plus extraction He on the data wavelenghts and normalising
            wave_masked,off_transit_stellar_spec_convolved  = disk_avg_stellar_spectrum(T,n,radius)
            # interpolate on data wavelengths
            disk_avg_spec = interp1d(wave_masked,off_transit_stellar_spec_convolved)(wave)
            # normalise by median
            disk_avg_spec /= np.median(disk_avg_spec)
            # center at 0 for fitting
            disk_avg_spec -= 1
            # multiply by a constant and add offset
            disk_avg_spec = mult_const*(1+disk_avg_spec)-1+offset
            return(disk_avg_spec)

        # Let's fit !

        # # First, we fit the entire spectrum with a multiplicative constant and an offset as free parameters, without adding He
        # # this is to ensure that the other PYSME lines are as close as possible to the data
        # # Multiplicativ constant won't work: trying with just an offset
        # def f(x,A,B):
        #     # Compute PYSME disk averaged spectrum in Earth RF without He
        #     spec = fitting_function(x,T=1e3,n=0,mult_const=A,offset=B,radius=10)
        #     return(spec)
        #
        # # set data
        # xdata = self.wave[0,0]
        # ydata = self.weighted_mean(self.data,transit_mask='off')[0] -1  # center around 0 for fit !!

        # Now fit the He
        def f2(x,T,n):
            # Compute PYSME disk averaged spectrum in Earth RF, we use a smaller radius for faster fit: don't go below 10 cells as the resulting spectrum will not be precise enough
            spec = fitting_function(x,T,n,mult_const=1,offset=0,radius=10)
            return(spec)

        # set data
        xdata = self.wave[0,0,:]
        mask = (xdata>1080)*(xdata<1085)
        xdata = xdata[mask]
        ydata = self.weighted_mean(self.data,transit_mask='off')[0,mask] -1  # center around 0 for fit !!

        p0  = (self.transit_dic['Teff'],1e10)
        bounds=[(1e2,0),(np.inf,np.inf)]
        print('Fitting the chromospheric stellar He line to the data')
        sol_Tn = optimize.curve_fit(f2,xdata,ydata,p0,bounds=bounds)
        T,n = sol_Tn[0]

        print()
        print(f'Fitting He on master-out done, best fit: T={T:.0f}K, n={n:.2e}cm^-2')

        # This is the final He line profile that we'll incorporate in PYSME spectra
        He_stellar_flux = He_triplet_line_wav(MyInterp.synth_wave*10,T,n,delta_d,choice) # wave should be in Angstroms

        if plots:
            print('Plotting the before/after He incorporation in PYSME')
            pysme_spec = fitting_function(xdata,T,n,mult_const=1,offset=0,radius=10)
            pysme_spec_noHe = fitting_function(xdata,T,n=0,mult_const=1,offset=0,radius=10)

            plt.figure()
            plt.plot(xdata,ydata,label="data")
            plt.plot(xdata,pysme_spec_noHe,label="pysme before He fit")
            plt.plot(xdata,pysme_spec,label="pysme after He fit")
            plt.legend()

        ######################################################################
        ### COMPUTING THE STELLAR DEFORMATION MAP INCLUDING THE STELLAR HE ###
        ######################################################################

        vsini = MyInterp.vsini # rotational velocity v.sin(i) of the stellar surface [m/s] (ref https://iopscience.iop.org/article/10.3847/1538-3881/153/1/21 for HD189733)

        # Instead of building the full grid, compute the off-transit spectra by summing up each cell spectra and immediatly free the memory of this cell after
        N_cell = radius*2+1 # total nb of cell along an axis (+1 to account for the 0)
        Vmap = np.zeros((N_cell,N_cell))

        print(f'Computing off-transit stellar spectra on a grid of {N_cell} x {N_cell} cells :')

        off_transit_cells_sum = np.zeros(MyInterp.synth_wave.size)
        Nb_valid_cell = 0 # count the nb of cells that are not nan. Used for the mean
        cell_spectra = total_interp([0,0])[0,:] # a dummy cell just to get the right size of a cell
        Nb_NaN_array = np.zeros(cell_spectra.shape) # a vector that count the nb of nan at each wavelength in all cells spectra

        # plot for verification
        if plots:
            plt.figure()

        k=0
        index_i = 0
        for i in range(-radius,radius+1):
            index_j = 0
            for j in range(-radius,radius+1):
                k+=1
                print(f'\r{k} / {(N_cell)**2}',end='',flush=True)
                # get mu corresponding to current cell
                r = np.sqrt( (i**2) + (j**2))
                r /= radius
                # if outside the stellar surface
                if r > 1:
                    index_j+=1
                    continue
                else:
                    # limb dark angle
                    alpha = np.arcsin(r)
                    mu = np.cos(alpha)

                    # spherical coordinate for RME
                    theta = np.arcsin(j / radius)
                    # inclination (0 = North Pole, pi/2 = equator)
                    phi = (np.pi/2) - np.arcsin(i / radius)

                    # doppler shift the spectra to account for stellar rotation (RME)
                    # see https://dynref.engr.illinois.edu/rvs.html for spherical coordinates/
                    Vrot = -1*vsini * np.sin(phi) * np.sin(theta)
                    Vmap[index_i,index_j] = Vrot

                    # compute the stellar spectra of this cell
                    cell_spectra = total_interp([-1*Vrot,mu])[0,:] # need to invert the sign of Vrot to get the blue & redshift on correct side of the star
                    # incremente the nb of invalid values for each wavelength of the spectra
                    invalid_mask = ~np.isfinite(cell_spectra) # contains a 1 at each wavelength storing an invalid number
                    Nb_NaN_array += invalid_mask
                    # replace invalid values in cell_spectra by 0, so that they will not be taken into account in the mean
                    cell_spectra[invalid_mask] = 0

                    # --------
                    # Add the Helium triplet: we shift the line by Vrot and multiply to get the same relative flux with limb darkening
                    # shift: we set a "+" on velocity to have the same shift as for the rest of the spectrum
                    He_stellar_flux_shift = interp1d(MyInterp.synth_wave/(1+Vrot/const.c.value),He_stellar_flux,fill_value=1,bounds_error=False)(MyInterp.synth_wave)
                    # multiply
                    cell_spectra*=He_stellar_flux_shift
                    # --------

                    # if the PYSME grid was generated with continuum fixed at 1 : apply a non-linear limb dark law on the continuum
                    if MyInterp.NormalizedByContinuum:
                        off_transit_cells_sum += limb_dark(mu)*cell_spectra
                    else:
                        off_transit_cells_sum += cell_spectra

                    Nb_valid_cell +=1
                    index_j+=1

                # plot for verification
                if plots:
                    if index_i==25 and index_j%5==0: plt.plot(MyInterp.synth_wave,cell_spectra,label=f'theta={theta:.2f}')

            index_i += 1

        if plots:
            plt.figure()
            plt.imshow(Vmap)

        # compute the mean by dividing each sum up wavelength by the nb of cells minus the nb of invalid values encounter at this specific wavelength
        off_transit_stellar_spec = off_transit_cells_sum / (Nb_valid_cell - Nb_NaN_array)
        # convolve to SPIRou resolution #
        wave_masked,off_transit_stellar_spec_convolved = convolve_SPIRou(MyInterp.synth_wave,off_transit_stellar_spec)
        # interpolate on SPIRou sampling wavelength
        interpoler = interp1d(wave_masked,off_transit_stellar_spec_convolved,bounds_error=False,fill_value=np.nan)
        final_disk_avg_off_spec = interpoler(self.wave)

        print('\nNow applying the correction on each observations :')

        # Compute the cells crossed by the planet during transit using pyasL.KeplerEllipse model
        pos = self.ke.xyzPos(self.time_from_mid)
        # x & y coordinate of the planet trajectory during transit, normalized in stellar radii
        a = pos[:,0]/self.transit_dic['Rs']
        y_a = pos[:,1]/self.transit_dic['Rs']

        final_disk_avg_on_spec = np.ones(self.shape)

        for obs in self.on_transit_mask:
            # get the masked cells for a given obs

            print(f'\r{obs-self.on_transit_mask[0]+1}/{len(self.on_transit_mask)}',end='',flush=True) # only show counter for on transit computation because the off-transit  is way faster
            # build the mask
            masked_cell = np.ones((N_cell,N_cell),dtype='bool')
            nan_mask = np.ones((N_cell,N_cell),dtype='bool')

            # for each masked cell : remove it from the sum of off-transit cells
            on_transit_stellar_spec = np.copy(off_transit_cells_sum)
            nb_masked_cell = 0

            # a vector that count the nb of nan at each wavelength in all cells spectra
            # Because the masked cell are computed exactly the same way as the cells used to build the off transit spectra, the invalid values are the same at a given cell and wavelength. Thus we can keep the same NaN_array for the computation of the mean on transit spectra
            # Nb_NaN_array = np.zeros(cell_spectra.shape)

            i_index = 0
            for i in range(-radius,radius+1):
                j_index = 0
                for j in range(-radius,radius+1):
                    r = np.sqrt( (i**2) + (j**2))
                    r /= radius
                    # if outside the stellar surface
                    if r > 1:
                        j_index += 1
                        continue
                    # if on a masked cell : remove it from spectra
                    # a cell is masked if it center lies within the radius of the planet over the stellar surface. /!\ i is index for rows, j for column ! a correspond to x coordinate, and _ya to y : we thus compare j (columns) to a and i (rows) to y_a !
                    elif (np.sqrt( ((j/radius) - a[obs])**2 + ((i/radius) - y_a[obs])**2 ) <= (self.transit_dic['Rp']/self.transit_dic['Rs'])):
                        # re-compute the corresponding cell to remove it from the mean spectra
                        # limb dark angle
                        alpha = np.arcsin(r)
                        mu = np.cos(alpha)

                        # spherical coordinate for RME
                        theta = np.arcsin(j / radius)
                        # inclination (0 = North Pole, pi/2 = equator)
                        phi = (np.pi/2) - np.arcsin(i / radius)

                        # doppler shift the spectra to account for stellar rotation (RME)
                        # see https://dynref.engr.illinois.edu/rvs.html for spherical coordinates/
                        Vrot = -1*vsini * np.sin(phi) * np.sin(theta)

                        # compute the stellar spectra of this cell
                        cell_spectra = total_interp([-1*Vrot,mu])[0,:] # need to invert the sign of Vrot to get the blue & redshift on correct side of the star

                        # --------
                        # Add the Helium triplet: we shift the line by Vrot and multiply to get the same relative flux with limb darkening
                        # shift: we set a "+" on velocity to have the same shift as for the rest of the spectrum
                        He_stellar_flux_shift = interp1d(MyInterp.synth_wave/(1+Vrot/const.c.value),He_stellar_flux,fill_value=1,bounds_error=False)(MyInterp.synth_wave)
                        # multiply
                        cell_spectra*=He_stellar_flux_shift
                        # --------

                        # incremente the nb of invalid values for each wavelength of the spectra
                        invalid_mask = ~np.isfinite(cell_spectra) # contains a 1 at each wavelength storing an invalid number
                        # Nb_NaN_array += invalid_mask
                        # replace invalid values in cell_spectra by 0, so that they will not be taken into account in the mean
                        cell_spectra[invalid_mask] = 0

                        # if the PYSME grid was generated with continuum fixed at 1 : apply a non-linear limb dark law on the continuum
                        if MyInterp.NormalizedByContinuum:
                            masked_spectra = limb_dark(mu)*cell_spectra
                        else:
                            masked_spectra = cell_spectra

                        on_transit_stellar_spec -= masked_spectra
                        nb_masked_cell += 1
                        j_index += 1
                    # if on stellar surface but not a masked cell : nothing happen
                    else:
                        j_index += 1
                        pass
                i_index += 1

            # here, invalid values were not removed from the off-transit spectra (because were set to 0), so they should not count in the soustraction of masked_cells in the mean)
            on_transit_stellar_spec /= (Nb_valid_cell - Nb_NaN_array - nb_masked_cell)
            # convolve to SPIRou resolution #
            wave_masked,on_transit_stellar_spec_convolved = convolve_SPIRou(MyInterp.synth_wave,on_transit_stellar_spec)
            # interpolate on data wavelengths
            final_disk_avg_on_spec[obs] = interp1d(wave_masked,on_transit_stellar_spec_convolved,bounds_error=False,fill_value=np.nan)(self.wave[obs]) # this is what we'll use to compute the deformation map

            # plot for verification
            if plots:
                if obs%5==0:
                    plt.figure()
                    plt.title(obs)
                    plt.plot(self.wave[obs,0],self.data[obs,0],label=f'data (earth RF)')
                    plt.plot(self.wave[obs,0],final_disk_avg_on_spec[obs,0]/np.median(final_disk_avg_on_spec[obs,0]),label=f'pysme in stellar RF')
                    plt.legend()

        # Then compute the correcting factor by dividing the on_transit by the off_transit
        stellar_correction = final_disk_avg_on_spec / final_disk_avg_off_spec
        stellar_correction[~np.isfinite(stellar_correction)] = 1. # set to 1. all pixels that are not defined

        # Plots the transit trajectory with on/off transit observations²
        if plots:
            plt.figure(figsize=(10,10))
            plt.title('Planet transit trajectory')
            # star
            circle1 = plt.Circle((0, 0), 1, color='yellow')
            plt.gca().add_patch(circle1)
            # velocity map of stellar surface
            # plt.imshow(Vmap)
            # Pyasl trajectory
            # only show the orbital path at negativ z i.e on the visible side of the LOS
            plt.plot(a,y_a,'k+',label='Planet center')
            # add circles to show planet radius
            for obs in range(self.shape[0]):
                if obs in self.on_transit_mask:
                    plt.gca().add_patch(plt.Circle((a[obs], y_a[obs]), self.transit_dic['Rp']/self.transit_dic['Rs'], color='green',alpha=0.5,fill=False))
                else:
                    plt.gca().add_patch(plt.Circle((a[obs], y_a[obs]), self.transit_dic['Rp']/self.transit_dic['Rs'], color='red',alpha=0.5,fill=False))
            # arrow to show direction
            dx = a[self.on_transit_mask[1]] - a[self.on_transit_mask[0]]
            dy = y_a[self.on_transit_mask[1]] - y_a[self.on_transit_mask[0]]
            plt.arrow(a[self.on_transit_mask[0]],y_a[self.on_transit_mask[0]],dx,dy,width=0.005,color='black')
            plt.xlim(-1.1,1.1)
            plt.ylim(-1.1,1.1)
            plt.xlabel('x (stellar radii)')
            plt.ylabel('y (stellar radii)')
            plt.hlines(0,-1,1,'k',ls='--')
            plt.vlines(0,-1,1,'k',ls='--')
            plt.legend()

        return stellar_correction

    def compute_kepler_orbit_old(self,show=False):
        '''
        Uses pyasl.KeplerEllipse to compute the planetary elliptical orbit and store it as an attribute
        transit (dic) : a dictionnary containing the informations of the transit (see transit_info.py)
        more information on pyasl.KeplerEllipse definition : https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/keplerOrbitAPI.html

        position and velocity can then be access with respect to time using self.ke.xyzPos(time) & self.xyzVel(time) for a given time.
        The unity corresponds to those provided in the transit dictionnary, e.g if transit['a'] is in m and transit['Porb'] in s, coordinates will be in m & velocity in m/s

        In order to align the origin of time for the KeplerEllipse orbit, we compute the time of mid transit corresponding to the model and store it in self.model_midpoint.
        Thus, time from mid transit can be convert from data_set time to keplerEllipse time using self.time_from_mid+self.model_midpoint

        if show = True, show the kepler orbit of the planet using pyasl.KeplerEllipse class.
        It spawns 4 figures:
            - 3D model of the planet
            - planet RV
            - planet coordinates
            - planet transit trajectory in front of the stellar surface
        '''

        # Set the model
        a = self.transit_dic['a'] # m
        per = self.transit_dic['Porb'] # BJDTBD
        e = self.transit_dic['e']
        Omega = self.transit_dic['lbda'] # °
        w = self.transit_dic['w'] # ° , in case of circular orbit, w is -90 if define in the planetary orbit, and +90 if define in the stellar orbit (see above documentation)
        i = self.transit_dic['i'] # °
        
        # the KeplerEllipse object is stored as an attribute
        self.ke = pyasl.KeplerEllipse(a=a, per=per, e=e, Omega=Omega, w=w, i=i, ks=pyasl.MarkleyKESolver)
        
        # find the mid-transit time corresponding to the models, defined as the time at which the modeled position is the closest to the stellar center. This time is found usnig scipy.minimize
        def f(time):
            pos = self.ke.xyzPos(time)
            r = np.sqrt(pos[:,0]**2 + pos[:,1]**2 )/self.transit_dic['Rs']
            return(r)
        # time in BJDTBD
        time = np.linspace(0,per,1000)
        pos = self.ke.xyzPos(time)
        # find model midpoint i.e corresponding to the minimal distance to the stellar surface in the xy plan
        # only select point during primary transit
        r = np.sqrt(pos[:,0]**2 + pos[:,1]**2 )/self.transit_dic['Rs'] # distance from center of star during primary transit in stellar radii
        m = np.logical_and(pos[:,2]<=0, r<=1.0) # z negative = planet in front of star, sqrt(x**2+y**2)<Rs = planet crossing stellar surface
        result = optimize.minimize(f, x0=time[m][m.sum()//2])
        self.model_midpoint = result.x # BJDTBD

        if show:
            time = np.linspace(0,per,1000)
            phase = (time-self.model_midpoint) / self.transit_dic['Porb']
            pos = self.ke.xyzPos(time)
            vel = self.ke.xyzVel(time)/(24*3600*1e3) # self.transit_dic['Porb'] is in days, so by default self.ke.xyzVel is in m/days. We convert it to km / s
            # Planet trajectory during transit in front of stellar surface
            plt.figure(figsize=(5,5))
            plt.title('Planet transit trajectory')
            plt.plot(pos[:,0]/self.transit_dic['Rs'],pos[:,1]/self.transit_dic['Rs'])
            circle1 = plt.Circle((0, 0), 1, color='r')
            plt.gca().add_patch(circle1)
            plt.xlim(-1.1,1.1)
            plt.ylim(-1.1,1.1)
            plt.xlabel('x (stellar radii)')
            plt.ylabel('y (stellar radii)')
            # Planet 3 coordinates wrt phase
            plt.figure()
            plt.title('Planet coordinates')
            l = ['x','y','z']
            for i in range(3):
                plt.plot(phase,pos[:,i]/const.au.value,label=l[i])
            plt.plot(phase,np.sqrt(np.sum(pos**2,axis=1))/const.au.value,label=r'|$\vec{r}$|')
            plt.xlabel('phase')
            plt.ylabel('Position (ua)')
            plt.legend()
            # Planet 3 RV coordinates wrt phase
            plt.figure()
            plt.title('Planet velocity component')
            l = ['vx','vy','vz (velocity along LOS)']
            for i in range(3):
                plt.plot(phase,vel[:,i],label=l[i])
            plt.plot(phase,np.sqrt(np.sum(vel**2,axis=1)),label=r'|$\vec{v}$|')
            plt.hlines(self.transit_dic['Kp']/1e3,phase[0],phase[-1],'k',label='planet\'s Kp value')
            plt.legend()
            plt.xlabel('phase')
            plt.ylabel('Velocity (km/s)')
            # planet 3D orbit
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title('Planet orbit 3D')
            ax.plot(pos[:,0]/const.au.value,pos[:,1]/const.au.value,pos[:,2]/const.au.value,label='orbit')
            ax.set_xlabel('x (au)')
            ax.set_ylabel('y (au)')
            ax.set_zlabel('z (au)')
            # plot the star
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            r = self.transit_dic['Rs'] / const.au.value
            x = r * np.outer(np.cos(u), np.sin(v))
            y = r * np.outer(np.sin(u), np.sin(v))
            z = r * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='r')
            ax.quiver(0,0,0,0,0,-0.01,length=1.0,label='Toward observer',color='k')
            # preserve aspect
            ax.set_xlim(-self.transit_dic['a']/const.au.value,self.transit_dic['a']/const.au.value)
            ax.set_ylim(-self.transit_dic['a']/const.au.value,self.transit_dic['a']/const.au.value)
            ax.set_zlim(-self.transit_dic['a']/const.au.value,self.transit_dic['a']/const.au.value)
            plt.legend()

        print('Compute Kepler orbit done')

    def compute_kepler_orbit(self,show=False):
        '''
        Uses radvel & pyasl.KeplerEllipse to compute the planetary elliptical orbit and store it as an attribute
        transit (dic) : a dictionnary containing the informations of the transit (see transit_info.py)
        more information on pyasl.KeplerEllipse definition : https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/keplerOrbitAPI.html

        position and velocity can then be access with respect to time using self.ke.xyzPos(time) & self.xyzVel(time) for a given time from mid-transit.
        To ensure correct time alignement with mid-transit, the time of periastron is computed with radvel and set in the Kepler Ellipse model.

        The unity corresponds to those provided in the transit dictionnary, e.g if transit['a'] is in m and transit['Porb'] in s, coordinates will be in m & velocity in m/s

        if show = True, show the kepler orbit of the planet using pyasl.KeplerEllipse class.
        It spawns 4 figures:
            - 3D model of the planet
            - planet RV
            - planet coordinates
            - planet transit trajectory in front of the stellar surface
        '''

        # Set the model
        a = self.transit_dic['a'] # m
        per = self.transit_dic['Porb'] # BJDTBD
        e = self.transit_dic['e']
        Omega = self.transit_dic['lbda'] # °
        w_star = self.transit_dic['w'] # in (°). KeplerEllipse uses the stellar 'w' to define the planet's orbit, so using directly the one in the transit dictionnary.
        # In case of circular orbit, KeplerEllipse assume w = -90° if defines the planetary orbit, and +90° if defines in the stellar orbit (see above documentation)
        w_planet = w_star + 180 # 'w' of the planet: is only used to get the planet's time of periastron. Radvel & Kepler Ellipse otherwise always use the stellar 'w' by default
        i = self.transit_dic['i'] # ° inclination
        
        # Use radvel to find time of periastron passage
        t_periastron = radvel.orbit.timetrans_to_timeperi(self.midpoint,   # time from mid (BJD-TDB) is converted to time from periastron by radvel
                                                        per, # days
                                                        e,
                                                        np.radians(w_planet)) # radians. This is the only function that requires the 'w' of the planet instead of the star (check radvel_test.ipynb notebook)

        # Initialise the pyasl's KeplerEllipse class
        # -> Since we initialized the KeplerEllipse with the time of periastron (computed with radvel from the time of mid-transit), we can now use the absolute time (self.time_vector) to find the planet's RV and position in the orbit
        self.ke = pyasl.KeplerEllipse(a=a, per=per, e=e, Omega=Omega, w=w_star, i=i, tau=t_periastron)

        if show:
            # show graphics for a full period centered at mid-transit
            time = np.linspace(-per/2,per/2,1000)
            phase = (time) / self.transit_dic['Porb']
            pos = self.ke.xyzPos(time)
            # velocity: add a -1 factor so that velocity is negative when planet is moving toward us
            vel = -1*self.ke.xyzVel(time)/(24*3600*1e3) # self.transit_dic['Porb'] is in days, so by default self.ke.xyzVel is in m/days. We convert it to km / s
            # Planet trajectory during transit in front of stellar surface
            plt.figure(figsize=(5,5))
            plt.title('Planet transit trajectory')
            plt.plot(pos[:,0]/self.transit_dic['Rs'],pos[:,1]/self.transit_dic['Rs'])
            circle1 = plt.Circle((0, 0), 1, color='r')
            plt.gca().add_patch(circle1)
            plt.xlim(-1.1,1.1)
            plt.ylim(-1.1,1.1)
            plt.xlabel('x (stellar radii)')
            plt.ylabel('y (stellar radii)')
            # Planet 3 coordinates wrt phase
            plt.figure()
            plt.title('Planet coordinates')
            l = ['x','y','z']
            for i in range(3):
                plt.plot(phase,pos[:,i]/const.au.value,label=l[i])
            plt.plot(phase,np.sqrt(np.sum(pos**2,axis=1))/const.au.value,label=r'|$\vec{r}$|')
            plt.xlabel('phase')
            plt.ylabel('Position (ua)')
            plt.legend()
            # Planet 3 RV coordinates wrt phase
            plt.figure()
            plt.title('Planet velocity component')
            l = ['vx','vy','vz (velocity along LOS)']
            for i in range(3):
                plt.plot(phase,vel[:,i],label=l[i])
            # Add the Vp as computed manually by the dataset
            Vp, Vtot = self.compute_RV(self.time_from_mid)
            plt.plot(self.time_from_mid/self.transit_dic['Porb'],Vp/1e3,label='RV in stellar, manual',ls='--')
            plt.plot(self.time_from_mid/self.transit_dic['Porb'],Vtot/1e3,label='RV in Earth, manual',ls='--')
            plt.plot(phase,np.sqrt(np.sum(vel**2,axis=1)),label=r'|$\vec{v}$|')
            plt.hlines(self.transit_dic['Kp']/1e3,phase[0],phase[-1],'k',label='planet\'s Kp value')
            plt.legend()
            plt.xlabel('phase')
            plt.ylabel('Velocity (km/s)')
            # planet 3D orbit
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title('Planet orbit 3D')
            ax.plot(pos[:,0]/const.au.value,pos[:,1]/const.au.value,pos[:,2]/const.au.value,label='orbit')
            ax.set_xlabel('x (au)')
            ax.set_ylabel('y (au)')
            ax.set_zlabel('z (au)')
            # plot the star
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            r = self.transit_dic['Rs'] / const.au.value
            x = r * np.outer(np.cos(u), np.sin(v))
            y = r * np.outer(np.sin(u), np.sin(v))
            z = r * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='r')
            ax.quiver(0,0,0,0,0,-0.01,length=1.0,label='Toward observer',color='k')
            # preserve aspect
            ax.set_xlim(-self.transit_dic['a']/const.au.value,self.transit_dic['a']/const.au.value)
            ax.set_ylim(-self.transit_dic['a']/const.au.value,self.transit_dic['a']/const.au.value)
            ax.set_zlim(-self.transit_dic['a']/const.au.value,self.transit_dic['a']/const.au.value)
            plt.legend()

        print('Compute Kepler orbit done')

    def compute_transit_window(self, show=False):

        # Batman's parameters
        params              = batman.TransitParams()                                    # object to store transit parameters
        params.t0           = 0.                                                        # time of inferior conjunction
        params.per          = self.transit_dic['Porb']                               # orbital period
        params.rp           = (self.transit_dic['Rp'] / self.transit_dic['Rs'])   # planet radius (in units of stellar radii)
        params.a            = (self.transit_dic['a']  / self.transit_dic['Rs'])   # semi-major axis (in units of stellar radii)
        params.inc          = self.transit_dic['i']                                  # orbital inclination (in degrees)
        params.ecc          = self.transit_dic['e']                                  # eccentricity
        params.w            = self.transit_dic['w']                                  # longitude of periastron (in degrees)
        params.limb_dark    = 'nonlinear'                                               # limb darkening model
        params.u            = self.transit_dic['c']                                  # limb darkening coefficients [u1, u2, u3, u4]
        if len(params.u)==0:
                params.limb_dark = 'uniform'
        elif len(params.u)==2:
            params.limb_dark = 'quadratic'
        # load transit weights with batman
        # initializes model
        self.batman_model = batman.TransitModel(params, self.time_from_mid)
        # calculates light curve
        self.light_curve = self.batman_model.light_curve(params)
        # transit window's weight
        W = (1-self.light_curve)
        W /= W.max()
        # following Bruno's indication : divide the weight by the average kimb darkening (because max limb darkening reached at mid transit is higher than 1) :
        if params.limb_dark == 'nonlinear':
            c1,c2,c3,c4 = params.u
            limb_avg = 1-c1-c2-c3-c4+0.8*c1+(2./3.)*c2+(4./7.)*c3+0.5*c4 # for non linear limb darkening only !
            W /= limb_avg
        self.transit_weight = W

        if show:
            plt.figure()
            plt.plot(W)
            plt.figure()
            plt.plot(self.light_curve)

        print('Compute transit window weight orbit done')

    def compute_RV(self, time, Kp=None, V0=0):
        '''
        Compute the planet's RV for the given absolute time (in days BJD-TDB) in the Stellar & Earth RF
        And for arbitrary Kp & V0 values (in m/s) if specified
        Takes:
            - absolute time vector (BJD-TDB)
            - Kp (optional) in m/s to compute RV with a custom Kp (is for example called when computing synthetics)
            - V0 (optional) in m/s to add a constant RV shift (is for example called when computing synthetics)
        Return: 
            - Vp: array of planet's RV in stellar RF
            - Vtot: array of planet's RV in EARTH RF
        '''

        # Get position vector (same units as sma) and true anomaly (radians) at given times from mid-transit
        r, f = self.ke.xyzPos(time, getTA=True)

        # grab other orbital parameters
        if Kp is None: Kp = self.transit_dic['Kp'] # m/s
        
        w_rad = np.radians(self.transit_dic['w']) # here we use the stellar 'w'. Weird since it's the planet's RV, but that's how the equation is defined (check this docs: https://github.com/California-Planet-Search/radvel/issues/51)
        e = self.transit_dic['e']

        # Planet instant velocity on it's orbit projected on line of sight [m/s]. DON'T MULTIPLY BY sin(i) (inclination) AS IT'S ALREADY IN THE KP !!!
        Vp = Kp * (np.cos(w_rad + f) + e * np.cos(w_rad)) # same units as Kp
        Vrv_star = -1 * self.transit_dic['Mp']/self.transit_dic['Ms'] * Vp

        # BERV & systemic velocity
        BERV = self.BERV
        Vs = self.transit_dic['Vs']

        # Total relative velocity between observer on Earth and the Planet Rest Frame [m/s] : Vtot < 0 means we're moving toward the planet, Vtot > 0 means we're moving backward wrt the planet
        Vtot = Vp + Vs - BERV + V0
        
        # return velocities: add -1 factors so that velocities are negative when planet is moving toward us
        return Vp, Vtot

    def remove_one(self, save = None, print_info = True):
        # Remove one to all data to center around 0. after the whole reduction is done
        # Current step ID
        step_ID = 'remove_one'
        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        # Retrieve current step parameters
        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')
        # Warn if step already found in history
        if(step_ID in self.history.keys()):
            warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            k = 0
            former_step_ID = step_ID
            while step_ID in self.history.keys():
                k+=1
                step_ID = former_step_ID+'_'+str(k)    #

                
        ## DO STUFF HERE ##
        self.data -= 1.
        ###################
    
        if print_info : print('\nSubstract 1. done !')
    
        # Save in history
        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")
        
        else :
            self.history[step_ID] = {'parameters' : parameters}

    def mask_null_RV_obs(self, threshold = 2.28e3, save = None, print_info = True):
        '''
        Mask the observations for which the absolute value of the planetary RV in Earth RF is less than threshold (in m/s, 2.28e3 by default corresponding to a SPIRou pixel)
        The masked observations will be set to NaN in the data, so this step has to be done at the end of reduction (just before CCF or Nested Sampling for example)
        Don't do it before since the reduction will still use these observations to properly fit the telluric & stellar signals.
        '''
        # Current step ID
        step_ID = 'mask_null_RV_obs'
        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        # Retrieve current step parameters
        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')
        # Warn if step already found in history
        if(step_ID in self.history.keys()):
            warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            k = 0
            former_step_ID = step_ID
            while step_ID in self.history.keys():
                k+=1
                step_ID = former_step_ID+'_'+str(k)    #

        ## DO STUFF HERE ##
        for obs in range(self.shape[0]):
            if np.abs(self.Vtot[obs]) <= threshold: self.data[obs] = np.nan # set the whole obs to NaN
        ###################
    
        if print_info : print('\Masking obs with low planetary RV in Earth RF done !')
    
        # Save in history
        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")
        
        else :
            self.history[step_ID] = {'parameters' : parameters}
            
    def apply_telluric_correction(self, plot_obs = None, threshold = 0.2, save = None, print_info = True):
        '''Simply apply the telluric correction stored in self.tellurics by dividing the data by the telluric template
        Telluric with transmission below provided threshold are not corrected and data are simply set to NaN instead. Does not replace
        the telluric transmission clipping (this one will remove the entire line based on the center value of the telluric).'''
        # Current step ID
        step_ID = 'apply_telluric_correction'
        # Default behaviour if no save parameter has been set
        if(save==None): save = self.save_history
        # Retrieve current step parameters
        # retrieve currently defined local variables in current scope : parameters and step_ID
        parameters = locals()
        # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        parameters.pop('self')
        # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        parameters.pop('step_ID')
        # Warn if step already found in history
        if(step_ID in self.history.keys()):
            warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            k = 0
            former_step_ID = step_ID
            while step_ID in self.history.keys():
                k+=1
                step_ID = former_step_ID+'_'+str(k)    #
        ## DO STUFF HERE ##
        if self.tellurics is None: raise NameError('No telluric correction has been loaded yet !')
        else:
            if plot_obs is not None:
                plt.figure()
                plt.plot(self.wave[plot_obs].flatten(),self.data[plot_obs].flatten(),label='Before correction')
            # remove values with less than 10% transmission to avoid division by 0
            tellurics = np.copy(self.tellurics)
            tellurics[tellurics<threshold] = np.nan
            self.data /= tellurics
        if plot_obs is not None:
            plt.plot(self.wave[plot_obs].flatten(),self.data[plot_obs].flatten(),label='After correction')
            plt.legend()
            plt.twinx().plot(self.wave[plot_obs].flatten(),self.tellurics[plot_obs].flatten(),color='r',alpha=0.2,label='Telluric template')
            plt.hlines(threshold,self.wave[plot_obs,0,0],self.wave[plot_obs,-1,-1],label='clipping threshold')
            plt.legend()
        ###################
    
        if print_info : print('\nApplying telluric correction done !')
    
        # Save in history
        if save:
            self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
            if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")
        
        else :
            self.history[step_ID] = {'parameters' : parameters}

    #<f
    #def dummy_function(self, save = None, print_info = True):
    #     # A DUMMY FUNCTION TO BE COPY AND PAST FOR EASILY ADDING A REDUCTION STEP WITH ALREADY WRITTEN SAVING AND OVERWRITING WARN PARTS
    #     # Current step ID
    #     step_ID = 'DUMMY_STEP_ID'
    #     # Default behaviour if no save parameter has been set
    #     if(save==None): save = self.save_history
    #     # Retrieve current step parameters
        # # retrieve currently defined local variables in current scope : parameters and step_ID
        # parameters = locals()
        # # remove the 'self' parameter (required in the definition of a class method but useless in history saving)
        # parameters.pop('self')
        # # remove the 'step_ID' variable added by locals() that contains all currently defined variables, thus parameters and step_ID
        # parameters.pop('step_ID')
    #     # Warn if step already found in history
    #     if(step_ID in self.history.keys()):
            # warn(f'\nWarning : {step_ID} has already been applied on data. Applying it again here : make sure this is what you want !\n')
            # # change step_ID to save it in the dictionnary without conflict with the same step that have been already done
            # k = 0
            # former_step_ID = step_ID
            # while step_ID in self.history.keys():
            #     k+=1
            #     step_ID = former_step_ID+'_'+str(k)    #
    #     ## DO STUFF HERE ##
    #
    #     ###################
    #
    #     if print_info : print('\nDUMMY FUNCTION done !')
    #
    #     # Save in history
    #     if save:
    #         self.history[step_ID] = {'data' : np.copy(self.data), 'parameters' : parameters}
    #         if print_info: print(f"\nThis step has been saved with ID : '{step_ID}'\n------------------------------")
        #
        # else :
        #     self.history[step_ID] = {'parameters' : parameters}
    #f>
