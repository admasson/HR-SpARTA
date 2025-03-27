################################################################################
## New module to create the synthetic spectra from petitRADTRANS
## Coding : utf-8
## Author : Adrien Masson (adrien.masson@obspm.fr)
## Date   : March 2025
################################################################################

#-------------------------------------------------------------------------------
# Import usefull libraries and modules
#-------------------------------------------------------------------------------

from HR_SpARTA.reduction import *

# set petitRADTRANS input path, and import petitRADTRANS
from petitRADTRANS.spectral_model import SpectralModel
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.planet import Planet
from petitRADTRANS.fortran_rebin import fortran_rebin as frebin
from scipy.signal import convolve
from scipy.interpolate import RectBivariateSpline

# from spectres import spectres # to easily integrate flux on spectral bins with flux conservation. Cite https://arxiv.org/abs/1705.05165

import json
import warnings

import threading

# some global variable for threading
lock = threading.Lock() # Define a lock for safe access
counter = 0 # counter for threading monitoring

# Safety check when module has been imported for debugging purpose
print(f'{datetime.datetime.now().time()} : AtmoModel.py has been loaded')

def lower_resolution(wave,flux,velocity_bin,nb_of_points):
    '''
    Lower a spectrum resolution by integrating it on a door function
    with a Radial Velocity size of "velocity_bin" in [m/s]. The integration
    is thus perform on "nb_of_points" points linearly ranging in the RV
    integration area.
    Can be used to:
    - lower to instrumental resolution, by setting velocity bin size to c / R or c / 2R (R being resolution, depends on whether we consider 2 element of resolution per bin for Nyquist)
    - broaden according to transit RV (in this case put the mean planet delta RV during a single observation)
    '''

    half_size = velocity_bin / 2
    pixel = np.linspace(-half_size,half_size,nb_of_points)

    integrated_model = np.zeros(flux.shape)

    f = interp1d(wave,flux,fill_value=np.nan)

    for v in pixel:
        # mask wavelength shifted outside the interpolation domain
        mask_down = (wave / (1 + v/const.c.value)) < wave.min()
        mask_up   = (wave / (1 + v/const.c.value)) > wave.max()
        mask = np.logical_or(mask_down,mask_up) # contains True where shifted wavelength are outside the valid interpolation domain
        integrated_model[~mask] += f(wave[~mask] / (1 + v/const.c.value))
        
        # replace values outside range by nan
        integrated_model[mask] = np.nan

    # normalise
    integrated_model /= len(pixel)

    # update model
    return integrated_model

def div_by_moving_average(spectrum,area):
    '''
    Using scipy convolution here since its faster than astropy, provide the same results, and we don't have NaN to handle here.
    astropy is used for the DataSet reduction since it properly account for NaN in the data.

    '''
    # create a matrix corresponding to a "door" function to convolve with our data
    kernel = np.ones(area)
    # normalize
    kernel /= kernel.sum()
    # computing the convolution of each spectra with this door function
    data_convolved = np.zeros(spectrum.shape) * np.nan
    for obs in range(spectrum.shape[0]):
        for order in range(spectrum.shape[1]):
            data_convolved[obs,order] = convolve(spectrum[obs,order,:],kernel,mode='same')
            # remove edges where convolution is invalid
            data_convolved[obs,order,:area] = np.nan
            data_convolved[obs,order,-area:] = np.nan
    # return result
    return spectrum / data_convolved


class AtmoModel:
    def __init__(self,
                 data_set,
                 species,
                 pressure=np.logspace(-8, 2, 130),
                 ref_pressure = 1e-2,   # ref pressure at which planet's radius & logg are defined. This is pRT default value in documentation
                 load_pRT=True,         # Can be set to False if we use an interpolation grid, since in this case we don't need to load the opacity grids from pRT
                 lbl_sampling=1,
                 ):
        
        '''
        Initialize the AtmoModel from a DataSet class and a given list of species to be loaded with petitRADTRANS
        - data_set -> a DataSet class (check reduction.py), contains the info about data (wavelength axis, radial velocities, planet parameters...)
        - species  -> a list of string with the species to be loaded by petitRADTRANS 
        '''
    
        # grab parameters
        self.pressure       = pressure
        self.species        = species

        self.transit        = data_set.transit_dic
        self.data_midpoint  = data_set.midpoint         # Transit midpoint [BJD-TBD]
        self.data_headers   = data_set.headers          # DataSet headers
        self.shape          = data_set.shape
        self.data_wave      = data_set.wave[0]          # meters
        self.transit_weight = data_set.transit_weight   # computed with Batman, 0. out of transit & 1. at mid-transit
        self.time_vector    = data_set.time_vector      # time axis in BJD-TDB
        self.data_set       = data_set                  # memory link to the data_set object, so we can access its functions 

        self.Vp_ref     = data_set.Vp                   # Planet's reference RV (with eccentricity) computed by the DataSet during reduction
        self.Kp_ref     = data_set.transit_dic['Kp']    # Planet's reference Kp used to compute Vp_ref during reduction

        self.BERV   = data_set.BERV                     # Barycentric Earth Radial Velocity
        self.Vsys   = data_set.transit_dic['Vs']        # Systemic velocity  

        # set a default name for this Model
        self.name = '_'.join(self.transit['data_dir'].replace('//','/').split('/')[-4:-1])

        # Variable to store the pRT interpolator when using interpolation grid
        self.pRT_interpolator = None # will be filled by load_interp_pRT()

        # compute planet ref gravity if missing (in cm/s²)
        if not 'planet_logg' in self.transit.keys():
            ref_gravity = const.G.cgs.value*(self.transit['Mp']*1e3/(self.transit['Rp']*100)**2)
            self.transit['planet_logg'] = np.log10(ref_gravity)
            print(f'logg was missing in the transit dic, calculating it: {self.transit["planet_logg"]} [cgs]')

        # Mass fractions dictionnary
        self.imposed_mass_fractions = {}
        for specie in self.species:
            self.imposed_mass_fractions[specie]=1e-2 # these can also be arrays of the same size as pressures
    
        # Setup petitRADTRANS atmosphere model with random values for the free parameters (will be updated when computing spectrum)
        # this avoid reloading all files each time a new model is computed and faster computation
        if load_pRT:

            self.spectral_model = SpectralModel(
                # Radtrans parameters
                pressures=self.pressure,
                line_species=self.species,
                rayleigh_species=['H2', 'He'],
                gas_continuum_contributors=['H2-H2', 'H2-He'],
                line_opacity_mode='lbl',
                line_by_line_opacity_sampling=lbl_sampling,

                # Planet parameters
                planet_radius=self.transit['Rp']*1e2, # cm
                reference_pressure=ref_pressure,
                reference_gravity=10**self.transit['planet_logg'],
                star_radius=self.transit['Rs']*1e2, # cm

                # Mass fractions
                # imposed_mass_fractions = self.imposed_mass_fractions,
                filling_species={  # automatically fill the atmosphere with H2 and He, such that the sum of MMRs is equal to 1 and H2/He = 37/12 -> Jupiter like ratio
                    'H2': 37,
                    'He': 12
                },
                
                # Observation parameters
                wavelength_boundaries=[self.data_wave.min()/1e3,self.data_wave.max()/1e3], # Here the boundaries are in microns: convert nm to microns
                
                # the above is not working when using data_wave.flatten, not sure why. 
                rebin_range_margin_power=1,  # used to set the wavelengths boundaries, adding a margin of ~1 Angstrom (1e-4 * ~1 µm)
            )

        # In the case where we don't want to compute petitRADTRANS directly but interpolate on a grid instead
        else:
            self.spectral_model = None
            print('You decided not to load the pRT spectrum: you MUST use an interpolation grid to compute the pRT spectrum !')

    def compute_pRT(self, atmo_parameters={}, plot=False, save=False, save_dir=''):
        '''
        atmo_parameters is a dic containing the parameters to update for the new spectrum calculation. Possible keys are:
        Tiso : float
        log_Pcloud: log of cloud pressure
        log_MMR : dictionnary {'specie': log abundance in MMR}

        if not given a parameter will take it's default value from the __init__ function call

        The result is the transit apparent radius of the planet at each wavelength, in meter.
        '''

        # reset parameters to avoid remnant values from previous call
        self.spectral_model.model_parameters['temperature'] = None
        self.spectral_model.model_parameters['opaque_cloud_top_pressure'] = None
        self.spectral_model.model_parameters['imposed_mass_fractions'] = None

        # update spectral model parameters
        if not len(atmo_parameters.keys())==0:
            for key, val in atmo_parameters.items():
                if key=='Tiso': self.spectral_model.model_parameters['temperature'] = val
                if key=='log_Pcloud': self.spectral_model.model_parameters['opaque_cloud_top_pressure'] = 10.**val
                if key=='log_MMR':
                    imposed_mass_fractions = {}
                    for specie,log_abund in val.items():
                        imposed_mass_fractions[specie]=10.**log_abund # these can also be arrays of the same size as pressures
                    self.spectral_model.model_parameters['imposed_mass_fractions']=imposed_mass_fractions
            self.spectral_model.update_model_functions_map(update_model_parameters=False)

        # Compute transmission spectrum with petitRADTRANS
        wavelengths, transit_radii = self.spectral_model.calculate_spectrum(
                                                mode='transmission',
                                                update_parameters=True,
                                                )

        wave = wavelengths.flatten() * 1e7 # oversampled wavelength in nm 
        model = transit_radii[0].flatten() / 1e2  # convert radii in cm to m. The [0] are for removing useless axis

        if plot:
            plt.figure()
            plt.plot(wave,model/const.R_jup.value,label='petitRADTRANS output') # convert to Jupiter radii
            plt.title('pRT Continuum')
            plt.ylabel('Apparent radius [Rjup]')
            plt.xlabel('Wavelength [nm]')

        # save
        if save:
            file_name = self.name
            for key, val in atmo_parameters.items():
                if key=='Tiso': file_name+=f'_Tiso[k]_{val}'
                if key=='log_Pcloud': file_name+=f'_logPcloud[bar]_{val}'
                if key=='log_MMR':
                    for specie,log_abund in val.items():
                        file_name+=f'_log{specie}[MMR]_{val}'
            file_name+='.dat'
            hdr = str(atmo_parameters)
            np.savetxt(save_dir+file_name,model,header=hdr)

        # return the model: wavelength in nm, and transit apparent radius (planet bulk asborption + atmosphere) in meters
        return(wave,model)      

    def compute_synthetic(self, Kp, V0, atmo_parameters={}, 
                          mav_area=None, interpolate_pRT=False, apply_svd=False, plot=False, lower_to_instrument_resolution=False, accurate_sampling=False, verbose=True):

        '''
        Kp, V0 -> m/s
        mav_area : if None (default) then no division by mav is done. If set to an odd int then div by mav is performed
        
        atmo_parameters is a dic containing the parameters to update for the new spectrum calculation. Possible keys are:
        Tiso : float
        log_Pcloud: log of cloud pressure
        log_MMR : dictionnary {'specie': log abundance in MMR}
        '''

        if apply_svd and (data_set is None): raise NameError('You must give a DataSet class object with apply_svd in its reduction history as "data_set" argument when apply_svd is True')

        # Compute petitRADTRANS model
        if interpolate_pRT: 
            raise NameError('To do')
        else:
            wave_model,transit_radius = self.compute_pRT(atmo_parameters,plot)

        # Lower to instrumental resolution            
        if not lower_to_instrument_resolution:
            if verbose: print('No lower_to_instrument_resolution parameter pass (expected SPIRou or ESPRESSO), skipping !')
        else:
            if lower_to_instrument_resolution == 'SPIRou': velocity_bin = 2*2.8e3 # SPIRou velocity bin size (R = 70 000), *2 for Nyquist criterion (?), in m/s
            elif lower_to_instrument_resolution == 'ESPRESSO': velocity_bin = 2*300   # ESPRESSO velocity bin size -> to re-check !
            else: raise NameError(lower_to_instrument_resolution, 'must be SPIRou or ESPRESSO')
            transit_radius = lower_resolution(wave_model,transit_radius,velocity_bin,nb_of_points=11)
            if plot: plt.plot(wave_model,transit_radius/const.R_jup.value, label='Instrument resolution')

        # Broadening induced by planet RV during the integration time
        delta_RV = np.mean(np.diff(self.Vp_ref))
        if verbose: print(f'Mean broadening induced by planet delta RV during integration time: {delta_RV:.2e} m/s')
        transit_radius = lower_resolution(wave_model, transit_radius, velocity_bin=delta_RV, nb_of_points=11)
        if plot: plt.plot(wave_model,transit_radius/const.R_jup.value, label='Integration time broadening')  

        # Broadening induced by planet rotation
        # -> TODO, can be a first quick fix by assuming planet is tidally locked & broadening with a kernel equal to planet instant RV
        if verbose: print('TODO: implement simple broadening due to planet rotation')

        ### Create the synthetic time series: shift for each observation and resample on data spectra bins ###
        synthetic = np.ones((self.shape))

        # compute list of RV from DataSet stored orbital geometry
        Vtot_list = self.compute_RV(Kp,V0)
        for obs in np.where(self.transit_weight != 0)[0]:

            # shift
            Vtot = Vtot_list[obs]
            wave_shifted = wave_model / (1 - (Vtot / const.c.value)) # V0 > 0 -> redshift, V0 < 0 -> blueshift
            transit_radius_shifted = interp1d(wave_shifted,transit_radius,bounds_error=False,fill_value=np.nan)(wave_model)

            if plot and obs == np.where(self.transit_weight != 0)[0][10]: plt.plot(wave_model,transit_radius_shifted/const.R_jup.value, label=f'Shifted @ Vtot = {Vtot:.2e} m/s')       
                  
            # sample on instrumental bins
            if accurate_sampling: # use spectres to integrate on the instrument spectral bins with flux conservation
                # instru_wave = self.data_wave.flatten() # flatten order-wise
                # transit_resampled = spectres(instru_wave,wave_model,transit_radius_shifted) # integrate on instrument spectral bins
                # transit_resampled = transit_resampled.reshape(self.data_wave.shape)
                raise NameError('Something weird\'s happening with spectres: constant values are included in some parts... Investigate')
            else:
                if verbose and obs == np.where(self.transit_weight != 0)[0][10]: print('Performing simple sampling here with no flux conservation !')
                transit_resampled = interp1d(wave_model,transit_radius_shifted,bounds_error=False,fill_value=np.nan)(self.data_wave)

            if plot and obs == np.where(self.transit_weight != 0)[0][10]: 
                plt.plot(self.data_wave.flatten(),transit_resampled.flatten()/const.R_jup.value, label=f'Resampled on instrumental bins')
                plt.legend()

            # compute transmission with window weight & store
            synthetic[obs] = 1 - self.transit_weight[obs]*((transit_resampled-self.transit['Rp'])**2 / self.transit['Rs']**2) # should we remove the planet's bulk absorption here ?

        # plot synthetic time series before post-processing (div by mav & SVD)
        if plot:
            plt.figure()
            plt.imshow(synthetic.reshape(self.shape[0],self.shape[1]*self.shape[2]), aspect='auto', origin='lower')
            plt.colorbar()

        if plot: 
            plt.figure()
            plt.plot(self.data_wave.flatten(),synthetic[obs-5].flatten(),label=f'Obs n°{obs-5}') # plotting a random in-transit observation, -5 to ensure it's not egress
            plt.ylabel('Normalised flux')
            plt.xlabel('Wavelength [nm]')

        # apply div by weighted avergage on in-transit data
        if mav_area is not None: 
            synthetic[self.transit_weight>0] = div_by_moving_average(synthetic[self.transit_weight>0],mav_area)
            if plot: 
                plt.plot(self.data_wave.flatten(),synthetic[obs-5].flatten(), label='Div by mav') 

        # Apply the svd on the synthetic Can be disabled for fast testing 
        if apply_svd: 
            synthetic = self.data_set.apply_pca_on_synth(synthetic)
            if plot: plt.plot(self.data_wave.flatten(),synthetic[obs-5].flatten(), label=f'Apply svd') 

        # remove NaN & inf
        synthetic = np.ma.masked_invalid(synthetic)
        synthetic.fill_value = 0.

        # center at 0.
        synthetic -= 1

        # add legend to previous plot        
        if plot: plt.legend()

        if plot: # some 2D plot
            plt.figure()
            plt.imshow(synthetic.reshape(self.shape[0],self.shape[1]*self.shape[2]), aspect='auto', origin='lower')
            plt.colorbar()

        # return synthetic
        return(synthetic)
    
    def compute_RV(self,Kp,V0):
        '''
        Small tricks to compute RV fast for any Kp,V0 value, by taking the already computed RV from the DataSet & simply changing the Kp factor in the formula:
        - DataSet.Vp is equal to: Kp * (np.cos(w_rad + f) + e * np.cos(w_rad))
        - So we compute the new Vp as: Vp_new = V0 + Kp_new * DataSet.Vp / Kp
        - And then add Vsys & remove BERV to get the new Vtot at a given Kp_new, V0_new

        -> return an array with same shape as velocities/time vectors (one element per observation)
        '''
        Vp_new = V0 + Kp * self.Vp_ref / self.Kp_ref
        Vtot = Vp_new + self.Vsys - self.BERV
        return Vtot

    def CCF_2D(self, wave_model, transit_radius,
                Kpmin=0,Kpmax=500e3,V0min=-150e3,V0max=+150e3,NKp=500,NV0=300,NVtot=500,
                divide_by_sigma = True,plot=False,apply_svd=False, mav_area=None, build_interp_grid=True, verbose=True,
                lower_to_instrument_resolution=False,
                Kp_ref=150e3,V0_ref=0,Nb_threads=0,ccf_mask=[-50e3,50e3,50e3,350e3]):
        
        '''
        Compute a petitRADTRANS model with the provided atmo parameters, and compute corresponding CCF map with the DataSet used during class initialisation.
        Kp_ref & V0_ref are used to plot an expected CCF position slope in the time serie and also show the 1D CCF at the given ref positions in the (Kp,V0) space
        Nb threads: put 0 if you don't want parallelization
        '''

        global counter
        counter = 0 # reset counter for threading

        ############### Prepare Data ###############
        # grab parameters from data_set
        data = np.copy(self.data_set.data)
        # center at 0 if isn't
        data -= np.nanmean(data)
        # get nb of obs
        N_obs = data.shape[0]
        # compute sigma
        sigma = self.data_set.weighted_std(self.data_set.data)
        if not divide_by_sigma: sigma = np.ones(sigma.shape)
        # divide by sigma**2: we do it once on data to save computation time when multiplying with the model in the ccf computation
        data_over_sigma2 = data / (sigma**2)
        data_over_sigma2 = np.ma.masked_invalid(data_over_sigma2)
        ##############################################



        # ######## Compute petitRADTRANS model ########
        if verbose: print('Convolving petitRADTRANS model to instrument & integration time broadening...')
        if plot:
            plt.figure()
            plt.plot(wave_model,transit_radius/const.R_jup.value,label='petitRADTRANS output') # convert to Jupiter radii
            plt.title('pRT Continuum')
            plt.ylabel('Apparent radius [Rjup]')
            plt.xlabel('Wavelength [nm]')

        # wave_model,transit_radius = self.compute_pRT(atmo_parameters,plot)

        # Lower to instrumental resolution            
        if not lower_to_instrument_resolution:
            if verbose: print('No lower_to_instrument_resolution parameter pass (expected SPIRou or ESPRESSO), skipping !')
        else:
            if lower_to_instrument_resolution == 'SPIRou': velocity_bin = 2*2.8e3 # SPIRou velocity bin size (R = 70 000), *2 for Nyquist criterion (?), in m/s
            elif lower_to_instrument_resolution == 'ESPRESSO': velocity_bin = 2*300   # ESPRESSO velocity bin size -> to re-check !
            else: raise NameError(lower_to_instrument_resolution, 'must be SPIRou or ESPRESSO')
            transit_radius = lower_resolution(wave_model,transit_radius,velocity_bin,nb_of_points=11)
            if plot: plt.plot(wave_model,transit_radius/const.R_jup.value, label='Instrument resolution')

        # Broadening induced by planet RV during the integration time
        delta_RV = np.mean(np.diff(self.Vp_ref))
        if verbose: print(f'Mean broadening induced by planet delta RV during integration time: {delta_RV:.2e} m/s')
        transit_radius = lower_resolution(wave_model, transit_radius, velocity_bin=delta_RV, nb_of_points=11)
        if plot: plt.plot(wave_model,transit_radius/const.R_jup.value, label='Integration time broadening')  

        # Broadening induced by planet rotation
        # -> TODO, can be a first quick fix by assuming planet is tidally locked & broadening with a kernel equal to planet instant RV
        if verbose: print('TODO: implement simple broadening due to planet rotation')
        ##############################################



        ######## Build the interpolation grid ########
        Vtot_min = 2*self.compute_RV(np.abs(Kpmin),V0min)[0              ] # taking abs to have negative Vtot_min even if Kp is negative (which is physcally impossible but can serve as a sanity check)
        Vtot_max = 2*self.compute_RV(Kpmax        ,V0max)[self.shape[0]-1]
        Vtot_range = np.linspace(Vtot_min,Vtot_max,NVtot)

        # interpolation grid to quickly compute a doppler shift of the oversampled transit_radius model computed with pRT
        if build_interp_grid:
            if verbose: print(f'Building the synthetic spectra grid for {NVtot} Vtot values between {Vtot_min/1e3:.0f} & {Vtot_max/1e3:.0f} km/s :\n')
            shifted_synth_list = np.zeros((NVtot,self.shape[1],self.shape[2]))
            j = 0 # just a loop counter
            for Vtot in Vtot_range:
                if verbose: print(f'\r{j+1}/{NVtot}',end='',flush=True)
                wave_shifted = wave_model / (1 - (Vtot / const.c.value)) # V0 > 0 -> redshift, V0 < 0 -> blueshift
                shifted_synth_list[j] = interp1d(wave_shifted,transit_radius,bounds_error=False,fill_value=np.nan)(self.data_wave) # directly resample on data spectral bins, approximation since we don't ensure flux conservation here
                j += 1
            if verbose: print()
            # Build the interpolation function on the grid
            if verbose: print('Creating the interpolation function on the grid : this step can take some time...')
            grid_interpoler = interp1d(Vtot_range,shifted_synth_list,bounds_error=False,fill_value=np.nan,axis=0)
            if verbose: print('Interpolation grid succesfully created !')
        ##############################################

        ########### Build CCF Time-Series ############
        def ComputeCCF_at_Vtot(CCF,index_list):
            global counter, lock

            # loop over index list (list of Vtot that current Thread must process)
            for index in index_list:
                # grab Vtot
                Vtot = Vtot_range[index]

                # shift model: we're computing the CCF time series, so here all models can have the same shift along the time axis. 
                # It's the signal in the data that is moving from one obs to the other, and we'll "align" it's signature properly in the (Kp,V0) frame when interpolating along the Kp/V0 ligns in the CCF Time Serie
                if build_interp_grid: transit_shifted_and_resampled = grid_interpoler(Vtot)
                else: 
                    wave_shifted = wave_model / (1 - (Vtot / const.c.value))
                    transit_shifted_and_resampled = interp1d(wave_shifted,transit_radius,bounds_error=False,fill_value=np.nan)(self.data_wave) # Here we both shift oversampled spectrum & resampled on data wave bin, with no guarantee on conserving flux !

                # apply transit weight
                synthetic = 1 - self.transit_weight[:,None,None]*((transit_shifted_and_resampled-self.transit['Rp'])**2 / self.transit['Rs']**2)[None,:] # should we remove the planet's bulk absorption here ?
                
                # apply div by weighted avergage on in-transit data
                if mav_area is not None: synthetic[self.transit_weight>0] = div_by_moving_average(synthetic[self.transit_weight>0],mav_area)

                # Apply the svd on the syntehtic Can be disabled for fast testing (in fact when injecting model before reduction the recovered model gives a worse CCF when applying PCA ?? We should investigate about SysREM & Gibson's method)
                if apply_svd : synthetic = self.data_set.apply_pca_on_synth(synthetic)

                # remove NaN & inf
                synthetic = np.ma.masked_invalid(synthetic)
                synthetic.fill_value = 0.
                
                # center at 0.
                synthetic -= 1

                # compute ccf in time-series (i.e (time,Vtot) space)
                ccf = np.ma.sum(data_over_sigma2 * synthetic, axis=(1,2))

                # update memory, using lock to avoid multiple threads accessing at the same time
                with lock:
                    # Then compute the CCF, sum(data**2/sigma**2) & sum(synth**2/sigma**2) for a given V0
                    CCF[:,index] = ccf
                    counter += 1

                print(f'\r{counter}/{NVtot}',end='',flush=True)

        # array to fill
        TimeSerie_CCF = {}
        CCF = np.zeros((N_obs,NVtot))
        
        # Compute CCF in (time,Vtot) space with multiprocess
        tstart = time.time()
        if Nb_threads: # if parallelization
            num_threads = Nb_threads
            threads = []
            # Divide the Vtot array in equal parts (one per thread)
            index_sub_arrays = np.array_split(np.arange(NVtot), num_threads)
            # Assign each thread with one part of the array
            for thread_idx in range(num_threads):
                index_list = index_sub_arrays[thread_idx] # list of index (in Vtot range & CCF) the thread will work with
                thread = threading.Thread(target=ComputeCCF_at_Vtot, args=(CCF,index_list))
                threads.append(thread)
                thread.start()
            # Wait for all threads to finish
            for thread in threads:
                thread.join()
        else: # without parallelisation
            index_list = np.arange(NVtot)
            ComputeCCF_at_Vtot(CCF,index_list)

        print()
        print(f'CCF Time-Series done in {time.time()-tstart}s')
        TimeSerie_CCF['data*model'] = CCF
        
        if plot:
            # 2D plots
            plt.figure(figsize=(10,10))
            plt.imshow(TimeSerie_CCF['data*model'],extent=[Vtot_min/1e3,Vtot_max/1e3,0,self.shape[0]],origin='lower',aspect='auto')
            plt.xlabel('Vtot [km/s]')
            plt.ylabel('Obs n°')
            # plt.ylim((data_set.on_transit_mask[0],data_set.on_transit_mask[-1]))
            plt.xlim((Vtot_min/1e3,Vtot_max/1e3))
            plt.title('CCF')
            c = plt.colorbar()
            c.set_label(f'CCF')
            # plot a Kp line
            plt.plot(self.compute_RV(Kp_ref,V0_ref)/1e3,np.arange(N_obs),'r',label=f'(Kp,V0) = ({Kp_ref:.0f},{V0_ref:.0f}) m/s')
            plt.plot(self.compute_RV((Kp_ref+100e3),V0_ref)/1e3,np.arange(N_obs),'--r',label=f'(Kp,V0) = ({Kp_ref+100e3:.0f},{V0_ref:.0f}) m/s')
            plt.plot(self.compute_RV((Kp_ref-100e3),V0_ref)/1e3,np.arange(N_obs),'-.r',label=f'(Kp,V0) = ({Kp_ref-100e3:.0f},{V0_ref:.0f}) m/s')

            plt.legend()
            plt.savefig('CCF_timeSeries')
        ##############################################



        ###### Interpolate CCF in (Kp,V0) space ######
        # reset counter for threading
        counter = 0 
        # set the (Kp,V0) grid range
        Kp_range = np.linspace(Kpmin,Kpmax,NKp)
        V0_range = np.linspace(V0min,V0max,NV0)

        f1 = RectBivariateSpline(Vtot_range,np.arange(N_obs),TimeSerie_CCF['data*model'].T)

        def interpolateCCF(Kp,V0):
            '''
            Interpolate CCF values along a diagonal defined by (Kp,V0) in the CCF time-series space (Vtot,time) and return the sum of the CCF interpolated on the diagonal
            '''
            return np.sum(np.diag(f1(self.compute_RV(Kp,V0), np.arange(N_obs))))

        def ComputeCCF(result,i,Kp,V0_list,progress):
            for j,V0 in enumerate(V0_list):
                result['data*model'][i,j] = interpolateCCF(Kp,V0)
                progress[0] += 1
            print(f'\r{100*progress[0]/(NKp*NV0):.0f} %',end='',flush=True)

        def CCF_KpV0_parallel(result,index_list):
            global counter
            for i in index_list:
                Kp = Kp_range[i]
                temp = np.zeros_like(result['data*model'][i])
                for j,V0 in enumerate(V0_range):
                    temp[j] = interpolateCCF(Kp,V0)
                with lock:
                    result['data*model'][i] = temp
                    counter += 1
                    print(f'\r{counter+1}/{NKp}',end='',flush=True)

        result = {}
        result['data*model'] = np.zeros((NKp,NV0))

        # # interpolate in (Kp,V0) space with multiprocess
        # tstart = time.time()
        # if Nb_threads: # if parallelization
        #     num_threads = Nb_threads
        #     threads = []
        #     # Divide the Kp array in equal parts (one per thread)
        #     index_sub_arrays = np.array_split(np.arange(NKp), num_threads)
        #     # Assign each thread with one part of the array
        #     for thread_idx in range(num_threads):
        #         index_list = index_sub_arrays[thread_idx] # list of index (in Vtot range & CCF) the thread will work with
        #         thread = threading.Thread(target=CCF_KpV0_parallel, args=(result,index_list))
        #         threads.append(thread)
        #         thread.start()
        #     # Wait for all threads to finish
        #     for thread in threads:
        #         thread.join()
        # else: # without parallelisation
        progress = np.zeros((1))
        for i,Kp in enumerate(Kp_range):
            ComputeCCF(result,i,Kp,V0_range,progress)

        # convert back to array
        result['data*model'] = np.array(result['data*model'])

        # store interpoler
        result['CCF_interp'] = f1

        # store parameters
        result['params'] = {}
        result['params']['V0min']    = V0min   
        result['params']['V0max']    = V0max   
        result['params']['Kpmin']    = Kpmin   
        result['params']['Kpmax']    = Kpmax   
        result['params']['Kp_ref']   = Kp_ref  
        result['params']['V0_ref']   = V0_ref  
        result['params']['Kp_range'] = Kp_range
        result['params']['V0_range'] = V0_range

        ### compute significance map ###
        CCF = np.copy(result['data*model'])
        # Compute noise outside signal's box
        Kp_signal_range = (Kp_range > ccf_mask[2]) * (Kp_range < ccf_mask[3])
        V0_signal_range = (V0_range > ccf_mask[0]) * (V0_range < ccf_mask[1]) 
        signal_mask = Kp_signal_range[...,None] * V0_signal_range.T[None,...] # has the shape of alpha & contains True where signal is located (the previously given box)
        # noise_alpha = np.nanstd(alpha[~signal_mask]) # compute SNR by dividing the signal (taken in a box around 50<Kp<350 & -10<V0<5) by the median noise outside the signal's box
        noise_CCF = np.std(CCF[~signal_mask])
        result['noise'] = noise_CCF
        if plot:
            # Compute the CCF projected along Kp_ref & V0_ref axes
            CCF_1D_Kp = [interpolateCCF(Kp_ref,v0)/noise_CCF for v0 in V0_range]  # 1D CCF at fixed Kp_ref
            CCF_1D_V0 = [interpolateCCF(kp,V0_ref)/noise_CCF for kp in Kp_range]  # 1D CCF at fixed V0_ref
            # plot mask
            plt.figure()
            plt.imshow(signal_mask,aspect='auto',origin='lower',extent=[V0min/1e3,V0max/1e3,Kpmin/1e3,Kpmax/1e3])
            print(signal_mask.shape)
            # plot CCF (2D + 1D projections)
            fig = plt.figure()
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1], wspace=0, hspace=0)
            # main plot: 2D CCF
            ax_main = fig.add_subplot(gs[0, 1])
            c = ax_main.imshow(CCF/noise_CCF,extent=[V0min/1e3,V0max/1e3,Kpmin/1e3,Kpmax/1e3],origin='lower',aspect='auto')
            ax_main.set_xlabel(f'V$_{0}$ [km/s]')
            ax_main.set_ylabel('Kp [km/s]')
            ax_main.hlines(Kp_ref/1e3,V0min/1e3,V0max/1e3,lw=0.5,ls='--',color='r')
            ax_main.vlines(V0_ref/1e3,Kpmin/1e3,Kpmax/1e3,lw=0.5,ls='--',color='r')
            ax_main.tick_params(labelbottom=False, labelleft=False)
            # Plot the projection along V0
            ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main)
            ax_left.plot(CCF_1D_V0, Kp_range/1e3, color='k')
            ax_left.set_xlabel("SNR")
            ax_left.set_ylabel(r"$K_p$ [km/s]")
            ax_left.invert_xaxis()  # Optional: Invert x-axis for better alignment
            # Plot the projection along Kp
            ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_main)
            ax_bottom.plot(V0_range/1e3, CCF_1D_Kp, color='k')
            ax_bottom.set_xlabel(r"$V_0$ [km/s]")
            # Hide unused axes
            fig.add_subplot(gs[1, 0]).axis("off")
            # add colorbar for main plot outside the entire grid
            cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
            fig.colorbar(c, cax=cbar_ax, label="SNR")
            # add plotting axes to result for enabling figure edition outside function
            result['plotting_axes'] = [fig,ax_main,ax_left,ax_bottom]
        ##############################################

        return(result,CCF/noise_CCF)

        



