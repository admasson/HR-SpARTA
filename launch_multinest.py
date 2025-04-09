
# Don't forget to set env variable for multinest:
# export LD_LIBRARY_PATH=$HOME/MultiNest/lib:$HOME/cuba/directory/:$LD_LIBRARY_PATH

# Then run with mpi in terminal:
# mpirun -np 16 -x OMP_NUM_THREADS=16 -x OPENBLAS_NUM_THREADS=16 -x MKL_NUM_THREADS=16 -x NUMEXPR_NUM_THREADS=16 python launch_multinest.py

# For local run we use 4 Threads:
# mpirun -np 4 -x OMP_NUM_THREADS=4 -x OPENBLAS_NUM_THREADS=4 -x MKL_NUM_THREADS=4 -x NUMEXPR_NUM_THREADS=4 python launch_multinest.py

# THINGS TO CHECK BEFORE LAUNCHING:
# - Data
# - Model definition (molecules)
# - Interpoler path
# - Are clouds fixed ?
# - Is sigma computation ok ?
# - Is division by mav ok ?
# - Has output name been updated ?

# If troubles with pymultinest (_os_gfortran_at_ ... error):
# follow instructions on https://johannesbuchner.github.io/pymultinest-tutorial/install.html#on-your-own-computer
# launch cmake with the following command: cmake -DCMAKE_EXE_LINKER_FLAGS="-lgfortran" -DCMAKE_SHARED_LINKER_FLAGS="-lgfortran" .. 

# Once the run is finished, we can create the corner plots with the following commands:
# $ python multinest_marginals_fancy.py path/to/chain-
# For example if results are saved with prefix 'example' in directory 'results', then run $ python multinest_marginals_fancy.py ./results/example/example-

####################################
########## IMPORT MODULES ##########
####################################

import time
time_start = time.time()

import os
import sys

import json
from scipy.interpolate import RectBivariateSpline
from plots     import *

from HR_SpARTA.transits_parameters import *
from HR_SpARTA.reduction import *
from HR_SpARTA.AtmoModel import AtmoModel
from HR_SpARTA.plots     import *
from HR_SpARTA.CCF import CCF_2D

# set petitRADTRANS input path, and import petitRADTRANS
from petitRADTRANS.spectral_model import SpectralModel
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.planet import Planet
from petitRADTRANS.config.configuration import petitradtrans_config_parser

import pymultinest

####################################
##########   PARAMETERS   ##########
####################################

# path to a DataSet object saved in .pkl format (DataSet.save method)
data_path = "/home/amasson/data/Kaperture_Science/Results/Injection_recovery/example_data_set.pkl"

# directory & subfolder names to store pymultinest results
save_path = '/home/amasson/data/Kaperture_Science/HR-SpARTA_priv/molecules/Nested_Sampling/results'
log_dir = 'nested_sampling_results'
name = 'example'

# directory to store temporary img
img_dir = 'images'

# which instrument are we simulating
instrument = 'SPIRou'

# set to true to compute a CCF map 
do_ccf = False

# set to true to perform a Nested Sampling run
do_nested = True

# set which molecules to consider in the model
molecules = [ # NEED TO BE IN CORRECT ORDER WRT THE INTERPOLATION GRID DIMENSIONS !!
    "H2O",
    'CO',
 ] # add -NatAbund for including isotopes with natural abundances

# path to an interpolation grid (if any)
interpoler_path = f'./toi2669_Na_FeH.pkl'

# set to true to load & use an interpolation grid instead of calling pRT at each run
use_interp = True

# Ref dictionnary containing the truth or expected value for the parameters. We can either fix specific value or let the rest of code set them (depending if a model is injected or not)
ref_value_dic = {
     'Kp [km/s]' : 120,
     'V0 [km/s]' : 5,
     'log_AbundNa [MMR]': -3,
     'log_AbundFeH [MMR]': -3,
     'Tiso [K]': 1200
} # -> will be use for plotting results

# Do we put a reference grey cloud deck?
ref_cloud = False # False or ref value (!=0) for injecting/plotting cloud position
if ref_cloud: ref_value_dic['log_Pcloud [bar]'] = -2

# Do we fix the cloud pressure (if any) or let it free ?
fix_cloud = False # if value != 0, fix the cloud at this pressure in the retrieval

# Do we apply SVD on the model? Better in theory, but slow things a lot !
do_svd = False # apply svd

####################################
##########  PREPARE DATA  ##########
####################################

# load DataSet object
with open(data_path,'rb') as file:
    data_set = pickle.load(file)

# set noise for retrieval: compute the std along time and use it as 1-sigma uncertainty
sigma = data_set.weighted_std(data_set.data)

print(10*'#')
print(f'WORKING ON DATA: {data_path}')
print(10*'#')
print(data_set.history.keys())

####################################
#########  PREPARE Model   #########
####################################

# initialise model: we just need to tell which molecules to load and give the data_set to get the shape and spectral bins
if use_interp: MyModel = AtmoModel(data_set,interpoler_path=interpoler_path)
else: MyModel = AtmoModel(data_set,molecules)

##############################################
##############  Check with CCF  ##############
##############################################
if do_ccf:
    injected_model_params = {
         'Tiso': 1200,
         'log_MMR': {
              'Na': -3,
              'FeH' : -3,
         },
        #  'log_Pcloud' : 1
    }
    
    if use_interp: wave_model, transit_radius = MyModel.interpolate_pRT(injected_model_params, plot=True)
    else: wave_model, transit_radius = MyModel.compute_pRT(injected_model_params, plot=True)

    plt.figure()
    plt.plot(wave_model,transit_radius)
    plt.savefig(img_dir+'/'+f'temp3.png')

    result,CCF = CCF_2D(wave_model, transit_radius, data_set, Kp_ref=ref_value_dic['Kp [km/s]']*1e3,V0_ref=ref_value_dic['V0 [km/s]']*1e3,
                    lower_to_instrument_resolution=instrument,
                    Kpmin=0,Kpmax=500e3,V0min=-150e3,V0max=+150e3,
                    NKp=500,NV0=300,NVtot=500,mav_area=119,
                    divide_by_sigma = True,plot=True,apply_svd=do_svd,build_interp_grid=False,
                    Nb_threads=8,ccf_mask=[-ref_value_dic['V0 [km/s]']*1e3-25e3,ref_value_dic['V0 [km/s]']*1e3+25e3,ref_value_dic['Kp [km/s]']*1e3-100e3,ref_value_dic['Kp [km/s]']*1e3+100e3]
                    )


    # add position of expected signature
    plt.hlines(ref_value_dic['Kp [km/s]'],-150,150,lw=0.5,ls='--',color='r')
    plt.vlines(ref_value_dic['V0 [km/s]'],0,500,lw=0.5,ls='--',color='r')
    plt.savefig(img_dir+'/'+f'{name}_CCF.png')

    plt.show()

##############################################
##########  PREPARE NestedSampling  ##########
##############################################
if not do_nested: raise NameError('Stopping here since do_nested = False')

# use global variables to avoid slowing logL estimation with unecessary calculus
data = data_set.data
# data = injected_synth.synthetic
N = data.size # number of data point

# Define the log-likelihood function from Gibson et al. 2020
def logL(theta):
    '''
    params: (Kp, V0, Tiso, **log_Abund, log_Pcloud)
    log_Abund is now a dictionnary with specie:abundance
    '''
    # grab parameters
    Kp = float(theta[0]) * 1e3 # convert km/s to m/s
    V0 = float(theta[1]) * 1e3 # convert km/s to m/s
    abund = {}
    for k,specie in enumerate(molecules):
        abund[specie] = float(theta[k+3])
    model_params = {
        'Tiso' : float(theta[2]),
        'log_MMR' : abund,
    }
    if ref_cloud: model_params['log_Pcloud'] = float(theta[-1])
    if fix_cloud: model_params['log_Pcloud'] = fix_cloud

    # compute the synthetic model with the new parameters
    model = MyModel.compute_synthetic(Kp,V0,model_params,mav_area=119, apply_svd=do_svd, plot=False, lower_to_instrument_resolution=instrument,verbose=False)
        
    # Gibson et al. 2020 definition has two scaling factor alpha & beta. However Hood et al. 2024 say they can be fixed at 1 in most cases (to check...)
    # alpha = theta[-2]
    # beta = theta[-1]
    alpha = 1
    beta = 1

    # compute chi2 & log-likelihood
    chi2 = np.ma.masked_invalid( (data - alpha*model)*(data - alpha*model) / (beta*beta*sigma*sigma) ) # x*x is faster than x**2
    logL = -N/2*np.log(2*np.pi) -N*np.log(beta) - np.sum(np.log(sigma)) - 0.5*np.sum(chi2)
    
    return logL

# Prior are defined through a transform function that takes in argument the parameter guess in the form of a quantile and return the physical value corresponding to this quantile for the given parameter. 
# We first define a function that will perform the transformation for a uniform prior given the two extremal values (low & high). Taking a quantile as argument, it returns the corresponding value of the variable
# For example, in the case of uniform prior, quantile = 1 will return the upper bound of the prior range (since the parameter has 100% chance of being under or equal this value), while quantile = 0.5 will return the mean of the range (since the parameter's value has 50% chance of being under this value)

# set the transform function that will provide priors for each of our parameters
def prior_transform(cube):
    # cube contains the quantile value for each parameter
    bounds = [  [50,200],       # Kp in km/s 
                [-10,10],       # V0 in km/s
                [500,2000],]    # Tiso in K      
    
    # let species range from 1e-6 to 1e-2 MMR
    for specie in molecules:
         bounds.append([-6,-2])

    # let cloud (if any) range from 1e-6 to 1 bar
    if ref_cloud: bounds.append([-6,1]) # logPcloud in bar

    # # if a & b parameters
    # bounds.append([0,100]) # a factor from Gibson paper
    # bounds.append([0,100]) # b factor from Gibson paper

    # transform prior to match pymultinest definition
    bounds = np.array(bounds)
    for k in range(bounds.shape[0]):
         cube[k] = cube[k]*(bounds[k,1]-bounds[k,0]) + bounds[k,0]
    return cube

# Setup the sampler: SPECIES NEED TO BE IN SAME ORDER AS IN THE INTERPOLATION GRID !!
parameter_names = ['Kp [km/s]','V0 [km/s]','Tiso [K]']
for specie in molecules:
    parameter_names.append(f'log_Abund{specie} [MMR]')
if ref_cloud: parameter_names.append('log_Pcloud [bar]')
# # if a & b free
# parameter_names.append('a')
# parameter_names.append('b')
print(parameter_names)

## For debugging: testing manual computation of logL & prior_transform to check for any error
# t0 = time.time()
# print(logL([100,-10,1300,-3,-3,-3]))
# print(prior_transform([1,1,1,1,1,1]))
# print(time.time()-t0)

# Setup the names for saving the run
if not f"{name}/" in os.listdir(f"{save_path}/{log_dir}/"):
     try: os.mkdir(f"{save_path}/{log_dir}/{name}/")
     except: pass
prefix = f"{save_path}/{log_dir}/{name}/{name}-"

# save the expected or truth (ref) value to be latter plotted in the corner plot
with open('%srefvalues.json' % prefix, 'w') as f:
     json.dump(ref_value_dic, f)

##############################
##########  RUN !!  ##########
##############################

t0 = time.time()
result = pymultinest.solve(logL, prior_transform, len(parameter_names),
                           outputfiles_basename = prefix,
                           resume = False, 
                           verbose = True,
                           n_live_points=400,
                           n_iter_before_update=1)

# Store the parameter names for plotting:
with open('%sparams.json' % prefix, 'w') as f:
	json.dump(parameter_names, f, indent=2)

# Print output:
print()
print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')
for name, col in zip(parameter_names, result['samples'].transpose()):
	print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
print(f'DONE IN {(time.time()-t0)//60}min {(time.time()-t0)%60}sec')
