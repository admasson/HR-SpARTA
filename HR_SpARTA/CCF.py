################################################################################
## Module to handle the CCF computation
## Coding : utf-8
## Author : Adrien Masson (adrien.masson@obspm.fr)
## Date   : December 2021
################################################################################

#-------------------------------------------------------------------------------
# Import usefull libraries and modules
#-------------------------------------------------------------------------------
#<f Import libraries
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import threading

# some global variable for threading
lock = threading.Lock() # Define a lock for safe access
counter = 0 # counter for threading monitoring

# Safety check when module has been imported for debugging purpose
print(f'{datetime.datetime.now().time()} : CCF.py has been loaded')

#-------------------------------------------------------------------------------
# Compute the Cross Correlation Function between a given DataSet (reduced data
# from SPIRou observations) and a SynthObject containing the shifted and
# interpolated model to use, along with the Kp and V0 parameters to explore
# with the CCF
#-------------------------------------------------------------------------------

def CCF_full_computation(synth,data_set,
                         Kpmin=0,Kpmax=500e3,V0min=-150e3,V0max=+150e3,NKp=500,NV0=300,NVtot=500,Ninterp=1000,
                         divide_by_sigma = True,plot=False,apply_svd=False, build_interp_grid=True,
                         Kp_ref=150e3,V0_ref=0,Nb_threads=0,ccf_mask=[-50e3,50e3,50e3,350e3]
                         ):
    '''
    Compute the CCF btw the given synthetic class & data class
    Kp_ref & V0_ref are used to plot an expected CCF position slope in the time serie and also show the 1D CCF at the given ref positions in the (Kp,V0) space
    Nb threads: put 0 if you don't want parallelization
    '''
    global counter
    counter = 0 # reset counter for threading
    # grab parameters from data_set
    data = np.copy(data_set.data)
    # center at 0 if isn't
    data -= np.nanmean(data)
    # get nb of obs
    N_obs = data.shape[0]
    ### build the interpolation grid ###
    # data in Earth RF
    Vtot_min = 2*synth.compute_RV(data_set.Vp,np.abs(Kpmin),V0min,0               ) # taking abs to have negative Vtot_min even if Kp is negative (which is physcally impossible but can serve as a sanity check)
    Vtot_max = 2*synth.compute_RV(data_set.Vp,Kpmax        ,V0max,synth.shape[0]-1)
    Vtot_range = np.linspace(Vtot_min,Vtot_max,NVtot)
    
    # sometimes we already build the intepr grid so there no use to compute it again
    # if build_interp_grid: synth.build_interp_grid(Vtot_min,Vtot_max,Ninterp) # with the oversampled model, comuting the interpolation grid is now longer and less effective than directly shifting the model on the fly during the CCF time series computation
    shape = data_set.shape
    
    # set the grid range
    Kp_range = np.linspace(Kpmin,Kpmax,NKp)
    V0_range = np.linspace(V0min,V0max,NV0)
    # compute sigma
    sigma = data_set.weighted_std(data_set.data)
    if not divide_by_sigma: sigma = np.ones(sigma.shape)
    # divide data by sigma**2
    # divide by sigma: we do it once on data to save computation time when multiplying with the model in the ccf computation
    data_over_sigma2 = data / (sigma**2)
    data_over_sigma2 = np.ma.masked_invalid(data_over_sigma2)

    def ComputeCCF_at_Vtot(CCF,index_list):
        global counter, lock
        # loop over index list
        for index in index_list:
            # grab Vtot
            Vtot = Vtot_range[index]

            synthetic = np.ones_like(data)

            # shift model: we're computing the CCF time series, so here all models can have the same shift along the time axis. It's the signal in the data that is moving from one obs to the other
            # synth_shifted = synth.grid_interpoler(Vtot)
            synth_shifted = synth.doppler_shift_and_sample(Vtot).T # Since we now parallelize the CCF time series, it's faster to directly shift the spectrum here rather than pre computing an inteprolation grid

            for obs in range(synth.shape[0]):
                synthetic[obs] = synth_shifted.T
                # apply transit weight
                # We want a flux at 1 out-of-transit and a drop in flux of W*delta_F (W: transit window weight & delta_F transmission spectrum of the planet (1-(Rp(lambda)/Rs)**2))
                # the spectrum from pRT is (1-(Rp(lambda)/Rs)²). We want (1 - W(Rp(lambda)/Rs)²):
                synthetic[obs] = 1-synth.transit_weight[obs]*(1-synthetic[obs])  
            
                # Above is true when we want to create a realistic signature (e.g. to inject it in data) that also contains the planet bulk transit. But for the CCF maybe we want to only have a weight proportional to the transit window applied on the lines ??

                # Now spectral model corresponds to the apparent planet radius in transit (bulk radius + wavelength excess of radius), so we remove the planet bulk contribution to only keep the wavelength transmission part
                # synthetic[obs] = 1 - synth.transit_weight[obs]*((synthetic[obs]-data_set.transit_dic['Rp'])**2 / data_set.transit_dic['Rs']**2)      
                # synthetic[obs] = 1 - synth.transit_weight[obs]*((synthetic[obs])**2 / data_set.transit_dic['Rs']**2)      

            # Apply the svd on the synth. Can be disabled for fast testing (in fact when injecting model before reduction the recovered model gives a worse CCF when applying PCA ?? We should investigate about SysREM & Gibson's method)
            if apply_svd : synthetic = data_set.apply_pca_on_synth(synthetic)

            # remove NaN & inf
            synthetic = np.ma.masked_invalid(synthetic)
            synthetic.fill_value = 0.
            
            # center at 0.
            synthetic -= 1

            # compute ccf
            ccf = np.ma.sum(data_over_sigma2 * synthetic,axis=(1,2))
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
    print(f'Done in {time.time()-tstart}s')

    counter = 0 # reset counter for threading
    TimeSerie_CCF['data*model'] = CCF

    if plot:
        # 2D plots
        plt.figure(figsize=(10,10))
        plt.imshow(TimeSerie_CCF['data*model'],extent=[Vtot_min/1e3,Vtot_max/1e3,0,shape[0]],origin='lower',aspect='auto')
        plt.xlabel('Vtot [km/s]')
        plt.ylabel('Obs n°')
        # plt.ylim((data_set.on_transit_mask[0],data_set.on_transit_mask[-1]))
        plt.xlim((Vtot_min/1e3,Vtot_max/1e3))
        plt.title('CCF')
        c = plt.colorbar()
        c.set_label(f'CCF')
        # plot a Kp line
        plt.plot(synth.compute_RV(data_set.Vp,Kp_ref,V0_ref,np.arange(N_obs))/1e3,np.arange(N_obs),'r',label=f'(Kp,V0) = ({Kp_ref:.0f},{V0_ref:.0f}) m/s')
        plt.plot(synth.compute_RV(data_set.Vp,(Kp_ref+100e3),V0_ref,np.arange(N_obs))/1e3,np.arange(N_obs),'--r',label=f'(Kp,V0) = ({Kp_ref+100e3:.0f},{V0_ref:.0f}) m/s')
        plt.plot(synth.compute_RV(data_set.Vp,(Kp_ref-100e3),V0_ref,np.arange(N_obs))/1e3,np.arange(N_obs),'-.r',label=f'(Kp,V0) = ({Kp_ref-100e3:.0f},{V0_ref:.0f}) m/s')

        plt.legend()
        plt.savefig('CCF_timeSeries')

    ### Interpolate CCF in (Kp,V0) space ###
    f1 = RectBivariateSpline(Vtot_range,np.arange(N_obs),TimeSerie_CCF['data*model'].T)

    def interpolateCCF(Kp,V0):
        '''
        Interpolate CCF values along a diagonal defined by (Kp,V0) in the CCF time-series space (Vtot,time) and return the sum of the CCF interpolated on the diagonal
        '''
        return np.sum(np.diag(f1(synth.compute_RV(data_set.Vp,Kp,V0,np.arange(N_obs)),np.arange(N_obs))))

    def ComputeCCF(result,i,Kp,V0_list,progress):
        for j,V0 in enumerate(V0_list):
            result['data*model'][i,j] = interpolateCCF(Kp,V0)
            progress[0] += 1
        print(f'\r{100*progress[0]/(NKp*NV0):.0f} %',end='',flush=True)
        
    result = {}
    result['data*model'] = np.zeros((NKp,NV0))
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
        
    return(result,CCF/noise_CCF)

def plot_CCF_results(synth,result_list,labels_list=None,ls_list=None):
    '''
    Plot a CCF 2D maps with projection plots based on a list of results from the CCF_full_computation function
    The plot contains a main figure showing the first result in the list in a 2D plot
    The other results in the list are plotted together with labels in the projection plot
    '''
    if labels_list is None: labels_list = ['None' for el in result_list]
    if ls_list is None: ls_list = ['-' for el in result_list] # list of linestyles

    def interpolateCCF(Kp,V0,ccf_interpoler):
        '''
        Interpolate CCF values along a diagonal defined by (Kp,V0) in the CCF time-series space (Vtot,time) and return the sum of the CCF interpolated on the diagonal
        '''
        N_obs = synth.shape[0]
        return np.sum(np.diag(ccf_interpoler(synth.compute_RV(synth.Vp,Kp,V0,np.arange(N_obs)),np.arange(N_obs))))

    # create plotting axes & figures
    fig = plt.figure(figsize=(10,8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1], wspace=0, hspace=0)
    ax_main = fig.add_subplot(gs[0, 1])
    ax_main.tick_params(labelbottom=False, labelleft=False)
    ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main)
    ax_left.set_xlabel("SNR")
    ax_left.set_ylabel(r"$K_p$ [km/s]")
    ax_left.invert_xaxis()  # Optional: Invert x-axis for better alignment
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_main)
    ax_bottom.set_xlabel(r"$V_0$ [km/s]")

    for index,result in enumerate(result_list):
        ### Grab CCF, noise, parameters ###
        CCF = np.copy(result['data*model'])
        noise_CCF = result['noise'] 
        f1 = result['CCF_interp']               # interpolate CCF in the (Vtot,time) space
        V0min    = result['params']['V0min']    # m/s
        V0max    = result['params']['V0max']    # m/s
        Kpmin    = result['params']['Kpmin']    # m/s
        Kpmax    = result['params']['Kpmax']    # m/s
        Kp_ref   = result['params']['Kp_ref']   # m/s
        V0_ref   = result['params']['V0_ref']   # m/s
        Kp_range = result['params']['Kp_range'] # m/s        
        V0_range = result['params']['V0_range'] # m/s        
        
        # Compute the CCF projected along Kp_ref & V0_ref axes
        CCF_1D_Kp = [interpolateCCF(Kp_ref,v0,f1)/noise_CCF for v0 in V0_range]  # 1D CCF at fixed Kp_ref
        CCF_1D_V0 = [interpolateCCF(kp,V0_ref,f1)/noise_CCF for kp in Kp_range]  # 1D CCF at fixed V0_ref

        if index==0:
            # main plot: 2D CCF, only plot for first result in list
            c = ax_main.imshow(CCF/noise_CCF,extent=[V0min/1e3,V0max/1e3,Kpmin/1e3,Kpmax/1e3],origin='lower',aspect='auto')
            ax_main.hlines(Kp_ref/1e3,V0min/1e3,V0max/1e3,lw=0.5,ls='--',color='r')
            ax_main.vlines(V0_ref/1e3,Kpmin/1e3,Kpmax/1e3,lw=0.5,ls='--',color='r')
            ax_main.set_title(f'CCF 2D: {labels_list[index]}')
            # add colorbar for main plot outside the entire grid
            cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
            fig.colorbar(c, cax=cbar_ax, label="SNR")
        # Plot the projection along V0
        ax_left.plot(CCF_1D_V0, Kp_range/1e3, label=labels_list[index],ls=ls_list[index])
        # Plot the projection along Kp
        ax_bottom.plot(V0_range/1e3, CCF_1D_Kp, label=labels_list[index],ls=ls_list[index])

    # Hide unused axes
    fig.add_subplot(gs[1, 0]).axis("off")
    ax_bottom.legend()

def plot_combined_results(synth,result_list,labels_list=None,ls_list=None):
    '''
    Combine results from a CCF result list and plot the result with projections
    The plot contains a main figure showing the first result in the list in a 2D plot
    The other results in the list are plotted together with labels in the projection plot
    '''
    if labels_list is None: labels_list = ['None' for el in result_list]
    if ls_list is None: ls_list = ['-' for el in result_list] # list of linestyles

    def interpolateCCF(Kp,V0,ccf_interpoler):
        '''
        Interpolate CCF values along a diagonal defined by (Kp,V0) in the CCF time-series space (Vtot,time) and return the sum of the CCF interpolated on the diagonal
        '''
        N_obs = synth.shape[0]
        return np.sum(np.diag(ccf_interpoler(synth.compute_RV(synth.Vp,Kp,V0,np.arange(N_obs)),np.arange(N_obs))))

    # create plotting axes & figures
    fig = plt.figure(figsize=(10,8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1], wspace=0, hspace=0)
    ax_main = fig.add_subplot(gs[0, 1])
    ax_main.tick_params(labelbottom=False, labelleft=False)
    ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main)
    ax_left.set_xlabel("SNR")
    ax_left.set_ylabel(r"$K_p$ [km/s]")
    ax_left.invert_xaxis()  # Optional: Invert x-axis for better alignment
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_main)
    ax_bottom.set_xlabel(r"$V_0$ [km/s]")

    # combine CCF
    noise_comb = 1 / np.sqrt(np.sum([1/res['noise']**2 for res in result_list],axis=0))
    CCF_comb = np.sum([res['data*model']/res['noise']**2 for res in result_list],axis=0) / np.sum([1/res['noise']**2 for res in result_list],axis=0)
    
    ### plot ###
    CCF = CCF_comb
    noise_CCF = noise_comb
    # f1 = result['CCF_interp']               # interpolate CCF in the (Vtot,time) space
    V0min    = result_list[0]['params']['V0min']    # m/s
    V0max    = result_list[0]['params']['V0max']    # m/s
    Kpmin    = result_list[0]['params']['Kpmin']    # m/s
    Kpmax    = result_list[0]['params']['Kpmax']    # m/s
    Kp_ref   = result_list[0]['params']['Kp_ref']   # m/s
    V0_ref   = result_list[0]['params']['V0_ref']   # m/s
    Kp_range = result_list[0]['params']['Kp_range'] # m/s        
    V0_range = result_list[0]['params']['V0_range'] # m/s        
    
    # # Compute the CCF projected along Kp_ref & V0_ref axes
    # CCF_1D_Kp = [interpolateCCF(Kp_ref,v0,f1)/noise_CCF for v0 in V0_range]  # 1D CCF at fixed Kp_ref
    # CCF_1D_V0 = [interpolateCCF(kp,V0_ref,f1)/noise_CCF for kp in Kp_range]  # 1D CCF at fixed V0_ref

    # main plot: 2D CCF, only plot for first result in list
    c = ax_main.imshow(CCF/noise_CCF,extent=[V0min/1e3,V0max/1e3,Kpmin/1e3,Kpmax/1e3],origin='lower',aspect='auto')
    ax_main.hlines(Kp_ref/1e3,V0min/1e3,V0max/1e3,lw=0.5,ls='--',color='r')
    ax_main.vlines(V0_ref/1e3,Kpmin/1e3,Kpmax/1e3,lw=0.5,ls='--',color='r')
    ax_main.set_title(f'CCF 2D combined')
    # add colorbar for main plot outside the entire grid
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
    fig.colorbar(c, cax=cbar_ax, label="SNR")

    # # Plot the projection along V0
    # ax_left.plot(CCF_1D_V0, Kp_range/1e3, label=labels_list[index],ls=ls_list[index])
    # # Plot the projection along Kp
    # ax_bottom.plot(V0_range/1e3, CCF_1D_Kp, label=labels_list[index],ls=ls_list[index])

    # Hide unused axes
    fig.add_subplot(gs[1, 0]).axis("off")
    ax_bottom.legend()

def CCF_1D(synth,data_set_list,Kp_ref,
            V0min=-150e3,V0max=+150e3,NV0=300,
            divide_by_sigma = True,plot=False,apply_svd=False,
            ccf_mask=[-50e3,50e3], labels = None, Nb_threads=False,
            ):
    '''
    Compute the 1D CCF btw the given synthetic class & data class at a fixed Kp_ref value
    data_set is a list of data_set that would be plotted together. In this case a list of labels can be provided for plotting the legend.
    Return Results: array of tuples (one per data_set in the list) containing the CCF array and it's std outside the provided mask
    Nb_threads: put 0 if don't want parallelization
    '''
    global counter
    counter = 0
    # grab parameters from data_set
    if labels is None: labels = [f'data_set #{k}' for k in range(len(data_set_list))]
    
    # Array to hold the resulting CCF array for each data set
    Results = []
    # set the grid range
    V0_range = np.linspace(V0min,V0max,NV0)

    for k,data_set in enumerate(data_set_list):
        print(f'Working on data_set #{k+1}/{len(data_set_list)}')
        print()

        # copy data
        data = np.copy(data_set.data)
        N_obs = data.shape[0]
        shape = data_set.shape
        # compute sigma
        sigma = data_set.weighted_std(data_set.data)
        if not divide_by_sigma: sigma = np.ones(sigma.shape)
        # divide by sigma: we do it once on data to save computation time when multiplying with the model in the ccf computation
        data_over_sigma2 = data / (sigma**2)
        data_over_sigma2 = np.ma.masked_invalid(data_over_sigma2)

        # build interp grid
        Vtot_min = 2*synth.compute_RV(data_set.Vp,Kp_ref,V0min,0         )
        Vtot_max = 2*synth.compute_RV(data_set.Vp,Kp_ref,V0max,shape[0]-1)
        synth.build_interp_grid(Vtot_min,Vtot_max,1000)

        # function to compute CCF at a given V0
        def CCF_V0(index, CCF):
            global counter, lock
            '''
            Index is the list of indexes of V0 to grab in the V0 list and the index where to store result in the CCF array
            ''' 
            for i in index:                            
                # grab V0
                V0 = V0_range[i]

                # build synth
                synthetic = np.zeros_like(data)

                # shift model
                for obs in np.where(synth.transit_weight != 0)[0]:
                    Vtot = synth.compute_RV(data_set.Vp,Kp_ref,V0,obs)
                    synth_shifted = synth.grid_interpoler(Vtot)
                    synthetic[obs] = synth_shifted.T
                    # apply transit weight
                    # We want a flux at 1 out-of-transit and a drop in flux of W*delta_F (W: transit window weight & delta_F transmission spectrum of the planet (1-(Rp(lambda)/Rs)**2))
                    # the spectrum from pRT is (1-(Rp(lambda)/Rs)²). We want (1 - W(Rp(lambda)/Rs)²):
                    synthetic[obs] = 1-synth.transit_weight[obs]*(1-synthetic[obs])        

                # Apply the svd on the synth. Can be disabled for fast testing (in fact when injecting model before reduction the recovered model gives a worse CCF when applying PCA ?? We should investigate about SysREM & Gibson's method)
                if apply_svd : synthetic = data_set.apply_pca_on_synth(synthetic)
                
                # remove NaN & inf
                synthetic = np.ma.masked_invalid(synthetic)
                synthetic.fill_value = 0.

                # center at 0.
                synthetic -= 1
                
                # compute CCF
                ccf = np.sum(data_over_sigma2 * synthetic)
                with lock: # Update the shared memory (lock ensure that only one thread get access to memory at time)
                    CCF[i] = ccf
                    counter += 1
                    print(f'\r{counter}/{NV0}',end='',flush=True)

        # CCF array to be filled
        CCF = np.zeros((NV0))

        if Nb_threads:
            # one thread per physicial CPU
            num_threads = Nb_threads
            threads = []
            # Divide the V0 array in equal parts (one per thread)
            index_sub_arrays = np.array_split(np.arange(NV0), num_threads)
            # Assign each thread with one part of the array
            for thread_idx in range(num_threads):
                index_list = index_sub_arrays[thread_idx] # list of index (in V0 range & CCF) the thread will work with
                thread = threading.Thread(target=CCF_V0, args=(index_list,CCF))
                threads.append(thread)
                thread.start()
            # Wait for all threads to finish
            for thread in threads:
                thread.join()
            print()

        else:
            index_list = np.arange(NV0)
            CCF_V0(index_list,CCF)
            print()

        # reset counter to 0
        counter = 0
        # convert to numpy array
        CCF = np.array(CCF)
        # compute CCF standard deviation outside mask
        signal_mask = (V0_range > ccf_mask[0]) * (V0_range < ccf_mask[1]) # True where the signal is thought to be 
        noise_CCF = np.std(CCF[~signal_mask])
        # append to results (one per provided dataset)
        Results.append((CCF,noise_CCF))
    
    # rest of plotting params
    if plot:
        plt.figure()
        for k,res in enumerate(Results):
            CCF, noise_CCF = res
            plt.plot(V0_range/1e3,CCF/noise_CCF,label=labels[k])
        plt.xlabel('V0 [km/s]')
        plt.ylabel('SNR')
        plt.title(f'Kp = {Kp_ref/1e3} km/s')
        plt.vlines(np.array(ccf_mask)/1e3,(CCF/noise_CCF).min(),(CCF/noise_CCF).max(),color='r',ls='--',label='SNR mask')
        plt.legend()

    return(Results)
