################################################################################
## Module holding the plotting tools
## Coding : utf-8
## Author : Adrien Masson (adrien.masson@obspm.fr)
## Date   : December 2021
################################################################################

#-------------------------------------------------------------------------------
# Import usefull libraries and modules
#-------------------------------------------------------------------------------
#<f Import libraries
import datetime
import matplotlib.pyplot as plt
import numpy as np
from warnings import warn
#f>

# Safety check when module has been imported for debugging purpose
print(f'{datetime.datetime.now().time()} : plots.py has been loaded')

#-------------------------------------------------------------------------------
# Plot all observations for 1 order or all orders for 1 obs in a grid
#-------------------------------------------------------------------------------
def plot_grid(Xarray,Yarray,xticks=[],xticks_labels=[],Xlabel='X',Ylabel='Y',title='',use_mask=True,ylims=[],xlims=[], show = True, **kwargs):
    '''
    Plot a grid given two 2D numpy arrays. For each array, the first
    dimension contains the 1D vectors to plot in each subplots, thus the
    number of subplots is given by the length of this first dimension.

    Parameters :
        - Xarray : TYPE NP.ARRAY - the 2D array containing the X vectors
        - Yarray : TYPE NP.ARRAY - the 2D array containing the Y vectors
        Note that Xarray and Yarray must have the same shape !

        Optionnal arguments :
        - xticks : TYPE LST - a list of x ticks to use for each of the subplots. Must have the same length as Xarray and Yarray first dimension
        - ylims : TYPE LST - a list [ymin, ymax] containing the y limits to use in all subplots. If not provided, matplotlib will automatically choose different limits for each subplots
        - Xlabel : TYPE STR - the label to put in X
        - Ylabel : TYPE STR - the label to put in Y
        - title  : TYPE STR - the title for the grid figure
        - use_mask : TYPE BOOL - wether to use a mask to adapt xlim to hide the edges where the plot is <= 0 （default is True)
        Note that you can add any arguments that plt.plot can understand as **kwargs

    Returns : TYPE LST - a list of all x ticks used to plot each subplots, usefull if you want to plot another grid with the same ticks

    Example :
        Let's say you have two 2D arrays : X that contains 50 vectors each of one correspond to an 100 coordinates x vector of 1 observation,
        and Y also contains 50 vectors, each of one is a 100 coordinates y vector.

        Simply call plot_grid(X,Y) and the function will plot a grid of 50 subplots each of one showing the corresponding x vector Vs y vector
    '''

    Yarray_isLST = isinstance(Yarray,list)

    if(Yarray_isLST):
        for k,Y in enumerate(Yarray):
            # raise an error if not same shape
            if(Xarray.shape != Y.shape) : raise NameError(f'Xarray and Yarray[{k}] must have the same shape ( {Xarray.shape} and {Y.shape} here )  !')
            if(len(xticks)>0):
                if(Xarray.shape[0] != len(xticks)) : raise NameError(f'xticks must have the same length as X and Y arrays first dimension ( {len(xticks)} and {Xarray.shape[0]} here )  !')

    else :
        # raise an error if not same shape
        if(Xarray.shape != Yarray.shape) : raise NameError(f'Xarray and Yarray must have the same shape ( {Xarray.shape} and {Yarray.shape} here )  !')
        if(len(xticks)>0):
            if(Xarray.shape[0] != len(xticks)) : raise NameError(f'xticks must have the same length as X and Y arrays first dimension ( {len(xticks)} and {Xarray.shape[0]} here )  !')

    # computing the grid dimensions
    n_vectors = Xarray.shape[0]
    n_cols = int(np.sqrt(n_vectors))
    n_rows = n_cols
    if n_rows*n_cols < n_vectors : n_rows+=1

    fig,ax = plt.subplots(n_rows,n_cols,figsize=(15,15))

    # list of x ticks for each subplot
    saved_x_ticks = []

    # loop over each vectors
    for vector_index in range(n_vectors):
        sub_ax = ax.flatten()[vector_index]

        if(Yarray_isLST):
            for Y in Yarray:
                sub_ax.plot(Xarray[vector_index,:],Y[vector_index,:],**kwargs)

        else:
            sub_ax.plot(Xarray[vector_index,:],Yarray[vector_index,:],**kwargs)

        sub_ax.text(0.8,0.85,vector_index,transform = sub_ax.transAxes,fontsize = 12) # plot order as a number at up-right edge #
        # sub_ax.set_title(f'Ordre {vector_index}',fontsize=22)

        # only show y labels on far left side
        if(vector_index%n_cols==0):    sub_ax.set_ylabel(Ylabel,fontsize = 10)

        # only show x labels on far bottom side
        if(vector_index>=(n_rows-1)*n_cols) :     sub_ax.set_xlabel(Xlabel,fontsize = 10)

        plt.setp(sub_ax.get_xticklabels(), fontsize=10)
        plt.setp(sub_ax.get_yticklabels(), fontsize=10)

        if (len(xticks)>0):
            ticks = xticks[vector_index]

        else:
            if(use_mask):

                # choosing 4 ticks to display within the area the spectra is not <= 0
                if(Yarray_isLST):
                    mask = Yarray[0][vector_index,:] > 0

                else:
                    mask = Yarray[vector_index,:] > 0

                # prevent errors from empty array by replacing empty mask with an array full of True
                if(not len(Xarray[vector_index,:][mask])==0):
                    xmin = Xarray[vector_index,:][mask].min()
                    xmax = Xarray[vector_index,:][mask].max()
                    sub_ax.set_xlim(xmin,xmax) # adding +-1 to make sure ticks will appear on the edges

                    xmin = 10 * np.floor(0.1 * xmin)
                    xmax = 10 * np.ceil (0.1 * xmax)
                    ticks = [xmin,(xmax+xmin)/2,xmax]

                else:
                    ticks = sub_ax.get_xticks()

            else:
                ticks = sub_ax.get_xticks()

        if (len(xticks_labels)>0):
            sub_ax.set_xticklabels(xticks_labels[vector_index])

        saved_x_ticks.append(ticks)
        sub_ax.set_xticks(ticks)
        sub_ax.tick_params(axis='x', rotation=45) # changing ticks rotation

        # changing y limits if provided
        if(len(ylims)==2):  sub_ax.set_ylim(ylims)
        elif(len(ylims)==0): pass
        else: raise NameError(f'ylims must be in the form [ylim, ymax] with only 2 elements provided (expected size 2, got {len(ylims)})')

        # changing x limits if provided
        if(len(xlims)==2):  sub_ax.set_xlim(xlims)
        elif(len(xlims)==0): pass
        else: raise NameError(f'xlims must be in the form [xlim, xmax] with only 2 elements provided (expected size 2, got {len(xlims)})')

    # Show !
    fig.suptitle(title)
    fig.tight_layout()
    if show: plt.show()

#-------------------------------------------------------------------------------
# Plot Yarray in a 2D imshow using the lines in Xarray as x coordinates for the
# lines in Yarray.
#-------------------------------------------------------------------------------
def plot_imshow(Yarray,extent=None,hlines=[],Xlabel='X',Ylabel='Y',clabel='',title='',**kwargs):
    '''
    Plot Yarray in a 2D imshow using the lines in Xarray as x coordinates for the lines in Yarray.

    Parameters:
        - Yarray : TYPE NP.ARRAY - a 2D array to show with plt.imshow

        Optionnal :
        - Xlabel : TYPE STR - label to put on the x axis
        - Ylabel : TYPE STR - label to put on the y axis
        - extent : TYPE LST - a list giving the x and y axis extensions : [xmin,xmax,ymin,ymax]
        - hlines : TYPE LST - a list of horizontal lines to plot, the list contains the corresponding y coordinates (e.g to plot the transit start and stop)
        - title  : TYPE STR - the figure title
        - clabel : TYPE STR - label for the colorbar
        - **kwargs : any arguments that plt.imshow() can understand

    Return: the plt.imshow() generated object
    '''

    fig = plt.figure(figsize=(10,10))

    im = plt.imshow(Yarray,extent=extent,**kwargs)

    # horizontal lines
    for y in hlines:
        plt.gca().axhline(y,color='r')

    c = plt.colorbar()
    c.set_label(clabel)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(title)

    return(im)
    # TO DO : add another function to plot a 2D imshow with xaxis = wavelength, yaxis = n° of observation (ie times) and color = flux

#-------------------------------------------------------------------------------
# Plot the fraction of removed values (i.e values set to nan or inf) for each
# reduction steps
#-------------------------------------------------------------------------------
def plot_rem_values_fraction(data_set):
    # plot the fraction of removed values (i.e values set to nan) for each reduction steps
    rem_values = [ (~np.isfinite(data_set.history[key]['data'])).sum() for key in data_set.history.keys()]
    rem_values_percentage = np.array(rem_values) / data_set.data.size
    plt.figure(figsize=(10,10))
    plt.plot(rem_values_percentage,'-+')
    plt.vlines(np.arange(0,len(rem_values_percentage)),rem_values_percentage.min()*0.95,rem_values_percentage.max()*1.05,'k',ls=':')
    plt.xticks(ticks= np.arange(len([key for key in data_set.history.keys()])),labels= [key for key in data_set.history.keys()],rotation=45 )
    plt.ylabel('% removed values in data')
    plt.xlabel('Reduction step')
    plt.title('% of removed values evolution during the reduction process')
