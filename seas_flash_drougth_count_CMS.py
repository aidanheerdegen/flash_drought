import xarray as xr
import numpy as np
from glob import glob
import time
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from mpl_toolkits.basemap import shiftgrid
from mpl_toolkits.basemap import Basemap
from numpy import meshgrid
import matplotlib.colors as colors
import os
# import cmaps

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is the main function where I try to get the timing when soil moisture drops from >=40th to <10th percentile
# within x days (dtime). It also measures the length until soil moisture is restored to the 30th percentile or more.


'''
From https://gist.github.com/Fnjn/b061b28c05b5b0e768c60964d2cafa8d

MIT License
Copyright (c) 2018 Fanjin Zeng
This work is licensed under the terms of the MIT license, see <https://opensource.org/licenses/MIT>.  
'''

def sliding_window_view(x, shape, step=None, subok=False, writeable=False):
    """
    Create sliding window views of the N dimensions array with the given window
    shape. Window slides across each dimension of `x` and provides subsets of `x`
    at any window position.

    Parameters
    ----------
    x : ndarray
        Array to create sliding window views.

    shape : sequence of int
        The shape of the window. Must have same length as number of input array dimensions.

    step: sequence of int, optional
        The steps of window shifts for each dimension on input array at a time.
        If given, must have same length as number of input array dimensions.
        Defaults to 1 on all dimensions.

    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise the returned
        array will be forced to be a base-class array (default).

    writeable : bool, optional
        If set to False, the returned array will always be readonly view.
        Otherwise it will return writable copies(see Notes).

    Returns
    -------
    view : ndarray
        Sliding window views (or copies) of `x`. view.shape = (x.shape - shape) // step + 1

    See also
    --------
    as_strided: Create a view into the array with the given shape and strides.
    broadcast_to: broadcast an array to a given shape.

    Notes
    -----
    ``sliding_window_view`` create sliding window views of the N dimensions array
    with the given window shape and its implementation based on ``as_strided``.
    Please note that if writeable set to False, the return is views, not copies
    of array. In this case, write operations could be unpredictable, so the return
    views is readonly. Bear in mind, return copies (writeable=True), could possibly
    take memory multiple amount of origin array, due to overlapping windows.

    For some cases, there may be more efficient approaches, such as FFT based algo discussed in #7753.

    Examples
    --------
    >>> i, j = np.ogrid[:3,:4]
    >>> x = 10*i + j
    >>> shape = (2,2)
    >>> sliding_window_view(x, shape)
    array([[[[ 0,  1],
             [10, 11]],

            [[ 1,  2],
             [11, 12]],

            [[ 2,  3],
             [12, 13]]],


           [[[10, 11],
             [20, 21]],

            [[11, 12],
             [21, 22]],

            [[12, 13],
             [22, 23]]]])


    >>> i, j = np.ogrid[:3,:4]
    >>> x = 10*i + j
    >>> shape = (2,2)
    >>> step = (1,2)
    >>> sliding_window_view(x, shape, step)
    array([[[[ 0,  1],
             [10, 11]],

            [[ 2,  3],
             [12, 13]]],


           [[[10, 11],
             [20, 21]],

            [[12, 13],
             [22, 23]]]])

    """
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    try:
        shape = np.array(shape, np.int)
    except:
        raise TypeError('`shape` must be a sequence of integer')
    else:
        if shape.ndim > 1:
            raise ValueError('`shape` must be one-dimensional sequence of integer')
        if len(x.shape) != len(shape):
            raise ValueError("`shape` length doesn't match with input array dimensions")
        if np.any(shape <= 0):
            raise ValueError('`shape` cannot contain non-positive value')

    if step is None:
        step = np.ones(len(x.shape), np.intp)
    else:
        try:
            step = np.array(step, np.intp)
        except:
            raise TypeError('`step` must be a sequence of integer')
        else:
            if step.ndim > 1:
                raise ValueError('`step` must be one-dimensional sequence of integer')
            if len(x.shape)!= len(step):
                raise ValueError("`step` length doesn't match with input array dimensions")
            if np.any(step <= 0):
                raise ValueError('`step` cannot contain non-positive value')

    o = (np.array(x.shape)  - shape) // step + 1 # output shape
    if np.any(o <= 0):
        raise ValueError('window shape cannot larger than input array shape')

    strides = x.strides
    view_strides = strides * step

    view_shape = np.concatenate((o, shape), axis=0)
    view_strides = np.concatenate((view_strides, strides), axis=0)
    view = np.lib.stride_tricks.as_strided(x, view_shape, view_strides, subok=subok, writeable=writeable)

    if writeable:
        return view.copy()
    else:
        return view

def find_fd1D_loop(array, dtime, percentiles, verbose=False):

    last_wet_day = None
    had_dry_day = False

    i = 0
    for i, val in enumerate(array):
        if verbose: print(i,val)
        if had_dry_day and val >= percentiles[1]:
            # Record the number of days it took to get wet(ish) again
            # at the position of the last wet day
            array[last_wet_day] = i - last_wet_day
            if verbose:
                print('Recorded wet day lag {} at {}'.format(array[last_wet_day], last_wet_day))
        elif last_wet_day is not None and not had_dry_day: 
            if (i - last_wet_day) < dtime:
                if val < percentiles[0]:
                    # Have had a wet day and found dry day within dtime
                    had_dry_day = True
                    if verbose:
                        print('Found dry day: {} {}'.format(i,val))
            else:
                if verbose:
                    print('Reset dry day: {} {}'.format((i-last_wet_day),val))
                last_wet_day = None
                had_dry_day = False

        if val >= percentiles[2]:
            # Reset the location of the last wet day
            last_wet_day = i
            had_dry_day = None
            if verbose:
                print('Reset last wet day: {} {}'.format(i,val))

        # Set all values of array to zero once their value has been checked
        # Locations matching wet days will be back-filed above
        array[i] = 0

    return array

def find_fd1D_mask(array, dtime, percentiles, verbose=False):

    indices = np.arange(len(array))

    # Return indices where wet and moist are true
    moist = np.where(array>=percentiles[1],indices,0).nonzero()[0]
    wet   = np.where(array>=percentiles[2],indices,0).nonzero()[0]

    duration = np.zeros_like(wet)

    # Cycle over all wet indices
    for ind, windex in enumerate(wet):
        if verbose: print('windex',windex)
        # Check for dry spell within dtime
        for i in range(windex+1,windex+dtime+1):
            if array[i] >= percentiles[2]:
                # Found another wet cell before a dry cell
                if verbose: print('found wet cell, abort search')
                break
            if array[i] < percentiles[0]:
                if verbose: print('dry is true')
                for j, loc in enumerate(moist):
                    if loc > i:
                        if verbose: print(j,loc,loc-windex)
                        duration[ind] = (loc-windex)
                        break
                # Delete all moist indices up to this point
                # so they don't need to be iterated again
                moist = np.delete(moist,j)
                break

    # Save durations in original array at wet points
    array[:] = 0
    array[wet] = duration

    return array




"""
# In the end I want to count the events depending on the season to get a return array of size 4 for each grid cell
# with the amount of events occured per season. The 'length' array saved the duration of each event and will vary in size
# for each grid cell. I'm not sure yet how to get a uniform output for this for each grid cell to save it into lat,lon array.

    seas_countfd = np.zeros(4)
    for seas, i in enumerate(seasons):

        seas_countfd[i] = array[fd_onset_index].sel(time=(array.time.dt.season==seas)).sum()

    return seas_countfd, length
"""
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is the previous funtion I was using which works on a 3D array but does not account for event length and counted
# events twice when soil moisture depleted too quickly and the statement (from >=40th to <10th percentile within x days)
# was true for two consecutive days.

def find_fd(x,dtime,percentile_low,percentile_norm):
    norm_x  = (x - x.min(dim='time')) / (x.max(dim='time') - x.min(dim='time'))
    x_low   = np.nanpercentile(norm_x,percentile_low,axis=0, keepdims=True)[0]
    x_norm  = np.nanpercentile(norm_x,percentile_norm,axis=0, keepdims=True)[0]

    mask    = np.ma.masked_greater_equal(norm_x[:-dtime,:,:],x_norm)
    mask2   = np.ma.masked_less(norm_x[dtime:,:,:],x_low)
    mask_fd = (mask.mask & mask2.mask)
    matrix  = np.zeros(mask_fd.shape)

    for t in range(1,mask_fd.shape[0]):
        matrix[t,:,:] = np.where(mask_fd[t,:,:] > mask_fd[t-1,:,:],1,0)

    matrix = xr.DataArray(matrix,coords=[x['time'][dtime:],x['lat'],x['lon']],dims=['time','lat','lon'])
    return matrix
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':

    idir_mrsos  = '/short/w35/dh4185/mrsos_merge/'
    CMIP5       = ['CanESM2','CSIRO-Mk3-6-0','GFDL-CM3','GFDL-ESM2G','GFDL-ESM2M','MIROC5']#
    seasons     = ['DJF','MAM','JJA','SON']

    dt = 10

    for i, model in enumerate(CMIP5):
        print(model+'\nRead data...')

        file        = glob(idir_mrsos+'mrsos_day_'+model+'_historical_r1i1p1_*.nc')[0]
        # mrsos       = xr.open_dataset(file).sel(time=slice('1867-01','2005-12'))
        mrsos       = xr.open_dataset(file).sel(time=slice('1867-01','1900-12'))

        find_fd1D(mrsos.mrsos.isel(lon=30,lat=40))

        result      = find_fd(mrsos['mrsos'],dt,low,norm)
        result_seas = xr.DataArray(np.zeros((len(seasons),mrsos.shape[1],mrsos.shape[2])),dims=['season','lat','lon'], coords=[seasons, mrsos['lat'], mrsos['lon']])

        # go through each lat and lon to get an 1D array for the find_fd1D function
        for lt in range(len(mrsos['lat'])):
            for ln in range(len(mrsos['lon'])):

                results_seas[:,lt,ln], length = find_fd1D(mrsos['mrsos'],dt)

            result_seas[j,:,:] = result.sel(time=(result.time.dt.season==seas)).sum(dim='time')

        # get events per decade
        result_seas = result_seas/(mrsos['mrsos'].shape[0]/365/10)

        # apply land sea mask
        mask_file   = xr.open_dataset('/short/w35/dh4185/landsea/topo_fx_'+model+'_historical_r0i0p0.nc').squeeze(dim='time')
        mask        = np.ma.masked_greater(mask_file['topo'],-.1)
        result_seas = result_seas.where(mask.mask)


    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # PLOTTING

        subplots = [221,222,223,224]

        fig = plt.figure(figsize=[17,7])
        map = Basemap(llcrnrlon=-180.,llcrnrlat=-60.,urcrnrlon=180.,urcrnrlat=78.,projection='cyl',lat_0=0, lon_0=0)
        x = result['lon'].data
        y = result['lat'].data
        xx, yy = meshgrid(x, y)
        # cmap=cmaps.MPL_s3pcpn
        plt.suptitle('Seasonal number of flash droughts for '+model+' mrsos (1868-2005)\nThreshold: SM from >='+str(norm)+'th prctl to <'+str(low)+'th prctl in '+str(dt)+' days')

        for j, plot in enumerate(subplots):
            ax = fig.add_subplot(plot)
            cs = map.contourf(xx, yy, result_seas[j,:,:],latlon=True,cmap=cmap,levels=np.linspace(-1,40,41),extend='max')
            cb = map.colorbar(location='right',label='FD events per decade')
            # cs.cmap.set_over("white")
            cb.set_ticks(np.round(np.arange(0,42,2),1))
            cb.set_ticklabels(np.round(np.arange(0,42,2),1))
            map.drawparallels(range(-90, 90, 30),labels=[1,0,0,0],fontsize=10)
            map.drawmeridians(range(-180, 180, 45),labels=[0,0,0,1],fontsize=10)
            map.drawcoastlines()
            textstr = r'$\mathrm{max}=%.2f$' % (np.nanmax(result_seas[j,:,:]))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.13, textstr, transform=ax.transAxes, fontsize=12,verticalalignment='top', bbox=props)
            plt.title(seasons[j])

        plt.tight_layout()
        # plt.show()
        plt.savefig('/short/w35/dh4185/EDDI/corr/plots/seas_flashdrought_'+model+'_low'+str(low)+'_norm'+str(norm)+'_dt'+str(dt)+'.png')
        print(model+' done.')
