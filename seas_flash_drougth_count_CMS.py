from glob import glob
import os
import time

import numpy as np
import xarray as xr

# Set this so all attributes are propagated with operations
xr.set_options(keep_attrs=True, enable_cftimeindex=True)

def copy_encoding(target, source):
    """
    Try and copy all encodings from a source to a target
    """
    vars = list(target.coords)
    if hasattr(target, 'data_vars'):
        vars.extend(list(target.data_vars))

    for v in vars:
        try:
            target[v].encoding = source[v].encoding
        except:
            print("Failed to copy encoding for {}".format(v))
    target.encoding = source.encoding

def find_fd1D_loop(array, dtime, percentiles, verbose=False):
    """
    Loop through all times in the array using state variables to
    track what to look for at each position
    """

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
            # Set this flag as drought now broken
            had_dry_day = False
        elif last_wet_day is not None and not had_dry_day: 
            if (i - last_wet_day) <= dtime:
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
        # Locations matching wet days will be back-filled above
        array[i] = 0

    return array

def find_fd1D_mask(array, dtime, percentiles, verbose=False):

    """
    Use masks to determine the indices of times that match the
    initial criteria, and only loop through a list of those positions
    """

    indices = np.arange(len(array))

    # Return indices where wet and moist are true
    moist = np.where(array>=percentiles[1],indices,0).nonzero()[0]
    wet   = np.where(array>=percentiles[2],indices,0).nonzero()[0]

    duration = np.zeros_like(wet)

    # Cycle over all wet indices
    for ind, windex in enumerate(wet):
        if verbose: print('windex',windex)
        # Check for dry spell within dtime
        for i in range(windex+1,min(windex+dtime+1,len(array))):
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


def wrap_find_fd(dataset, dtime=7, function=find_fd1D_mask, numpy=True, verbose=False):
    """
    Call the correct find function with precomputed percentiles 
    """
    if numpy:
        # Pass data as underlying numpy array (.values). A bug in numpy (currently 1.16.2)
        # makes this operation 500x faster if numpy array passed directly. This loses the
        # xarray metadata, so need to use the .copy function explicitly
        # https://github.com/numpy/numpy/issues/8562
        result = dataset.mrsos.copy(data=function(dataset.mrsos.values, dtime, dataset.percentiles.values))
    else:
        result = function(dataset.mrsos, dtime, dataset.percentiles)

    return result


if __name__ == '__main__':

    idir_mrsos  = '/short/w35/dh4185/mrsos_merge/'
    # CMIP5       = ['CanESM2','CSIRO-Mk3-6-0','GFDL-CM3','GFDL-ESM2G','GFDL-ESM2M','MIROC5']#
    CMIP5       = ['CanESM2',]
    seasons     = ['DJF','MAM','JJA','SON']

    dt = 10

    time_slice = slice('1867-01','2005-12')

    for model in CMIP5:

        file        = glob(os.path.join(idir_mrsos,'mrsos_day_'+model+'_historical_r1i1p1_*.nc'))[0]
        print('{}\n Read data from {}...'.format(model, file))
        # Use chunking to reduce the memory overhead
        mrsos       = xr.open_dataset(file, chunks={'lat':10,'lon':10}).sel(time=time_slice)

        # Create a stacked (2D) version of the dataset, which creates a new axis called latlon,
        # which is a combination of the lat and lon axes. This just adds another index, so it is
        # a fast operation. Do this to make it simple to mask the data later
        mrsos_stack = mrsos.mrsos.stack(latlon=('lat','lon'))

        print('Size of raw data {}'.format(mrsos_stack.shape))

        # Get the max and min value in time at every point in the new stacked latlon axis
        mrsos_max = mrsos_stack.max(dim='time')
        mrsos_min = mrsos_stack.min(dim='time')

        # Make a masked version of the data and pull out only those pixels where the value changes 
        # over time. In polar regions it is set to the same value constantly, and in oceans it 
        # is always zero. This reduces the data to 23% of the original
        mrsos_masked = mrsos_stack.where(mrsos_max != mrsos_min, drop=True)

        print('Size of masked data {}'.format(mrsos_masked.shape))

        # It is fast to make one call to quantile for the entire dataset. Takes just over 1s
        # for a 52925 x 1958 array (first dim is time, second is masked lat*lon)
        print('Pre-calculate percentiles')
        percentiles = mrsos_masked.load().quantile([0.1,0.3,0.4]).rename('percentiles')

        # Make a dataset with the rainfall data and the percentiles so we can iterate over it
        # and access the pre-computed percentiles in the same way
        mrsos_perc = xr.merge([mrsos_masked, percentiles])

        print('Find droughts')
        # Use a groupby here which effectively just loops over all the locations and applies
        # the wrap_find_fd function
        result = mrsos_perc.groupby('latlon').apply(wrap_find_fd, dtime=dt)

        # Make the dataset 3D again by unstacking the latlon index
        result = result.unstack()

        # Now some tidying. The longitude axis loses it ordering in the above calls, so re-sort
        result = result.unstack().reindex(lon=sorted(result.lon.values))

        # Because it was masked, there are also whole bands of lon/lat missing in the output, where
        # there was no data. So make set the original data to NaN, and combine with the calculated
        # data. This will add back missing lat/lon which exist in the original data, and fill the missing
        # locations with the data from the original data, which was set to NaN
        # When data is chunked must do a load before assignment
        mrsos.mrsos.load()
        mrsos.mrsos[:] = np.nan
        result = result.combine_first(mrsos.mrsos)

        # Write out the result array which has the length of the drought at each point that
        # satisfied the criteria
        print('Writing result')
        # result.mrsos.encoding = mrsos.mrsos.encoding
        copy_encoding(result, mrsos)
        result.to_netcdf(path='result_{}.nc'.format(model))

        # To find the number of occurences of drought by season can use a sum if all the values if
        # the length of each drought is set to 1. 
        result_season = result.where(result>=1,1).groupby('time.season').sum(dim='time')
        result_season.to_netcdf(path='result_{}_season.nc'.format(model))

        # To get events per decade, add an index for decade and group by that
        result.coords['decade'] = (result.time.dt.year // 10) * 10
        result_decade = result.groupby('decade').sum(dim='time')
        result_decade.to_netcdf(path='result_{}_decadal.nc'.format(model))
