from glob import glob
import os
import sys
import time

import numpy as np
import xarray as xr

start_time = time.time()
last_time  = start_time
def laptime():
    global last_time
    now = time.time()
    print("--- {:.2f} seconds ---".format(now - last_time))
    last_time = now

# Set this so all attributes are propagated with operations
xr.set_options(keep_attrs=True, enable_cftimeindex=True)

#===============================================================================
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



def find_fd1D_mask(array, dtime, percentiles, verbose=False):

    """
    Use masks to determine the indices of times that match the
    initial criteria, and only loop through a list of those positions
    """

    indices = np.arange(len(array))

    # Return indices where wet and moist are true
    moist = np.where(array>=percentiles[1],indices,0).nonzero()[0]
    wet   = np.where(array>=percentiles[2],indices,0).nonzero()[0]

    duration = np.zeros_like(array)

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
                        duration[i] = (loc-windex)
                        break
                # Delete all moist indices up to this point
                # so they don't need to be iterated again
                moist = np.delete(moist,j)
                break

    return duration



def wrap_find_fd(dataset, varname, dtime=7, function=find_fd1D_mask, numpy=True, verbose=False):
    """
    Call the correct find function with precomputed percentiles
    """
    if numpy:
        # Pass data as underlying numpy array (.values). A bug in numpy (currently 1.16.2)
        # makes this operation 500x faster if numpy array passed directly. This loses the
        # xarray metadata, so need to use the .copy function explicitly
        # https://github.com/numpy/numpy/issues/8562
        result = dataset[varname].copy(data=function(dataset.mrsos_est_30d.values, dtime, dataset.percentiles.values))
    else:
        result = function(dataset[varname], dtime, dataset.percentiles)

    return result


def main(models):

    idir_SSI  = '/g/data/w35/dh4185/data/SSI/'
    odir      = '/g/data/w35/dh4185/data/fd_count/prctl_104040_dt14/'

    # Make sure the output directory exists, if it does will error, so 
    # wrap in try/except
    try:
        os.makedirs(odir)
    except:
        pass

    dt    = 14
    prctl = [0.1,0.4,0.4]
    ndays = 30
    varname = 'mrsos_est_{}d'.format(ndays)

    time_slice = slice('1867-01','2005-12')

    for model in models:

        if os.path.isfile('{}result_FDcount_dt{}_{}.nc'.format(odir,dt,model)):
            print('File exists.')
        else:
            file        = glob(os.path.join(idir_SSI,'SSI_agg{}d_{}_historical_r1i1p1_*.nc'.format(ndays,model)))[0]
            print('{}\n Read data from {}...'.format(model, file))
            # Use chunking to reduce the memory overhead
            mrsos       = xr.open_dataset(file).sel(time=time_slice)

            # Create a stacked (2D) version of the dataset, which creates a new axis called latlon,
            # which is a combination of the lat and lon axes. This just adds another index, so it is
            # a fast operation. Do this to make it simple to mask the data later
            mrsos_stack = mrsos[varname].stack(latlon=('lat','lon'))

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
            percentiles = mrsos_masked.load().quantile(prctl, dim='time').rename('percentiles')
            laptime()

            # Make a dataset with the rainfall data and the percentiles so we can iterate over it
            # and access the pre-computed percentiles in the same way
            mrsos_perc = xr.merge([mrsos_masked, percentiles])

            # Use a groupby here which effectively just loops over all the locations and applies
            # the wrap_find_fd function
            print('Find droughts')
            result = mrsos_perc.groupby('latlon').apply(wrap_find_fd, dtime=dt, varname=varname)
            laptime()

            # Make the dataset 3D again by unstacking the latlon index
            print('Unstack')
            result = result.unstack()
            laptime()

            # Now some tidying. The longitude axis loses it ordering in the above calls, so re-sort
            print('Sort axis')
            result = result.unstack().reindex(lon=sorted(result.lon.values))
            laptime()

            # Because it was masked, there are also whole bands of lon/lat missing in the output, where
            # there was no data. So make set the original data to NaN, and combine with the calculated
            # data. This will add back missing lat/lon which exist in the original data, and fill the missing
            # locations with the data from the original data, which was set to NaN
            # When data is chunked must do a load before assignment
            print('Fill in missing bands')
            mrsos[varname].load()
            mrsos[varname][:] = np.nan
            result = result.combine_first(mrsos[varname])
            laptime()

            # Write out the result array which has the length of the drought at each point that
            # satisfied the criteria
            print('Writing result')
            # result.mrsos.encoding = mrsos[varname].encoding
            copy_encoding(result, mrsos)
            result.to_netcdf(path=os.path.join(odir,'result_FDcount_dt{}_{}.nc'.format(dt,model)))
            laptime()

            # # To find the number of occurences of drought by season can use a sum if all the values if
            # # the length of each drought is set to 1.
            # result_season = result.where(result>=1,1).groupby('time.season').sum(dim='time')
            # result_season.to_netcdf(path='{}result_FDcount_dt{}_{}_season.nc'.format(odir,dt,model))

            # # To get events per decade, add an index for decade and group by that
            # result.coords['decade'] = (result.time.dt.year // 10) * 10
            # result_decade = result.groupby('decade').sum(dim='time')
            # result_decade.to_netcdf(path='{}result_FDcount_dt{}_{}_decadal.nc'.format(odir,dt,model))

if __name__ == '__main__':

    # Pass model names as arguments on command line. Possible values are
    # 'CanESM2','CSIRO-Mk3-6-0','GFDL-CM3','GFDL-ESM2G','GFDL-ESM2M','MIROC5'
    main(sys.argv[1:])
