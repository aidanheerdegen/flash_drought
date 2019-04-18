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
import cmaps

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is the main function where I try to get the timing when soil moisture drops from >=40th to <10th percentile
# within x days (dtime). It also measures the length until soil moisture is restored to the 30th percentile or more.

def find_fd1D(array, dtime):

    fd_onset_index, length, count = [], [], 0
    seasons      = ['DJF','MAM','JJA','SON']
    norm_x       = (array - array.min()) / (array.max() - array.min())
    condition_10 = (norm_x<np.nanpercentile(norm_x,10,axis=0,keepdims=True))
    condition_30 = (norm_x<np.nanpercentile(norm_x,30,axis=0,keepdims=True))
    condition_40 = (norm_x>=np.nanpercentile(norm_x,40,axis=0,keepdims=True))
    dtime        = dtime
    time_len     = len(array['time'])

    i = dtime
    while i < time_len:

        if condition_40[i-dtime]==True and condition_10[i]==True:
            fd_onset_index.append(i)

            while condition_30[i]==True and (i < time_len-1)==True:
                i += 1
                count+= 1
            length.append(count)
            count = 0

        if i==time_len-1 and count>0:
            length.append(count)
            break
        i += 1

# In the end I want to count the events depending on the season to get a return array of size 4 for each grid cell
# with the amount of events occured per season. The 'length' array saved the duration of each event and will vary in size
# for each grid cell. I'm not sure yet how to get a uniform output for this for each grid cell to save it into lat,lon array.

    seas_countfd = np.zeros(4)
    for seas, i in enumerate(seasons):

        seas_countfd[i] = array[fd_onset_index].sel(time=(array.time.dt.season==seas)).sum()

    return seas_countfd, length
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

idir_mrsos  = '/short/w35/dh4185/mrsos_merge/'
CMIP5       = ['CanESM2','CSIRO-Mk3-6-0','GFDL-CM3','GFDL-ESM2G','GFDL-ESM2M','MIROC5']#
seasons     = ['DJF','MAM','JJA','SON']

dt = 10

for i, model in enumerate(CMIP5):
    print(model+'\nRead data...')

    file        = glob(idir_mrsos+'mrsos_day_'+model+'_historical_r1i1p1_*.nc')[0]
    mrsos       = xr.open_dataset(file).sel(time=slice('1867-01','2005-12'))

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
    cmap=cmaps.MPL_s3pcpn
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
