import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from mpl_toolkits.basemap import shiftgrid
from mpl_toolkits.basemap import Basemap
from numpy import meshgrid
import matplotlib.colors as colors
import os

subplots = [221,222,223,224]

result_seas = xr.open_dataset('result_seasons.nc')

fig = plt.figure(figsize=[17,7])
map = Basemap(llcrnrlon=-180.,llcrnrlat=-60.,urcrnrlon=180.,urcrnrlat=78.,projection='cyl',lat_0=0, lon_0=0)
x = result_seas['lon'].data
y = result_seas['lat'].data
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
