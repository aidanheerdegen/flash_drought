import numpy as np
import pytest
import xarray

from seas_flash_drougth_count_new import *

def test_find_fd1D_mask():

    # Make a test array of "rainfall"
    a = np.array([10,99,43,35,27,10,5,5,5,40,70,5,35])

    assert((find_fd1D_mask(a, 7, [10, 30, 40],verbose=False) == 
                     np.array([0,0,0,0,0,0,7,0,0,0,0,2,0])).all())

    a = np.array([10,99,43,35,27,10,5,5,5,40,70,5,35])

    assert((find_fd1D_mask(a, 4, [10, 30, 40], verbose=False) == 
                     np.array([0,0,0,0,0,0,7,0,0,0,0,2,0])).all())

    a = np.array([10,99,43,35,27,10,5,5,5,40,70,5,35])

    assert((find_fd1D_mask(a, 3, [10, 30, 40], verbose=False) == 
                     np.array([0,0,0,0,0,0,0,0,0,0,0,2,0])).all())

def test_dubbo():
    """
    Test one time series of CanESM data from near Wagga
    """

    filename = 'dubbo.nc'
     
    ds = xr.open_dataset(filename).isel(lat=0,lon=0)

    percentiles = ds.mrsos.quantile([0.1, 0.3, 0.4])

    res = find_fd1D_mask(ds.mrsos.values, 7, percentiles.values)

    assert(np.all( [np.min(res),np.max(res),np.sum(res),np.count_nonzero(res) ] 
                     == [0.0, 33.0, 100.0, 5,]) )

    assert(np.all( res.nonzero()[0] == [14585, 32807, 33206, 51476, 52185] ))

    res = ds.mrsos.copy(data=res)
    res.to_netcdf('dubbo_res.nc')
