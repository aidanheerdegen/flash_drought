import numpy as np
import pytest
import xarray

from seas_flash_drougth_count_CMS import *

def test_find_fd1D_loop():

    # Make a test array of "rainfall"
    a = np.array([10,99,43,35,27,5,5,5,5,5,70,5,35])

    assert((find_fd1D_loop(a, 7, [10, 30, 40], verbose=False) == 
                     np.array([0,0,8,0,0,0,0,0,0,0,2,0,0])).all())

def test_find_fd1D_mask():

    # Make a test array of "rainfall"
    a = np.array([10,99,43,35,27,5,5,5,5,5,70,5,35])

    assert((find_fd1D_mask(a, 7, [10, 30, 40],verbose=False) == 
                     np.array([0,0,8,0,0,0,0,0,0,0,2,0,0])).all())

def test_dubbo():

    filename = 'mrsos_day_CanESM2_historical_r1i1p1_18610101-20051231_dubbo.nc'
     
    ds = xr.open_dataset(filename)