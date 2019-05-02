# Detect Flash Droughts using xarray

The code can be tested using `pytest`:

    pytest -s test_algorithm.py

This will run the functions contained in `test_algorithm.py` that begin with `test_` (which is all of them).

There are two functions used for detecting the drought events. One uses a traditional looping structure, the other uses loop over a smaller number of pre-identified matching locations to speed up execution.

As tested it takes about 12 minutes to calculate the required metrics from a single CMIP5 dataset which is 128x64 pixels with 50735 data points at each pixel.

Run the program like so:

    python seas_flash_drougth_count_CMS.py

There are three output files:

1. `result_{model}.nc` : the same dimensionality as the input data with moisture levels replaced by duration of a flash drought at the date the drought started (last wet day)

2. `result_{model}_seasonal.nc` : a count of the number of flash droughts per season

3. `result_{model}_decadal.nc` : a count of the number of flash droughts per decade
