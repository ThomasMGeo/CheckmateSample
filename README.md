# CheckmateSample

This Python module provides a function to generate a checkerboard pattern using NumPy. The function make_checkerboard creates a 2D NumPy array representing a checkerboard with customizable board size and square (and rectangle) sizes. Can be used for geospatial ML problems to try and combat autocorrelation. The functions can export two (testing/training) and three (testing/training/validation) checkerboard patterns


## Parameters

board_size (tuple[int, int]): Size of the board in (rows, columns).

square_size (tuple[int, int]): Size of each square in (rows, columns).

validation (bool): If True, use a ternary (0, 1, 2) pattern; if False, use a binary (0, 1) pattern

xarray only:

dim_names (dict): Dictionary specifying the names of x and y dimensions
da (xr.DataArray): Input DataArray to apply the checkerboard pattern to

## Dependencies

numpy and xarray

## Useage

# Create a NumPy checkerboard
checkerboard = make_checkerboard((10, 10), (2, 2))

# Create an xarray DataArray
da = xr.DataArray(np.random.rand(100, 100), dims=['lat', 'lon'])

# Apply checkerboard pattern to DataArray
result = make_checkerboard_xr(da, (10, 10), keep_pattern=1)
