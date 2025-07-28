# CheckmateSample

A Python module for generating checkerboard patterns to combat spatial autocorrelation in geospatial machine learning. Supports both NumPy arrays and xarray DataArrays with binary (train/test) and ternary (train/test/validation) patterns.

## Installation

```bash
pip install CheckmateSample
```

## Parameters 

board_size (tuple[int, int]): Size of the board in (rows, columns).

square_size (tuple[int, int]): Size of each square in (rows, columns).

separation_size (int): Size of separation distance in pixels between squares.

validation (bool): If True, use a ternary (0, 1, 2) pattern; if False, use a binary (0, 1) pattern

verbose flag: True / False 

an example of this is:

```
checkerboard_01 = make_checkerboard(board_size=(70,70), square_size=(10,10), separation_size=2)
```

xarray only:

dim_names (dict): Dictionary specifying the names of x and y dimensions
da (xr.DataArray): Input DataArray to apply the checkerboard pattern to

an example of this:


```
ds = xr.tutorial.load_dataset('air_temperature')

air_temp = ds.air.isel(time=0)

square_size = (7, 5)  # 7x5 pixel squares
checkerboard_temp_0 = make_checkerboard_xr(air_temp, square_size, separation_size=3, keep_pattern=0, validation=False)
```

## Dependencies

- numpy >= 1.26.0
- xarray >= 2023.8.0

## Usage

```python
# Create a NumPy checkerboard
checkerboard = make_checkerboard((10, 10), (2, 2))

# Create an xarray DataArray
da = xr.DataArray(np.random.rand(100, 100), dims=['lat', 'lon'])

# Apply checkerboard pattern to DataArray
result = make_checkerboard_xr(da, (10, 10), keep_pattern=1)
```
