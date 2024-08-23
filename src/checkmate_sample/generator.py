import numpy as np
import xarray as xr

def make_checkerboard(board_size: tuple[int, int], square_size: tuple[int, int], validation: bool = False) -> np.ndarray:
    """
    Create a checkerboard pattern.
    Props to stackoverflow user Blubberguy22, posted March 17, 2020 at 19:00
    https://stackoverflow.com/questions/2169478/how-to-make-a-checkerboard-in-numpy

    Parameters:
        board_size (tuple[int, int]): Size of the board in rows and columns.
        square_size (tuple[int, int]): Size of each square in rows and columns.
        validation (bool): If True, use 0, 1, 2 pattern; if False, use 0, 1 pattern.

    Returns:
        np.ndarray: Checkerboard pattern as a NumPy array.
    """
    rows, cols = board_size
    sq_rows, sq_cols = square_size

    # Check for invalid inputs
    if rows <= 0 or cols <= 0 or sq_rows <= 0 or sq_cols <= 0:
        raise ValueError("All input dimensions must be positive integers.")

    # Add warning if inputs are not the same
    if rows != cols or sq_rows != sq_cols:
        print("Warning: The inputs for board_size or square_size are not the same. This may result in a non-square checkerboard.")
    
    # Calculate the checkerboard pattern efficiently
    row_indices = np.arange(rows) // sq_rows
    col_indices = np.arange(cols) // sq_cols
    
    if validation:
        # Create a 3-value checkerboard pattern (0, 1, 2)
        checkerboard = (row_indices[:, np.newaxis] + col_indices) % 3
    else:
        # Create a 2-value checkerboard pattern (0, 1)
        checkerboard = (row_indices[:, np.newaxis] + col_indices) % 2

    return checkerboard.astype('float32')

def make_checkerboard_xr(
    da: xr.DataArray, 
    square_size: tuple[int, int], 
    keep_pattern: int = 1, 
    validation: bool = False,
    dim_names: dict = None
) -> xr.DataArray:
    """
    Apply a checkerboard pattern to an existing xarray DataArray.

    Parameters:
        da (xr.DataArray): Input DataArray to apply the checkerboard pattern to.
        square_size (tuple[int, int]): Size of each square in pixels (y, x).
        keep_pattern (int): Which part of the pattern to keep. 0, 1 for binary pattern; 0, 1, 2 for ternary pattern.
        validation (bool): If True, use a ternary (0, 1, 2) pattern; if False, use a binary (0, 1) pattern.
        dim_names (dict): Dictionary specifying the names of x and y dimensions. 
                          Format: {'x': 'x_dim_name', 'y': 'y_dim_name'}
                          If None, will attempt to automatically detect dimensions.

    Returns:
        xr.DataArray: Input DataArray with checkerboard pattern applied.
    """
    sq_y, sq_x = square_size

    # Check for invalid inputs
    if sq_y <= 0 or sq_x <= 0:
        raise ValueError("Square size dimensions must be positive integers.")
    if validation and keep_pattern not in [0, 1, 2]:
        raise ValueError("For validation (ternary pattern), keep_pattern must be 0, 1, or 2.")
    elif not validation and keep_pattern not in [0, 1]:
        raise ValueError("For non-validation (binary pattern), keep_pattern must be 0 or 1.")

    # Determine x and y dimensions
    if dim_names is None:
        # Try to automatically detect dimensions
        possible_y_dims = ['y', 'lat', 'latitude']
        possible_x_dims = ['x', 'lon', 'longitude']
        
        y_dim = next((dim for dim in da.dims if dim in possible_y_dims), None)
        x_dim = next((dim for dim in da.dims if dim in possible_x_dims), None)
        
        if y_dim is None or x_dim is None:
            raise ValueError("Could not automatically detect x and y dimensions. Please specify using dim_names.")
    else:
        y_dim = dim_names.get('y')
        x_dim = dim_names.get('x')
        
        if y_dim is None or x_dim is None:
            raise ValueError("Both 'x' and 'y' must be specified in dim_names.")
        
        if y_dim not in da.dims or x_dim not in da.dims:
            raise ValueError(f"Specified dimensions {y_dim} and {x_dim} not found in DataArray.")

    y_size, x_size = da.sizes[y_dim], da.sizes[x_dim]

    # Add warning if inputs are not the same
    if y_size != x_size or sq_y != sq_x:
        print("Warning: The inputs for board_size or square_size are not the same. This may result in a non-square checkerboard.")

    # Calculate the checkerboard pattern efficiently
    y_indices = (np.arange(y_size) // sq_y)[:, np.newaxis]
    x_indices = np.arange(x_size) // sq_x
    
    if validation:
        checkerboard = (y_indices + x_indices) % 3
    else:
        checkerboard = (y_indices + x_indices) % 2

    # Create a DataArray with the checkerboard pattern
    checkerboard_da = xr.DataArray(
        data=checkerboard,
        dims=[y_dim, x_dim],
        coords={dim: da.coords[dim] for dim in [y_dim, x_dim]}
    )

    # Broadcast the checkerboard pattern to match the input DataArray's shape
    checkerboard_da = checkerboard_da.broadcast_like(da)

    # Apply the checkerboard pattern to the input DataArray
    result = da.where(checkerboard_da == keep_pattern, np.nan)

    # Add metadata
    result.attrs.update(da.attrs)
    result.attrs['checkerboard_applied'] = 'True'
    result.attrs['checkerboard_square_size'] = str(square_size)
    result.attrs['checkerboard_keep_pattern'] = str(keep_pattern)
    result.attrs['checkerboard_validation'] = str(validation)
    result.attrs['checkerboard_dims'] = f"y: {y_dim}, x: {x_dim}"

    return result
