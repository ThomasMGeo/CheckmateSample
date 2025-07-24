import numpy as np
import xarray as xr


def make_checkerboard(
    board_size: tuple[int, int],
    square_size: tuple[int, int],
    separation_size: int = None,
    validation: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Create a checkerboard pattern.

    This function generates a checkerboard pattern that can be used for spatial
    cross-validation in machine learning to combat spatial autocorrelation.
    Props to stackoverflow user Blubberguy22, posted March 17, 2020 at 19:00
    https://stackoverflow.com/questions/2169478/how-to-make-a-checkerboard-in-numpy

    Parameters:
        board_size (tuple[int, int]): Size of the board in (rows, columns).
            Must be positive integers.
        square_size (tuple[int, int]): Size of each square in (rows, columns).
            Must be positive integers.
        separation_size (int, optional): Size of the separation mask between squares
            in pixels. If provided, creates a buffer zone with value -1 between
            checkerboard squares. Must be a non-negative integer. Default is None.
        validation (bool, optional): If True, creates a ternary pattern with values
            (0, 1, 2) for train/test/validation splits. If False, creates a binary
            pattern with values (0, 1) for train/test splits. Default is False.
        verbose (bool, optional): If True, prints warnings for non-square inputs.
            Default is False.

    Returns:
        np.ndarray: Checkerboard pattern as a 2D NumPy array with dtype float32.
            Values are 0 and 1 for binary pattern, 0, 1, and 2 for ternary pattern,
            and -1 for separation areas when separation_size is specified.

    Raises:
        ValueError: If any dimension is non-positive or separation_size is negative.

    Examples:
        >>> # Basic binary checkerboard
        >>> board = make_checkerboard((8, 8), (2, 2))
        >>> board.shape
        (8, 8)

        >>> # Ternary pattern for train/test/validation
        >>> board = make_checkerboard((10, 10), (2, 2), validation=True)
        >>> np.unique(board)
        array([0., 1., 2.])

        >>> # With separation between squares
        >>> board = make_checkerboard((10, 10), (3, 3), separation_size=1)
        >>> -1 in board  # Check for separation areas
        True
    """
    rows, cols = board_size
    sq_rows, sq_cols = square_size
    sep = separation_size

    # Check for invalid inputs
    if rows <= 0 or cols <= 0 or sq_rows <= 0 or sq_cols <= 0:
        raise ValueError("All input dimensions must be positive integers.")
    if sep and sep < 0:
        raise ValueError("Separation size must be a non-negative, non-zero integer")

    # Add warning if inputs are not the same
    if verbose and (rows != cols or sq_rows != sq_cols):
        print(
            "Warning: The inputs for board_size or square_size are not the same. This may result in a non-square checkerboard."
        )

    # if separation is given, modify the row and column sizes
    if sep:
        sq_rows += sep
        sq_cols += sep

    # Calculate the checkerboard pattern efficiently
    row_indices = np.arange(rows) // sq_rows
    col_indices = np.arange(cols) // sq_cols

    if validation:
        # Create a 3-value checkerboard pattern (0, 1, 2)
        checkerboard = (row_indices[:, np.newaxis] + col_indices) % 3
    else:
        # Create a 2-value checkerboard pattern (0, 1)
        checkerboard = (row_indices[:, np.newaxis] + col_indices) % 2

    # mask the board with a separation mask if given
    if sep:
        for i in range(0, rows + sep, sq_rows):
            checkerboard[i - sep : i] = -1
        for j in range(0, cols + sep, sq_cols):
            checkerboard[:, j - sep : j] = -1

    return checkerboard.astype("float32")


def make_checkerboard_xr(
    da: xr.DataArray,
    square_size: tuple[int, int],
    separation_size: int = None,
    keep_pattern: int = 1,
    validation: bool = False,
    dim_names: dict = None,
    verbose: bool = False,
) -> xr.DataArray:
    """
    Apply a checkerboard pattern to an existing xarray DataArray.

    This function applies a checkerboard mask to an xarray DataArray, keeping only
    the specified pattern areas and setting others to NaN. Useful for spatial
    cross-validation in geospatial machine learning applications.

    Parameters:
        da (xr.DataArray): Input DataArray to apply the checkerboard pattern to.
            Must have at least 2 spatial dimensions.
        square_size (tuple[int, int]): Size of each square in pixels (y, x).
            Must be positive integers. Determines the size of checkerboard squares.
        separation_size (int, optional): Size of separation between squares in pixels.
            If provided, creates buffer zones between squares. Must be non-negative.
            Default is None.
        keep_pattern (int, optional): Which part of the pattern to keep.
            For binary patterns: 0 or 1. For ternary patterns: 0, 1, or 2.
            Areas not matching this pattern will be set to NaN. Default is 1.
        validation (bool, optional): If True, creates a ternary (0, 1, 2) pattern
            for train/test/validation splits. If False, creates a binary (0, 1)
            pattern for train/test splits. Default is False.
        dim_names (dict, optional): Dictionary specifying the names of x and y dimensions.
            Format: {'x': 'x_dim_name', 'y': 'y_dim_name'}
            If None, automatically detects common dimension names like 'x', 'y',
            'lat', 'lon', 'latitude', 'longitude'. Default is None.
        verbose (bool, optional): If True, prints warnings for non-square inputs.
            Default is False.

    Returns:
        xr.DataArray: Input DataArray with checkerboard pattern applied.
            Areas not matching keep_pattern are set to NaN. Original attributes
            are preserved with additional checkerboard metadata added.

    Raises:
        ValueError: If square_size dimensions are non-positive, keep_pattern is
            invalid for the chosen validation mode, separation_size is negative,
            or required dimensions cannot be found in the DataArray.

    Examples:
        >>> import xarray as xr
        >>> import numpy as np
        >>>
        >>> # Create sample DataArray
        >>> da = xr.DataArray(np.random.rand(20, 30),
        ...                   dims=['lat', 'lon'],
        ...                   coords={'lat': range(20), 'lon': range(30)})
        >>>
        >>> # Apply binary checkerboard, keep pattern 0 (training data)
        >>> train_data = make_checkerboard_xr(da, (5, 5), keep_pattern=0)
        >>>
        >>> # Apply binary checkerboard, keep pattern 1 (test data)
        >>> test_data = make_checkerboard_xr(da, (5, 5), keep_pattern=1)
        >>>
        >>> # Ternary pattern with validation set
        >>> val_data = make_checkerboard_xr(da, (4, 4), keep_pattern=2,
        ...                                validation=True)
        >>>
        >>> # Custom dimension names
        >>> custom_da = xr.DataArray(np.random.rand(15, 25),
        ...                          dims=['custom_y', 'custom_x'])
        >>> result = make_checkerboard_xr(custom_da, (3, 3),
        ...                               dim_names={'y': 'custom_y', 'x': 'custom_x'})
    """
    sq_y, sq_x = square_size
    sep = separation_size

    # Check for invalid inputs
    if sq_y <= 0 or sq_x <= 0:
        raise ValueError("Square size dimensions must be positive integers.")
    if validation and keep_pattern not in [0, 1, 2]:
        raise ValueError("For validation (ternary pattern), keep_pattern must be 0, 1, or 2.")
    elif not validation and keep_pattern not in [0, 1]:
        raise ValueError("For non-validation (binary pattern), keep_pattern must be 0 or 1.")
    if sep and sep < 0:
        raise ValueError("Separation size must be a non-negative, non-zero integer")

    # Determine x and y dimensions
    if dim_names is None:
        # Try to automatically detect dimensions
        possible_y_dims = ["y", "lat", "latitude"]
        possible_x_dims = ["x", "lon", "longitude"]

        y_dim = next((dim for dim in da.dims if dim in possible_y_dims), None)
        x_dim = next((dim for dim in da.dims if dim in possible_x_dims), None)

        if y_dim is None or x_dim is None:
            raise ValueError("Could not automatically detect x and y dimensions. Please specify using dim_names.")
    else:
        y_dim = dim_names.get("y")
        x_dim = dim_names.get("x")

        if y_dim is None or x_dim is None:
            raise ValueError("Both 'x' and 'y' must be specified in dim_names.")

        if y_dim not in da.dims or x_dim not in da.dims:
            raise ValueError(f"Specified dimensions {y_dim} and {x_dim} not found in DataArray.")

    y_size, x_size = da.sizes[y_dim], da.sizes[x_dim]

    # Add warning if inputs are not the same
    if verbose and (y_size != x_size or sq_y != sq_x):
        print(
            "Warning: The inputs for board_size or square_size are not the same. This may result in a non-square checkerboard."
        )

    # if separation is given, modify the row and column sizes
    if sep:
        sq_x += sep
        sq_y += sep

    # Calculate the checkerboard pattern efficiently
    y_indices = (np.arange(y_size) // sq_y)[:, np.newaxis]
    x_indices = np.arange(x_size) // sq_x

    if validation:
        checkerboard = (y_indices + x_indices) % 3
    else:
        checkerboard = (y_indices + x_indices) % 2

    # mask the board with a separation mask if given
    if sep:
        for i in range(0, y_size + sep, sq_y):
            checkerboard[i - sep : i] = -1
        for j in range(0, x_size + sep, sq_x):
            checkerboard[:, j - sep : j] = -1

    # Create a DataArray with the checkerboard pattern
    checkerboard_da = xr.DataArray(
        data=checkerboard,
        dims=[y_dim, x_dim],
        coords={dim: da.coords[dim] for dim in [y_dim, x_dim]},
    )

    # Broadcast the checkerboard pattern to match the input DataArray's shape
    checkerboard_da = checkerboard_da.broadcast_like(da)

    # Apply the checkerboard pattern to the input DataArray
    result = da.where(checkerboard_da == keep_pattern, np.nan)

    # Add metadata
    result.attrs.update(da.attrs)
    result.attrs["checkerboard_applied"] = "True"
    result.attrs["checkerboard_square_size"] = str(square_size)
    result.attrs["checkerboard_keep_pattern"] = str(keep_pattern)
    result.attrs["checkerboard_validation"] = str(validation)
    result.attrs["checkerboard_mask_nosample"] = str(-1)
    result.attrs["checkerboard_dims"] = f"y: {y_dim}, x: {x_dim}"

    return result
