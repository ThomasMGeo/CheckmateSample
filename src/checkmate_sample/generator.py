import numpy as np

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
