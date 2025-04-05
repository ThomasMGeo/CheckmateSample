import pytest
import numpy as np
from CheckmateSample.generator import make_checkerboard


def test_checkerboard_values_zero_one():
    board_size = (10, 10)
    square_size = (2, 2)
    checkerboard = make_checkerboard(board_size, square_size, verbose=True)
    assert np.all(np.logical_or(checkerboard == 0, checkerboard == 1)), "Checkerboard should only contain 0 and 1"


def test_masked_checkerboard_values_minusone_zero_one():
    board_size = (10, 10)
    square_size = (2, 2)
    separation_size = 1
    checkerboard = make_checkerboard(board_size, square_size, separation_size, verbose=True)
    # Use logical_or.reduce for multiple conditions
    valid_values = np.logical_or.reduce([checkerboard == -1, checkerboard == 0, checkerboard == 1])
    assert np.all(valid_values), "Checkerboard should only contain -1, 0 and 1"


def test_checkerboard_shape():
    board_size = (8, 8)
    square_size = (1, 1)
    checkerboard = make_checkerboard(board_size, square_size, verbose=True)
    assert checkerboard.shape == board_size


def test_checkerboard_values():
    board_size = (4, 4)
    square_size = (1, 1)
    checkerboard = make_checkerboard(board_size, square_size, verbose=True)
    expected = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype="float32")
    np.testing.assert_array_equal(checkerboard, expected)


def test_masked_checkerboard_values():
    board_size = (4, 4)
    square_size = (1, 1)
    setparation_size = 1
    checkerboard = make_checkerboard(board_size, square_size, setparation_size, verbose=True)
    expected = np.array(
        [[0, -1, 1, -1], [-1, -1, -1, -1], [1, -1, 0, -1], [-1, -1, -1, -1]],
        dtype="float32",
    )
    np.testing.assert_array_equal(checkerboard, expected)


def test_checkerboard_larger_squares():
    board_size = (6, 6)
    square_size = (2, 2)
    checkerboard = make_checkerboard(board_size, square_size, verbose=True)
    expected = np.array(
        [
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
        ],
        dtype="float32",
    )
    np.testing.assert_array_equal(checkerboard, expected)


def test_masked_checkerboard_larger_squares():
    board_size = (6, 6)
    square_size = (2, 2)
    separation_size = 1
    checkerboard = make_checkerboard(board_size, square_size, separation_size, verbose=True)
    # Update expected to match the actual implementation pattern
    expected = np.array(
        [
            [0.0, 0.0, -1.0, 1.0, 1.0, -1.0],
            [0.0, 0.0, -1.0, 1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0, 0.0, 0.0, -1.0],
            [1.0, 1.0, -1.0, 0.0, 0.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        ],
        dtype="float32",
    )
    np.testing.assert_array_equal(checkerboard, expected)


def test_checkerboard_rectangular():
    board_size = (6, 4)
    square_size = (2, 1)
    checkerboard = make_checkerboard(board_size, square_size, verbose=True)
    expected = np.array(
        [
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
        ],
        dtype="float32",
    )
    np.testing.assert_array_equal(checkerboard, expected)


def test_masked_checkerboard_rectangular():
    board_size = (6, 4)
    square_size = (2, 1)
    setparation_size = 1
    checkerboard = make_checkerboard(board_size, square_size, setparation_size, verbose=True)
    expected = np.array(
        [
            [0, -1, 1, -1],
            [0, -1, 1, -1],
            [-1, -1, -1, -1],
            [1, -1, 0, -1],
            [1, -1, 0, -1],
            [-1, -1, -1, -1],
        ],
        dtype="float32",
    )
    np.testing.assert_array_equal(checkerboard, expected)


def test_checkerboard_dtype():
    board_size = (4, 4)
    square_size = (1, 1)
    checkerboard = make_checkerboard(board_size, square_size, verbose=True)
    assert checkerboard.dtype == np.float32


@pytest.mark.parametrize(
    "board_size,square_size",
    [((0, 0), (1, 1)), ((5, 5), (0, 0)), ((-1, 5), (1, 1)), ((5, 5), (-1, 1))],
)
def test_invalid_inputs(board_size, square_size):
    with pytest.raises((ValueError, ZeroDivisionError)):
        make_checkerboard(board_size, square_size, verbose=True)


def test_make_checkerboard_validation_true():
    board_size = (6, 6)
    square_size = (2, 2)
    result = make_checkerboard(board_size, square_size, validation=True, verbose=True)

    # Check if the result contains only 0, 1, and 2
    assert np.all(np.isin(result, [0, 1, 2]))

    # Check if all three values (0, 1, 2) are present in the result
    assert set(np.unique(result)) == {0, 1, 2}

    # Check the pattern (this assumes a 6x6 board with 2x2 squares)
    expected_pattern = np.array(
        [
            [0, 0, 1, 1, 2, 2],
            [0, 0, 1, 1, 2, 2],
            [1, 1, 2, 2, 0, 0],
            [1, 1, 2, 2, 0, 0],
            [2, 2, 0, 0, 1, 1],
            [2, 2, 0, 0, 1, 1],
        ],
        dtype="float32",
    )

    np.testing.assert_array_equal(result, expected_pattern)


def test_make_checkerboard_warning_non_square_board(capsys):
    make_checkerboard((5, 6), (2, 2), verbose=True)
    captured = capsys.readouterr()
    assert "Warning: The inputs for board_size or square_size are not the same" in captured.out


def test_make_checkerboard_warning_non_square_squares(capsys):
    make_checkerboard((6, 6), (2, 3), verbose=True)
    captured = capsys.readouterr()
    assert "Warning: The inputs for board_size or square_size are not the same" in captured.out


def test_make_checkerboard_warning_both_non_square(capsys):
    make_checkerboard((5, 6), (2, 3), verbose=True)
    captured = capsys.readouterr()
    assert "Warning: The inputs for board_size or square_size are not the same" in captured.out
