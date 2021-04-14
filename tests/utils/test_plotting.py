from typing import List
import rasa.utils.plotting
import numpy as np
import pytest


@pytest.mark.parametrize(
    "data, num_bins, expected_bins",
    [
        ([[1, 3, 8], [2, 3, 3]], 7, list(range(1, 9 + 1))),
        ([[3, 8], [2, 3, 3]], 6, list(range(2, 9 + 1))),
        ([[3, 7], [2, 3, 3]], 5, list(range(2, 8 + 1))),
        ([[3.0, 7.0], [3.0, 7.0]], 2, [3.0, 5.0, 7.0, 9.0]),
    ],
)
def test_paired_histogram_specification_bins(
    data: List[List[float]], num_bins: int, expected_bins: List[float]
):
    """Bin list should run from the lowest data value to the highest + bin_width"""
    for density in [False, True]:
        bins, _, _, _, = rasa.utils.plotting._paired_histogram_specification(
            data,
            num_bins=num_bins,
            density=density,
            x_pad_fraction=0,
            y_pad_fraction=0,
        )
        assert np.all(bins == expected_bins)


@pytest.mark.parametrize(
    "data, num_bins, density, expected_histograms",
    [
        (
            [[1, 3, 8], [2, 3, 3]],
            7,
            False,
            [[1, 0, 1, 0, 0, 0, 0, 1], [0, 1, 2, 0, 0, 0, 0, 0]],
        ),
        (
            [[1, 3, 8], [2, 3, 3]],
            7,
            True,
            [[1 / 3, 0, 1 / 3, 0, 0, 0, 0, 1 / 3], [0, 1 / 3, 2 / 3, 0, 0, 0, 0, 0]],
        ),
        ([[3.0, 7.0], [3.0, 7.0]], 2, False, [[1, 0, 1], [1, 0, 1]]),
        ([[3.0, 7.0], [3.0, 7.0]], 2, True, [[1 / 4, 0, 1 / 4], [1 / 4, 0, 1 / 4]]),
        ([[3.0, 8.0], [3.0, 8.0]], 2, True, [[1 / 5, 0, 1 / 5], [1 / 5, 0, 1 / 5]]),
        ([[3.0, -1.0], [3.0, 7.0]], 4, False, [[1, 0, 1, 0, 0], [0, 0, 1, 0, 1]]),
        (
            [[3.0, -1.0], [3.0, 7.0]],
            4,
            True,
            [[1 / 4, 0, 1 / 4, 0, 0], [0, 0, 1 / 4, 0, 1 / 4]],
        ),
        ([[3.0, 7.0], [3.0, 7.0, 8.5]], 2, False, [[1, 1, 0], [1, 1, 1]]),
    ],
)
def test_paired_histogram_specification_histograms(
    data: List[List[float]],
    num_bins: int,
    density: bool,
    expected_histograms: List[List[float]],
):
    _, histograms, _, _, = rasa.utils.plotting._paired_histogram_specification(
        data, num_bins=num_bins, density=density, x_pad_fraction=0, y_pad_fraction=0,
    )
    assert np.all(histograms[0] == expected_histograms[0])
    assert np.all(histograms[1] == expected_histograms[1])


@pytest.mark.parametrize(
    "data, num_bins, density, x_pad_fraction, expected_ranges",
    [
        ([[1, 3, 8], [2, 3, 3]], 100, False, 0.0, [1.0, 2.0]),
        ([[1, 3, 8], [2, 3, 3, 3, 3]], 100, False, 0.0, [1.0, 4.0]),
        ([[1, 3, 8], [2, 3, 3]], 7, True, 0.0, [2 / 3, 2 / 3]),
    ],
)
def test_paired_histogram_specification_x_ranges(
    data: List[List[float]],
    num_bins: int,
    density: bool,
    x_pad_fraction: float,
    expected_ranges: List[float],
):
    _, histograms, x_ranges, _, = rasa.utils.plotting._paired_histogram_specification(
        data, num_bins=num_bins, density=density, x_pad_fraction=0, y_pad_fraction=0,
    )
    assert np.all(x_ranges == expected_ranges)
