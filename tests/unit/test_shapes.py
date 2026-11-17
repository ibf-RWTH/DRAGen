import numpy as np
import pytest

from dragen.analysis.shapes import grain_shapes


def test_box_shaped_grain_is_convex_and_elongated():
    labels = np.zeros((20, 20, 20), dtype=int)
    labels[2:14, 5:9, 5:9] = 1                       # 12 x 4 x 4 voxels along x
    s = grain_shapes(labels, (1, 1, 1), (False,) * 3).iloc[0]
    assert s.solidity == pytest.approx(1.0)
    assert s.elongation_2 == pytest.approx(s.elongation_3, rel=1e-6)
    assert s.elongation_2 == pytest.approx(4 / 12, rel=0.05)
    assert s.slope == pytest.approx(0.0, abs=1e-6) or s.slope == pytest.approx(180.0, abs=1e-6)


def test_l_shaped_grain_is_not_convex():
    labels = np.zeros((10, 10, 4), dtype=int)
    labels[0:8, 0:2, :] = 1
    labels[0:2, 0:8, :] = 1
    s = grain_shapes(labels, (1, 1, 1), (False,) * 3).iloc[0]
    assert s.solidity == pytest.approx(28 / 46)          # L of 28 cells, hull pentagon of area 46


def test_slope_of_grain_along_y_is_90():
    labels = np.zeros((10, 20, 10), dtype=int)
    labels[4:6, 2:18, 4:6] = 1
    assert grain_shapes(labels, (1, 1, 1), (False,) * 3).iloc[0].slope == pytest.approx(90.0, abs=1e-6)


def test_grain_across_periodic_boundary_matches_unwrapped_one():
    inside = np.zeros((20, 20, 20), dtype=int)
    inside[6:14, 5:9, 5:9] = 1
    wrapped = np.roll(inside, 10, axis=0)            # now split over x = 19 | 0
    a = grain_shapes(inside, (1, 1, 1), (True,) * 3).iloc[0]
    b = grain_shapes(wrapped, (1, 1, 1), (True,) * 3).iloc[0]
    for col in ('solidity', 'elongation_2', 'elongation_3', 'volume'):
        assert a[col] == pytest.approx(b[col])
