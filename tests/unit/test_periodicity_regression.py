"""Regression test documenting a known bug: non-cubic periodicity settings are silently ignored.

dragen/utilities/Helpers.py:466 has a self-flagged FIXME ("full periodicity applies for the
x != y != z case, which is presumably not 100% reasonable") in make_periodic_3D_new's branch for
when both box_size_y and box_size_z are set. Empirically, that branch's shift computation is
commented out and it falls back to exactly the same np.roll(...) call as the fully-cubic/periodic
branch -- so setting box_size_y/box_size_z (which InputInfo.py documents as turning periodicity
OFF in those directions) currently has NO effect on this function's output at all.
"""
import numpy as np
import pytest

from dragen.utilities.Helpers import HelperFunctions
from dragen.utilities.InputInfo import RveInfo


@pytest.mark.xfail(
    strict=True,
    reason="Helpers.py:466 FIXME -- box_size_y/box_size_z are ignored by make_periodic_3D_new; "
           "the non-cubic branch collapses to the fully-periodic/cubic case.",
)
def test_non_cubic_box_size_changes_periodic_wrapping():
    RveInfo.bin_size = 1.0
    RveInfo.n_pts = 10

    arr = np.zeros((10, 10, 10), dtype='int16')
    arr[1, 1, 1] = 5
    x0, y0, z0 = 3, 3, 3
    helpers = HelperFunctions()

    RveInfo.box_size_y = None
    RveInfo.box_size_z = None
    cubic_result = helpers.make_periodic_3D_new(arr, x0, y0, z0)

    RveInfo.box_size_y = 6
    RveInfo.box_size_z = 4
    non_cubic_result = helpers.make_periodic_3D_new(arr, x0, y0, z0)

    assert not np.array_equal(cubic_result, non_cubic_result), (
        "setting box_size_y/box_size_z should change periodic wrapping behavior, but it doesn't"
    )
