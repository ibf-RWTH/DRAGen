"""Regression test documenting a known bug: RveInfo doesn't fully reset between Run() calls.

dragen/run.py's Run.__init__ only reassigns RveInfo.n_pts_y/n_pts_z *conditionally* (inside
`if box_size_y:` / `if box_size_z:`), while RveInfo.box_size_y/box_size_z themselves are always
reset unconditionally. So a Run() with a sheet/non-cubic box followed by a Run() with a cubic box
in the same process silently keeps the previous run's n_pts_y/n_pts_z, even though
box_size_y/box_size_z correctly report the new (cubic) configuration.

This constructs Run() directly rather than relying on the autouse RveInfo snapshot/restore
fixture (tests/conftest.py), which only resets *between* pytest tests, not within a single test --
this test needs to reproduce the leak between two Run() calls in one process, exactly as it can
happen today when tests/scenarios/run_all_test_cases.py runs multiple scenarios in one session.
"""
import pytest

from dragen.run import Run
from dragen.utilities.InputInfo import RveInfo


def _run_kwargs(box_size_y):
    return dict(
        dimension=3, box_size=8, box_size_y=box_size_y, box_size_z=None, resolution=1,
        number_of_rves=1, slope_offset=0, abaqus_flag=False, damask_flag=False, moose_flag=False,
        calibration_rve_flag=False, element_type='HEX8', pbc_flag=True, submodel_flag=False,
        phase2iso_flag={1: True}, smoothing_flag=False, xfem_flag=False, gui_flag=False,
        anim_flag=False, root='./', info_box_obj=None, progress_obj=None,
        phase_ratio={1: 1}, file_dict={1: None}, phases=['Ferrite'], number_of_bands=0,
        upper_band_bound=1, lower_band_bound=1, band_orientation='xy', band_filling=1,
        subs_flag=False, subs_file_flag=False, subs_file='', equiv_d=5, p_sigma=0.1, t_mu=1.0,
        b_sigma=0.1, decreasing_factor=0.95, lower=1.5, upper=2.5, circularity=1, plt_name='p.png',
        save=False, plot=False, filename='f.png', orientation_relationship='KS',
    )


@pytest.mark.xfail(
    strict=True,
    reason="dragen/run.py:155-163 -- n_pts_y/n_pts_z are only reset conditionally, so a stale "
           "value leaks from a prior Run() into the next one in the same process.",
)
def test_n_pts_y_resets_when_box_size_y_is_dropped():
    Run(**_run_kwargs(box_size_y=6))
    assert RveInfo.n_pts_y == 6

    Run(**_run_kwargs(box_size_y=None))
    assert RveInfo.box_size_y is None
    assert RveInfo.n_pts_y is None
