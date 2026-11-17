import numpy as np
import pytest

from dragen.analysis import metrics


def test_fill_ratio_counts_negative_markers_as_assigned():
    labels = np.array([[0, 1], [-200, 2]])
    assert metrics.fill_ratio(labels) == 0.75


def test_grain_voxel_counts_ignores_zero_and_negative():
    labels = np.array([0, 1, 1, 2, -200, 3, 3, 3])
    assert metrics.grain_voxel_counts(labels).tolist() == [0, 2, 1, 3]


def test_phase_fractions():
    labels = np.array([1, 1, 2, 3, 3, 3])
    phase_of_grain = np.array([0, 1, 2, 2])
    assert metrics.phase_fractions(labels, phase_of_grain) == pytest.approx({1: 2 / 6, 2: 4 / 6})


def test_grain_split_across_periodic_boundary_is_one_component():
    labels = np.zeros((6, 4, 4), dtype=int)
    labels[0, :, :] = 1
    labels[5, :, :] = 1
    labels[1:5, :, :] = 2
    periodic = metrics.components_per_grain(labels, periodic=(True, True, True))
    open_x = metrics.components_per_grain(labels, periodic=(False, True, True))
    assert periodic[1] == 1
    assert open_x[1] == 2
    assert periodic[2] == open_x[2] == 1


def test_diagonal_contact_is_not_connected():
    labels = np.array([[1, 0], [0, 1]])
    assert metrics.components_per_grain(labels, periodic=(False, False))[1] == 2


def test_report_on_perfect_field():
    labels = np.zeros((4, 4, 4), dtype=int)
    labels[:2] = 1
    labels[2:] = 2
    rep = metrics.report(labels, target_volume=np.array([0, 32.0, 32.0]),
                         phase_of_grain=np.array([0, 1, 2]), voxel_volume=1.0,
                         periodic=(True, True, True))
    assert rep.fill_ratio == 1.0
    assert rep.empty_grains == 0
    assert rep.fragmented_grains == 0
    assert rep.volume_ratio_median == 1.0
    assert rep.phase_fractions == {1: 0.5, 2: 0.5}
