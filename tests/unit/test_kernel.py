import numpy as np
import pytest

from dragen.core.kernel import _rotated_quadratic, build_kernel


def sphere_kernel(radius=4.0, shrink=0.5, max_epoch=20, spacing=(0.5, 0.5, 0.5)):
    rate = np.full(len(spacing), spacing[0] / (radius * shrink))
    return build_kernel([radius] * len(spacing), 0.0, spacing, shrink, rate, max_epoch)


def test_centre_first_and_epochs_sorted():
    k = sphere_kernel()
    assert k.offsets[0].tolist() == [0, 0, 0]
    assert k.entry_epoch[0] == 0
    assert np.all(np.diff(k.entry_epoch) >= 0)
    assert 0 < k.seed_size < k.size


def test_target_matches_continuous_sphere_volume():
    k = sphere_kernel(radius=4.0)
    continuous = 4 / 3 * np.pi * 4.0 ** 3 / 0.5 ** 3
    assert k.target_voxels == pytest.approx(continuous, rel=0.05)


def test_entry_epoch_is_first_epoch_inside_growing_ellipsoid():
    radii, shrink, spacing = np.array([4.0, 2.0, 3.0]), 0.6, np.array([0.5, 0.5, 0.5])
    rate = np.array([0.1, 0.2, 0.15])
    k = build_kernel(radii, 30.0, spacing, shrink, rate, max_epoch=12)
    phys = k.offsets * spacing
    for e in (1, 5, 12):
        r_e = radii * shrink * (1 + e * rate)
        r_prev = radii * shrink * (1 + (e - 1) * rate)
        sel = k.entry_epoch == e
        assert np.all(_rotated_quadratic(phys[sel], r_e, 30.0) <= 1)
        assert np.all(_rotated_quadratic(phys[sel], r_prev, 30.0) > 1)


def test_rotation_by_90_degrees_swaps_x_and_y_extent():
    rate = np.array([0.1, 0.1, 0.1])
    k0 = build_kernel([4, 1, 1], 0.0, (0.5, 0.5, 0.5), 1.0, rate, max_epoch=0)
    k90 = build_kernel([4, 1, 1], 90.0, (0.5, 0.5, 0.5), 1.0, rate, max_epoch=0)
    extent = lambda k: np.ptp(k.offsets, axis=0)
    assert extent(k0)[0] > extent(k0)[1]
    assert extent(k90)[1] > extent(k90)[0]


def test_two_dimensional_kernel():
    k = sphere_kernel(spacing=(0.5, 0.5))
    assert k.offsets.shape[1] == 2
    assert k.target_voxels == pytest.approx(np.pi * 4.0 ** 2 / 0.25, rel=0.05)


def test_tiny_grain_still_owns_its_centre():
    k = build_kernel([0.01, 0.01, 0.01], 0.0, (1, 1, 1), 0.5, np.array([0.1] * 3), max_epoch=3)
    assert k.seed_size >= 1
    assert k.target_voxels >= 1


def test_seed_of_thin_rotated_grain_is_face_connected():
    from scipy import ndimage
    rng = np.random.default_rng(0)
    for _ in range(300):
        r = rng.uniform(0.3, 2.0, 3)
        r[2] = r[0]
        k = build_kernel(r, rng.uniform(0, 180), (0.5,) * 3, 0.3 ** (1 / 3), np.full(3, 0.05), 40)
        loc = k.seed - k.seed.min(axis=0)
        mask = np.zeros(loc.max(axis=0) + 1, dtype=bool)
        mask[tuple(loc.T)] = True
        assert ndimage.label(mask)[1] == 1
