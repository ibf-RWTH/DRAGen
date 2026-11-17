import numpy as np
import pytest

from dragen.analysis.metrics import components_per_grain
from dragen.config import RsaConfig, TessellationConfig
from dragen.core import VoxelDomain
from dragen.core.kernel import EllipsoidKernel, build_kernel
from dragen.stages.repair import merge_detached_pieces
from dragen.stages.rsa import RsaError, RsaPlacer, _connected
from dragen.stages.tessellation import Tessellator


def kernels_for(radii_list, spacing, shrink=0.6, margin=2.0):
    radii_list = np.asarray(radii_list, dtype=float)
    rate = min(spacing) / (radii_list.max(axis=0) * shrink)
    max_epoch = int(np.ceil((margin / shrink - 1) / rate.min()))
    return [build_kernel(r, 0.0, spacing, shrink, rate, max_epoch) for r in radii_list]


# ---------------------------------------------------------------- RSA

def test_rsa_places_every_seed_without_overwriting():
    dom = VoxelDomain((30, 30, 30), (0.5,) * 3, (True,) * 3)
    kernels = kernels_for([[3, 2, 2]] * 20, dom.spacing)
    labels = dom.new_labels()
    placer = RsaPlacer(dom, RsaConfig(max_overlap=0.0), np.random.default_rng(0))
    centers = placer.place(labels, kernels)
    counts = np.bincount(labels.ravel(), minlength=21)[1:]
    assert np.all(counts == [k.seed_size for k in kernels])      # no overlap allowed -> full seeds
    assert np.all(labels[tuple(centers.T)] == np.arange(1, 21))   # every centre owns itself
    assert np.all(components_per_grain(labels, (True,) * 3)[1:] == 1)


def test_rsa_reports_when_seeds_do_not_fit():
    dom = VoxelDomain((10, 10, 10), (1.0,) * 3, (True,) * 3)
    kernels = kernels_for([[4, 4, 4]] * 10, dom.spacing, shrink=1.0)
    placer = RsaPlacer(dom, RsaConfig(max_overlap=0.0, max_attempts_per_grain=50), np.random.default_rng(0))
    with pytest.raises(RsaError):
        placer.place(dom.new_labels(), kernels)


def test_connected_helper():
    assert _connected(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]))
    assert not _connected(np.array([[0, 0, 0], [1, 1, 0]]))       # edge contact only


# ---------------------------------------------------------------- tessellation

def wall_setup(connected: bool):
    """Grain 1 grows from the left; grain 2 is a full-height wall at x = 6 with nothing to grow."""
    dom = VoxelDomain((20, 6), (1.0, 1.0), periodic=(False, True))
    labels = dom.new_labels()
    labels[2, 3] = 1
    labels[6, :] = 2
    big = build_kernel([14, 14], 0.0, dom.spacing, 0.1, np.array([0.5, 0.5]), max_epoch=30)
    wall = EllipsoidKernel(offsets=np.zeros((1, 2), np.int16), entry_epoch=np.zeros(1, np.int32),
                           seed_size=1, target_voxels=6)
    tess = Tessellator(dom, TessellationConfig(connected_growth=connected), np.random.default_rng(0))
    tess.run(labels, [big, wall], np.array([[2, 3], [6, 0]]))
    return labels, tess


def test_connected_growth_does_not_jump_over_a_neighbour():
    labels, tess = wall_setup(connected=True)
    assert np.all(labels[:6] == 1)
    assert np.all(labels[7:] == 2)                     # right side filled from the wall, not from grain 1
    assert tess.stats.filled_by_dilation > 0


def test_legacy_growth_jumps_over_a_neighbour():
    labels, _ = wall_setup(connected=False)
    assert np.any(labels[7:] == 1)


def test_tessellation_fills_box_with_single_body_grains():
    dom = VoxelDomain((24, 24, 24), (0.5,) * 3, (True,) * 3)
    rng = np.random.default_rng(3)
    radii = rng.uniform(1.0, 2.5, size=(40, 3))
    kernels = kernels_for(radii, dom.spacing)
    labels = dom.new_labels()
    centers = RsaPlacer(dom, RsaConfig(), rng).place(labels, kernels)
    Tessellator(dom, TessellationConfig(), rng).run(labels, kernels, centers)
    assert np.all(labels > 0)
    assert np.all(components_per_grain(labels, (True,) * 3)[1:] == 1)


def test_tessellation_is_reproducible():
    def run(seed):
        dom = VoxelDomain((16, 16, 16), (0.5,) * 3, (True,) * 3)
        rng = np.random.default_rng(seed)
        kernels = kernels_for(rng.uniform(1.0, 2.0, size=(15, 3)), dom.spacing)
        labels = dom.new_labels()
        centers = RsaPlacer(dom, RsaConfig(), rng).place(labels, kernels)
        Tessellator(dom, TessellationConfig(), rng).run(labels, kernels, centers)
        return labels
    assert np.array_equal(run(5), run(5))
    assert not np.array_equal(run(5), run(6))


# ---------------------------------------------------------------- repair

def test_repair_moves_detached_piece_to_main_neighbour():
    dom = VoxelDomain((8, 8), (1.0, 1.0), (False, False))
    labels = np.full((8, 8), 2)
    labels[:, :3] = 1
    labels[5, 6] = 1                                    # stray voxel of grain 1 inside grain 2
    moved = merge_detached_pieces(labels, dom)
    assert moved == 1
    assert labels[5, 6] == 2
    assert np.all(components_per_grain(labels, (False, False))[1:] == 1)


def test_repair_leaves_connected_field_untouched():
    dom = VoxelDomain((6, 6), (1.0, 1.0), (True, True))
    labels = np.repeat(np.arange(1, 4), 2)[:, None] * np.ones((1, 6), int)
    before = labels.copy()
    assert merge_detached_pieces(labels, dom) == 0
    assert np.array_equal(labels, before)


def test_kernel_wider_than_half_the_box_still_fills_it():
    dom = VoxelDomain((12, 12, 12), (1.0,) * 3, (True,) * 3)
    rng = np.random.default_rng(0)
    kernels = kernels_for([[9, 9, 9], [2, 2, 2], [2, 2, 2]], dom.spacing)   # first one wraps onto itself
    labels = dom.new_labels()
    centers = RsaPlacer(dom, RsaConfig(), rng).place(labels, kernels)
    tess = Tessellator(dom, TessellationConfig(), rng)
    tess.run(labels, kernels, centers)
    assert np.all(labels > 0)
    assert tess.stats.filled_after_rsa + tess.stats.filled_regular + tess.stats.filled_reset \
        + tess.stats.filled_by_dilation == dom.n_voxels


def test_grains_without_fill_up_do_not_take_part_in_it():
    dom = VoxelDomain((24, 24, 24), (0.5,) * 3, (True,) * 3)

    def run(excluded: bool):
        rng = np.random.default_rng(2)
        # 30 grains whose ellipsoids together hold about the box volume (12^3 um^3)
        kernels = kernels_for(rng.uniform(2.0, 2.8, size=(30, 3)), dom.spacing)
        labels = dom.new_labels()
        centers = RsaPlacer(dom, RsaConfig(), rng).place(labels, kernels)
        no_fill_up = np.zeros(30, dtype=bool)
        no_fill_up[:10] = excluded
        Tessellator(dom, TessellationConfig(), rng).run(labels, kernels, centers, no_fill_up=no_fill_up)
        assert np.all(labels > 0)
        return np.bincount(labels.ravel(), minlength=31)[1:11].sum()

    assert run(excluded=True) < run(excluded=False)
