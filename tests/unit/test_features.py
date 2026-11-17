import numpy as np
import pytest

from dragen.config import BandConfig, RsaConfig
from dragen.core import VoxelDomain
from dragen.core.kernel import build_kernel
from dragen.stages.bands import Band, BandError, place_bands
from dragen.stages.inclusions import InclusionPlacer
from dragen.stages.rsa import RsaPlacer


def test_band_mask_wraps_on_periodic_axis():
    dom = VoxelDomain((4, 4, 10), (1.0, 1.0, 1.0), (True, True, True))
    mask = Band(axis=2, center=0.0, thickness=4.0).mask(dom).reshape(dom.shape)
    assert np.flatnonzero(mask[0, 0]).tolist() == [0, 1, 8, 9]


def test_band_mask_on_open_axis_does_not_wrap():
    dom = VoxelDomain((4, 4, 10), (1.0, 1.0, 1.0), (True, True, False))
    mask = Band(axis=2, center=0.0, thickness=4.0).mask(dom).reshape(dom.shape)
    assert np.flatnonzero(mask[0, 0]).tolist() == [0, 1]


def test_bands_do_not_overlap_and_stay_inside_open_box():
    dom = VoxelDomain((10, 10, 40), (0.5,) * 3, (True, True, False))
    cfg = BandConfig(count=3, source='b.csv', thickness=(2, 3), normal_axis=2, filling=0.5)
    for seed in range(20):
        bands = place_bands(cfg, dom, np.random.default_rng(seed))
        for i, a in enumerate(bands):
            assert a.thickness / 2 <= a.center <= dom.size[2] - a.thickness / 2
            for b in bands[i + 1:]:
                assert abs(a.center - b.center) >= (a.thickness + b.thickness) / 2


def test_too_many_bands_raise():
    dom = VoxelDomain((10, 10, 10), (1.0,) * 3, (True,) * 3)
    with pytest.raises(BandError):
        place_bands(BandConfig(count=5, source='b.csv', thickness=(3, 3), filling=0.5), dom, np.random.default_rng(0))


def test_rsa_keeps_band_seeds_inside_the_band():
    dom = VoxelDomain((40, 40, 40), (0.5,) * 3, (True,) * 3)
    band = Band(axis=2, center=10.0, thickness=3.0)
    region = band.mask(dom)
    kernels = [build_kernel([1.2, 0.8, 0.6], 30.0, dom.spacing, 0.7, np.full(3, 0.1), 5) for _ in range(30)]
    labels = dom.new_labels()
    RsaPlacer(dom, RsaConfig(), np.random.default_rng(1)).place(labels, kernels, region=region, max_outside=0.1)
    seeded = labels.ravel() > 0
    assert region[seeded].mean() >= 0.9


def test_inclusion_lies_inside_one_grain_with_a_shell():
    dom = VoxelDomain((20, 20, 20), (1.0,) * 3, (True,) * 3)
    labels = np.ones(dom.shape, dtype=np.int32)
    labels[10:] = 2                                    # two grains, boundary planes at x = 10 and x = 0
    kernels = [build_kernel([2.0, 2.0, 2.0], 0.0, dom.spacing, 1.0, np.zeros(3), 0) for _ in range(4)]
    placer = InclusionPlacer(dom, np.random.default_rng(0))
    placer.place(labels, kernels, first_id=3)
    assert placer.stats.placed == 4
    for i in range(3, 7):
        xs = np.unique(np.nonzero(labels == i)[0])
        host = 1 if xs.max() < 10 else 2
        assert np.all(labels[xs.min() - 1] != 3 - host) and np.all(labels[(xs.max() + 1) % 20] != 3 - host)


def test_inclusion_too_large_for_every_grain_is_reported():
    dom = VoxelDomain((10, 10, 10), (1.0,) * 3, (True,) * 3)
    labels = (np.indices(dom.shape).sum(axis=0) % 2 + 1).astype(np.int32)   # checkerboard of two grains
    kernels = [build_kernel([2.0, 2.0, 2.0], 0.0, dom.spacing, 1.0, np.zeros(3), 0)]
    placer = InclusionPlacer(dom, np.random.default_rng(0), max_attempts=50)
    placed, centers = placer.place(labels, kernels, first_id=3)
    assert placer.stats.not_placed == 1
    assert placed.size == 0 and centers.shape == (0, 3)
    assert not np.any(labels == 3)


def test_inclusion_does_not_cut_off_voxels_attached_only_through_it():
    """Host: centre voxel c, its six face neighbours and a strand towards -x. The neighbours +x,
    +-y, +-z hang on c alone. A shell of face neighbours only is all host here, so an inclusion
    at c would be accepted and cut them off; the full 26-neighbour shell rejects it."""
    from dragen.analysis.metrics import components_per_grain
    dom = VoxelDomain((12, 12, 12), (1.0,) * 3, (True,) * 3)
    # every background voxel is its own grain, so c is the only place an inclusion could go
    labels = (10 + np.arange(dom.n_voxels)).reshape(dom.shape).astype(np.int32)
    c = np.array([6, 6, 6])
    for d in np.vstack([np.zeros(3, int), np.eye(3, dtype=int), -np.eye(3, dtype=int)]):
        labels[tuple(c + d)] = 1
    labels[1:6, 6, 6] = 1                                  # strand to -x, keeps the host one body
    kernel = build_kernel([0.3, 0.3, 0.3], 0.0, dom.spacing, 1.0, np.zeros(3), 0)   # a single voxel
    before = components_per_grain(labels, (True,) * 3)[1]
    InclusionPlacer(dom, np.random.default_rng(0), max_attempts=500).place(labels, [kernel], first_id=5000)
    assert components_per_grain(labels, (True,) * 3)[1] == before == 1


def test_band_grains_are_merged_into_one_grain_per_band():
    from dragen.core.grains import GrainTable
    from dragen.pipeline import merge_bands
    dom = VoxelDomain(shape=(6, 6, 6), spacing=(1.0, 1.0, 1.0), periodic=(True, True, True))
    labels = np.array([1, 2, 3, 4, 5, 0] * 36, dtype=np.int32).reshape(6, 6, 6)
    grains = GrainTable(radii=np.ones((5, 3)), alpha=np.zeros(5), euler=np.arange(15.0).reshape(5, 3),
                        phase=np.array([2, 2, 2, 1, 1]), band=np.array([0, 0, 1, -1, -1]),
                        centers=np.zeros((5, 3), dtype=int))
    bands = [Band(axis=2, center=1.0, thickness=1.0), Band(axis=2, center=4.0, thickness=1.0)]
    new, table, targets, merged = merge_bands(labels, grains, [10, 20, 30, 40, 50], bands, dom)
    # band 0 = old grains 1 and 2, band 1 = old grain 3, matrix grains 4, 5 -> 3, 4
    assert np.array_equal(np.unique(new), [0, 1, 2, 3, 4])
    assert np.array_equal(new.ravel()[:6], [1, 1, 2, 3, 4, 0])
    assert merged == [{'merged': 2, 'detached': 0}, {'merged': 1, 'detached': 0}] and targets == [30, 30, 40, 50]
    assert np.array_equal(table.band, [0, 1, -1, -1]) and np.array_equal(table.phase, [2, 2, 1, 1])
    assert np.allclose(table.volume[:2], [2 * grains.volume[0], grains.volume[2]])
    assert np.array_equal(table.euler[0], grains.euler[0])


def test_band_grain_without_contact_stays_a_grain():
    from dragen.core.grains import GrainTable
    from dragen.pipeline import merge_bands
    dom = VoxelDomain(shape=(8, 1, 1), spacing=(1.0, 1.0, 1.0), periodic=(False, False, False))
    # band grains 1, 2 touch, band grain 3 is separated from them by matrix grain 4
    labels = np.array([1, 1, 2, 2, 4, 3, 4, 4], dtype=np.int32).reshape(8, 1, 1)
    grains = GrainTable(radii=np.ones((4, 3)), alpha=np.zeros(4), euler=np.zeros((4, 3)),
                        phase=np.array([2, 2, 2, 1]), band=np.array([0, 0, 0, -1]),
                        centers=np.zeros((4, 3), dtype=int))
    new, table, targets, stats = merge_bands(labels, grains, [2, 2, 1, 3], [Band(axis=2, center=0.5, thickness=1.0)], dom)
    assert new.ravel().tolist() == [1, 1, 1, 1, 3, 2, 3, 3]
    assert stats == [{'merged': 2, 'detached': 1}]
    assert table.band.tolist() == [0, 0, -1] and targets == [4, 1, 3]
