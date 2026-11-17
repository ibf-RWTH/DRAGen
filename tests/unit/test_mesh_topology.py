"""Voxel clean-up and topology of the grain boundary network (M1)."""
import os
import sys

import numpy as np
import pytest
from scipy import ndimage

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'benchmarks'))

from dragen.analysis.metrics import components_per_grain  # noqa: E402
from dragen.mesh import corpus, quality  # noqa: E402
from dragen.mesh.cleanup import ManifoldCleaner, critical_blocks  # noqa: E402
from dragen.mesh.surface import staircase_mesh  # noqa: E402
from dragen.mesh.topology import IMPRINT, build_topology  # noqa: E402

CASES = corpus.cases()
WITH_TOPOLOGY = [c for c in CASES if 'topology' in c.expected]


def random_fields(n, seed=0):
    rng = np.random.default_rng(seed)
    for t in range(n):
        shape = rng.integers(3, 8, 3)
        labels = rng.integers(1, rng.integers(2, 6) + 1, size=shape)
        if t % 2:
            labels = ndimage.median_filter(labels, 3)
        yield np.unique(labels, return_inverse=True)[1].reshape(shape) + 1


# ------------------------------------------------------------------------------------- clean-up

def test_block_criterion_equals_nonmanifold_surface():
    """A grain violates the block criterion exactly when its staircase surface is non-manifold."""
    for labels in random_fields(150):
        grains = quality.grain_surfaces(staircase_mesh(labels, (1, 1, 1))).set_index('grain')
        nonmanifold = set(grains.index[(grains['nonmanifold_edges'] + grains['nonmanifold_vertices']) > 0])
        assert set(critical_blocks(labels)[1].tolist()) == nonmanifold


@pytest.mark.parametrize('case', CASES, ids=[c.name for c in CASES])
def test_cleanup_of_corpus(case):
    cleaned, stats = ManifoldCleaner(case.periodic).run(case.labels)
    assert stats.critical_after == 0
    grains = quality.grain_surfaces(staircase_mesh(cleaned, (1, 1, 1)))
    assert grains['closed_manifold'].all()
    needs_repair = any(case.expected.get('nonmanifold_edges', {}).values()) \
        or any(case.expected.get('nonmanifold_vertices', {}).values())
    if needs_repair:
        assert 0 < stats.changed_voxels <= 4
    else:
        assert stats.changed_voxels == 0
        assert np.array_equal(cleaned, case.labels)


def weighted_voronoi_fields(n, seed=0):
    """Curved, non-convex grains that are single bodies, like generated RVEs."""
    rng = np.random.default_rng(seed)
    while n:
        shape = rng.integers(8, 16, 3)
        k = rng.integers(4, 20)
        seeds = rng.uniform(0, 1, (k, 3)) * shape
        weight = rng.uniform(0.5, 2.0, k)
        grid = np.stack(np.meshgrid(*[np.arange(m) + 0.5 for m in shape], indexing='ij'), -1)
        labels = (np.linalg.norm(grid[..., None, :] - seeds, axis=-1) / weight).argmin(-1)
        labels = np.unique(labels, return_inverse=True)[1].reshape(shape) + 1
        if np.all(components_per_grain(labels, (False,) * 3)[1:] == 1):
            n -= 1
            yield labels


def test_cleanup_of_random_microstructures():
    resolved = 0
    for labels in weighted_voronoi_fields(40, seed=1):
        cleaned, stats = ManifoldCleaner((False,) * 3).run(labels)
        assert stats.critical_after == 0
        assert np.all(components_per_grain(cleaned, (False,) * 3)[1:] == 1)
        resolved += stats.critical_before > 0
    assert resolved > 30


def test_cleanup_of_noise_improves_without_losing_grains():
    """Pure noise is not always fully repairable (documented limit), but never made worse."""
    for labels in random_fields(20, seed=2):
        cleaned, stats = ManifoldCleaner((False,) * 3).run(labels)
        assert stats.critical_after <= stats.critical_before
        assert np.all(np.bincount(cleaned.ravel(), minlength=labels.max() + 1)[1:] > 0)


# --------------------------------------------------------------------------------------- topology

@pytest.mark.parametrize('case', WITH_TOPOLOGY, ids=[c.name for c in WITH_TOPOLOGY])
def test_topology_matches_hand_counts(case):
    topo = build_topology(case.labels, (1, 1, 1), case.periodic)
    summary = topo.summary()
    for key, value in case.expected['topology'].items():
        assert summary[key] == value, key


@pytest.mark.parametrize('case', CASES, ids=[c.name for c in CASES])
def test_topology_invariants_on_corpus(case):
    cleaned, _ = ManifoldCleaner(case.periodic).run(case.labels)
    check_invariants(build_topology(cleaned, (1, 1, 1), case.periodic))


def check_invariants(topo):
    anchors = set(topo.anchors.tolist())
    assert topo.euler_check() == 1
    assert topo.isolated_anchors() == 0
    for line in topo.lines:
        assert line.vertices[0] in anchors and line.vertices[-1] in anchors
        assert not anchors.intersection(line.vertices[1:-1].tolist())
        assert line.order >= 3 or line.imprinted
    # every edge belongs to at most one line, and every line edge is on exactly one line
    used = np.concatenate([np.stack([ln.vertices[:-1], ln.vertices[1:]], axis=1) for ln in topo.lines]) \
        if topo.lines else np.zeros((0, 2), int)
    used = np.sort(used, axis=1)
    assert len(np.unique(used, axis=0)) == len(used) == np.count_nonzero(topo.edge_line >= 0)
    # periodic images: anchors closed, and every line has an image line with the same vertex classes
    cls = topo.periodic_class
    assert set(cls[topo.anchors].tolist()) == set(cls[np.isin(cls, cls[topo.anchors])].tolist())
    by_classes = {}
    for line in topo.lines:
        by_classes.setdefault(tuple(sorted(cls[line.vertices].tolist())), []).append(line)
    for line in topo.lines:
        corner = topo.mesh.corner[line.vertices]
        for axis, per in enumerate(topo.periodic):
            if per and (np.all(corner[:, axis] == 0) or np.all(corner[:, axis] == topo.shape[axis])):
                assert len(by_classes[tuple(sorted(cls[line.vertices].tolist()))]) >= 2


def test_ring_anchors_of_imprinted_ring_are_periodic_images():
    case = corpus.periodic_imprint()
    topo = build_topology(case.labels, (1, 1, 1), case.periodic)
    rings = [ln for ln in topo.lines if ln.closed]
    assert len(rings) == 2
    real, imprinted = sorted(rings, key=lambda ln: ln.imprinted)
    assert real.regions == (-1, 1, 3) and IMPRINT in imprinted.regions
    a, b = topo.mesh.corner[real.vertices[0]], topo.mesh.corner[imprinted.vertices[0]]
    assert a[0] == 0 and b[0] == 6 and np.array_equal(a[1:], b[1:])
    assert set(topo.periodic_class[real.vertices]) == set(topo.periodic_class[imprinted.vertices])


def test_higher_order_line_is_one_line_not_a_chain_of_anchors():
    topo = build_topology(corpus.four_grain_edge().labels, (1, 1, 1), (False,) * 3)
    quad = [ln for ln in topo.lines if ln.order == 4]
    assert len(quad) == 1 and len(quad[0].vertices) == 9
    assert quad[0].regions == (1, 2, 3, 4)


@pytest.fixture(scope='module')
def generated():
    import common
    from dragen.config import BoxConfig, PhaseConfig, RveConfig
    from dragen.pipeline import generate
    # seed 2 contains one pinched patch
    config = RveConfig(box=BoxConfig((15,) * 3, 2),
                       phases=(PhaseConfig('ferrite', common.lognormal_input(), 1.0),), seed=2)
    return generate(config)


def test_generated_rve_cleanup_and_topology(generated):
    periodic = generated.domain.periodic
    cleaned, stats = ManifoldCleaner(periodic).run(generated.labels)
    assert stats.critical_before > 0 and stats.critical_after == 0
    assert stats.changed_voxels < 0.01 * cleaned.size
    assert np.all(components_per_grain(cleaned, periodic)[1:] == 1)
    topo = build_topology(cleaned, generated.domain.spacing, periodic)
    assert topo.pinched_patches() == 1
    assert topo.summary()['imprinted_lines'] > 0
    check_invariants(topo)
