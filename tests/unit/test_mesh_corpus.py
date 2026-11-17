"""The meshing test corpus (M0): staircase meshes of the special cases must give the answers that
follow from the construction of each case, and generated RVEs must pass the same invariants."""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'benchmarks'))

from dragen.analysis.metrics import components_per_grain  # noqa: E402
from dragen.mesh import corpus, quality  # noqa: E402
from dragen.mesh.surface import corner_grain_counts, staircase_mesh  # noqa: E402

CASES = corpus.cases()


def diagonal_edge_contacts(labels):
    """Per grain, voxel edges around which the grain holds exactly two diagonal voxels.

    Brute force on the voxel field, independent of the mesh code.
    """
    out = np.zeros(labels.max() + 1, dtype=int)
    for a in range(3):
        plane = np.moveaxis(labels, a, 0)
        p00, p10 = plane[:, :-1, :-1], plane[:, 1:, :-1]
        p01, p11 = plane[:, :-1, 1:], plane[:, 1:, 1:]
        for d1, d2, o1, o2 in ((p00, p11, p10, p01), (p10, p01, p00, p11)):
            hit = (d1 == d2) & (o1 != d1) & (o2 != d1)
            np.add.at(out, d1[hit], 1)
    return out


@pytest.mark.parametrize('case', CASES, ids=[c.name for c in CASES])
def test_staircase_matches_expected_topology(case):
    mesh = staircase_mesh(case.labels, (1, 1, 1))
    grains = quality.grain_surfaces(mesh).set_index('grain')
    for key in ('components', 'nonmanifold_edges', 'nonmanifold_vertices'):
        for g, value in case.expected.get(key, {}).items():
            assert grains.loc[g, key] == value, (key, g)
    for g, value in case.expected.get('euler', {}).items():
        assert list(grains.loc[g, 'euler']) == value, g
    assert quality.interface_patches(mesh) == case.expected['patches']
    bodies = components_per_grain(case.labels, case.periodic)
    for g, value in case.expected.get('bodies', {}).items():
        assert bodies[g] == value, g


@pytest.mark.parametrize('case', CASES, ids=[c.name for c in CASES])
def test_staircase_invariants(case):
    spacing = (0.5, 0.25, 1.0)
    mesh = staircase_mesh(case.labels, spacing)
    counts = np.bincount(case.labels.ravel()) * np.prod(spacing)
    reference = {g: counts[g] for g in range(1, len(counts))}
    rep = quality.interface_report(mesh, reference_volume=reference, intersections=True)
    assert rep['open_edges'] == rep['misoriented_edges'] == 0
    assert rep['duplicate_triangles'] == rep['degenerate_triangles'] == rep['same_region_triangles'] == 0
    assert rep['self_intersections'] == 0
    assert rep['volume_deviation']['max_abs'] < 1e-12
    assert rep['nonmanifold_edges'] == diagonal_edge_contacts(case.labels).sum()


def test_staircase_of_periodic_case_mirrors_box_faces():
    case = corpus.fragmented_grain(periodic=True)
    mesh = staircase_mesh(case.labels, (1, 1, 1))
    box_tris = mesh.triangles[(mesh.grains < 0).any(axis=1)]
    upper = np.array(case.labels.shape, float)
    result = quality.periodic_mismatch(mesh.points, box_tris, (0, 0, 0), upper, axes=(0, 1, 2))
    for axis in range(3):
        assert result[axis]['unmatched_nodes'] == 0
        assert result[axis]['unmatched_facets'] == 0
        assert result[axis]['nodes_lower'] == result[axis]['nodes_upper'] > 0


def test_corner_grain_counts_classify_junctions():
    four = corner_grain_counts(corpus.four_grain_edge().labels, (False,) * 3)
    assert np.all(four[4, 4, :] == 4)
    assert four.max() == 4
    eight = corner_grain_counts(corpus.eight_grain_point().labels, (False,) * 3)
    assert eight[3, 3, 3] == 8
    assert np.count_nonzero(eight == 8) == 1
    ring = corner_grain_counts(corpus.ring_triple_line().labels, (False,) * 3)
    loop = ring[3:8, 3:8, 5]
    assert np.count_nonzero(ring >= 4) == 0
    assert np.all(loop[[0, -1], :] == 3) and np.all(loop[:, [0, -1]] == 3)
    slab = corpus.fragmented_grain(periodic=True).labels
    # on the periodic x faces, corners inside grain 1 see only grain 1; on an open box they too
    assert corner_grain_counts(slab, (True,) * 3)[0, 2, 2] == 1
    assert corner_grain_counts(slab, (False,) * 3)[2, 2, 2] == 2


@pytest.fixture(scope='module')
def generated():
    import common
    from dragen.config import BoxConfig, PhaseConfig, RveConfig
    from dragen.pipeline import generate
    config = RveConfig(box=BoxConfig((10,) * 3, 2),
                       phases=(PhaseConfig('ferrite', common.lognormal_input(), 1.0),), seed=1)
    return generate(config)


def test_generated_rve_staircase(generated):
    labels, domain = generated.labels, generated.domain
    mesh = staircase_mesh(labels, domain.spacing)
    counts = np.bincount(labels.ravel()) * domain.voxel_volume
    rep = quality.interface_report(mesh, reference_volume={g: counts[g] for g in range(1, len(counts))},
                                   intersections=True)
    assert rep['open_edges'] == rep['misoriented_edges'] == rep['self_intersections'] == 0
    assert rep['volume_deviation']['max_abs'] < 1e-12
    assert rep['nonmanifold_edges'] == diagonal_edge_contacts(labels).sum()
    box_tris = mesh.triangles[(mesh.grains < 0).any(axis=1)]
    per = quality.periodic_mismatch(mesh.points, box_tris, (0, 0, 0), domain.size, axes=(0, 1, 2))
    assert all(v['unmatched_nodes'] == 0 and v['unmatched_facets'] == 0 for v in per.values())
