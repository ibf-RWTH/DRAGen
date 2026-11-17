"""Hierarchical smoothing (M3): rules of the author, volumes, no crossings or folds."""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'benchmarks'))

from dragen.mesh import corpus, quality  # noqa: E402
from dragen.mesh.cleanup import ManifoldCleaner  # noqa: E402
from dragen.mesh.model import ON_LINE, build_model  # noqa: E402
from dragen.mesh.smoothing import smooth  # noqa: E402
from dragen.mesh.surface import staircase_mesh  # noqa: E402
from dragen.mesh.topology import build_topology  # noqa: E402

CASES = corpus.cases()


def smoothed(labels, spacing, periodic):
    labels, _ = ManifoldCleaner(periodic).run(labels)
    model = build_model(build_topology(labels, spacing, periodic))
    return model, smooth(model, spacing)


def check_rules(model, result, shape, spacing, periodic):
    mesh, x = model.mesh, result.points
    # anchors fixed, box-face coordinates unchanged
    assert np.array_equal(x[model.points], mesh.points[model.points])
    assert np.array_equal(x[model.box_lock], mesh.points[model.box_lock])
    # no crossings, no folds, every grain within 1 % or one voxel volume
    assert quality.self_intersections(x, mesh.triangles) == 0
    assert quality.folded_triangles(mesh.points, x, model.oriented) == 0
    assert result.stats['volume_final']['grains_off_tolerance'] == 0
    # periodic images still mirror each other
    box = np.array(shape) * np.array(spacing)
    box_tris = mesh.triangles[(mesh.grains < 0).any(axis=1)]
    per = quality.periodic_mismatch(x, box_tris, (0, 0, 0), box, [a for a in range(3) if periodic[a]])
    assert all(v['unmatched_nodes'] == 0 and v['unmatched_facets'] == 0 for v in per.values())


@pytest.mark.parametrize('case', CASES, ids=[c.name for c in CASES])
def test_smoothing_keeps_the_rules_on_the_corpus(case):
    model, result = smoothed(case.labels, (1, 1, 1), case.periodic)
    check_rules(model, result, case.labels.shape, (1, 1, 1), case.periodic)


def test_flat_boundaries_do_not_move():
    for case in (corpus.two_halves(), corpus.thin_grain(), corpus.four_grain_edge()):
        model, result = smoothed(case.labels, (1, 1, 1), case.periodic)
        assert np.allclose(result.points, model.mesh.points)


def test_ring_is_rounded_in_its_plane_and_cube_keeps_its_volume():
    case = corpus.ring_triple_line()
    model, result = smoothed(case.labels, (1, 1, 1), case.periodic)
    ring = [ln for ln in model.curves if ln.closed][0]
    moved = result.points[ring.vertices] - model.mesh.points[ring.vertices]
    assert np.abs(moved[:, 2]).max() < 1e-12            # the ring stays in the plane z = 5
    assert np.linalg.norm(moved, axis=1).max() > 0.1     # but its corners are cut
    kink0 = quality.surface_kinks(model.mesh.points, model.oriented, model.topology.patch)
    kink = quality.surface_kinks(result.points, model.oriented, model.topology.patch)
    assert np.mean(kink >= 89.9) < np.mean(kink0 >= 89.9)
    assert result.grain_volume_target[3] == pytest.approx(64)
    assert result.stats['volume_final']['max_abs'] < 1e-6


def test_displacement_limit_and_moving_lines():
    case = corpus.two_contact_patches()
    model, result = smoothed(case.labels, (1, 1, 1), case.periodic)
    d = np.linalg.norm(result.points - model.mesh.points, axis=1)
    assert d.max() <= 1.0 + 1e-9                         # at most one voxel edge
    assert d[model.vertex_kind == ON_LINE].max() > 0.1   # the pillar rings are rounded


def test_tolerance_is_one_percent_or_one_voxel():
    # a one-voxel grain may lose up to one voxel volume and is kept smoothed ...
    case = corpus.single_voxel()
    model, result = smoothed(case.labels, (1, 1, 1), case.periodic)
    assert result.stats['reverted_grains'] == 0
    assert 0.01 < result.stats['volume_final']['max_abs'] < 1.0
    # ... but with a tolerance below one voxel it falls back to the staircase, which is exact
    from dragen.mesh.smoothing import SmoothingConfig, Smoother
    strict = Smoother(model, (1, 1, 1), SmoothingConfig())
    strict._allowed = lambda volume: 0.01 * volume
    result = strict.run()
    assert result.stats['reverted_grains'] == 1
    assert result.stats['volume_final']['max_abs'] < 1e-12


def test_kinks_and_folds_on_the_staircase():
    mesh = staircase_mesh(corpus.ring_triple_line().labels, (1, 1, 1))
    model = build_model(build_topology(corpus.ring_triple_line().labels, (1, 1, 1), (False,) * 3))
    kink = quality.surface_kinks(model.mesh.points, model.oriented, model.topology.patch)
    assert set(np.round(kink).astype(int)) <= {0, 90}
    assert quality.folded_triangles(mesh.points, mesh.points, mesh.triangles) == 0
    assert quality.folded_triangles(model.mesh.points, model.mesh.points[:, [1, 0, 2]],
                                    model.oriented) == len(model.oriented)


@pytest.fixture(scope='module')
def generated():
    import common
    from dragen.config import BoxConfig, PhaseConfig, RveConfig
    from dragen.pipeline import generate
    config = RveConfig(box=BoxConfig((15,) * 3, 2),
                       phases=(PhaseConfig('ferrite', common.lognormal_input(), 1.0),), seed=1)
    return generate(config)


def test_generated_rve_smoothing(generated):
    domain = generated.domain
    model, result = smoothed(generated.labels, domain.spacing, domain.periodic)
    check_rules(model, result, generated.labels.shape, domain.spacing, domain.periodic)
    kink0 = quality.surface_kinks(model.mesh.points, model.oriented, model.topology.patch)
    kink = quality.surface_kinks(result.points, model.oriented, model.topology.patch)
    assert np.mean(kink0 >= 89.9) > 0.2 and np.mean(kink >= 89.9) < 0.02
