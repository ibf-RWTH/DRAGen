"""Volume meshing with gmsh (M4): surface kept, conforming tet10, exact volumes and periodic pairs."""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'benchmarks'))

pytest.importorskip('gmsh')

from dragen.mesh import corpus  # noqa: E402
from dragen.mesh.cleanup import ManifoldCleaner  # noqa: E402
from dragen.mesh.model import build_model  # noqa: E402
from dragen.mesh.smoothing import smooth  # noqa: E402
from dragen.mesh.tetmesh import TetMeshConfig, check, grain_bodies, periodic_node_pairs, tet_mesh  # noqa: E402
from dragen.mesh.topology import build_topology  # noqa: E402

CASES = corpus.cases()
QUIET = TetMeshConfig(verbosity=0)


def meshed(labels, spacing, periodic, config=QUIET):
    labels, _ = ManifoldCleaner(periodic).run(labels)
    model = build_model(build_topology(labels, spacing, periodic))
    smoothed = smooth(model, spacing)
    result = tet_mesh(model, smoothed.points, spacing, config)
    rep = check(model, smoothed.points, result.mesh, labels.shape, spacing, periodic,
                smoothed.grain_volume_target)
    return model, smoothed, result, rep


def assert_valid(rep):
    for key in ('inverted_elements', 'open_inner_faces', 'overused_faces', 'midside_mismatch',
                'unused_points', 'interface_faces_missing',
                'interface_faces_extra', 'periodic_unmatched_nodes', 'periodic_unmatched_facets'):
        assert rep[key] == 0, key
    assert rep['surface_nodes_moved'] == 0.0
    assert rep['periodic_max_distance'] == 0.0
    # the element volume of every grain is the volume enclosed by its smoothed surface (the voxel
    # volume only within the smoothing tolerance)
    assert rep['volume_vs_surface']['max_abs'] < 1e-9


@pytest.mark.parametrize('case', CASES, ids=[c.name for c in CASES])
def test_corpus(case):
    model, smoothed, result, rep = meshed(case.labels, (1, 1, 1), case.periodic)
    assert rep['cell_type'] == 'tet10'
    assert_valid(rep)


def test_bodies_and_cavities():
    # grain 1 around grain 2: one body with a cavity
    case = corpus.grain_in_grain()
    model = build_model(build_topology(case.labels, (1, 1, 1), case.periodic))
    bodies = {b.grain: b for b in grain_bodies(model, model.mesh.points)}
    assert len(bodies[1].shells) == 2 and len(bodies[2].shells) == 1
    # torus: the ring grain 2 lies in grain 1, which also fills the hole of the ring
    case = corpus.torus()
    model = build_model(build_topology(case.labels, (1, 1, 1), case.periodic))
    bodies = grain_bodies(model, model.mesh.points)
    assert sorted((b.grain, len(b.shells)) for b in bodies) == [(1, 2), (2, 1)]
    # a periodic slab of grain 1 is cut by the box into two bodies; a fragmented grain has two
    for case, n in ((corpus.fragmented_grain(periodic=True), 2), (corpus.fragmented_grain(), 2)):
        model = build_model(build_topology(case.labels, (1, 1, 1), case.periodic))
        bodies = grain_bodies(model, model.mesh.points)
        assert sum(b.grain == 1 for b in bodies) == n


def test_periodic_pairs_cover_midside_nodes():
    case = corpus.fragmented_grain(periodic=True)
    _, _, result, _ = meshed(case.labels, (1, 1, 1), case.periodic)
    mesh = result.mesh
    pairs = periodic_node_pairs(mesh.points, case.labels.shape, (0, 1, 2))
    midside = np.unique(mesh.cells[:, 4:])
    for axis, p in pairs.items():
        assert len(p) > 0
        assert np.isin(p.ravel(), midside).any()
        d = mesh.points[p[:, 1]] - mesh.points[p[:, 0]]
        assert np.array_equal(d[:, axis], np.full(len(p), case.labels.shape[axis], dtype=float))
        assert np.array_equal(np.delete(d, axis, axis=1), np.zeros((len(p), 2)))


def test_elements_get_coarser_inside():
    labels = np.ones((16, 16, 16), dtype=np.int32)
    labels[8:] = 2
    fine = meshed(labels, (1, 1, 1), (False,) * 3, TetMeshConfig(growth=0.0, verbosity=0))[3]
    graded = meshed(labels, (1, 1, 1), (False,) * 3, TetMeshConfig(growth=0.5, verbosity=0))[3]
    assert graded['n_elements'] < 0.6 * fine['n_elements']
    assert graded['longest_edge_voxels']['max'] > 2 * fine['longest_edge_voxels']['median']


@pytest.fixture(scope='module')
def generated():
    import common
    from dragen.config import BoxConfig, PhaseConfig, RveConfig
    from dragen.pipeline import generate
    config = RveConfig(box=BoxConfig((15,) * 3, 2),
                       phases=(PhaseConfig('ferrite', common.lognormal_input(), 1.0),), seed=1)
    return generate(config)


def test_generated_rve(generated):
    domain = generated.domain
    model, smoothed, result, rep = meshed(generated.labels, domain.spacing, domain.periodic)
    assert_valid(rep)
    assert rep['grains_with_several_bodies'] == 0            # pieces joined across the periodic box
    assert result.stats['bodies'] > len(model.shells)        # grains cut by the periodic box
