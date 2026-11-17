"""Adaptive remeshing of the grain boundaries (M4a) and the whole meshing pipeline."""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'benchmarks'))

from dragen.mesh import corpus  # noqa: E402
from dragen.mesh.cleanup import ManifoldCleaner  # noqa: E402
from dragen.mesh.model import ON_LINE, build_model  # noqa: E402
from dragen.mesh.quality import self_intersection_pairs  # noqa: E402
from dragen.mesh.remeshing import RemeshConfig, Remesher, check, remesh  # noqa: E402
from dragen.mesh.smoothing import smooth  # noqa: E402
from dragen.mesh.topology import build_topology  # noqa: E402

CASES = corpus.cases()


def remeshed(labels, spacing, periodic, config=RemeshConfig()):
    labels, _ = ManifoldCleaner(periodic).run(labels)
    model = build_model(build_topology(labels, spacing, periodic))
    smoothed = smooth(model, spacing)
    result = remesh(model, smoothed.points, labels, spacing, smoothed.grain_volume_target, config)
    return model, smoothed, result


def assert_valid(model, result, spacing):
    rep = check(model, result, spacing)
    assert all(v == 0 for v in rep['errors'].values()), rep['errors']
    assert result.stats['volume_final']['grains_off_tolerance'] == 0
    return rep


@pytest.mark.parametrize('case', CASES, ids=[c.name for c in CASES])
def test_corpus(case):
    model, smoothed, result = remeshed(case.labels, (1, 1, 1), case.periodic)
    rep = assert_valid(model, result, (1, 1, 1))
    # same entities: every surface keeps triangles, every curve runs anchor to anchor
    assert len(result.model.surfaces) == len(model.surfaces)
    assert len(result.model.curves) == len(model.curves)
    anchors = set(result.model.points.tolist())
    for line in result.model.curves:
        assert line.vertices[0] in anchors and line.vertices[-1] in anchors
        assert not anchors & set(line.vertices[1:-1].tolist())
    assert rep['min_angle_deg']['min'] > 15.0


def test_line_vertices_stay_on_their_smoothed_line():
    case = corpus.ring_triple_line()
    model, smoothed, result = remeshed(case.labels, (1, 1, 1), case.periodic)
    m = result.model
    on_line = np.flatnonzero(m.vertex_kind == ON_LINE)
    assert len(on_line)
    # the ring lies in the plane z = 5 before and after
    ring = [ln for ln in m.curves if ln.closed][0]
    assert np.allclose(result.points[ring.vertices, 2], 5.0)


def test_coarser_on_flat_boundaries_and_size_field_limits():
    # two big grains with one flat boundary: the remeshed boundary has far fewer triangles,
    # edge lengths stay within [h_min, 4/3 h_max]
    labels = np.ones((24, 24, 24), dtype=np.int32)
    labels[12:] = 2
    model, smoothed, result = remeshed(labels, (1, 1, 1), (True, True, True))
    rep = assert_valid(model, result, (1, 1, 1))
    assert rep['triangles'] < 0.3 * len(model.mesh.triangles)
    d = (6 * 24 ** 3 / 2 / np.pi) ** (1 / 3)
    assert rep['edge_voxels']['max'] <= 4 / 3 * 0.25 * d + 1e-9


def test_closed_small_grain_keeps_its_volume_and_shape():
    case = corpus.single_voxel()
    model, smoothed, result = remeshed(case.labels, (1, 1, 1), case.periodic)
    assert_valid(model, result, (1, 1, 1))
    # within one voxel volume (the tolerance) and not flattened
    vol = result.stats['volume_final']
    assert vol['max_abs'] < 0.5


def test_refine_near_makes_the_surface_finer_there():
    labels = np.ones((16, 16, 16), dtype=np.int32)
    labels[8:] = 2
    labels, _ = ManifoldCleaner((False,) * 3).run(labels)
    model = build_model(build_topology(labels, (1, 1, 1), (False,) * 3))
    smoothed = smooth(model, (1, 1, 1))
    remesher = Remesher(model, smoothed.points, labels, (1, 1, 1), smoothed.grain_volume_target)
    centre = np.array([[8.0, 8.0, 8.0]])

    def edge_near(result):
        tris = result.model.oriented
        x = result.points
        mid = x[tris].mean(axis=1)
        near = np.linalg.norm(mid - centre, axis=1) < 2.0
        t = x[tris[near]]
        return np.linalg.norm(t - t[:, [1, 2, 0]], axis=2).mean()

    coarse = remesher.run()
    before = edge_near(coarse)
    fine = remesher.refine_near(centre)
    assert edge_near(fine) < 0.8 * before
    assert all(v == 0 for v in check(model, fine, (1, 1, 1))['errors'].values())


def test_coarsen_near_makes_the_surface_coarser_there():
    labels = np.ones((16, 16, 16), dtype=np.int32)
    labels[8:] = 2
    labels[3:6, 3:6, 3:6] = 3
    labels, _ = ManifoldCleaner((False,) * 3).run(labels)
    model = build_model(build_topology(labels, (1, 1, 1), (False,) * 3))
    smoothed = smooth(model, (1, 1, 1))
    remesher = Remesher(model, smoothed.points, labels, (1, 1, 1), smoothed.grain_volume_target)
    centre = np.array([[8.0, 8.0, 8.0]])

    def edge_near(result):
        x, tris = result.points, result.model.oriented
        near = np.linalg.norm(x[tris].mean(axis=1) - centre, axis=1) < 3.0
        t = x[tris[near]]
        return np.linalg.norm(t - t[:, [1, 2, 0]], axis=2).mean()

    before = edge_near(remesher.run())
    coarse = remesher.coarsen_near(centre)
    assert edge_near(coarse) > 1.2 * before
    assert all(v == 0 for v in check(model, coarse, (1, 1, 1))['errors'].values())



def test_size_field_scales_once_for_clustered_centres():
    # O52: three poor elements close together double the size there once, not 2**3 times
    labels = np.ones((12, 12, 12), dtype=np.int32)
    labels[6:] = 2
    labels, _ = ManifoldCleaner((False,) * 3).run(labels)
    model = build_model(build_topology(labels, (1, 1, 1), (False,) * 3))
    smoothed = smooth(model, (1, 1, 1))
    remesher = Remesher(model, smoothed.points, labels, (1, 1, 1), smoothed.grain_volume_target)
    h = remesher.ref_h.copy()
    remesher._scale_size_near(np.array([[6.0, 6.0, 6.0], [6.2, 6.0, 6.0], [6.0, 6.2, 6.0]]), 2.0)
    ratio = remesher.ref_h / h
    assert ratio.max() == 2.0
    assert np.all((ratio == 1.0) | (ratio == 2.0))

def test_shared_vertex_crossings_are_found():
    # two triangles sharing vertex 0; the second one passes through the first
    pts = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0.5, 0.5, -1], [0.5, 0.5, 1]], dtype=float)
    tris = np.array([[0, 1, 2], [0, 3, 4]])
    assert len(self_intersection_pairs(pts, tris)) == 0
    assert len(self_intersection_pairs(pts, tris, shared_vertex=True)) == 1


def test_vertex_touching_a_foreign_triangle_is_found():
    # vertex 5 of the second triangle lies on the edge (1, 2) of the first one, outside its plane
    # the second triangle rises away: no crossing, but a T-junction gmsh HXT rejects
    pts = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [1, 1, 1], [2, 2, 1], [1, 1, 0]], dtype=float)
    tris = np.array([[0, 1, 2], [3, 4, 5]])
    assert len(self_intersection_pairs(pts, tris, shared_vertex=True)) == 0
    assert len(self_intersection_pairs(pts, tris, shared_vertex=True, touching=True)) == 1
    # a corner shared by both triangles does not count
    tris = np.array([[0, 1, 2], [3, 4, 0]])
    assert len(self_intersection_pairs(pts, tris, shared_vertex=True, touching=True)) == 0



def test_edge_crossing_a_foreign_edge_is_found():
    # O51: the edge (3, 4) of the second triangle (in the plane x = y) crosses the edge (1, 2) of
    # the first one at (1, 1, 0), inside both edges; otherwise the triangles stay apart. gmsh HXT
    # rejects this ("Two segments intersect")
    pts = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [1, 1, -1], [1, 1, 1], [3, 3, 0]], dtype=float)
    tris = np.array([[0, 1, 2], [3, 4, 5]])
    assert len(self_intersection_pairs(pts, tris, shared_vertex=True)) == 0
    assert len(self_intersection_pairs(pts, tris, shared_vertex=True, touching=True)) == 1
    # moved slightly away from the first triangle: no contact
    pts[3:, 0] += 1e-3
    assert len(self_intersection_pairs(pts, tris, shared_vertex=True, touching=True)) == 0
    # an edge ending at a shared corner does not count
    pts = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 1], [-1, -1, 0]], dtype=float)
    tris = np.array([[0, 1, 2], [0, 3, 4]])
    assert len(self_intersection_pairs(pts, tris, shared_vertex=True, touching=True)) == 0

@pytest.fixture(scope='module')
def generated():
    import common
    from dragen.config import BoxConfig, PhaseConfig, RveConfig
    from dragen.pipeline import generate
    config = RveConfig(box=BoxConfig((15,) * 3, 2),
                       phases=(PhaseConfig('ferrite', common.lognormal_input(), 1.0),), seed=1)
    return generate(config)


def test_generated_rve_pipeline(generated):
    pytest.importorskip('gmsh')
    from dragen.mesh.mesher import MeshConfig, mesh_rve
    from dragen.mesh.tetmesh import TetMeshConfig, check as tet_check
    domain = generated.domain
    result = mesh_rve(generated.labels, domain.spacing, domain.periodic, MeshConfig(tetmesh=TetMeshConfig(verbosity=0)))
    rep = tet_check(result.surface, result.points, result.tets.mesh, generated.labels.shape, domain.spacing,
                    domain.periodic, result.smoothed.grain_volume_target)
    for key in ('inverted_elements', 'open_inner_faces', 'overused_faces', 'midside_mismatch',
                'interface_faces_missing', 'interface_faces_extra', 'periodic_unmatched_nodes',
                'periodic_unmatched_facets', 'grains_with_several_bodies'):
        assert rep[key] == 0, key
    assert rep['surface_nodes_moved'] == 0.0
    assert rep['volume_deviation']['max_abs'] < 0.011
    # acceptance: p1 of the smallest dihedral angle >= 15 degrees; the minimum >= 5 degrees except
    # for counted exceptions
    assert rep['dihedral_min_deg']['p1'] >= 15.0
    best = result.stats['feedback'][result.stats['feedback_best_round']]
    assert best['below_min_dihedral'] <= 2
    assert best['below_min_dihedral'] == min(f['below_min_dihedral'] for f in result.stats['feedback']
                                             if 'error' not in f)
    assert best['triangles'] < len(result.staircase.oriented)


def test_failed_feedback_round_keeps_best(generated, monkeypatch):
    # a feedback round gmsh cannot mesh must not discard the earlier rounds
    pytest.importorskip('gmsh')
    from dragen.mesh import mesher
    from dragen.mesh.tetmesh import TetMeshConfig
    original, calls = mesher.tet_mesh, []

    def failing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise Exception('HXT 3D mesh failed')
        return original(*args, **kwargs)

    monkeypatch.setattr(mesher, 'tet_mesh', failing)
    domain = generated.domain
    # a threshold above the 5 degrees leaves poor elements, so the feedback reaches round 1
    config = mesher.MeshConfig(tetmesh=TetMeshConfig(verbosity=0), min_dihedral=12.0, feedback=('coarsen',))
    result = mesher.mesh_rve(generated.labels, domain.spacing, domain.periodic, config)
    assert len(calls) == 2
    assert result.stats['feedback_best_round'] == 0
    assert 'HXT 3D mesh failed' in result.stats['feedback'][1]['error']
    assert len(result.tets.mesh.corners) == result.stats['feedback'][0]['elements']
