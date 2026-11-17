"""Discrete surface model (M2): loops, shells, periodic links and vertex constraints."""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'benchmarks'))

from dragen.mesh import corpus  # noqa: E402
from dragen.mesh.cleanup import ManifoldCleaner  # noqa: E402
from dragen.mesh.model import FIXED, ON_LINE, ON_SURFACE, build_model  # noqa: E402
from dragen.mesh.topology import build_topology  # noqa: E402

CASES = corpus.cases()
WITH_MODEL = [c for c in CASES if 'model' in c.expected]
MUST_BE_ZERO = ('unused_curves', 'curves_used_once', 'slit_curves', 'unbalanced_shell_curves',
                'non_integer_genus', 'misoriented_triangles', 'unmatched_periodic_surfaces',
                'inexact_periodic_links')


def model_of(case):
    labels, _ = ManifoldCleaner(case.periodic).run(case.labels)
    return build_model(build_topology(labels, (1, 1, 1), case.periodic))


@pytest.mark.parametrize('case', CASES, ids=[c.name for c in CASES])
def test_model_is_consistent(case):
    check = model_of(case).check()
    assert {k: check[k] for k in MUST_BE_ZERO} == dict.fromkeys(MUST_BE_ZERO, 0)


@pytest.mark.parametrize('case', WITH_MODEL, ids=[c.name for c in WITH_MODEL])
def test_model_matches_hand_counts(case):
    check = model_of(case).check()
    for key, value in case.expected['model'].items():
        assert check[key] == value, key


def test_ring_surface_has_two_loops_and_points_out_of_its_larger_grain():
    model = model_of(corpus.ring_triple_line())
    annulus = [s for s in model.surfaces if s.regions == (2, 1)]
    assert len(annulus) == 1 and len(annulus[0].loops) == 2
    # the 1-2 boundary lies in the plane z = 5; its normal must point out of grain 2, i.e. down
    p = model.mesh.points[model.oriented[annulus[0].triangles]]
    normals = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    assert np.all(normals[:, 2] < 0) and np.allclose(normals[:, :2], 0)


def test_shell_of_every_grain_encloses_its_voxel_volume():
    case = corpus.two_contact_patches()
    model = model_of(case)
    p = model.mesh.points
    for grain, shell in model.shells.items():
        volume = 0.0
        for s, sign in shell:
            t = p[model.oriented[model.surfaces[s].triangles]]
            volume += sign * np.einsum('ij,ij->i', t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6
        assert volume == pytest.approx(np.count_nonzero(case.labels == grain))


def test_vertex_constraints_follow_the_box_rules():
    model = model_of(corpus.ring_triple_line())
    corner, n = model.mesh.corner, 10
    on_box = (corner == 0) | (corner == n)
    assert np.array_equal(model.box_lock, on_box)
    # box corners are fixed, box-edge vertices lock two coordinates, face vertices one
    corners = np.flatnonzero(on_box.sum(axis=1) == 3)
    assert len(corners) == 8 and np.all(model.vertex_kind[corners] == FIXED)
    edge = np.flatnonzero(on_box.sum(axis=1) == 2)
    assert np.all(model.vertex_kind[edge] != ON_SURFACE)
    # the ring around the cube: one anchor, the other ring vertices move along the ring
    ring = [i for i, ln in enumerate(model.curves) if ln.closed][0]
    ring_vertices = model.curves[ring].vertices[1:-1]
    assert np.all(model.vertex_kind[ring_vertices] == ON_LINE)
    assert np.all(model.vertex_entity[ring_vertices] == ring)
    inner = np.flatnonzero(~on_box.any(axis=1) & (model.vertex_kind == ON_SURFACE))
    assert len(inner) > 0
    assert all(s.regions[0] > 0 for s in model.surfaces)


def test_periodic_links_translate_exactly():
    case = corpus.fragmented_grain(periodic=True)
    model = model_of(case)
    corner = model.mesh.corner
    for link in model.periodic_surfaces:
        slave = np.unique(model.oriented[model.surfaces[link.slave].triangles])
        master = np.unique(model.oriented[model.surfaces[link.master].triangles])
        moved = {tuple(c) for c in corner[master] + link.translation}
        assert moved == {tuple(c) for c in corner[slave]}
        # masters lie on the lower faces
        assert all(t >= 0 for t in link.translation) and sum(link.translation) > 0
    # the eight box corners: one master at the origin, seven slaves
    origin = [l for l in model.periodic_points if np.all(corner[model.points[l.master]] == 0)]
    assert sorted(l.translation for l in origin) == sorted(
        (8 * i, 4 * j, 4 * k) for i in (0, 1) for j in (0, 1) for k in (0, 1) if i or j or k)


def test_imprinted_ring_is_slave_of_the_real_ring():
    model = model_of(corpus.periodic_imprint())
    rings = {i for i, ln in enumerate(model.curves) if ln.closed}
    links = [l for l in model.periodic_curves if l.slave in rings]
    assert len(links) == 1
    assert not model.curves[links[0].master].imprinted and model.curves[links[0].slave].imprinted
    assert links[0].translation == (6, 0, 0)


def test_pyvista_export(tmp_path):
    pytest.importorskip('pyvista')
    model = model_of(corpus.ring_triple_line())
    blocks = model.to_pyvista()
    assert blocks['surfaces'].n_cells == len(model.mesh.triangles)
    assert blocks['curves'].n_cells == len(model.curves)
    assert blocks['points'].n_points == len(model.points)
    blocks.save(str(tmp_path / 'model.vtm'))


@pytest.fixture(scope='module')
def generated():
    import common
    from dragen.config import BoxConfig, PhaseConfig, RveConfig
    from dragen.pipeline import generate
    config = RveConfig(box=BoxConfig((15,) * 3, 2),
                       phases=(PhaseConfig('ferrite', common.lognormal_input(), 1.0),), seed=2)
    return generate(config)


def test_generated_rve_model(generated):
    """Seed 2 contains a pinched patch; its loops must still be clean."""
    periodic = generated.domain.periodic
    labels, _ = ManifoldCleaner(periodic).run(generated.labels)
    topo = build_topology(labels, generated.domain.spacing, periodic)
    assert topo.pinched_patches() == 1
    model = build_model(topo)
    check = model.check()
    assert {k: check[k] for k in MUST_BE_ZERO} == dict.fromkeys(MUST_BE_ZERO, 0)
    assert check['periodic_surfaces'] > 0 and check['periodic_curves'] > 0
    # every surface on a periodic box face is master or slave
    assert check['unmatched_periodic_surfaces'] == 0


def test_check_detects_a_flipped_triangle():
    model = model_of(corpus.two_halves())
    # a triangle away from every line: all three edges are shared inside its surface
    inner = np.flatnonzero(np.all(model.vertex_kind[model.oriented] == ON_SURFACE, axis=1))[0]
    model.oriented[inner] = model.oriented[inner, ::-1]
    assert model.check()['misoriented_triangles'] == 3
