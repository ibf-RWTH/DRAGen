"""2D triangle mesh with smoothed grain boundaries (dragen.mesh.mesh2d)."""
import numpy as np
import pytest

import common
from dragen.config import BoxConfig, PhaseConfig, RveConfig
from dragen.mesh.mesh2d import Mesh2DConfig, boundary_segments, cleanup_pinches, mesh_rve_2d, trace_curves
from dragen.mesh.tetmesh import periodic_node_pairs
from dragen.pipeline import generate

pytest.importorskip('gmsh')


def edges_of(cells):
    tri = cells[:, :3]
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    return np.unique(e, axis=0, return_counts=True)


def check_mesh(m, size, periodic, spacing, n_grains):
    """Conforming (every edge in one or two triangles, single edges only on the box), counter-
    clockwise, grain areas equal to the pixel areas, periodic node pairs complete and exact."""
    q = m.stats
    assert q['inverted'] == 0 and q['min_angle_deg'] > 15
    assert q['mesh_area_deviation_max'] < 0.005 and q['total_area_error'] < 1e-9
    edges, count = edges_of(m.cells)
    assert set(count) <= {1, 2}
    single = edges[count == 1]
    p = m.points[single]
    on_box = np.zeros(len(single), dtype=bool)
    for axis in range(2):
        for target in (0.0, size[axis]):
            on_box |= np.isclose(p[:, :, axis], target).all(axis=1)
    assert on_box.all(), 'free edges inside the box: the grains are not connected'
    assert set(np.unique(m.grain)) == set(range(1, n_grains + 1))
    axes = [a for a in range(2) if periodic[a]]
    pairs = periodic_node_pairs(m.points, size, axes)          # raises if a node has no partner
    for a, pr in pairs.items():
        shift = np.zeros(2)
        shift[a] = size[a]
        assert np.allclose(m.points[pr[:, 0]] + shift, m.points[pr[:, 1]], atol=1e-9)
    if m.cells.shape[1] == 6:                                    # straight sides: midside at the middle
        c = m.cells
        for mid, (i, j) in zip((3, 4, 5), ((0, 1), (1, 2), (2, 0))):
            assert np.allclose(m.points[c[:, mid]], 0.5 * (m.points[c[:, i]] + m.points[c[:, j]]))


@pytest.mark.parametrize('periodic,order', [((True, True), 2), ((True, False), 1), ((False, False), 2)])
def test_generated_rve(periodic, order):
    config = RveConfig(box=BoxConfig((30.0, 24.0), 2.0, periodic=periodic),
                       phases=(PhaseConfig('ferrite', common.lognormal_input(), 1.0),), seed=2)
    r = generate(config)
    m = mesh_rve_2d(r.labels, config.box.spacing, periodic, Mesh2DConfig(order=order))
    assert m.cell_type == ('triangle6' if order == 2 else 'triangle')
    check_mesh(m, config.box.size, periodic, config.box.spacing, len(r.grains))
    assert m.stats['smoothing_fallbacks'] == 0


def test_smoothing_moves_the_staircase():
    """A diagonal staircase between two grains becomes a nearly straight line."""
    n = 24
    i, j = np.indices((n, n))
    labels = np.where(i + j < n, 1, 2)
    m = mesh_rve_2d(labels, (1.0, 1.0), (False, False), Mesh2DConfig(order=1))
    interior = [c for c in m.curves if not (np.isclose(c[:, 0], 0) | np.isclose(c[:, 0], n) |
                                            np.isclose(c[:, 1], 0) | np.isclose(c[:, 1], n)).all()]
    assert len(interior) == 1
    # grain 1 has n (n + 1) / 2 pixels: the straight line with that area is x + y = sqrt(n (n + 1));
    # the staircase deviates from it by up to 0.5 (in x + y: 1)
    p = interior[0][3:-3]
    assert np.abs(p.sum(axis=1) - np.sqrt(n * (n + 1))).max() < 0.1
    check_mesh(m, (n, n), (False, False), (1.0, 1.0), 2)


def test_island_grain_is_a_hole():
    """A grain enclosed by another one: closed curve without anchor, outer grain with a hole."""
    labels = np.ones((20, 20), dtype=int)
    i, j = np.indices(labels.shape)
    labels[(i - 9.5) ** 2 + (j - 9.5) ** 2 < 25] = 2
    segs = boundary_segments(labels)
    curves, anchor = trace_curves(segs, 21 * 21)
    closed = [c for c, l, r in curves if c[0] == c[-1] and min(l, r) > 0]
    assert len(closed) == 1
    m = mesh_rve_2d(labels, (0.5, 0.5), (True, True), Mesh2DConfig())
    check_mesh(m, (10.0, 10.0), (True, True), (0.5, 0.5), 2)
    u = m.points[m.cells[:, 1]] - m.points[m.cells[:, 0]]
    v = m.points[m.cells[:, 2]] - m.points[m.cells[:, 0]]
    area = np.bincount(m.grain, weights=0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]))
    assert area[2] == pytest.approx(np.count_nonzero(labels == 2) * 0.25, rel=0.005)


def test_grain_cut_by_the_periodic_boundary():
    labels = np.ones((16, 16), dtype=int)
    labels[:4, 5:11] = 2
    labels[12:, 5:11] = 2                                        # one body across x = 0
    m = mesh_rve_2d(labels, (1.0, 1.0), (True, True), Mesh2DConfig())
    check_mesh(m, (16.0, 16.0), (True, True), (1.0, 1.0), 2)


def test_pinch_cleanup():
    """a b / c a: grain a touches itself only at a corner; afterwards it is one face-connected body."""
    labels = np.array([[1, 2, 2],
                       [3, 1, 2],
                       [3, 3, 2]])
    clean, changed = cleanup_pinches(labels, (False, False))
    assert changed >= 1
    from dragen.analysis.metrics import components_per_grain
    assert np.all(components_per_grain(clean, (False, False))[1:][np.bincount(clean.ravel())[1:] > 0] == 1)


def test_four_grains_meet_at_a_point():
    labels = np.ones((12, 12), dtype=int)
    labels[6:, :6] = 2
    labels[:6, 6:] = 3
    labels[6:, 6:] = 4
    m = mesh_rve_2d(labels, (1.0, 1.0), (False, False), Mesh2DConfig())
    check_mesh(m, (12.0, 12.0), (False, False), (1.0, 1.0), 4)
    assert np.any(np.all(np.isclose(m.points, 6.0), axis=1))           # the quadruple point stays
