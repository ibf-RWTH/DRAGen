"""The mesh quality metrics (M0) must detect the defects they are named after."""
import numpy as np
import pytest

from dragen.mesh import corpus, quality
from dragen.mesh.surface import InterfaceMesh, staircase_mesh
from dragen.mesh.volume import (VolumeMesh, element_volumes, grain_boundary_mesh, hex_to_tets,
                                scaled_jacobian, tet4_to_tet10, tet_dihedral_angles, tet_radius_ratio,
                                voxel_hex_mesh)

REGULAR_TET = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)


def two_halves_mesh():
    return staircase_mesh(corpus.two_halves().labels, (1, 1, 1))


def inner_triangle(mesh):
    return int(np.flatnonzero((mesh.grains > 0).all(axis=1))[0])


# -------------------------------------------------------------------------------------- surfaces

def test_flipped_triangle_is_misoriented_for_both_grains():
    mesh = two_halves_mesh()
    f = inner_triangle(mesh)
    mesh.triangles[f] = mesh.triangles[f, ::-1]
    grains = quality.grain_surfaces(mesh).set_index('grain')
    assert grains.loc[1, 'misoriented_edges'] == grains.loc[2, 'misoriented_edges'] == 3
    assert grains.loc[1, 'open_edges'] == 0


def test_missing_triangle_opens_both_grains():
    mesh = two_halves_mesh()
    f = inner_triangle(mesh)
    keep = np.arange(len(mesh.triangles)) != f
    mesh = InterfaceMesh(mesh.points, mesh.triangles[keep], mesh.grains[keep])
    grains = quality.grain_surfaces(mesh).set_index('grain')
    assert grains.loc[1, 'open_edges'] == grains.loc[2, 'open_edges'] == 3
    assert not grains['closed_manifold'].any()


def test_duplicate_and_same_region_triangles_are_counted():
    mesh = two_halves_mesh()
    f = inner_triangle(mesh)
    mesh = InterfaceMesh(mesh.points, np.vstack([mesh.triangles, mesh.triangles[f]]),
                         np.vstack([mesh.grains, [[1, 1]]]))
    rep = quality.interface_report(mesh)
    assert rep['duplicate_triangles'] == 1
    assert rep['same_region_triangles'] == 1


def test_self_intersection_of_crossing_triangles():
    points = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0],
                       [0.5, 0.5, -1], [0.5, 0.5, 1], [3, 3, 0],
                       [5, 5, 5], [6, 5, 5], [5, 6, 5]], dtype=float)
    crossing = np.array([[0, 1, 2], [3, 4, 5]])
    assert quality.self_intersections(points, crossing) == 1
    apart = np.array([[0, 1, 2], [6, 7, 8]])
    assert quality.self_intersections(points, apart) == 0
    sharing_vertex = np.array([[0, 1, 2], [0, 4, 5]])
    assert quality.self_intersections(points, sharing_vertex) == 0


def test_triangle_quality_of_equilateral_and_right_triangles():
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0], [0, 1, 0]])
    q = quality.triangle_quality(points, np.array([[0, 1, 2], [0, 1, 3]]))
    assert q['min_angle'] == pytest.approx([60, 45])
    assert q['radius_ratio'][0] == pytest.approx(1)
    assert q['radius_ratio'][1] == pytest.approx(2 * (np.sqrt(2) - 1))


def test_enclosed_volume_follows_moved_vertex():
    mesh = staircase_mesh(corpus.single_voxel().labels, (1, 1, 1))
    grains = quality.grain_surfaces(mesh).set_index('grain')
    assert grains.loc[2, 'volume'] == pytest.approx(1)
    # pull one corner of the single voxel outward along the diagonal: volume grows, grain 1 shrinks
    corner = np.flatnonzero(np.all(mesh.points == [3, 3, 3], axis=1))[0]
    mesh.points[corner] += 0.3
    grains = quality.grain_surfaces(mesh).set_index('grain')
    assert grains.loc[2, 'volume'] > 1
    assert grains.loc[1, 'volume'] + grains.loc[2, 'volume'] == pytest.approx(125)


# ---------------------------------------------------------------------------------- volume meshes

def tet_mesh(points, cells, cell_type='tet4'):
    return VolumeMesh(np.asarray(points, float), np.asarray(cells), cell_type, np.ones(len(cells), int))


def test_regular_tetrahedron_scores_one():
    mesh = tet_mesh(REGULAR_TET, [[0, 1, 2, 3]])
    if element_volumes(mesh)[0] < 0:
        mesh = tet_mesh(REGULAR_TET, [[0, 2, 1, 3]])
    assert scaled_jacobian(mesh)[0] == pytest.approx(1)
    assert tet_radius_ratio(mesh)[0] == pytest.approx(1)
    assert tet_dihedral_angles(mesh)[0] == pytest.approx(np.full(6, np.degrees(np.arccos(1 / 3))))


def test_inverted_tetrahedron_is_counted():
    mesh = tet_mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 1, 2, 3], [0, 2, 1, 3]])
    rep = quality.volume_report(mesh, (0, 0, 0), (1, 1, 1))
    assert rep['inverted_elements'] == 1


def test_unit_cube_hex_scores_one_and_has_exact_volume():
    mesh = voxel_hex_mesh(np.ones((1, 1, 1), int), (2, 3, 4))
    assert scaled_jacobian(mesh)[0] == pytest.approx(1)
    assert element_volumes(mesh)[0] == pytest.approx(24)


@pytest.mark.parametrize('case', [corpus.edge_contact(), corpus.torus(), corpus.two_contact_patches()],
                         ids=lambda c: c.name)
def test_reference_volume_meshes_are_exact_and_conforming(case):
    spacing = (0.5, 1.0, 0.25)
    counts = np.bincount(case.labels.ravel()) * np.prod(spacing)
    reference = {g: counts[g] for g in range(1, len(counts))}
    upper = np.array(case.labels.shape) * spacing
    hexes = voxel_hex_mesh(case.labels, spacing)
    for mesh in (hexes, hex_to_tets(hexes), tet4_to_tet10(hex_to_tets(hexes))):
        rep = quality.volume_report(mesh, (0, 0, 0), upper, reference_volume=reference)
        assert rep['inverted_elements'] == 0
        assert rep['open_inner_faces'] == rep['overused_faces'] == rep['unused_points'] == 0
        assert rep['volume_deviation']['max_abs'] < 1e-12
        expected_split = sum(1 for n in case.expected.get('bodies', {}).values() if n > 1)
        assert rep['grains_with_several_bodies'] == expected_split
        if mesh.cell_type == 'tet10':
            assert rep['midside_mismatch'] == 0
            assert rep['jacobian_ratio']['min'] == pytest.approx(1)


def test_differently_split_neighbours_leave_gaps():
    hexes = voxel_hex_mesh(np.array([[[1]], [[2]]]), (1, 1, 1))
    # turn the second hex by 180 degrees about z: the diagonal on the shared face changes
    hexes.cells[1] = hexes.cells[1][[2, 3, 0, 1, 6, 7, 4, 5]]
    tets = hex_to_tets(hexes)
    rep = quality.volume_report(tets, (0, 0, 0), (2, 1, 1))
    assert rep['open_inner_faces'] > 0
    assert rep['inverted_elements'] == 0


def test_midside_mismatch_and_curved_tet10():
    hexes = voxel_hex_mesh(np.ones((2, 1, 1), int), (1, 1, 1))
    tet10 = tet4_to_tet10(hex_to_tets(hexes))
    points = np.vstack([tet10.points, tet10.points[tet10.cells[0, 4]]])
    cells = tet10.cells.copy()
    cells[0, 4] = len(points) - 1           # same position, but a node of its own
    split = VolumeMesh(points, cells, 'tet10', tet10.grain)
    assert quality.volume_report(split, (0, 0, 0), (2, 1, 1))['midside_mismatch'] == 1
    curved = VolumeMesh(tet10.points.copy(), tet10.cells, 'tet10', tet10.grain)
    a, b = curved.cells[0, :2]
    curved.points[curved.cells[0, 4]] += 0.45 * np.cross(curved.points[b] - curved.points[a], [0, 0, 1])
    rep = quality.volume_report(curved, (0, 0, 0), (2, 1, 1))
    assert rep['jacobian_ratio']['min'] < 1


# ------------------------------------------------------------------------------ periodic, anchors

def test_periodic_mismatch_detects_moved_node():
    case = corpus.fragmented_grain(periodic=True)
    mesh = staircase_mesh(case.labels, (1, 1, 1))
    box_tris = mesh.triangles[(mesh.grains < 0).any(axis=1)]
    node = np.flatnonzero((mesh.points[:, 0] == 8) & (mesh.points[:, 1] == 2) & (mesh.points[:, 2] == 2))[0]
    mesh.points[node, 1] += 0.1
    result = quality.periodic_mismatch(mesh.points, box_tris, (0, 0, 0), (8, 4, 4), axes=(0,))[0]
    assert result['unmatched_nodes'] == 2
    assert result['unmatched_facets'] > 0


def test_anchor_displacement():
    before = np.zeros((3, 3))
    after = before.copy()
    after[1] = [0, 3, 4]
    d = quality.anchor_displacement(before, after, np.array([0, 1]))
    assert d == {'n': 2, 'max': 5.0, 'mean': 2.5, 'moved': 1}


@pytest.mark.parametrize('case', [corpus.ring_triple_line(), corpus.vertex_contact()], ids=lambda c: c.name)
def test_grain_boundary_of_voxel_meshes_equals_staircase(case):
    upper = np.array(case.labels.shape, float)
    reference = quality.grain_surfaces(staircase_mesh(case.labels, (1, 1, 1))).set_index('grain')
    hexes = voxel_hex_mesh(case.labels, (1, 1, 1))
    for mesh in (hexes, hex_to_tets(hexes)):
        boundary = grain_boundary_mesh(mesh, (0, 0, 0), upper)
        got = quality.grain_surfaces(boundary).set_index('grain')
        assert got['volume'].to_numpy() == pytest.approx(reference['volume'].to_numpy())
        assert list(got['euler']) == list(reference['euler'])
        assert (got['open_edges'] == 0).all() and (got['misoriented_edges'] == 0).all()
        assert quality.interface_patches(boundary) == case.expected['patches']


def test_gap_between_grains_shows_as_open_surface():
    hexes = voxel_hex_mesh(np.array([[[1]], [[2]]]), (1, 1, 1))
    hexes.cells[1] = hexes.cells[1][[2, 3, 0, 1, 6, 7, 4, 5]]
    boundary = grain_boundary_mesh(hex_to_tets(hexes), (0, 0, 0), (2, 1, 1))
    # each grain keeps a closed surface of its own, but the two do not share their triangles
    rep = quality.interface_report(boundary)
    assert rep['gap_triangles'] == 4
    assert rep['open_edges'] == 0


def test_grain_across_periodic_boundary_is_one_body():
    case = corpus.fragmented_grain(periodic=True)
    hexes = voxel_hex_mesh(case.labels, (1, 1, 1))
    open_box = quality.volume_report(hexes, (0, 0, 0), (8, 4, 4))
    periodic = quality.volume_report(hexes, (0, 0, 0), (8, 4, 4), periodic_axes=(0, 1, 2))
    assert open_box['grains_with_several_bodies'] == 1
    assert periodic['grains_with_several_bodies'] == 0


def test_box_faces_of_periodic_hex_mesh_mirror_each_other():
    case = corpus.fragmented_grain(periodic=True)
    hexes = voxel_hex_mesh(case.labels, (1, 1, 1))
    boundary = grain_boundary_mesh(hexes, (0, 0, 0), (8, 4, 4))
    box_tris = boundary.triangles[(boundary.grains < 0).any(axis=1)]
    result = quality.periodic_mismatch(boundary.points, box_tris, (0, 0, 0), (8, 4, 4), axes=(0, 1, 2))
    assert all(r['unmatched_nodes'] == 0 and r['unmatched_facets'] == 0 for r in result.values())
