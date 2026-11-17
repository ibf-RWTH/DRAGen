"""Martensite substructure on the voxel field (dragen.substructure): packets, blocks, orientations,
the RVEResult with one region per block, and the KS variant table."""
import numpy as np
import pandas as pd
import pytest

from dragen.analysis.metrics import component_labels
from dragen.api import RVEResult, euler_to_quaternion
from dragen.substructure import SubstructureConfig, apply_substructure, generate_substructure
from dragen.substructure import orientation
from dragen.substructure.core import _unwrapped

FERRITE, MARTENSITE = 1, 2
PARENT = (200.0, 40.0, 100.0)


def synthetic(labels, phases, periodic=(True, True, False), spacing_um=0.5):
    """RVEResult from a hand-made label field; grain 1 has (10, 20, 30), the others PARENT."""
    labels = np.asarray(labels, dtype=np.int32)
    g = int(labels.max())
    euler = np.array([(10.0, 20.0, 30.0)] + [PARENT] * (g - 1))
    return RVEResult(grain_ids=labels, orientations=euler_to_quaternion(euler), euler_bunge_deg=euler,
                     phase_ids=np.asarray(phases, dtype=np.int32),
                     phase_names={1: 'ferrite', 2: 'martensite', 6: 'inclusions'},
                     box_size_m=tuple(s * spacing_um * 1e-6 for s in labels.shape), periodic=periodic)


@pytest.fixture(scope='module')
def rve():
    """24^3 ferrite with a 14^3 martensite grain inside and a martensite slab cut by the
    periodic x boundary (x in [20, 24) and [0, 4))."""
    labels = np.ones((24, 24, 24), dtype=np.int32)
    labels[5:19, 5:19, 5:19] = 2
    labels[20:, :, 2:8] = 3
    labels[:4, :, 2:8] = 3
    return synthetic(labels, [FERRITE, MARTENSITE, MARTENSITE])


@pytest.fixture(scope='module')
def sub(rve):
    return generate_substructure(rve, SubstructureConfig(block_thickness=1.0, seed=3))


def test_only_substructured_phases_get_packets_and_blocks(rve, sub):
    martensite = np.isin(rve.grain_ids, [2, 3])
    assert np.all(sub.packet_ids[martensite] > 0) and np.all(sub.packet_ids[~martensite] == -1)
    assert np.all(sub.block_ids[martensite] > 0) and np.all(sub.block_ids[~martensite] == -1)


@pytest.mark.parametrize('name', ['packet_ids', 'block_ids'])
def test_ids_are_gapless_and_one_based(sub, name):
    ids = np.unique(getattr(sub, name))
    ids = ids[ids > 0]
    assert ids[0] == 1 and np.array_equal(ids, np.arange(1, ids[-1] + 1))


def test_blocks_nest_in_packets_and_packets_in_grains(rve, sub):
    for b in range(1, sub.block_ids.max() + 1):
        mask = sub.block_ids == b
        assert np.unique(sub.packet_ids[mask]).tolist() == [sub.block_packet[b - 1]]
        assert np.unique(rve.grain_ids[mask]).tolist() == [sub.block_grain[b - 1]]
    for p in range(1, sub.packet_ids.max() + 1):
        assert np.unique(rve.grain_ids[sub.packet_ids == p]).tolist() == [sub.packet_grain[p - 1]]


def test_interior_grain_is_split_into_packets_and_blocks(sub):
    assert len(np.unique(sub.packet_ids[5:19, 5:19, 5:19])) == 4               # 2744 voxels: 4 packets
    assert len(np.unique(sub.block_ids[5:19, 5:19, 5:19])) > 8


def test_packets_of_a_grain_cut_by_the_boundary_are_bodies(rve, sub):
    """Coordinates are unwrapped, so k-means does not cut a packet out of both box ends."""
    packets = sub.packet_ids * (rve.grain_ids == 3)
    pieces = component_labels(packets, rve.periodic)
    for p in np.unique(packets[packets > 0]):
        assert len(np.unique(pieces[(packets == p).ravel()])) == 1


def test_unwrapping_makes_a_cut_grain_contiguous(rve):
    coords = _unwrapped(rve.grain_ids, rve.periodic)[:, 0].reshape(rve.grain_ids.shape)
    x = np.unique(coords[rve.grain_ids == 3])
    assert np.array_equal(x, np.arange(x.min(), x.min() + 8))              # 8 adjacent layers


def test_small_blocks_are_merged(rve, sub):
    counts = np.bincount(sub.block_ids[sub.block_ids > 0])[1:]
    assert np.count_nonzero(counts < 5) == sub.stats['blocks_below_min'] <= 2
    assert sub.stats['blocks_merged'] > 0


def test_ks_orientations(rve, sub):
    assert np.all((sub.block_variant >= 0) & (sub.block_variant <= 23))
    assert len(sub.packet_class) == sub.packet_ids.max() and set(sub.packet_class) <= {0, 1, 2, 3}
    g = orientation.euler_bunge_to_matrix(*PARENT)
    for euler, v in zip(sub.block_euler, sub.block_variant):
        expected = orientation.project_to_SO3(orientation.T_LIST[v] @ g)
        assert np.allclose(orientation.euler_bunge_to_matrix(*euler), expected, atol=1e-8)
    # blocks of one packet share the habit plane class: their variants lie in one packet group
    for p in range(1, sub.packet_ids.max() + 1):
        variants = sub.block_variant[sub.block_packet == p]
        assert len({v // 6 for v in variants}) == 1


def test_same_seed_same_substructure(rve, sub):
    again = generate_substructure(rve, SubstructureConfig(block_thickness=1.0, seed=3))
    assert np.array_equal(again.block_ids, sub.block_ids) and np.allclose(again.block_euler, sub.block_euler)
    other = generate_substructure(rve, SubstructureConfig(block_thickness=1.0, seed=4))
    assert not np.array_equal(other.block_ids, sub.block_ids)


def test_thicker_blocks_give_fewer_blocks(rve, sub):
    thick = generate_substructure(rve, SubstructureConfig(block_thickness=3.0, seed=3))
    assert thick.block_ids.max() < sub.block_ids.max()
    assert np.all(sub.block_thickness_um == 1.0)


def test_block_thickness_from_file(rve, tmp_path):
    values = np.r_[np.full(50, 0.05), np.linspace(0.8, 2.0, 200), np.full(5, 50.0)]
    pd.DataFrame({'block_thickness': values}).to_csv(tmp_path / 'b.csv', index=False)
    s = generate_substructure(rve, SubstructureConfig(block_file=str(tmp_path / 'b.csv'), seed=1))
    lo, hi = np.percentile(values[values >= 0.1], [5, 95])
    assert np.all((s.block_thickness_um >= lo) & (s.block_thickness_um <= hi))
    assert s.block_ids.max() > 10


def test_experimental_orientation_relationship(rve, tmp_path):
    """Children measured at a fixed transformation T from their parents: every block gets T g."""
    T = orientation.euler_bunge_to_matrix(30.0, 45.0, 10.0)
    parents = np.array([[0.0, 0.0, 0.0], [90.0, 30.0, 45.0]])
    rows = []
    for gid, p in enumerate(parents, start=1):
        rows.append((gid, *orientation.matrix_to_euler_bunge(T @ orientation.euler_bunge_to_matrix(*p))))
    pd.DataFrame(parents, columns=['phi1', 'PHI', 'phi2']).to_csv(tmp_path / 'p.csv', index=False)
    pd.DataFrame(rows, columns=['grain_id', 'phi1', 'PHI', 'phi2']).to_csv(tmp_path / 'c.csv', index=False)
    s = generate_substructure(rve, SubstructureConfig(block_thickness=1.0, orientation_mode='experimental',
                                                      parent_orientation_file=str(tmp_path / 'p.csv'),
                                                      child_orientation_file=str(tmp_path / 'c.csv')))
    expected = T @ orientation.euler_bunge_to_matrix(*PARENT)
    for euler in s.block_euler:
        assert np.allclose(orientation.euler_bunge_to_matrix(*euler), expected, atol=1e-8)


def test_config_is_checked(tmp_path):
    with pytest.raises(ValueError, match='either'):
        SubstructureConfig()
    with pytest.raises(ValueError, match='either'):
        SubstructureConfig(block_thickness=1.0, block_file='x.csv')
    with pytest.raises(ValueError, match='experimental'):
        SubstructureConfig(block_thickness=1.0, orientation_mode='experimental')
    with pytest.raises(ValueError):
        SubstructureConfig(block_thickness=1.0, phases=('unobtainium',))
    c = SubstructureConfig(block_thickness=1.0, phases=(2, 'bainite'))
    assert c.phase_ids == (2, 4) and SubstructureConfig.from_dict(c.to_dict()) == c


# ---------------------------------------------------------------------- RVEResult with blocks

def test_apply_substructure(rve, sub, tmp_path):
    b = apply_substructure(rve, sub)
    assert b.n_grains == 1 + sub.block_ids.max()
    assert np.array_equal(b.parent_grain_ids[b.grain_ids - 1], rve.grain_ids)    # same grains
    assert b.parent_grain_ids[0] == 1 and b.packet_ids[0] == -1                  # ferrite kept, first
    assert np.allclose(b.euler_bunge_deg[0], (10.0, 20.0, 30.0))
    assert np.all(b.phase_ids[1:] == MARTENSITE)
    # one orientation per block, the block's own
    for region in range(2, b.n_grains + 1):
        blocks = np.unique(sub.block_ids[b.grain_ids == region])
        assert len(blocks) == 1 and np.allclose(b.euler_bunge_deg[region - 1], sub.block_euler[blocks[0] - 1])
    assert np.array_equal(b.packet_ids[1:], sub.block_packet)
    b.save(tmp_path)
    again = RVEResult.load(tmp_path)
    assert np.array_equal(again.parent_grain_ids, b.parent_grain_ids)


def test_exports_of_blocks(rve, sub, tmp_path):
    from dragen.export import export_abaqus, grain_table
    b = apply_substructure(rve, sub)
    table = grain_table(b)
    assert {'parent_grain_id', 'packet_id'} <= set(table) and len(table) == b.n_grains
    inp = export_abaqus(b, tmp_path / 'rve')['inp'].read_text()
    assert '*Elset, elset=PARENT_2' in inp and '*Elset, elset=PACKET_1\n' in inp
    assert inp.count('*Solid Section') == b.n_grains


def test_tet_elements_get_the_block_at_their_centroid(rve, sub):
    """A coarse tet mesh of the voxel box (6 tets per voxel) keeps conformity; every element gets a
    block of its own grain."""
    from dragen.api import TetMesh
    from dragen.export.mesh import voxel_mesh
    hexes = voxel_mesh(rve)
    split = np.array([[0, 1, 3, 4], [1, 2, 3, 6], [1, 3, 4, 6], [1, 4, 5, 6], [3, 4, 6, 7]])
    cells = hexes.cells[:, split].reshape(-1, 4)
    grains = np.repeat(hexes.grain_ids, len(split))
    with_tet = RVEResult(**{**rve.__dict__, 'tet_mesh': TetMesh(hexes.points, cells, grains, hexes.periodic_pairs)})
    b = apply_substructure(with_tet, sub)
    assert np.array_equal(b.parent_grain_ids[b.tet_mesh.grain_ids - 1], grains)
    centroid_voxel = np.repeat(np.arange(rve.grain_ids.size), len(split))
    assert np.mean(b.tet_mesh.grain_ids == b.grain_ids.ravel()[centroid_voxel]) == 1.0


# ---------------------------------------------------------------------- KS table (legacy values)

def _is_rotation(matrix) -> bool:
    """Tabulated to 3 decimals: det 1 +- 2e-3, orthogonal to 3e-3."""
    return abs(np.linalg.det(matrix) - 1.0) < 2e-3 and np.abs(matrix @ matrix.T - np.eye(3)).max() < 3e-3


def test_ks_variants_except_0_are_rotations():
    for i, matrix in enumerate(orientation.T_LIST):
        if i != 0:
            assert _is_rotation(matrix), f'variant {i} is not a proper rotation'


@pytest.mark.xfail(strict=True, reason='KS variant 0 of the legacy table is no rotation (det 0.119); '
                                       'project_to_SO3 turns it into a 7.3 deg rotation. Needs a decision '
                                       'from the materials side (process.md O56).')
def test_all_ks_variants_are_rotations():
    for i, matrix in enumerate(orientation.T_LIST):
        assert _is_rotation(matrix), f'variant {i} is not a proper rotation'


def test_ks_variants_are_distinct():
    for i in range(24):
        for j in range(i + 1, 24):
            assert not np.allclose(orientation.T_LIST[i], orientation.T_LIST[j], atol=1e-6)


def test_euler_matrix_round_trip():
    for euler in [(10.0, 20.0, 30.0), PARENT, (359.0, 179.0, 1.0)]:
        m = orientation.euler_bunge_to_matrix(*euler)
        assert np.allclose(orientation.euler_bunge_to_matrix(*orientation.matrix_to_euler_bunge(m)), m, atol=1e-8)
