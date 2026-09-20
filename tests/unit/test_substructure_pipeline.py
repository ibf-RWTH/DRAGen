"""Unit tests for the substructure pipeline on a small synthetic RVE.

A full Run() takes minutes and needs a solver export to feed the pipeline, so this builds the
10x10x10 voxel mesh the pipeline actually consumes (GrainID + phaseID cell data) directly and drives
packets -> blocks -> orientations on it. That covers both packet regimes in one mesh:

    grain 1  ferrite,     fills the box, touches every face  -> not transformable, stays at -1
    grain 2  martensite,  a 6^3 block strictly inside grain 1 -> k-means packet split

The assertions are the invariants every stage claims to maintain, so a port slip anywhere in
dragen/substructure/ surfaces here rather than three hours into a scenario run.
"""
import numpy as np
import pyvista as pv
import pytest

from dragen.substructure import blocks, orientation, packets
from dragen.substructure.config import SubsConfig

N_CELLS_PER_EDGE = 10
INCLUSION_SLICE = slice(2, 8)   # 6^3 interior block, touches no face
FERRITE, MARTENSITE = 1, 2


def _synthetic_rve() -> pv.UnstructuredGrid:
    """A cube of ferrite with one interior martensite grain, in the pipeline's input format."""
    n = N_CELLS_PER_EDGE
    grid = pv.ImageData(dimensions=(n + 1, n + 1, n + 1), spacing=(1.0, 1.0, 1.0))

    phase = np.full((n, n, n), FERRITE, dtype=np.int32)
    phase[INCLUSION_SLICE, INCLUSION_SLICE, INCLUSION_SLICE] = MARTENSITE

    mesh = grid.cast_to_unstructured_grid()
    # ImageData cell order is x-fastest, matching Fortran order over (nx, ny, nz).
    mesh.cell_data['phaseID'] = phase.flatten(order='F')
    mesh.cell_data['GrainID'] = np.where(
        mesh.cell_data['phaseID'] == MARTENSITE, 2, 1).astype(np.int32)

    return mesh


def _config(tmp_path) -> SubsConfig:
    """Abaqus-flavoured config: parent orientations come from a graindata.inp we write here."""
    (tmp_path / 'graindata.inp').write_text(
        'Grain: 1: 10.0: 20.0: 30.0: 5.0\n'
        'Grain: 2: 200.0: 40.0: 100.0: 5.0\n'
    )
    (tmp_path / 'Postprocessing' / 'Substructure').mkdir(parents=True)

    return SubsConfig(
        store_path=str(tmp_path),
        solver='abaqus',
        length_unit='micrometer',      # mesh spacing is 1.0, so thicknesses are in cells
        block_generation_mode='user',
        average_block_thickness=2.0,
        min_packet_cells=100,
        min_block_cells=10,
        min_cells_per_packet=5,
        min_cells_per_block=5,
    )


@pytest.fixture
def labelled_rve(tmp_path):
    """Run the full labelling chain once; the tests below inspect the result."""
    cfg = _config(tmp_path)
    mesh = _synthetic_rve()

    mesh = packets.gen_packets(cfg, mesh)
    mesh = packets.merge_small_packets(cfg, mesh)
    mesh = blocks.gen_blocks(cfg, mesh)
    mesh = blocks.merge_small_blocks(cfg, mesh)
    mesh = orientation.assign_orientations(cfg, mesh)

    return cfg, mesh


def _arrays(mesh):
    return (np.asarray(mesh.cell_data['SubstructureFlag']).astype(int),
            np.asarray(mesh.cell_data['PacketID']).astype(int),
            np.asarray(mesh.cell_data['BlockID']).astype(int))


def test_only_transformable_phases_are_substructured(labelled_rve):
    _, mesh = labelled_rve
    sub_flag, _, _ = _arrays(mesh)
    phase = np.asarray(mesh.cell_data['phaseID']).astype(int)

    assert np.array_equal(sub_flag == 1, phase == MARTENSITE)
    assert np.count_nonzero(sub_flag == 1) == 6 ** 3


def test_substructure_flag_invariant(labelled_rve):
    _, mesh = labelled_rve
    sub_flag, packet_id, block_id = _arrays(mesh)

    for ids in (packet_id, block_id):
        assert np.all(ids[sub_flag == 1] > 0)
        assert np.all(ids[sub_flag == 0] == -1)


@pytest.mark.parametrize('name', ['PacketID', 'BlockID'])
def test_ids_are_gapless_and_one_based(labelled_rve, name):
    _, mesh = labelled_rve
    ids = np.asarray(mesh.cell_data[name]).astype(int)

    positive = np.unique(ids[ids > 0])
    assert positive.size > 0
    assert np.array_equal(positive, np.arange(1, positive.max() + 1))


def test_blocks_are_nested_in_packets_and_grains(labelled_rve):
    _, mesh = labelled_rve
    _, packet_id, block_id = _arrays(mesh)
    grain_id = np.asarray(mesh.cell_data['GrainID']).astype(int)

    for bid in np.unique(block_id[block_id > 0]):
        mask = block_id == bid
        assert np.unique(packet_id[mask]).size == 1
        assert np.unique(grain_id[mask]).size == 1

    for pid in np.unique(packet_id[packet_id > 0]):
        assert np.unique(grain_id[packet_id == pid]).size == 1


def test_interior_grain_is_split_into_several_packets_and_blocks(labelled_rve):
    _, mesh = labelled_rve
    _, packet_id, block_id = _arrays(mesh)

    # 216 cells / min_packet_cells=100 -> 3 k-means packets, each sliced into 2 um thick blocks.
    assert len(np.unique(packet_id[packet_id > 0])) == 3
    assert len(np.unique(block_id[block_id > 0])) > 3


def test_no_block_stays_below_the_merge_threshold(labelled_rve):
    cfg, mesh = labelled_rve
    _, _, block_id = _arrays(mesh)

    sizes = [np.count_nonzero(block_id == bid) for bid in np.unique(block_id[block_id > 0])]
    assert min(sizes) >= cfg.min_cells_per_block


def test_one_orientation_per_block(labelled_rve):
    _, mesh = labelled_rve
    _, _, block_id = _arrays(mesh)

    eulers = np.column_stack([mesh.cell_data['phi1'],
                              mesh.cell_data['PHI'],
                              mesh.cell_data['phi2']])

    for bid in np.unique(block_id[block_id > 0]):
        assert np.unique(eulers[block_id == bid], axis=0).shape[0] == 1


def test_unsubstructured_cells_keep_their_parent_orientation(labelled_rve):
    _, mesh = labelled_rve
    sub_flag, _, _ = _arrays(mesh)

    ferrite = sub_flag == 0
    assert np.allclose(mesh.cell_data['phi1'][ferrite], 10.0)
    assert np.allclose(mesh.cell_data['PHI'][ferrite], 20.0)
    assert np.allclose(mesh.cell_data['phi2'][ferrite], 30.0)


def test_variant_ids_are_valid_ks_variants(labelled_rve):
    _, mesh = labelled_rve
    sub_flag, _, _ = _arrays(mesh)
    variant_id = np.asarray(mesh.cell_data['VariantID']).astype(int)

    assert np.all((variant_id[sub_flag == 1] >= 0) & (variant_id[sub_flag == 1] <= 23))
    assert np.all(variant_id[sub_flag == 0] == -1)


def test_block_orientations_differ_from_the_parent_grain(labelled_rve):
    """A KS variant is a real rotation away from the PAG -- not the identity."""
    _, mesh = labelled_rve
    sub_flag, _, _ = _arrays(mesh)

    martensite = sub_flag == 1
    parent = np.array([200.0, 40.0, 100.0])
    eulers = np.column_stack([mesh.cell_data['phi1'],
                              mesh.cell_data['PHI'],
                              mesh.cell_data['phi2']])[martensite]

    assert not np.any(np.all(np.isclose(eulers, parent), axis=1))


def _is_rotation(matrix) -> bool:
    """Tabulated to 3 decimals, so the tolerance has to absorb that rounding (det comes out 1.0009)."""
    return (abs(np.linalg.det(matrix) - 1.0) < 2e-3
            and np.abs(matrix @ matrix.T - np.eye(3)).max() < 3e-3)


BROKEN_KS_VARIANTS = [0]


def test_most_ks_variant_matrices_are_rotations():
    for i, matrix in enumerate(orientation.T_LIST):
        if i in BROKEN_KS_VARIANTS:
            continue
        assert _is_rotation(matrix), f'variant {i} is not a proper rotation'


@pytest.mark.xfail(
    strict=True,
    reason="KS variant 0 in dragen/substructure/orientation.py::T_LIST is not a proper rotation "
           "(det 0.119, rows 0 and 1 nearly parallel), carried over verbatim from the source "
           "implementation. matrix_to_euler_bunge() runs it through project_to_SO3(), so it is "
           "silently coerced instead of raising: it collapses to a 7.3 deg rotation, i.e. blocks "
           "drawing it end up almost parallel to their parent grain rather than at a KS "
           "misorientation. It has variant 15's magnitudes with the minus signs lost, but "
           "restoring them would duplicate variant 15, and several 3-sign-flip candidates are "
           "valid KS variants missing from the table -- the intended entry is not recoverable "
           "from the table alone. Needs a decision from the materials side. (Variant 16 had a "
           "similar, unambiguous one-sign typo at (0,1) and has been fixed.)",
)
def test_all_ks_variant_matrices_are_rotations():
    for i, matrix in enumerate(orientation.T_LIST):
        assert _is_rotation(matrix), f'variant {i} is not a proper rotation'


def test_ks_variants_are_distinct():
    """24 variants means 24 *different* rotations; a duplicated row would bias variant selection."""
    for i in range(len(orientation.T_LIST)):
        for j in range(i + 1, len(orientation.T_LIST)):
            assert not np.allclose(orientation.T_LIST[i], orientation.T_LIST[j], atol=1e-6), (
                f'KS variants {i} and {j} are identical')


def test_euler_matrix_roundtrip():
    for phi1, PHI, phi2 in [(10.0, 20.0, 30.0), (200.0, 40.0, 100.0), (359.0, 179.0, 1.0)]:
        matrix = orientation.euler_bunge_to_matrix(phi1, PHI, phi2)
        recovered = orientation.matrix_to_euler_bunge(matrix)
        assert np.allclose(orientation.euler_bunge_to_matrix(*recovered), matrix, atol=1e-8)
