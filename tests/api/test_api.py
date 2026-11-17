"""Library interface dragen.api and the exports in dragen.export: small RVEs, no GUI."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dragen.analysis.metrics import component_labels
from dragen.api import PhaseSpec, RVESpec, euler_to_quaternion, generate_rve

ROOT = Path(__file__).resolve().parents[2]
FERRITE = ROOT / 'ExampleInput/Ferrite/TrainedData_Ferrite.pkl'          # with phi1, PHI, phi2
MARTENSITE = ROOT / 'ExampleInput/Martensite/TrainedData_Martensite.pkl'  # without: random texture
GOSS = (0.0, 45.0, 0.0)

pytest.importorskip('torch', reason="GAN sources need the extra 'ml'")


def spec(**kwargs) -> RVESpec:
    """16^3 voxels, about 20 grains: 70 % ferrite, 30 % martensite."""
    args = dict(box_size_m=5.5e-6, resolution=16, seed=1,
                phases=[PhaseSpec('ferrite', 0.7, FERRITE), PhaseSpec('martensite', 0.3, MARTENSITE)])
    args.update(kwargs)
    return RVESpec(**args)


@pytest.fixture(scope='module')
def voxel():
    return generate_rve(spec())


@pytest.fixture(scope='module')
def tet():
    return generate_rve(spec(mesh='tet'))


# ---------------------------------------------------------------------- result layout

def test_shapes_and_types(voxel):
    g = voxel.n_grains
    assert voxel.grain_ids.shape == (16, 16, 16) and voxel.grain_ids.dtype == np.int32
    assert voxel.orientations.shape == (g, 4) and voxel.orientations.dtype == np.float64
    assert voxel.euler_bunge_deg.shape == (g, 3)
    assert voxel.phase_ids.shape == (g,) and voxel.phase_ids.dtype == np.int32
    assert voxel.box_size_m == (5.5e-6,) * 3
    assert np.allclose(voxel.voxel_size_m, 5.5e-6 / 16)
    assert voxel.tet_mesh is None
    assert 10 <= g <= 40


def test_every_grain_has_one_unit_quaternion(voxel):
    assert np.array_equal(np.unique(voxel.grain_ids), np.arange(1, voxel.n_grains + 1))
    assert np.allclose(np.linalg.norm(voxel.orientations, axis=1), 1)
    assert np.all(voxel.orientations[:, 0] >= 0)
    assert voxel.orientation_of(voxel.grain_ids).shape == (16, 16, 16, 4)
    assert np.array_equal(voxel.orientation_of([1, voxel.n_grains]), voxel.orientations[[0, -1]])


def test_phase_fractions(voxel):
    fractions = voxel.metadata['phase_fractions']
    assert set(voxel.phase_names[int(p)] for p in voxel.phase_ids) == {'ferrite', 'martensite'}
    assert fractions['ferrite'] == pytest.approx(0.7, abs=0.1)
    assert fractions['martensite'] == pytest.approx(0.3, abs=0.1)
    assert sum(fractions.values()) == pytest.approx(1)


def test_metadata(voxel):
    meta = voxel.metadata
    assert meta['seed'] == 1 and meta['n_grains'] == voxel.n_grains
    assert len(meta['spec_sha256']) == 64
    assert meta['seconds']['total'] > 0
    assert meta['spec']['phases'][0]['name'] == 'ferrite'


def test_same_seed_same_rve_other_seed_other_rve(voxel):
    again = generate_rve(spec())
    assert np.array_equal(again.grain_ids, voxel.grain_ids)
    assert np.array_equal(again.orientations, voxel.orientations)
    assert again.metadata['spec_sha256'] == voxel.metadata['spec_sha256']
    other = generate_rve(spec(seed=2))
    assert not np.array_equal(other.grain_ids, voxel.grain_ids)


def test_periodic_grains_continue_across_the_faces(voxel):
    labels = voxel.grain_ids
    pieces = component_labels(labels, (True, True, True))
    assert len(np.unique(pieces)) == voxel.n_grains            # every grain is one body with wrap ...
    open_pieces = component_labels(labels, (False, False, False))
    assert len(np.unique(open_pieces)) > voxel.n_grains          # ... and some are cut by the faces
    for axis in range(3):                                       # opposite faces carry the same grains
        lo, hi = np.take(labels, 0, axis=axis), np.take(labels, -1, axis=axis)
        assert np.mean(lo == hi) > 0.5


def test_no_output_to_stdout(capfd):
    generate_rve(spec(seed=3))
    assert capfd.readouterr().out == ''


def test_invalid_specs_are_rejected():
    with pytest.raises(ValueError, match='unknown phase'):
        PhaseSpec('steel', 1.0, FERRITE)
    with pytest.raises(ValueError, match='add up to 1'):
        spec(phases=[PhaseSpec('ferrite', 0.5, FERRITE)])
    with pytest.raises(ValueError, match='mesh'):
        spec(mesh='hex')


def test_import_is_headless():
    code = ('import sys, dragen, dragen.api, dragen.export; '
            "print(sorted(m for m in ('matplotlib', 'PyQt5', 'torch', 'pyvista', 'damask', 'gmsh', "
            "'cv2', 'netCDF4', 'geomloss') if m in sys.modules))")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True, cwd=ROOT)
    assert out.stdout.strip() == '[]'


# ---------------------------------------------------------------------- orientations

def test_goss_quaternion_convention():
    q = euler_to_quaternion([GOSS])[0]
    assert np.allclose(q, [np.cos(np.pi / 8), np.sin(np.pi / 8), 0, 0])
    crystal_to_sample = Rotation.from_quat(q[[1, 2, 3, 0]])
    assert np.allclose(crystal_to_sample.apply([1, 0, 0]), [1, 0, 0])                  # [100] || RD
    assert np.allclose(crystal_to_sample.apply(np.array([0, 1, 1]) / np.sqrt(2)), [0, 0, 1])  # [011] || ND


def test_quaternions_are_the_damask_numbers():
    damask = pytest.importorskip('damask')
    euler = np.random.default_rng(0).uniform([0, 0, 0], [360, 180, 360], (50, 3))
    assert np.allclose(euler_to_quaternion(euler), damask.Rotation.from_Euler_angles(euler, degrees=True).quaternion)


def test_quaternions_match_euler(voxel):
    assert np.allclose(voxel.orientations, euler_to_quaternion(voxel.euler_bunge_deg))


def test_orientation_file_sets_the_texture_of_its_phase(voxel, tmp_path):
    goss = tmp_path / 'goss.csv'
    goss.write_text('phi1,PHI,phi2\n' + '0,45,0\n' * 3)
    textured = generate_rve(spec(phases=[PhaseSpec('ferrite', 0.7, FERRITE, orientations=goss),
                                         PhaseSpec('martensite', 0.3, MARTENSITE)]))
    assert np.array_equal(textured.grain_ids, voxel.grain_ids)        # geometry does not change
    ferrite = textured.phase_ids == 1
    assert np.allclose(textured.euler_bunge_deg[ferrite], GOSS)
    assert np.array_equal(textured.orientations[~ferrite], voxel.orientations[~ferrite])


# ---------------------------------------------------------------------- tetrahedral mesh

def test_tet_mesh(tet):
    m = tet.tet_mesh
    assert m.cells.dtype == np.int64 and m.cells.shape[1] == 10 and m.cell_type == 'tetra10'
    assert m.points.dtype == np.float64 and m.points.shape[1] == 3
    assert m.points.min() >= -1e-12 and m.points.max() <= 5.5e-6 * (1 + 1e-9)    # metres
    assert m.grain_ids.dtype == np.int32 and len(m.grain_ids) == len(m.cells)
    assert set(np.unique(m.grain_ids)) <= set(range(1, tet.n_grains + 1))
    assert tet.metadata['tet_mesh']['elements'] == len(m.cells)


def test_tet_periodic_node_pairs_match(tet):
    m = tet.tet_mesh
    assert set(m.periodic_pairs) == {0, 1, 2}
    for axis, pairs in m.periodic_pairs.items():
        assert len(pairs) > 0
        shift = np.zeros(3)
        shift[axis] = tet.box_size_m[axis]
        assert np.allclose(m.points[pairs[:, 0]] + shift, m.points[pairs[:, 1]], atol=1e-15)


def test_linear_tets_without_periodicity():
    r = generate_rve(spec(mesh='tet', tet_order=1, periodic=(True, True, False), seed=4))
    assert r.tet_mesh.cells.shape[1] == 4 and set(r.tet_mesh.periodic_pairs) == {0, 1}


# ---------------------------------------------------------------------- exports and saving

def test_damask_round_trip(voxel, tmp_path):
    damask = pytest.importorskip('damask')
    from dragen.export import export_damask
    files = export_damask(voxel, tmp_path / 'rve')
    grid = damask.GeomGrid.load(files['grid'])
    assert np.array_equal(grid.material + 1, voxel.grain_ids)
    assert np.allclose(grid.size, voxel.box_size_m)
    material = damask.ConfigMaterial.load(files['material'])
    assert len(material['material']) == voxel.n_grains
    o = np.array([m['constituents'][0]['O'] for m in material['material']])
    assert np.allclose(o, voxel.orientations)
    phases = [m['constituents'][0]['phase'] for m in material['material']]
    assert phases == [voxel.phase_names[int(p)].capitalize() for p in voxel.phase_ids]


def test_tet_exports_round_trip(tet, tmp_path):
    import meshio
    from dragen.export import export_gmsh, export_vtu
    vtu = meshio.read(export_vtu(tet, tmp_path / 'rve'))
    assert np.array_equal(vtu.cell_data['grain_id'][0], tet.tet_mesh.grain_ids)
    assert np.allclose(vtu.points, tet.tet_mesh.points)
    msh = meshio.read(export_gmsh(tet, tmp_path / 'rve'))
    grains = np.concatenate(msh.cell_data['gmsh:physical'])
    cells = np.concatenate([c.data for c in msh.cells if c.type == 'tetra10'])
    assert np.array_equal(grains, tet.tet_mesh.grain_ids) and np.array_equal(cells, tet.tet_mesh.cells)


def test_exodus_has_one_block_per_grain(tet, tmp_path):
    netCDF4 = pytest.importorskip('netCDF4')
    import meshio
    from dragen.export import export_exodus
    path = export_exodus(tet, tmp_path / 'rve')
    ids, counts = np.unique(tet.tet_mesh.grain_ids, return_counts=True)
    with netCDF4.Dataset(path) as ds:
        assert np.array_equal(ds['eb_prop1'][:], ids)                       # block ID = grain ID
        assert list(netCDF4.chartostring(ds['eb_names'][:])) == [f'grain_{g}' for g in ids]
        assert [ds.dimensions[f'num_el_in_blk{i}'].size for i in range(1, len(ids) + 1)] == list(counts)
    e = meshio.read(path)                    # meshio reads the blocks in file order, not by number
    assert sorted(len(c.data) for c in e.cells) == sorted(counts)


def test_exports_without_mesh_say_so(voxel, tmp_path):
    from dragen.export import export_vtu
    with pytest.raises(ValueError, match="mesh='tet'"):
        export_vtu(voxel, tmp_path / 'x', mesh='tet')


def test_save(tet, tmp_path):
    tet.save(tmp_path)
    data = np.load(tmp_path / 'rve.npz')
    assert np.array_equal(data['grain_ids'], tet.grain_ids)
    assert np.array_equal(data['tet_periodic_pairs_2'], tet.tet_mesh.periodic_pairs[2])
    assert (tmp_path / 'metadata.json').exists()
