"""dragen.export (E1): voxel and tetrahedral solver meshes, Exodus II with box-face sets, Abaqus
input with periodic equations, grain table, DAMASK phases and orientations of bands and inclusions
(findings 08, 09, 12). The files are read back with independent parsers (netCDF4, a small Abaqus
reader below), not with the code that wrote them. CSV sources only, so no extra 'ml' is needed."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dragen.api import BandSpec, InclusionSpec, PhaseSpec, RVESpec, generate_rve
from dragen.export import (export_abaqus, export_damask, export_exodus, export_grains, export_vtu,
                           grain_table, solver_mesh, voxel_mesh)
from dragen.export.mesh import BOX_FACES

BOX = 10e-6


@pytest.fixture(scope='module')
def sources(tmp_path_factory):
    d = tmp_path_factory.mktemp('src')
    rng = np.random.default_rng(0)
    a = rng.uniform(1.5, 2.5, 2000)
    pd.DataFrame({'a': a, 'b': a * rng.uniform(0.6, 1, 2000), 'c': a, 'phi1': rng.uniform(0, 360, 2000),
                  'PHI': rng.uniform(0, 180, 2000), 'phi2': rng.uniform(0, 360, 2000)}).to_csv(d / 'grains.csv', index=False)
    a = rng.uniform(0.5, 0.7, 2000)
    pd.DataFrame({'a': a, 'b': a, 'c': a}).to_csv(d / 'inclusions.csv', index=False)
    return d / 'grains.csv', d / 'inclusions.csv'


def spec(sources, **kwargs):
    grains, inclusions = sources
    args = dict(box_size_m=BOX, resolution=20, seed=1, phases=[PhaseSpec('ferrite', 1.0, grains)],
                bands=BandSpec(1, (3e-6, 3.5e-6), grains, phase='martensite'),
                inclusions=InclusionSpec(inclusions, 0.01))
    args.update(kwargs)
    return RVESpec(**args)


@pytest.fixture(scope='module')
def voxel(sources):
    return generate_rve(spec(sources))


@pytest.fixture(scope='module')
def tet(sources):
    """Quadratic tets, periodic in x and y only, no bands (keeps the mesh small)."""
    return generate_rve(spec(sources, resolution=14, bands=None, inclusions=None, mesh='tet', periodic=(True, True, False)))


# ---------------------------------------------------------------------- API: bands and inclusions

def test_bands_and_inclusions_reach_the_result(voxel):
    assert voxel.band_ids.dtype == np.int32 and len(voxel.band_ids) == voxel.n_grains
    assert voxel.band_ids[0] == 0 and voxel.phase_ids[0] == 2           # band 0 is grain 1, martensite
    assert np.count_nonzero(voxel.phase_ids == 6) > 0                     # inclusions
    assert set(np.unique(voxel.grain_ids)) == set(range(1, voxel.n_grains + 1))
    assert voxel.radii_m.shape == (voxel.n_grains, 3) and voxel.radii_m.max() < BOX
    assert voxel.metadata['spec']['bands']['count'] == 1


def test_band_and_inclusion_specs_are_checked(sources):
    with pytest.raises(ValueError, match='band phase'):
        BandSpec(1, (3e-6, 4e-6), sources[0], phase='bands')
    with pytest.raises(ValueError):
        spec(sources, bands=BandSpec(1, (4e-6, 3e-6), sources[0]))
    with pytest.raises(ValueError):
        spec(sources, inclusions=InclusionSpec(sources[1], 1.5))


def test_spec_hash_covers_band_source(sources, tmp_path):
    from dragen.api import _spec_hash
    copy = tmp_path / 'grains.csv'
    copy.write_bytes(sources[0].read_bytes())
    s = spec(sources, bands=BandSpec(1, (3e-6, 3.5e-6), copy))
    before = _spec_hash(s)
    copy.write_text(copy.read_text() + '\n')
    assert _spec_hash(s) != before


# ---------------------------------------------------------------------- solver mesh

def _hex_volume_sign(points, cells):
    p = points[cells]
    return np.einsum('ij,ij->i', p[:, 1] - p[:, 0], np.cross(p[:, 3] - p[:, 0], p[:, 4] - p[:, 0]))


def test_voxel_mesh_matches_labels(voxel):
    m = voxel_mesh(voxel)
    assert m.cells.shape == (voxel.grain_ids.size, 8) and m.cell_type == 'hexahedron'
    assert np.all(_hex_volume_sign(m.points, m.cells) > 0)
    centre = m.points[m.cells].mean(axis=1)
    ijk = np.floor(centre / np.array(voxel.voxel_size_m)).astype(int)
    assert np.array_equal(voxel.grain_ids[tuple(ijk.T)], m.grain_ids)
    assert m.points.max(axis=0).tolist() == list(voxel.box_size_m)        # exact upper faces


@pytest.mark.parametrize('kind', ['voxel', 'tet'])
def test_periodic_classes(voxel, tet, kind):
    r = voxel if kind == 'voxel' else tet
    m = solver_mesh(r)
    slave, master, offset = m.periodic_classes()
    assert len(np.unique(slave)) == len(slave) and not np.isin(master, slave).any()
    L = np.array(r.box_size_m)
    assert np.allclose(m.points[slave], m.points[master] + offset * L, atol=1e-9 * L.max())
    assert set(np.unique(offset)) <= {0, 1} and np.all(offset.sum(axis=1) >= 1)
    # every node on an upper periodic face is a slave, and only those
    upper = np.zeros(len(m.points), bool)
    for a in m.periodic_pairs:
        upper |= m.on_face(a, 1)
    assert np.array_equal(np.sort(slave), np.flatnonzero(upper))
    if kind == 'voxel':                                                     # fully periodic box corner
        origin = np.flatnonzero(np.all(m.points == 0, axis=1))[0]
        assert np.count_nonzero(master == origin) == 7


def test_mesh_kind_choice(voxel, tet):
    assert solver_mesh(tet).cell_type == 'tetra10' and solver_mesh(tet, 'voxel').cell_type == 'hexahedron'
    assert solver_mesh(voxel).cell_type == 'hexahedron'
    with pytest.raises(ValueError, match="mesh='tet'"):
        solver_mesh(voxel, 'tet')


# ---------------------------------------------------------------------- Exodus II

EXODUS_FACES = {'HEX8': [(1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7), (1, 5, 8, 4), (1, 4, 3, 2), (5, 6, 7, 8)],
                'TETRA4': [(1, 2, 4), (2, 3, 4), (1, 4, 3), (1, 3, 2)]}
EXODUS_FACES['TETRA10'] = EXODUS_FACES['TETRA4']


@pytest.mark.parametrize('kind', ['voxel', 'tet'])
def test_exodus_blocks_and_box_face_sets(voxel, tet, kind, tmp_path):
    """Faces from the Exodus II standard numbering (written out here, not taken from the code):
    every side of a side set lies in its box face and its normal points out of the box."""
    netCDF4 = pytest.importorskip('netCDF4')
    r = voxel if kind == 'voxel' else tet
    m = solver_mesh(r)
    path = export_exodus(r, tmp_path / 'rve')
    with netCDF4.Dataset(path) as ds:
        pts = np.column_stack([ds['coordx'][:], ds['coordy'][:], ds['coordz'][:]])
        assert np.allclose(pts, m.points)
        ids = np.asarray(ds['eb_prop1'][:])
        assert list(ids) == sorted(set(m.grain_ids.tolist()))
        conn, etype = [], None
        for i, g in enumerate(ids, start=1):
            c = np.asarray(ds[f'connect{i}'][:]) - 1
            etype = ds[f'connect{i}'].elem_type
            assert np.all(m.grain_ids[_rows(m.cells, c)] == g)
            conn.append(c)
        conn = np.concatenate(conn)                                         # element k+1 = row k
        assert len(conn) == len(m.cells)
        ss_names = list(netCDF4.chartostring(ds['ss_names'][:]))
        ns_names = list(netCDF4.chartostring(ds['ns_names'][:]))
        assert ss_names == ns_names == [n for n, _, _ in BOX_FACES]
        L = np.array(r.box_size_m)
        for i, (name, axis, side) in enumerate(BOX_FACES, start=1):
            nodes = np.asarray(ds[f'node_ns{i}'][:]) - 1
            target = 0 if side == 0 else L[axis]
            assert len(nodes) and np.allclose(pts[nodes, axis], target, atol=1e-12)
            assert len(nodes) == np.count_nonzero(np.isclose(pts[:, axis], target, atol=1e-12))
            elems = np.asarray(ds[f'elem_ss{i}'][:]) - 1
            sides = np.asarray(ds[f'side_ss{i}'][:]) - 1
            assert len(elems) > 0
            faces = np.array([EXODUS_FACES[etype][s] for s in sides]) - 1
            fn = conn[elems[:, None], faces]
            assert np.allclose(pts[fn, axis], target, atol=1e-12)
            p = pts[fn]
            normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, axis]
            assert np.all(np.sign(normal) == (1 if side else -1))
        # the side sets cover the box faces: area per face = box face area
            area = 0.5 * np.abs(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, axis])
            if etype == 'HEX8':
                area *= 2
            assert np.isclose(area.sum(), np.prod(np.delete(L, axis)), rtol=1e-9)


def _rows(cells, conn):
    """Indices of the rows of `conn` in `cells`."""
    lookup = {tuple(c): i for i, c in enumerate(cells.tolist())}
    return np.array([lookup[tuple(c)] for c in conn.tolist()])


# ---------------------------------------------------------------------- Abaqus

def read_inp(path):
    """Minimal reader for the keywords export_abaqus writes: part nodes and elements, sets,
    sections, assembly nodes and equations."""
    lines = Path(path).read_text().splitlines()
    out = dict(nodes={}, elements=[], etype=None, elsets={}, nsets={}, sections={}, rp={}, equations=[])
    key, opts, in_assembly, i = None, {}, False, 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.startswith('**') or not line.strip():
            continue
        if line.startswith('*'):
            parts = [p.strip() for p in line[1:].split(',')]
            key = parts[0].lower()
            opts = dict(p.lower().split('=', 1) if '=' in p else (p.lower(), '') for p in parts[1:])
            if key == 'assembly':
                in_assembly = True
            if key == 'element':
                out['etype'] = opts['type'].upper()
            if key == 'solid section':
                out['sections'][opts['elset'].upper()] = parts[2].split('=')[1]
            if key == 'equation':
                n = int(lines[i])
                i += 1
                terms = []
                while len(terms) < n:
                    t = [x.strip() for x in lines[i].split(',') if x.strip()]
                    terms += [(t[j], int(t[j + 1]), float(t[j + 2])) for j in range(0, len(t), 3)]
                    i += 1
                out['equations'].append(terms)
            continue
        values = [v.strip() for v in line.split(',') if v.strip()]
        if key == 'node':
            target = out['rp'] if in_assembly else out['nodes']
            target[int(values[0])] = [float(v) for v in values[1:]]
        elif key == 'element':
            out['elements'].append([int(v) for v in values])
        elif key == 'elset':
            out['elsets'].setdefault(opts['elset'].upper(), []).extend(int(v) for v in values)
        elif key == 'nset':
            out['nsets'].setdefault(opts['nset'].upper(), []).extend(int(v) for v in values)
    return out


@pytest.mark.parametrize('kind', ['voxel', 'tet'])
def test_abaqus_input(voxel, tet, kind, tmp_path):
    r = voxel if kind == 'voxel' else tet
    m = solver_mesh(r)
    files = export_abaqus(r, tmp_path / 'rve', length_unit='um')
    inp = read_inp(files['inp'])
    assert inp['etype'] == {'hexahedron': 'C3D8', 'tetra10': 'C3D10'}[m.cell_type]
    pts = np.array([inp['nodes'][k] for k in range(1, len(m.points) + 1)])
    assert np.allclose(pts, m.points * 1e6, rtol=1e-9, atol=1e-9)
    elements = np.array(inp['elements'])
    assert np.array_equal(elements[:, 0], np.arange(1, len(m.cells) + 1))
    assert np.array_equal(elements[:, 1:], m.cells + 1)

    # element sets: grains partition the mesh, phase and band sets follow from them
    grain_sets = {k: v for k, v in inp['elsets'].items() if k.startswith('GRAIN_')}
    members = np.concatenate([v for v in grain_sets.values()])
    assert np.array_equal(np.sort(members), np.arange(1, len(m.cells) + 1))
    for k, v in grain_sets.items():
        g = int(k.split('_')[1])
        assert np.all(m.grain_ids[np.array(v) - 1] == g)
        assert inp['sections'][k] == f'{r.phase_names[int(r.phase_ids[g - 1])].capitalize()}_{g}'
    elem_phase = r.phase_ids[m.grain_ids - 1]
    for p in np.unique(elem_phase):
        assert sorted(inp['elsets'][r.phase_names[int(p)].upper()]) == list(np.flatnonzero(elem_phase == p) + 1)
    if kind == 'voxel':
        assert sorted(inp['elsets']['BAND_0']) == list(np.flatnonzero(m.grain_ids == 1) + 1)
        assert 'INCLUSIONS' in inp['elsets']
    for name, axis, side in BOX_FACES:
        target = 0 if side == 0 else r.box_size_m[axis] * 1e6
        assert sorted(inp['nsets'][name.upper()]) == list(np.flatnonzero(np.isclose(pts[:, axis], target)) + 1)

    # equations: u_s - u_m - sum k_a u(RP_a) = 0, geometrically right, one per slave and DOF
    rp_name = {'RP_X': 0, 'RP_Y': 1, 'RP_Z': 2}
    axes = sorted(m.periodic_pairs)
    assert sorted(n for n in inp['nsets'] if n.startswith('RP_')) == sorted(['RP_X', 'RP_Y', 'RP_Z'][a] for a in axes)
    L = np.array(r.box_size_m) * 1e6
    first = []
    for eq in inp['equations']:
        (s, dof, c1), (mst, dof2, c2), *rps = eq
        assert c1 == 1 and c2 == -1 and all(c == -1 and d == dof for _, d, c in rps) and dof2 == dof
        si, mi = int(s.split('.')[1]), int(mst.split('.')[1])
        shift = np.zeros(3)
        for name, _, _ in rps:
            shift[rp_name[name]] += L[rp_name[name]]
        assert np.allclose(pts[si - 1], pts[mi - 1] + shift, atol=1e-6)
        first.append((si, dof))
    assert len(set(first)) == len(first)                                   # no node eliminated twice
    eliminated = {s for s, _ in first}
    assert not eliminated & {int(t[1][0].split('.')[1]) for t in inp['equations']}
    upper = np.zeros(len(m.points), bool)
    for a in axes:
        upper |= m.on_face(a, 1)
    assert sorted(eliminated) == list(np.flatnonzero(upper) + 1)
    assert pts[inp['nsets']['ORIGIN'][0] - 1].tolist() == [0, 0, 0]


def test_abaqus_graindata(voxel, tmp_path):
    files = export_abaqus(voxel, tmp_path / 'rve', length_unit='um')
    lines = files['graindata'].read_text().splitlines()
    assert lines[0] == '!MMM Crystal Plasticity Input File'
    rows = [l.split(':') for l in lines[1:]]
    grains = [int(r[1]) for r in rows]
    assert grains == [g for g in range(1, voxel.n_grains + 1) if voxel.phase_ids[g - 1] != 6]
    angles = np.array([[float(x) for x in r[2:5]] for r in rows])
    assert np.allclose(angles, voxel.euler_bunge_deg[np.array(grains) - 1], atol=1e-6)
    voxels = np.bincount(voxel.grain_ids.ravel())[np.array(grains)]
    radius = np.cbrt(voxels * (voxel.voxel_size_m[0] * 1e6) ** 3 * 3 / (4 * np.pi))
    assert np.allclose([float(r[5]) for r in rows], radius, rtol=1e-8)


def test_abaqus_options(voxel, tmp_path):
    inp = read_inp(export_abaqus(voxel, tmp_path / 'a', element_type='C3D8R', material_per='phase')['inp'])
    assert inp['etype'] == 'C3D8R' and set(inp['sections'].values()) == {'Ferrite', 'Martensite', 'Inclusions'}
    with pytest.raises(ValueError, match='does not fit'):
        export_abaqus(voxel, tmp_path / 'b', element_type='C3D10')
    with pytest.raises(ValueError, match='length_unit'):
        export_abaqus(voxel, tmp_path / 'c', length_unit='inch')


def test_abaqus_without_periodicity(sources, tmp_path):
    r = generate_rve(spec(sources, resolution=10, bands=None, inclusions=None, periodic=False))
    inp = read_inp(export_abaqus(r, tmp_path / 'rve')['inp'])
    assert inp['equations'] == [] and not any(n.startswith('RP_') for n in inp['nsets'])


# ---------------------------------------------------------------------- grain table, DAMASK, VTU

def test_grain_table_has_all_phases(voxel, tmp_path):
    """Finding 12: the legacy sampled_grains.csv held only the last phase."""
    table = pd.read_csv(export_grains(voxel, tmp_path / 'grains'))
    assert len(table) == voxel.n_grains and list(table.grain_id) == list(range(1, voxel.n_grains + 1))
    assert set(table.phase) == {'ferrite', 'martensite', 'inclusions'}
    assert table.voxels.sum() == voxel.grain_ids.size
    assert np.allclose(table[['qw', 'qx', 'qy', 'qz']].to_numpy(), voxel.orientations)
    assert table.band_id.tolist() == voxel.band_ids.tolist()
    assert np.allclose(table.volume_m3.sum(), np.prod(voxel.box_size_m))
    assert grain_table(voxel).shape == table.shape


def test_damask_phases_and_orientations_of_all_grains(voxel, sources, tmp_path):
    """Finding 08 (band/inclusion phase names swapped) and 09 (martensite, pearlite, bainite got
    random orientations): every material has its grain's phase and orientation; the band grain's
    orientation is one of the source's measured orientations."""
    damask = pytest.importorskip('damask')
    files = export_damask(voxel, tmp_path / 'damask')
    material = damask.ConfigMaterial.load(files['material'])['material']
    phases = [mat['constituents'][0]['phase'] for mat in material]
    expected = {1: 'Ferrite', 2: 'Martensite', 6: 'Inclusions'}
    assert phases == [expected[int(p)] for p in voxel.phase_ids]
    o = np.array([mat['constituents'][0]['O'] for mat in material])
    assert np.allclose(o, voxel.orientations)
    source = pd.read_csv(sources[0])[['phi1', 'PHI', 'phi2']].to_numpy()
    band = voxel.euler_bunge_deg[0]
    assert np.min(np.abs(source - band).max(axis=1)) < 1e-9


def test_vtu_of_voxel_mesh(voxel, tmp_path):
    import meshio
    out = meshio.read(export_vtu(voxel, tmp_path / 'rve'))
    assert out.cells[0].type == 'hexahedron'
    assert np.array_equal(out.cell_data['grain_id'][0], voxel.grain_ids.ravel())
    assert np.array_equal(out.cell_data['band_id'][0], voxel.band_ids[voxel.grain_ids.ravel() - 1])


# ---------------------------------------------------------------------- 2D

@pytest.fixture(scope='module')
def rve2d(sources):
    return generate_rve(RVESpec(box_size_m=(10e-6, 8e-6), resolution=20, seed=3,
                                phases=[PhaseSpec('ferrite', 1.0, sources[0])], periodic=(True, False),
                                bands=BandSpec(1, (2e-6, 2.5e-6), sources[0])))


@pytest.fixture(scope='module')
def tri2d(sources):
    return generate_rve(RVESpec(box_size_m=(10e-6, 8e-6), resolution=20, seed=3, mesh='tet',
                                phases=[PhaseSpec('ferrite', 1.0, sources[0])], periodic=(True, True)))


def test_2d_result(rve2d, tri2d):
    assert rve2d.grain_ids.shape == (20, 16) and rve2d.periodic == (True, False)
    assert rve2d.band_ids[0] == 0                                            # band normal to y
    rows = (rve2d.grain_ids == 1).any(axis=0)                                # y layers of the band
    assert 0 < rows.sum() < 12 and (rve2d.grain_ids == 1).any(axis=1).mean() > 0.8   # along x
    t = tri2d.tet_mesh
    assert t.cell_type == 'triangle6' and t.points.shape[1] == 2 and set(t.periodic_pairs) == {0, 1}
    assert tri2d.metadata['tri_mesh']['inverted'] == 0


EXODUS_EDGES = {'QUAD4': [(1, 2), (2, 3), (3, 4), (4, 1)], 'TRI3': [(1, 2), (2, 3), (3, 1)],
                'TRI6': [(1, 2), (2, 3), (3, 1)]}


@pytest.mark.parametrize('kind', ['pixel', 'tri'])
def test_2d_exodus(rve2d, tri2d, kind, tmp_path):
    """Exodus standard edge numbering written out here: every side lies on its box side with the
    element inside the box, and the sides cover the box side."""
    netCDF4 = pytest.importorskip('netCDF4')
    r = rve2d if kind == 'pixel' else tri2d
    path = export_exodus(r, tmp_path / 'rve')
    with netCDF4.Dataset(path) as ds:
        assert ds.dimensions['num_dim'].size == 2
        pts = np.column_stack([ds['coordx'][:], ds['coordy'][:]])
        n_blocks = ds.dimensions['num_el_blk'].size
        conn = np.concatenate([np.asarray(ds[f'connect{i}'][:]) - 1 for i in range(1, n_blocks + 1)])
        etype = ds['connect1'].elem_type
        names = list(netCDF4.chartostring(ds['ss_names'][:]))
        assert names == ['left', 'right', 'bottom', 'top']
        L = np.array(r.box_size_m)
        for i, (axis, side) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)], start=1):
            elems = np.asarray(ds[f'elem_ss{i}'][:]) - 1
            sides = np.asarray(ds[f'side_ss{i}'][:]) - 1
            edge = np.array([EXODUS_EDGES[etype][s] for s in sides]) - 1
            nodes = conn[elems[:, None], edge]
            target = side * L[axis]
            assert np.allclose(pts[nodes, axis], target, atol=1e-12)
            # element interior (centroid of its corners) on the inner side of the box side
            n_corner = 4 if etype == 'QUAD4' else 3
            centroid = pts[conn[elems, :n_corner]].mean(axis=1)[:, axis]
            assert np.all((centroid > target) if side == 0 else (centroid < target))
            length = np.abs(pts[nodes[:, 1], 1 - axis] - pts[nodes[:, 0], 1 - axis]).sum()
            assert np.isclose(length, L[1 - axis])


@pytest.mark.parametrize('kind', ['pixel', 'tri'])
def test_2d_abaqus(rve2d, tri2d, kind, tmp_path):
    r = rve2d if kind == 'pixel' else tri2d
    m = solver_mesh(r)
    inp = read_inp(export_abaqus(r, tmp_path / 'rve', length_unit='um')['inp'])
    assert inp['etype'] == {'quad': 'CPE4', 'triangle6': 'CPE6'}[m.cell_type]
    pts = np.array([inp['nodes'][k] for k in range(1, len(m.points) + 1)])
    assert pts.shape[1] == 2 and np.allclose(pts, m.points * 1e6)
    assert {'LEFT', 'RIGHT', 'BOTTOM', 'TOP'} <= set(inp['nsets']) and 'BACK' not in inp['nsets']
    L = np.array(r.box_size_m) * 1e6
    rp_axis = {'RP_X': 0, 'RP_Y': 1}
    for eq in inp['equations']:
        (s, dof, _), (mst, _, _), *rps = eq
        assert dof in (1, 2)
        shift = np.zeros(2)
        for name, _, _ in rps:
            shift[rp_axis[name]] += L[rp_axis[name]]
        assert np.allclose(pts[int(s.split('.')[1]) - 1], pts[int(mst.split('.')[1]) - 1] + shift, atol=1e-6)
    upper = np.zeros(len(m.points), bool)
    for a in m.periodic_pairs:
        upper |= m.on_face(a, 1)
    assert len(inp['equations']) == 2 * upper.sum()
    text = (tmp_path / 'rve.inp').read_text()
    thickness = text.split('*Solid Section')[1].splitlines()[1]
    assert float(thickness.rstrip(',')) == pytest.approx(r.voxel_size_m[0] * 1e6)


def test_2d_damask_and_grains(rve2d, tmp_path):
    damask = pytest.importorskip('damask')
    grid = damask.GeomGrid.load(export_damask(rve2d, tmp_path / 'd')['grid'])
    assert grid.cells.tolist() == [20, 16, 1] and np.array_equal(grid.material[:, :, 0] + 1, rve2d.grain_ids)
    assert np.allclose(grid.size, [10e-6, 8e-6, 0.5e-6])
    table = grain_table(rve2d)
    assert 'area_m2' in table and np.isclose(table.area_m2.sum(), 80e-12)
    assert np.allclose(table.eq_diameter_m, 2 * np.sqrt(table.area_m2 / np.pi))


def test_2d_vtu(tri2d, tmp_path):
    import meshio
    out = meshio.read(export_vtu(tri2d, tmp_path / 'rve'))
    assert out.cells[0].type == 'triangle6' and np.allclose(out.points[:, :2], tri2d.tet_mesh.points)
