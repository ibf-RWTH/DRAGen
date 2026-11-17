"""Mesh files of an RVEResult: VTU, Gmsh MSH 2.2 and Exodus II (MOOSE). Each takes the
tetrahedral mesh if the RVE has one, otherwise the voxel mesh (mesh='auto'); mesh='voxel' or
'tet' chooses explicitly (see dragen.export.mesh)."""
from pathlib import Path

import numpy as np

from dragen.export.mesh import EXODUS_SIDES, MeshKind, SolverMesh, box_faces, solver_mesh

_EXODUS_TYPE = {'hexahedron': 'HEX8', 'tetra': 'TETRA4', 'tetra10': 'TETRA10',
                'quad': 'QUAD4', 'triangle': 'TRI3', 'triangle6': 'TRI6'}


def _cell_data(result, m: SolverMesh):
    grains = m.grain_ids
    return {'grain_id': grains.astype(np.int32),
            'phase_id': result.phase_ids[grains - 1].astype(np.int32),
            'band_id': result.band_ids[grains - 1].astype(np.int32),
            'orientation': result.orientation_of(grains)}


def export_vtu(result, path: Path, mesh: MeshKind = 'auto') -> Path:
    """<path>.vtu with cell data grain_id, phase_id, band_id (-1 outside bands) and orientation
    (quaternion w, x, y, z)."""
    import meshio
    m = solver_mesh(result, mesh)
    path = Path(path).with_suffix('.vtu')
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: [v] for k, v in _cell_data(result, m).items()}
    meshio.Mesh(_xyz(m.points), [(m.cell_type, m.cells)], cell_data=data).write(path)
    return path


def export_gmsh(result, path: Path, mesh: MeshKind = 'auto') -> Path:
    """<path>.msh, Gmsh format 2.2 (ASCII), read by gmsh, meshio and MOOSE/libMesh. Physical and
    geometrical tag of every element = its grain ID."""
    import meshio
    m = solver_mesh(result, mesh)
    path = Path(path).with_suffix('.msh')
    path.parent.mkdir(parents=True, exist_ok=True)
    grains = m.grain_ids.astype(np.int32)
    out = meshio.Mesh(_xyz(m.points), [(m.cell_type, m.cells)],
                      cell_data={'gmsh:physical': [grains], 'gmsh:geometrical': [grains]})
    meshio.write(path, out, file_format='gmsh22', binary=False)
    return path


def export_exodus(result, path: Path, mesh: MeshKind = 'auto') -> Path:
    """<path>.e, Exodus II for MOOSE.

    Element blocks: one per grain, block ID = grain ID (MOOSE subdomain ID), name grain_<ID>;
    elements numbered block by block. Node sets and side sets: the six box faces with the names
    of MOOSE's GeneratedMesh, IDs 0-5 as there: left/right (x = 0 / x = L), bottom/top (y),
    back/front (z); node sets include the midside nodes. 2D: QUAD4/TRI3/TRI6, two coordinates,
    the four sides left/right/bottom/top (IDs 0-3). MOOSE's periodic boundaries pair these
    side sets (e.g. left - right). Written directly (netCDF classic 64-bit offset format, 32-bit
    integers) with scipy's netCDF writer: meshio drops block IDs, names and sets, and netCDF4
    defines variables in quadratic time in the number of dimensions (two per block: 30 s for
    2000 grains, against 0.1 s with scipy).
    """
    from scipy.io import netcdf_file
    m = solver_mesh(result, mesh)
    path = Path(path).with_suffix('.e')
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(m.points) >= 2 ** 31 or len(m.cells) >= 2 ** 31:
        raise ValueError('mesh too large for 32-bit Exodus IDs')

    order = np.argsort(m.grain_ids, kind='stable')
    ids, starts = np.unique(m.grain_ids[order], return_index=True)
    blocks = np.split(order, starts[1:])
    new_number = np.empty(len(m.cells), dtype=np.int64)       # 1-based element number in the file
    new_number[order] = np.arange(1, len(m.cells) + 1)
    node_sets = m.face_nodes()
    side_sets = m.face_sides(EXODUS_SIDES[m.cell_type])

    dims = [('time_step', None), ('len_string', 33), ('len_line', 81), ('four', 4), ('len_name', 33),
            ('num_dim', m.ndim), ('num_nodes', len(m.points)), ('num_elem', len(m.cells)),
            ('num_el_blk', len(ids)), ('num_node_sets', len(node_sets)), ('num_side_sets', len(side_sets))]
    variables = []                                   # (name, type, dims, attributes, data)

    def names(var, dim, values):
        chars = np.array(values, dtype='S33').view('S1').reshape(len(values), 33)
        variables.append((var, 'c', (dim, 'len_name'), {}, chars))

    def id_props(prefix, dim, values):
        variables.append((f'{prefix}_status', 'i4', (dim,), {}, np.ones(len(values), np.int32)))
        variables.append((f'{prefix}_prop1', 'i4', (dim,), {'name': 'ID'}, np.asarray(values, np.int32)))

    variables.append(('time_whole', 'f8', ('time_step',), {}, None))
    names('coor_names', 'num_dim', ['x', 'y', 'z'][:m.ndim])
    for axis, var in enumerate(['coordx', 'coordy', 'coordz'][:m.ndim]):
        variables.append((var, 'f8', ('num_nodes',), {}, m.points[:, axis]))
    id_props('eb', 'num_el_blk', ids)
    names('eb_names', 'num_el_blk', [f'grain_{g}' for g in ids])
    for i, block in enumerate(blocks, start=1):
        dims += [(f'num_el_in_blk{i}', len(block)), (f'num_nod_per_el{i}', m.cells.shape[1])]
        variables.append((f'connect{i}', 'i4', (f'num_el_in_blk{i}', f'num_nod_per_el{i}'),
                          {'elem_type': _EXODUS_TYPE[m.cell_type]}, (m.cells[block] + 1).astype(np.int32)))
    faces = box_faces(m.ndim)
    set_names = [name for name, _, _ in faces]
    for prefix, count in (('ns', 'num_node_sets'), ('ss', 'num_side_sets')):
        id_props(prefix, count, np.arange(len(faces)))
        names(f'{prefix}_names', count, set_names)
    for i, name in enumerate(set_names, start=1):
        nodes = node_sets[name]
        elems, faces = side_sets[name]
        dims += [(f'num_nod_ns{i}', len(nodes)), (f'num_side_ss{i}', len(elems))]
        variables += [(f'node_ns{i}', 'i4', (f'num_nod_ns{i}',), {}, (nodes + 1).astype(np.int32)),
                      (f'elem_ss{i}', 'i4', (f'num_side_ss{i}',), {}, new_number[elems].astype(np.int32)),
                      (f'side_ss{i}', 'i4', (f'num_side_ss{i}',), {}, (faces + 1).astype(np.int32))]

    with netcdf_file(path, 'w', version=2) as ds:
        ds.title = f'DRAGen RVE, spec {result.metadata.get("spec_sha256", "")}'[:80].encode('ascii')
        ds.api_version = np.float32(8.03)
        ds.version = np.float32(8.03)
        ds.floating_point_word_size = np.int32(8)
        ds.file_size = np.int32(1)
        ds.maximum_name_length = np.int32(32)
        ds.int64_status = np.int32(0)
        for name, size in dims:
            ds.createDimension(name, size)
        for name, kind, var_dims, attrs, data in variables:
            v = ds.createVariable(name, kind, var_dims)
            for key, value in attrs.items():
                setattr(v, key, value.encode('ascii') if isinstance(value, str) else value)
            if data is not None:
                v[:] = data
    return path


def _xyz(points: np.ndarray) -> np.ndarray:
    """2D points with z = 0 (VTU and MSH store three coordinates)."""
    return points if points.shape[1] == 3 else np.column_stack([points, np.zeros(len(points))])
