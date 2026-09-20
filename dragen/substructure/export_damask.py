"""Write the substructured RVE back out as a DAMASK material.yaml + grid.vti.

One DAMASK material per block for substructure cells, one per grain for everything else, so each
block carries its own orientation. The phase definitions and the homogenization from the
material.yaml that dragen/generation/spectral.py already wrote are kept -- only the material list
is replaced.
"""
import os

import numpy as np
import pyvista as pv
import damask
import yaml

from dragen.substructure.config import SubsConfig
from dragen.utilities.InputInfo import RveInfo

HOMOGENIZATION = 'SX'


def _phase_id_to_name(store_path: str) -> dict:
    """Invert the phase-name -> phaseID mapping that spectral.write_grid used for `phases`.

    That function classifies each material.yaml phase by substring, so 'Band' becomes phaseID 6 and
    the inclusion phase ('ThirdPhase') falls through to 7 -- note this is *not* RveInfo.PHASENUM,
    where 6 is Inclusions and 7 is Bands. Deriving the map from the file that produced grid.vti
    keeps the two in step instead of hardcoding either convention.
    """
    with open(os.path.join(store_path, 'material.yaml'), 'r', encoding='utf-8') as f:
        matdata = yaml.safe_load(f)

    # Same order and substrings as dragen/generation/spectral.py::write_grid.
    ordered_keys = [(1, 'Ferrite'), (2, 'Martensite'), (3, 'Pearlite'),
                    (4, 'Bainite'), (5, 'Austenite'), (6, 'Band')]

    phase_id_to_name = {}

    for name in matdata.get('phase', {}):
        for phase_id, key in ordered_keys:
            if key in name:
                phase_id_to_name[phase_id] = name
                break
        else:
            phase_id_to_name[7] = name

    RveInfo.LOGGER.info(f'DAMASK phase mapping: {phase_id_to_name}')

    return phase_id_to_name


def _check_one_orientation_per_material(material_id, phi1, PHI, phi2) -> None:
    for mid in np.unique(material_id):
        mask = material_id == mid
        unique_values = np.unique(np.column_stack([phi1[mask], PHI[mask], phi2[mask]]), axis=0)

        if unique_values.shape[0] != 1:
            raise ValueError(f'MaterialID {mid} has {unique_values.shape[0]} distinct '
                             'orientations, expected 1.')

    RveInfo.LOGGER.info('Orientation per MaterialID: OK')


def build_material_regions(mesh: pv.DataSet):
    """One 0-based DAMASK material per block (substructure) or per grain (everything else)."""
    grain_id = np.asarray(mesh.cell_data['GrainID']).astype(int)
    phase_id = np.asarray(mesh.cell_data['phaseID']).astype(int)
    block_id = np.asarray(mesh.cell_data['BlockID']).astype(int)
    sub_flag = np.asarray(mesh.cell_data['SubstructureFlag']).astype(int)

    key_to_material = {}
    region_keys = []
    material_per_cell = np.full(mesh.n_cells, -1, dtype=np.int32)

    for cell_id in range(mesh.n_cells):
        if sub_flag[cell_id] == 1:
            key = ('block', int(block_id[cell_id]))
        else:
            key = ('grain', int(phase_id[cell_id]), int(grain_id[cell_id]))

        if key not in key_to_material:
            key_to_material[key] = len(region_keys)
            region_keys.append(key)

        material_per_cell[cell_id] = key_to_material[key]

    if np.any(material_per_cell < 0):
        raise RuntimeError('Some cells did not receive a material ID.')

    return material_per_cell, region_keys


def write(cfg: SubsConfig, mesh: pv.DataSet) -> None:
    required = ['GrainID', 'phaseID', 'BlockID', 'SubstructureFlag', 'phi1', 'PHI', 'phi2']
    for array in required:
        if array not in mesh.cell_data:
            raise KeyError(f'Missing cell_data array: {array}')

    phase_id = np.asarray(mesh.cell_data['phaseID']).astype(int)
    block_id = np.asarray(mesh.cell_data['BlockID']).astype(int)
    sub_flag = np.asarray(mesh.cell_data['SubstructureFlag']).astype(int)

    phi1 = np.asarray(mesh.cell_data['phi1'], dtype=float)
    PHI = np.asarray(mesh.cell_data['PHI'], dtype=float)
    phi2 = np.asarray(mesh.cell_data['phi2'], dtype=float)

    if not (np.all(np.isfinite(phi1)) and np.all(np.isfinite(PHI)) and np.all(np.isfinite(phi2))):
        raise ValueError('Euler angles contain NaN or Inf values.')

    if np.any(PHI < 0) or np.any(PHI > 360):
        raise ValueError('PHI contains values outside 0..360 degrees.')

    if np.any(block_id[sub_flag == 1] < 1):
        raise ValueError('Some substructure cells have BlockID < 1.')

    if np.any(block_id[sub_flag == 0] != -1):
        raise ValueError('Some non-substructure cells have BlockID != -1.')

    phase_id_to_name = _phase_id_to_name(cfg.store_path)

    material_per_cell, region_keys = build_material_regions(mesh)
    mesh.cell_data['MaterialID'] = material_per_cell

    n_materials = len(region_keys)
    RveInfo.LOGGER.info(f'DAMASK materials after substructuring: {n_materials}')

    _check_one_orientation_per_material(material_per_cell, phi1, PHI, phi2)

    eulers = []
    phase_names = []

    for material_id in range(n_materials):
        mask = material_per_cell == material_id

        unique_values = np.unique(np.column_stack([phi1[mask], PHI[mask], phi2[mask]]), axis=0)
        eulers.append(unique_values[0])

        phases_in_material = np.unique(phase_id[mask])
        if phases_in_material.size != 1:
            raise ValueError(f'MaterialID {material_id} contains multiple phaseIDs: '
                             f'{phases_in_material}')

        pid = int(phases_in_material[0])
        if pid not in phase_id_to_name:
            raise KeyError(f'phaseID {pid} has no phase in material.yaml. '
                           f'Known: {phase_id_to_name}')

        phase_names.append(phase_id_to_name[pid])

    eulers = np.asarray(eulers, dtype=float)
    orientations = damask.Rotation.from_Euler_angles(eulers, degrees=True)

    _write_material_yaml(cfg, phase_names, orientations)
    _write_grid(cfg, mesh, material_per_cell)


def _write_material_yaml(cfg: SubsConfig, phase_names, orientations) -> None:
    material_file = os.path.join(cfg.store_path, 'material.yaml')

    matdata = damask.ConfigMaterial.load(material_file)

    available = set(matdata['phase'].keys())
    for name in set(phase_names):
        if name not in available:
            raise KeyError(f"Phase '{name}' is not defined in material.yaml. "
                           f'Available: {sorted(available)}')

    if HOMOGENIZATION not in matdata['homogenization']:
        raise KeyError(f"Homogenization '{HOMOGENIZATION}' not found in material.yaml.")

    matdata['material'] = []
    matdata = matdata.material_add(phase=phase_names, O=orientations,
                                   homogenization=HOMOGENIZATION)

    RveInfo.LOGGER.info(f'material.yaml complete: {matdata.is_complete}, '
                        f'valid: {matdata.is_valid}')

    matdata.save(material_file)
    RveInfo.LOGGER.info(f'Saved: {material_file}')


def _write_grid(cfg: SubsConfig, mesh: pv.DataSet, material_per_cell: np.ndarray) -> None:
    """Rebuild the regular voxel grid from the cell centres, in material.yaml order."""
    centers = np.round(mesh.cell_centers().points, 12)

    xs, ys, zs = (np.unique(centers[:, i]) for i in range(3))
    nx, ny, nz = len(xs), len(ys), len(zs)

    if nx * ny * nz != mesh.n_cells:
        raise RuntimeError(f'Not a complete regular voxel grid: nx*ny*nz = {nx * ny * nz}, '
                           f'mesh.n_cells = {mesh.n_cells}')

    i = np.searchsorted(xs, centers[:, 0])
    j = np.searchsorted(ys, centers[:, 1])
    k = np.searchsorted(zs, centers[:, 2])

    material = np.full((nx, ny, nz), -1, dtype=np.int32)
    material[i, j, k] = material_per_cell

    if np.any(material < 0):
        raise RuntimeError('Some grid positions were not filled.')

    bounds = mesh.bounds
    origin = np.array([bounds[0], bounds[2], bounds[4]], dtype=float)
    size = np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]],
                    dtype=float)

    grid_file = os.path.join(cfg.store_path, 'grid.vti')
    damask.GeomGrid(material=material, size=size, origin=origin).save(grid_file, compress=True)

    RveInfo.LOGGER.info(f'Saved: {grid_file} | shape {material.shape} | '
                        f'{len(np.unique(material))} materials')
