"""Bring a meshed RVE into the single internal format the substructure pipeline works on.

Internal format: a `pyvista.UnstructuredGrid` with two int32 cell_data arrays,

    GrainID  -- 1-based parent (austenite) grain id
    phaseID  -- DRAGen phase id, see RveInfo.PHASENUM

Everything downstream only ever sees this, so the DAMASK and Abaqus paths differ in exactly one step.
"""
import os

import numpy as np
import pyvista as pv
import yaml

from dragen.utilities.InputInfo import RveInfo


def _read_mesh_safely(filename: str) -> pv.DataSet:
    """Abaqus/meshio VTK files can carry types (vtktypeint32) that trip up pv.read()."""
    try:
        mesh = pv.read_meshio(filename)
        RveInfo.LOGGER.info(f'Loaded with pv.read_meshio(): {filename}')
        return mesh
    except Exception as read_meshio_error:
        RveInfo.LOGGER.info(f'pv.read_meshio() failed ({read_meshio_error}), trying pv.read()')
        return pv.read(filename)


def _log_composition(mesh: pv.DataSet) -> None:
    grain_id = mesh.cell_data['GrainID']
    phase_id = mesh.cell_data['phaseID']

    RveInfo.LOGGER.info(f'Substructure input mesh: {mesh.n_cells} cells, '
                        f'{len(np.unique(grain_id))} grains')
    for pid in np.unique(phase_id):
        RveInfo.LOGGER.info(f'  phaseID {int(pid)}: {int(np.count_nonzero(phase_id == pid))} cells')


def from_damask_grid(store_path: str) -> pv.UnstructuredGrid:
    """Read the DAMASK grid written by dragen/generation/spectral.py::write_grid.

    That file carries `material` (0-based, one entry per grain) and `phases` (DRAGen phase ids);
    the pipeline wants 1-based GrainIDs, hence the +1.
    """
    vti_file = os.path.join(store_path, 'grid.vti')
    material_file = os.path.join(store_path, 'material.yaml')

    mesh = pv.read(vti_file)
    RveInfo.LOGGER.info(f'Loaded: {vti_file}')

    for array in ('material', 'phases'):
        if array not in mesh.cell_data:
            raise KeyError(f"Array '{array}' not found in grid.vti cell_data.")

    material_id = np.asarray(mesh.cell_data['material']).astype(int)
    phase_id = np.asarray(mesh.cell_data['phases']).astype(int)

    # Consistency check against material.yaml: the export stage rewrites that file, so a mismatch
    # here would silently shift every orientation.
    with open(material_file, 'r', encoding='utf-8') as f:
        matdata = yaml.safe_load(f)

    n_material_yaml = len(matdata.get('material', []))
    unique_material_ids = np.unique(material_id)

    if unique_material_ids.min() != 0:
        raise ValueError('Material IDs in grid.vti do not start from 0. '
                         'Check the material-to-GrainID mapping.')

    if unique_material_ids.max() >= n_material_yaml:
        raise ValueError('grid.vti contains a material ID larger than the number of '
                         'material.yaml entries.')

    if len(unique_material_ids) != n_material_yaml:
        RveInfo.LOGGER.warning(
            f'grid.vti uses {len(unique_material_ids)} material IDs but material.yaml defines '
            f'{n_material_yaml} materials.')

    mesh = mesh.cast_to_unstructured_grid()
    mesh.cell_data['GrainID'] = (material_id + 1).astype(np.int32)
    mesh.cell_data['phaseID'] = phase_id.astype(np.int32)

    _log_composition(mesh)
    return mesh


def from_abaqus_mesh(store_path: str) -> pv.UnstructuredGrid:
    """Read the smoothed hex mesh written by dragen/generation/Mesher3D.py::AbaqusMesher.run().

    Cell order in rve-part.vtk is the Abaqus element order, which export_abaqus relies on when it
    writes element sets (element id == cell index + 1).
    """
    input_vtk = os.path.join(store_path, 'rve-part.vtk')

    mesh = _read_mesh_safely(input_vtk)

    for array in ('GrainID', 'phaseID'):
        if array not in mesh.cell_data:
            raise KeyError(f"Array '{array}' not found in rve-part.vtk cell_data.")

    grain_id = np.asarray(mesh.cell_data['GrainID']).astype(np.int32)
    phase_id = np.asarray(mesh.cell_data['phaseID']).astype(np.int32)

    for name, values in (('GrainID', grain_id), ('phaseID', phase_id)):
        if values.size != mesh.n_cells:
            raise ValueError(f'{name} length {values.size} != number of cells {mesh.n_cells}')

    mesh.cell_data.clear()
    mesh.cell_data['GrainID'] = grain_id
    mesh.cell_data['phaseID'] = phase_id

    _log_composition(mesh)
    return mesh
