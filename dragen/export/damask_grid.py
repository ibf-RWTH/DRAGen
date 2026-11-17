"""DAMASK grid solver input: geometry (.vti) and the material list of material.yaml."""
from pathlib import Path
from typing import Dict

import numpy as np

from dragen._extras import require

HOMOGENIZATION = 'SX'


def export_damask(result, path: Path) -> Dict[str, Path]:
    """Write <path>/grid.vti and <path>/material.yaml; returns both paths.

    grid.vti: material index g - 1 for grain g (DAMASK counts from 0), size in metres, origin 0.
    2D: one layer of cells, as thick as a pixel is wide.
    material.yaml: one material per grain in grain order, with its orientation (the quaternion of
    the RVEResult, which DAMASK reads as the same rotation, see dragen.api), its phase (Ferrite,
    Martensite, ...) and homogenization 'SX'. The phase and homogenization entries are left
    empty (null): their constitutive parameters come from the caller. Reads back with
    damask.GeomGrid.load and damask.ConfigMaterial.load.
    """
    damask = require('damask', 'damask')
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    material = np.asarray(result.grain_ids, dtype=np.int64) - 1
    size = np.asarray(result.box_size_m, dtype=float)
    if material.ndim == 2:                       # 2D: one layer of cubic cells
        material = material[:, :, None]
        size = np.append(size, result.voxel_size_m[0])
    grid = damask.GeomGrid(material=material, size=size,
                           comments=[f'DRAGen {result.metadata.get("dragen_version")}, '
                                     f'seed {result.metadata.get("seed")}, '
                                     f'spec {result.metadata.get("spec_sha256")}'])
    grid_file = path / 'grid.vti'
    grid.save(grid_file)

    phases = [result.phase_names[int(p)].capitalize() for p in result.phase_ids]
    material = damask.ConfigMaterial().material_add(
        O=damask.Rotation.from_quaternion(result.orientations), phase=phases,
        homogenization=HOMOGENIZATION)
    material_file = path / 'material.yaml'
    material.save(material_file)
    return {'grid': grid_file, 'material': material_file}
