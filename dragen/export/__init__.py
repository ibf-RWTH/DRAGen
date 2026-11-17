"""Exports of an RVEResult (dragen.api), all with the signature export_xxx(result, path, ...).

export_damask   DAMASK grid: <dir>/grid.vti + <dir>/material.yaml          (extra 'damask')
export_abaqus   Abaqus model data <path>.inp (mesh, sets, sections, periodic equations) and
                graindata.inp for the crystal plasticity UMAT
export_vtu      mesh as VTK unstructured grid, <path>.vtu
export_gmsh     mesh as Gmsh MSH 2.2, <path>.msh; physical group = grain ID
export_exodus   mesh as Exodus II, <path>.e (MOOSE): one element block per grain, node and side
                sets of the box faces
export_grains   grain table <path>.csv (all phases, bands, inclusions)

The mesh exports take the tetrahedral mesh if the RVE has one, else one hexahedron per voxel;
mesh='voxel' or 'tet' chooses explicitly (dragen.export.mesh). Lengths in metres, as in the
RVEResult, except where a function takes a length unit.
"""
from dragen.export.abaqus import export_abaqus
from dragen.export.damask_grid import export_damask
from dragen.export.files import export_exodus, export_gmsh, export_vtu
from dragen.export.grains import export_grains, grain_table
from dragen.export.mesh import SolverMesh, solver_mesh, voxel_mesh

__all__ = ['export_damask', 'export_abaqus', 'export_vtu', 'export_gmsh', 'export_exodus',
           'export_grains', 'grain_table', 'solver_mesh', 'voxel_mesh', 'SolverMesh']
