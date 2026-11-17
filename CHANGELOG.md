# Changelog

## 0.2.0 (branch `mst-integration`, based on `rework`)

### Added
- `pyproject.toml`: package `dragen` (with `InputGenerator`), Python >= 3.12. Lean core
  (numpy, scipy, pandas, pyyaml, gmsh, meshio); extras `ml`, `damask`, `extra`, `gui`, `test`.
- `dragen.api`: headless library interface `generate_rve(RVESpec) -> RVEResult` over the reworked
  generator and mesher. Metres, axes x = RD / y = TD / z = ND, grain IDs from 1, orientations as
  unit quaternions (w, x, y, z) crystal -> sample, optional tetrahedral mesh with periodic node
  pairs, metadata (version, git commit, seed, SHA-256 of spec and inputs, run times, warnings).
  Optional per-phase orientation file with measured Bunge angles.
- `dragen.export`: `export_damask` (grid.vti + material.yaml, reads back with
  `damask.GeomGrid.load`), `export_vtu`, `export_gmsh` (MSH 2.2, physical group = grain),
  `export_exodus` (one element block per grain, block ID = grain ID).
- `tests/api`: tests of interface and exports (about 30 s).
- E1: `export_abaqus` (part with nodes, C3D8/C3D4/C3D10 elements, element sets per grain, phase
  and band, node sets of the box faces, one solid section per grain, periodic equations to
  reference nodes RP_X/RP_Y/RP_Z with one equation per slave node and DOF, `graindata.inp` in
  the legacy UMAT format; materials and steps from the user). `export_grains` (grain table of all
  phases, replaces the legacy `sampled_grains.csv` that held only the last phase).
  `dragen.export.mesh`: voxel mesh (one hexahedron per voxel), box-face node and side sets,
  periodic node classes. `BandSpec`, `InclusionSpec` and `RVEResult.band_ids`, `radii_m`,
  `alpha_deg` in the API.
- README section "Use as a library".

### Changed
- Missing optional packages raise an ImportError that names the extra.
- E1: VTU, MSH and Exodus also write the voxel mesh (default when the RVE has no tetrahedral
  mesh, `mesh='voxel'|'tet'` chooses). Exodus is written with scipy's netCDF writer (no netCDF4
  needed; 100^3 voxels with 2000 grains in about 1 s instead of 6 min with netCDF4) and has node
  and side sets left/right/bottom/top/back/front for MOOSE.
- The GAN source loads the GAN code with matplotlib on Agg and routes its prints to `logging`.
- Legacy tests that need `pyvista` are skipped with a reason when the extra `gui` is missing.

### Fixed
- numpy 2: the temporary MSH file for gmsh contained `np.float64(...)` in the bounding boxes, so
  every tetrahedral meshing failed.
- pandas 3: the legacy GAN reader (`HelperFunctions.read_input_gan`) used positional `Series[int]`
  access and chained assignment (which no longer writes through).
