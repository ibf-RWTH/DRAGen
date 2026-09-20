"""Entry point of the substructure pipeline.

`Run().run_damask(store_path)` and `Run().run_abaqus(store_path)` are called from
dragen/main3D.py once the respective solver export has written its mesh. Both drive the same
stages; only the first (read the mesh) and the last (write the solver deck) differ:

    adapters      grid.vti / rve-part.vtk  ->  mesh with GrainID + phaseID
    packets       PacketID, SubstructureFlag        (+ merge packets that came out too small)
    blocks        BlockID                           (+ merge blocks that came out too small)
    orientation   phi1/PHI/phi2 per block, KS variants or a measured OR
    plots         screenshots of every ID level
    export_*      material.yaml + grid.vti  /  substructure element sets, materials, sections

Every intermediate mesh is dumped to <store_path>/Postprocessing/Substructure/ for inspection.
"""
import os

import pyvista as pv

from dragen.substructure import adapters, blocks, export_abaqus, export_damask, orientation, packets, plots
from dragen.substructure.config import SubsConfig
from dragen.utilities.InputInfo import RveInfo


class Run:

    def run_damask(self, store_path: str) -> pv.DataSet:
        cfg = SubsConfig.from_rve_info(store_path, solver='damask')
        mesh = self._generate(cfg, adapters.from_damask_grid(store_path))

        export_damask.write(cfg, mesh)

        RveInfo.LOGGER.info('DAMASK substructure generation finished.')
        return mesh

    def run_abaqus(self, store_path: str) -> pv.DataSet:
        cfg = SubsConfig.from_rve_info(store_path, solver='abaqus')
        mesh = self._generate(cfg, adapters.from_abaqus_mesh(store_path))

        export_abaqus.write(cfg, mesh)

        RveInfo.LOGGER.info('Abaqus substructure generation finished.')
        return mesh

    # ------------------------------------------------------------------ #

    def _generate(self, cfg: SubsConfig, mesh: pv.DataSet) -> pv.DataSet:
        """Solver-independent part: packets -> blocks -> orientations -> plots."""
        os.makedirs(cfg.subs_dir, exist_ok=True)
        cfg.dump()

        RveInfo.LOGGER.info('-' * 78)
        RveInfo.LOGGER.info(f'substructure generation begins ({cfg.solver}, '
                            f'{cfg.block_generation_mode} blocks, {cfg.orientation_mode} '
                            f'orientations)')
        RveInfo.LOGGER.info('-' * 78)

        self._dump(cfg, mesh, 'rve_from_grid.vtk')

        mesh = packets.gen_packets(cfg, mesh)
        self._dump(cfg, mesh, 'rve_with_packets.vtk')

        mesh = packets.merge_small_packets(cfg, mesh)
        self._dump(cfg, mesh, 'rve_with_packets_fixed.vtk')

        mesh = blocks.gen_blocks(cfg, mesh)
        self._dump(cfg, mesh, 'rve_with_blocks.vtk')

        mesh = blocks.merge_small_blocks(cfg, mesh)
        self._dump(cfg, mesh, 'rve_with_blocks_fixed.vtk')

        mesh = orientation.assign_orientations(cfg, mesh)
        self._dump(cfg, mesh, 'rve_with_orientations.vtk')

        plots.save_substructure_plots(cfg, mesh)

        return mesh

    @staticmethod
    def _dump(cfg: SubsConfig, mesh: pv.DataSet, filename: str) -> None:
        mesh.save(cfg.path(filename))
        RveInfo.LOGGER.info(f'Saved: {cfg.path(filename)}')
