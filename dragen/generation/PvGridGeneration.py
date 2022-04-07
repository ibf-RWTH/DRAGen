import numpy as np
import pandas as pd
import pyvista as pv
import sys
import logging
import tetgen
from dragen.utilities.InputInfo import RveInfo


class MeshingHelper:
    def __init__(self, rve_shape: tuple = None, rve: pd.DataFrame = None, grains_df: pd.DataFrame = None):

        self.rve = rve
        self.grains_df = grains_df

        self.x_max = int(max(rve.x))
        self.x_min = int(min(rve.x))
        self.y_max = int(max(rve.y))
        self.y_min = int(min(rve.y))
        self.z_max = int(max(rve.z))
        self.z_min = int(min(rve.z))
        self.n_grains = int(max(rve.GrainID))

        self.n_pts_x = rve_shape[0]
        if isinstance(rve.box_size, pd.Series):
            self.bin_size = rve.box_size[0] / self.n_pts_x  # test
        else:
            self.bin_size = rve.box_size / self.n_pts_x  # test

        if RveInfo.box_size_y is not None:
            self.box_size_y = RveInfo.box_size_y
            self.n_pts_y = rve_shape[1]
        else:
            self.box_size_y = RveInfo.box_size
            self.n_pts_y = self.n_pts_x

        if RveInfo.box_size_z is not None:
            self.box_size_z = RveInfo.box_size_z
            self.n_pts_z = rve_shape[2]
        else:
            self.box_size_z = RveInfo.box_size
            self.n_pts_z = self.n_pts_x


    def gen_blocks(self) -> pv.UnstructuredGrid:

        """this function generates a structured grid
        in py-vista according to the rve"""
        xrng = np.linspace(0, RveInfo.box_size/1000, self.n_pts_x+1, endpoint=True)
        yrng = np.linspace(0, self.box_size_y/1000, self.n_pts_y+1, endpoint=True)
        zrng = np.linspace(0, self.box_size_z/1000, self.n_pts_z+1, endpoint=True)
        grid = pv.RectilinearGrid(xrng, yrng, zrng)
        grid = grid.cast_to_unstructured_grid()
        return grid

    def gen_grains(self, grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:

        """the grainIDs are written on the cell_array"""
        self.rve.sort_values(by=['z', 'y', 'x'], inplace=True) # This sorting is important! Keep it that way

        # Add the data values to the cell data
        grid.cell_data["GrainID"] = self.rve['GrainID'].to_numpy()
        grid.cell_data["phaseID"] = self.rve['phaseID'].to_numpy()
        print(self.rve['phaseID'])
        # Now plot the grid!
        if RveInfo.anim_flag:
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(grid, scalars='phaseID',
                             show_edges=True, interpolate_before_map=True)
            plotter.add_axes()
            plotter.show(interactive=True, auto_close=True, window_size=[800, 600],
                         screenshot=RveInfo.store_path + '/Figs/pyvista_Hex_Mesh_phases.png')
            plotter.close()

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(grid, scalars='GrainID',
                             show_edges=True, interpolate_before_map=True)
            plotter.add_axes()
            plotter.show(interactive=True, auto_close=True, window_size=[800, 600],
                         screenshot=RveInfo.store_path + '/Figs/pyvista_Hex_Mesh_grains.png')
            plotter.close()

        return grid

    def smoothen_mesh(self, grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:

        """information about grainboundary elements of hex-mesh
        is extracted here and stored in pv.Polydata and
        in a pd.Dataframe"""
        x_max = max(grid.points[:, 0])
        x_min = min(grid.points[:, 0])
        y_max = max(grid.points[:, 1])
        y_min = min(grid.points[:, 1])
        z_max = max(grid.points[:, 2])
        z_min = min(grid.points[:, 2])
        numberOfGrains = self.n_grains

        gid_list = list()
        pid_list = list()

        ######################################
        assert RveInfo.element_type in ['C3D8', 'HEX8', 'C3D10', 'C3D4']
        if RveInfo.element_type != 'C3D8' and RveInfo.element_type != 'HEX8':
            old_grid = grid.copy()
            grid_tet = pv.UnstructuredGrid()
            for i in range(1, numberOfGrains + 1):
                phase = self.rve.loc[self.rve['GrainID'] == i].phaseID.values[0]
                grain_grid_tet = old_grid.extract_cells(np.where(np.asarray(old_grid.cell_data.values())[0] == i))
                grain_surf_tet = grain_grid_tet.extract_surface(pass_pointid=True, pass_cellid=True)
                grain_surf_tet.triangulate(inplace=True)

                tet = tetgen.TetGen(grain_surf_tet)
                if RveInfo.element_type == 'C3D4':
                    tet.tetrahedralize(order=1, mindihedral=10, minratio=1.5, supsteiner_level=0, steinerleft=0)
                elif RveInfo.element_type == 'C3D10':
                    sys.exit('Element type Error! C3D10 currently not supported! Chose C3D4')
                    node, elem = tet.tetrahedralize(order=2, mindihedral=10, minratio=1.5, supsteiner_level=0, steinerleft=0)

                tet_grain_grid = tet.grid
                ncells = tet_grain_grid.n_cells

                if RveInfo.gui_flag:
                    RveInfo.progress_obj.emit(75+(100*(i+1)/self.n_grains/4))
                grainIDList = [i]
                grainID_array = grainIDList * ncells
                gid_list.extend(grainID_array)

                phaseIDList = [phase]
                phaseID_array = phaseIDList * ncells
                pid_list.extend(phaseID_array)
                if i == 1:
                    grid_tet = tet_grain_grid
                else:
                    grid_tet = tet_grain_grid.merge(grid_tet, merge_points=True)

            grid_tet.cell_data['GrainID'] = np.asarray(gid_list)
            grid_tet.cell_data['phaseID'] = np.asarray(pid_list)
            grid = grid_tet.copy()

        all_points_df = pd.DataFrame(grid.points, columns=['x', 'y', 'z'])
        all_points_df['ori_idx'] = all_points_df.index

        all_points_df_old = all_points_df.copy()
        all_points_df_old['x_min'] = False
        all_points_df_old['y_min'] = False
        all_points_df_old['z_min'] = False
        all_points_df_old['x_max'] = False
        all_points_df_old['y_max'] = False
        all_points_df_old['z_max'] = False
        all_points_df_old.loc[(all_points_df_old.x == x_min), 'x_min'] = True
        all_points_df_old.loc[(all_points_df_old.y == y_min), 'y_min'] = True
        all_points_df_old.loc[(all_points_df_old.z == z_min), 'z_min'] = True
        all_points_df_old.loc[(all_points_df_old.x == x_max), 'x_max'] = True
        all_points_df_old.loc[(all_points_df_old.y == y_max), 'y_max'] = True
        all_points_df_old.loc[(all_points_df_old.z == z_max), 'z_max'] = True

        old_grid = grid.copy()  # copy doesn't copy the dynamically assigned new property...
        for i in range(1, numberOfGrains + 1):
            grain_grid = old_grid.extract_cells(np.where(old_grid.cell_data['GrainID'] == i))
            grain_surf = grain_grid.extract_surface()
            grain_surf_df = pd.DataFrame(data=grain_surf.points, columns=['x', 'y', 'z'])
            merged_pts_df = grain_surf_df.join(all_points_df_old.set_index(['x', 'y', 'z']), on=['x', 'y', 'z'])
            grain_surf_smooth = grain_surf.smooth(n_iter=250)
            smooth_pts_df = pd.DataFrame(data=grain_surf_smooth.points, columns=['x', 'y', 'z'])
            all_points_df.loc[merged_pts_df['ori_idx'], ['x', 'y', 'z']] = smooth_pts_df.values
            grain_vol = grain_grid.volume
            self.grains_df.loc[self.grains_df['GrainID'] == i-1, 'meshed_conti_volume'] = grain_vol * 10 ** 9

        self.grains_df[['GrainID', 'meshed_conti_volume', 'phaseID']].\
            to_csv(RveInfo.store_path + '/Generation_Data/grain_data_output_conti.csv', index=False)

        all_points_df.loc[all_points_df_old['x_min'], 'x'] = x_min
        all_points_df.loc[all_points_df_old['y_min'], 'y'] = y_min
        all_points_df.loc[all_points_df_old['z_min'], 'z'] = z_min
        all_points_df.loc[all_points_df_old['x_max'], 'x'] = x_max
        all_points_df.loc[all_points_df_old['y_max'], 'y'] = y_max
        all_points_df.loc[all_points_df_old['z_max'], 'z'] = z_max

        grid.points = all_points_df[['x', 'y', 'z']].values

        return grid
