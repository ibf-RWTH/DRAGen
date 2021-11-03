import numpy as np
import pandas as pd
import pyvista as pv
import tetgen
import datetime
import os
import logging
class Mesher:

    def __init__(self, box_size_x: int, box_size_y: int = None, box_size_z: int = None, rve_shape: tuple = None,
                 rve: pd.DataFrame = None, grains_df: pd.DataFrame = None, store_path: str = None,
                 phase_two_isotropic=True, animation=True, infobox_obj=None,
                 progress_obj=None, gui=True, element_type='C3D4'):
        self.box_size_x = box_size_x
        self.box_size_y = box_size_y
        self.box_size_z = box_size_z
        self.rve = rve
        self.grains_df = grains_df
        self.store_path = store_path
        self.phase_two_isotropic = phase_two_isotropic
        self.animation = animation
        self.infobox_obj = infobox_obj
        self.progress_obj = progress_obj
        self.tex_phi1 = grains_df['phi1'].tolist()
        self.tex_PHI = grains_df['PHI'].tolist()
        self.tex_phi2 = grains_df['phi2'].tolist()
        self.x_max = int(max(rve.x))
        self.x_min = int(min(rve.x))
        self.y_max = int(max(rve.y))
        self.y_min = int(min(rve.y))
        self.z_max = int(max(rve.z))
        self.z_min = int(min(rve.z))
        self.n_grains = int(max(rve.GrainID))

        self.n_pts_x = rve_shape[0]+1
        if isinstance(rve.box_size,pd.Series):
            self.bin_size = rve.box_size[0] / (self.n_pts_x)  ## test
        else:
            self.bin_size = rve.box_size / (self.n_pts_x) ## test

        if self.box_size_y is not None:
            self.n_pts_y = rve_shape[1]
        else:
            self.box_size_y = self.box_size_x
            self.n_pts_y =self.n_pts_x

        if self.box_size_z is not None:
            self.n_pts_z = rve_shape[2]
        else:
            self.box_size_z = self.box_size_x
            self.n_pts_z = self.n_pts_x

        self.logger = logging.getLogger("RVE-Gen")
        self.gui = gui
        self.element_type = element_type
        self.roughness = True

    def gen_blocks(self) -> pv.UnstructuredGrid:

        """this function generates a structured grid
        in py-vista according to the rve"""

        xrng = np.linspace(0, self.box_size_x/1000, self.n_pts_x+1, endpoint=True)
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
        # Now plot the grid!
        if self.animation:
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(grid, scalars='phaseID',
                             show_edges=True, interpolate_before_map=True)
            plotter.add_axes()
            plotter.show(interactive=True, auto_close=True, window_size=[800, 600],
                         screenshot=self.store_path + '/Figs/pyvista_Hex_Mesh_phases.png')
            plotter.close()

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(grid, scalars='GrainID',
                             show_edges=True, interpolate_before_map=True)
            plotter.add_axes()
            plotter.show(interactive=True, auto_close=True, window_size=[800, 600],
                         screenshot=self.store_path + '/Figs/pyvista_Hex_Mesh_grains.png')
            plotter.close()

        return grid

    def smoothen_mesh(self, grid: pv.UnstructuredGrid, element_type: str = 'C3D8') -> pv.UnstructuredGrid:

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
        if self.element_type != 'C3D8':
            old_grid = grid.copy()
            grid_tet = pv.UnstructuredGrid()
            for i in range(1, numberOfGrains + 1):
                phase = self.rve.loc[self.rve['GrainID'] == i].phaseID.values[0]
                grain_grid_tet = old_grid.extract_cells(np.where(np.asarray(old_grid.cell_data.values())[0] == i))
                grain_surf_tet = grain_grid_tet.extract_surface(pass_pointid=True, pass_cellid=True)
                grain_surf_tet.triangulate(inplace=True)

                tet = tetgen.TetGen(grain_surf_tet)
                if self.element_type == 'C3D4':
                    tet.tetrahedralize(order=1, mindihedral=10, minratio=1.5, supsteiner_level=0, steinerleft=0)
                elif self.element_type == 'C3D10':
                    tet.tetrahedralize(order=2, mindihedral=10, minratio=1.5, supsteiner_level=0, steinerleft=0)
                tet_grain_grid = tet.grid
                ncells = tet_grain_grid.n_cells

                if self.gui:
                    self.progress_obj.emit(75+(100*(i+1)/self.n_grains/4))
                grainIDList = [i]
                grainID_array = grainIDList * ncells
                gid_list.extend(grainID_array)

                phaseIDList = [phase]
                phaseID_array = phaseIDList * ncells
                pid_list.extend(phaseID_array)
                if i == 0:
                    grid_tet = tet_grain_grid
                else:
                    grid_tet = tet_grain_grid.merge(grid_tet, merge_points=True)


            grid_tet.cell_data['GrainID'] = np.asarray(gid_list)
            grid_tet.cell_data['PhaseID'] = np.asarray(pid_list)
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

        old_grid = grid.copy() #copy doenst copy the dynamically assigned new property...
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

        self.grains_df[['GrainID','meshed_conti_volume', 'phaseID']].\
            to_csv(self.store_path + '/Generation_Data/grain_data_output_conti.csv', index=False)

        all_points_df.loc[all_points_df_old['x_min'], 'x'] = x_min
        all_points_df.loc[all_points_df_old['y_min'], 'y'] = y_min
        all_points_df.loc[all_points_df_old['z_min'], 'z'] = z_min
        all_points_df.loc[all_points_df_old['x_max'], 'x'] = x_max
        all_points_df.loc[all_points_df_old['y_max'], 'y'] = y_max
        all_points_df.loc[all_points_df_old['z_max'], 'z'] = z_max

        grid.points = all_points_df[['x', 'y', 'z']].values


        return grid

    def make_assembly(self) -> None:

        """simple function to write the assembly definition in the input file"""

        f = open(self.store_path + '/RVE_smooth.inp', 'a')
        f.write('*End Part\n')
        f.write('**\n')
        f.write('** ASSEMBLY\n')
        f.write('**\n')
        f.write('*Assembly, name=Assembly\n')
        f.write('**\n')
        f.write('*Instance, name=Part-1-1, part=Part-1\n')
        f.write('*End Instance\n')
        f.write('**\n')
        f.write('*Include, Input=Nsets.inp\n')
        f.write('*Include, input=LeftToRight.inp\n')
        f.write('*Include, input=BottomToTop.inp\n')
        f.write('*Include, input=FrontToRear.inp\n')
        f.write('*Include, input=Edges.inp\n')
        f.write('*Include, input=Corners.inp\n')
        f.write('*Include, input=VerticeSets.inp\n')
        f.write('*End Assembly\n')
        f.close()

    def pbc(self, rve: pv.UnstructuredGrid, grid_hull_df: pd.DataFrame) -> None:

        """function to define the periodic boundary conditions
        if errors appear or equations are wrong check ppt presentation from ICAMS
        included in the docs folder called PBC_docs"""

        min_x = min(rve.points[:, 0])
        min_y = min(rve.points[:, 1])
        min_z = min(rve.points[:, 2])
        max_x = max(rve.points[:, 0])
        max_y = max(rve.points[:, 1])
        max_z = max(rve.points[:, 2])

        numberofgrains = self.n_grains
        ########## write Equation - sets ##########
        grid_hull_df.sort_values(by=['x', 'y', 'z'], inplace=True)
        grid_hull_df.index.rename('pointNumber', inplace=True)
        grid_hull_df = grid_hull_df.reset_index()
        grid_hull_df.index.rename('Eqn-Set', inplace=True)
        grid_hull_df = grid_hull_df.reset_index()

        ########## Define Corner Sets ###########
        corner_df = grid_hull_df.loc[((grid_hull_df['x'] == max_x) | (grid_hull_df['x'] == min_x)) &
                                     ((grid_hull_df['y'] == max_y) | (grid_hull_df['y'] == min_y)) &
                                     ((grid_hull_df['z'] == max_z) | (grid_hull_df['z'] == min_z))]

        V1_df = corner_df.loc[(corner_df['x'] == min_x) & (corner_df['y'] == min_y) & (corner_df['z'] == max_z)]
        V1 = V1_df['pointNumber'].values[0]
        V1Eqn = V1_df['Eqn-Set'].values[0]

        V2_df = corner_df.loc[(corner_df['x'] == max_x) & (corner_df['y'] == min_y) & (corner_df['z'] == max_z)]
        V2 = V2_df['pointNumber'].values[0]
        V2Eqn = V2_df['Eqn-Set'].values[0]

        V3_df = corner_df.loc[(corner_df['x'] == max_x) & (corner_df['y'] == max_y) & (corner_df['z'] == max_z)]
        V3 = V3_df['pointNumber'].values[0]
        V3Eqn = V3_df['Eqn-Set'].values[0]

        V4_df = corner_df.loc[(corner_df['x'] == min_x) & (corner_df['y'] == max_y) & (corner_df['z'] == max_z)]
        V4 = V4_df['pointNumber'].values[0]
        V4Eqn = V4_df['Eqn-Set'].values[0]

        H1_df = corner_df.loc[(corner_df['x'] == min_x) & (corner_df['y'] == min_y) & (corner_df['z'] == min_z)]
        H1 = H1_df['pointNumber'].values[0]
        H1Eqn = H1_df['Eqn-Set'].values[0]

        H2_df = corner_df.loc[(corner_df['x'] == max_x) & (corner_df['y'] == min_y) & (corner_df['z'] == min_z)]
        H2 = H2_df['pointNumber'].values[0]
        H2Eqn = H2_df['Eqn-Set'].values[0]

        H3_df = corner_df.loc[(corner_df['x'] == max_x) & (corner_df['y'] == max_y) & (corner_df['z'] == min_z)]
        H3 = H3_df['pointNumber'].values[0]
        H3Eqn = H3_df['Eqn-Set'].values[0]

        H4_df = corner_df.loc[(corner_df['x'] == min_x) & (corner_df['y'] == max_y) & (corner_df['z'] == min_z)]
        H4 = H4_df['pointNumber'].values[0]
        H4Eqn = H4_df['Eqn-Set'].values[0]

        ############ Define Edge Sets ###############
        edges_df = grid_hull_df.loc[(((grid_hull_df['x'] == max_x) | (grid_hull_df['x'] == min_x)) &
                                     ((grid_hull_df['y'] == max_y) | (grid_hull_df['y'] == min_y)) &
                                     ((grid_hull_df['z'] != max_z) & (grid_hull_df['z'] != min_z))) |

                                    (((grid_hull_df['x'] == max_x) | (grid_hull_df['x'] == min_x)) &
                                     ((grid_hull_df['y'] != max_y) & (grid_hull_df['y'] != min_y)) &
                                     ((grid_hull_df['z'] == max_z) | (grid_hull_df['z'] == min_z))) |

                                    (((grid_hull_df['x'] != max_x) & (grid_hull_df['x'] != min_x)) &
                                     ((grid_hull_df['y'] == max_y) | (grid_hull_df['y'] == min_y)) &
                                     ((grid_hull_df['z'] == max_z) | (grid_hull_df['z'] == min_z)))]
        # edges_df.sort_values(by=['x', 'y', 'z'], inplace=True)

        # Top front Edge
        E_T1 = edges_df.loc[(edges_df['y'] == max_y) & (edges_df['z'] == max_z)]['Eqn-Set'].to_list()

        # Top right Edge
        E_T2 = edges_df.loc[(edges_df['x'] == max_x) & (edges_df['y'] == max_y)]['Eqn-Set'].to_list()

        # Top back Edge
        E_T3 = edges_df.loc[(edges_df['y'] == max_y) & (edges_df['z'] == min_z)]['Eqn-Set'].to_list()

        # Top left Edge
        E_T4 = edges_df.loc[(edges_df['x'] == min_x) & (edges_df['y'] == max_y)]['Eqn-Set'].to_list()

        # bottm front edge
        E_B1 = edges_df.loc[(edges_df['y'] == min_y) & (edges_df['z'] == max_z)]['Eqn-Set'].to_list()

        # bottm right edge
        E_B2 = edges_df.loc[(edges_df['x'] == max_x) & (edges_df['y'] == min_y)]['Eqn-Set'].to_list()

        # bottm back edge
        E_B3 = edges_df.loc[(edges_df['y'] == min_y) & (edges_df['z'] == min_z)]['Eqn-Set'].to_list()

        # bottm left edge
        E_B4 = edges_df.loc[(edges_df['x'] == min_x) & (edges_df['y'] == min_y)]['Eqn-Set'].to_list()

        # left front edge
        E_M1 = edges_df.loc[(edges_df['x'] == min_x) & (edges_df['z'] == max_z)]['Eqn-Set'].to_list()

        # right front edge
        E_M2 = edges_df.loc[(edges_df['x'] == max_x) & (edges_df['z'] == max_z)]['Eqn-Set'].to_list()

        # left rear edge
        E_M4 = edges_df.loc[(edges_df['x'] == min_x) & (edges_df['z'] == min_z)]['Eqn-Set'].to_list()

        # right rear edge
        E_M3 = edges_df.loc[(edges_df['x'] == max_x) & (edges_df['z'] == min_z)]['Eqn-Set'].to_list()

        ######### Define Surface Sets #############
        faces_df = grid_hull_df.loc[(((grid_hull_df['x'] == max_x) | (grid_hull_df['x'] == min_x)) &
                                     ((grid_hull_df['y'] != max_y) & (grid_hull_df['y'] != min_y)) &
                                     ((grid_hull_df['z'] != max_z) & (grid_hull_df['z'] != min_z))) |

                                    (((grid_hull_df['x'] != max_x) & (grid_hull_df['x'] != min_x)) &
                                     ((grid_hull_df['y'] != max_y) & (grid_hull_df['y'] != min_y)) &
                                     ((grid_hull_df['z'] == max_z) | (grid_hull_df['z'] == min_z))) |

                                    (((grid_hull_df['x'] != max_x) & (grid_hull_df['x'] != min_x)) &
                                     ((grid_hull_df['y'] == max_y) | (grid_hull_df['y'] == min_y)) &
                                     ((grid_hull_df['z'] != max_z) & (grid_hull_df['z'] != min_z)))]

        # left set
        LeftSet = faces_df.loc[faces_df['x'] == min_x]['Eqn-Set'].to_list()
        # right set
        RightSet = faces_df.loc[faces_df['x'] == max_x]['Eqn-Set'].to_list()
        # bottom set
        BottomSet = faces_df.loc[faces_df['y'] == min_y]['Eqn-Set'].to_list()
        # top set
        TopSet = faces_df.loc[faces_df['y'] == max_y]['Eqn-Set'].to_list()
        # front set
        RearSet = faces_df.loc[faces_df['z'] == min_z]['Eqn-Set'].to_list()
        # rear set
        FrontSet = faces_df.loc[faces_df['z'] == max_z]['Eqn-Set'].to_list()

        OutPutFile = open(self.store_path + '/Nsets.inp', 'w')
        for i in grid_hull_df.index:
            OutPutFile.write('*Nset, nset=Eqn-Set-{}, instance=PART-1-1\n'.format(i + 1))
            OutPutFile.write(' {},\n'.format(int(grid_hull_df.loc[i]['pointNumber'] + 1)))
        OutPutFile.close()

        ############### Define Equations ###################################
        OutPutFile = open(self.store_path + '/LeftToRight.inp', 'w')

        OutPutFile.write('**** X-DIR \n')
        for i in range(len(LeftSet)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(RightSet[i] + 1) + ',1, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(LeftSet[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')

        OutPutFile.write('**** \n')
        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(LeftSet)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(RightSet[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(LeftSet[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** \n')
        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(LeftSet)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(RightSet[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(LeftSet[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')
        OutPutFile.close()

        OutPutFile = open(self.store_path + '/BottomToTop.inp', 'w')

        OutPutFile.write('**** X-DIR \n')
        for i in range(len(BottomSet)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(BottomSet[i] + 1) + ',1,1 \n')
            OutPutFile.write('Eqn-Set-' + str(TopSet[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',1,1 \n')

        OutPutFile.write('**** \n')
        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(BottomSet)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(BottomSet[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(TopSet[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** \n')
        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(BottomSet)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(BottomSet[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(TopSet[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',3, 1 \n')
        OutPutFile.close()

        OutPutFile = open(self.store_path + '/FrontToRear.inp', 'w')

        OutPutFile.write('**** X-DIR \n')
        for i in range(len(RearSet)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(RearSet[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(FrontSet[i] + 1) + ',1,1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',1,1 \n')

        OutPutFile.write('**** \n')
        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(RearSet)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(RearSet[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(FrontSet[i] + 1) + ',2,1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',2,1 \n')

        OutPutFile.write('**** \n')
        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(RearSet)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(RearSet[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(FrontSet[i] + 1) + ',3,1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',3,1 \n')
        OutPutFile.close()

        OutPutFile = open(self.store_path + '/Edges.inp', 'w')

        # Edges in x-y Plane
        # right top edge to left top edge
        OutPutFile.write('**** X-DIR \n')
        for i in range(len(E_T2)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T2[i] + 1) + ',1, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T4[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')

        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(E_T2)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T2[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T4[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(E_T2)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T2[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T4[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # right bottom edge to left bottom edge
        OutPutFile.write('**** X-DIR \n')
        for i in range(len(E_B2)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B2[i] + 1) + ',1, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B4[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')

        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(E_B2)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B2[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B4[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(E_B2)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B2[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B4[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # left top edge to left bottom edge
        OutPutFile.write('**** X-DIR \n')
        for i in range(len(E_T4)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T4[i] + 1) + ',1, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B4[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')

        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(E_T4)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T4[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B4[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(E_T4)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T4[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B4[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # Edges in y-z Plane
        # top back edge to top front edge
        OutPutFile.write('**** X-DIR \n')
        for i in range(len(E_T3)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T3[i] + 1) + ',1, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T1[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')

        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(E_T3)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T3[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T1[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(E_T3)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T3[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T1[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # Botom back edge to bottom front edge
        OutPutFile.write('**** X-DIR \n')
        for i in range(len(E_B3)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B3[i] + 1) + ',1, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B1[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')

        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(E_B3)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B3[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B1[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(E_B3)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B3[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B1[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # top front edge to bottom front edge
        OutPutFile.write('**** X-DIR \n')
        for i in range(len(E_T1)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T1[i] + 1) + ',1, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B1[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')

        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(E_T1)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T1[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B1[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(E_T1)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T1[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B1[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # Edges in x-z Plane
        # rear right edge to rear left edge
        OutPutFile.write('**** X-DIR \n')
        for i in range(len(E_M3)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M3[i] + 1) + ',1, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M4[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')

        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(E_M3)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M3[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M4[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(E_M3)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M3[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M4[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # front right edge to front left edge
        OutPutFile.write('**** X-DIR \n')
        for i in range(len(E_M2)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M2[i] + 1) + ',1, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M1[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')

        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(E_M2)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M2[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M1[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(E_M2)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M2[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M1[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # top front edge to bottom front edge
        OutPutFile.write('**** X-DIR \n')
        for i in range(len(E_M4)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M4[i] + 1) + ',1, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M1[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')

        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(E_M4)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M4[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M1[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(E_M4)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M4[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M1[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')
        OutPutFile.close()

        OutPutFile = open(self.store_path + '/Corners.inp', 'w')

        # V3 zu V4
        OutPutFile.write('**** X-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(V3Eqn + 1) + ',1, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',1,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',1,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')
        OutPutFile.write('**** y-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(V3Eqn + 1) + ',2, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',2,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',2,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')
        OutPutFile.write('**** z-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(V3Eqn + 1) + ',3, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',3,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',3,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # H4 zu V4
        OutPutFile.write('**** X-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(H4Eqn + 1) + ',1, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',1,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',1,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')
        OutPutFile.write('**** y-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(H4Eqn + 1) + ',2, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',2,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',2,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')
        OutPutFile.write('**** z-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(H4Eqn + 1) + ',3, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',3,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',3,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # H3 zu V3
        OutPutFile.write('**** X-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(H3Eqn + 1) + ',1, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V3Eqn + 1) + ',1,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',1,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')
        OutPutFile.write('**** y-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(H3Eqn + 1) + ',2, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V3Eqn + 1) + ',2,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',2,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')
        OutPutFile.write('**** z-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(H3Eqn + 1) + ',3, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V3Eqn + 1) + ',3,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',3,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # H2 zu V2
        OutPutFile.write('**** X-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(H2Eqn + 1) + ',1, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',1,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',1,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')
        OutPutFile.write('**** y-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(H2Eqn + 1) + ',2, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',2,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',2,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')
        OutPutFile.write('**** z-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(H2Eqn + 1) + ',3, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',3,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',3,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')
        OutPutFile.close()

        OutPutFile = open(self.store_path + '/VerticeSets.inp', 'w')
        OutPutFile.write('*Nset, nset=V1, instance=PART-1-1\n')
        OutPutFile.write(' {},\n'.format(V1 + 1))
        OutPutFile.write('*Nset, nset=V2, instance=PART-1-1\n')
        OutPutFile.write(' {},\n'.format(V2 + 1))
        OutPutFile.write('*Nset, nset=V3, instance=PART-1-1\n')
        OutPutFile.write(' {},\n'.format(V3 + 1))
        OutPutFile.write('*Nset, nset=V4, instance=PART-1-1\n')
        OutPutFile.write(' {},\n'.format(V4 + 1))
        OutPutFile.write('*Nset, nset=H1, instance=PART-1-1\n')
        OutPutFile.write(' {},\n'.format(H1 + 1))
        OutPutFile.write('*Nset, nset=H2, instance=PART-1-1\n')
        OutPutFile.write(' {},\n'.format(H2 + 1))
        OutPutFile.write('*Nset, nset=H3, instance=PART-1-1\n')
        OutPutFile.write(' {},\n'.format(H3 + 1))
        OutPutFile.write('*Nset, nset=H4, instance=PART-1-1\n')
        OutPutFile.write(' {},\n'.format(H4 + 1))
        OutPutFile.close()

    def write_material_def(self) -> None:

        """simple function to write material definition in Input file
        needs to be adjusted for multiple phases"""
        phase1_idx = 0
        phase2_idx = 0
        numberofgrains = self.n_grains

        phase = [self.rve.loc[self.rve['GrainID'] == i].phaseID.values[0] for i in range(1, numberofgrains+1)]
        f = open(self.store_path + '/RVE_smooth.inp', 'a')

        f.write('**\n')
        f.write('** MATERIALS\n')
        f.write('**\n')
        for i in range(numberofgrains):
            ngrain = i+1
            if not self.phase_two_isotropic:
                if phase[i] == 1:
                    phase1_idx += 1
                    f.write('*Material, name=Ferrite_{}\n'.format(phase1_idx))
                    f.write('*Depvar\n')
                    f.write('    176,\n')
                    f.write('*User Material, constants=2\n')
                    f.write('{}.,3.\n'.format(ngrain))
                elif phase[i] == 2:
                    phase2_idx += 1
                    f.write('*Material, name=Martensite_{}\n'.format(phase2_idx))
                    f.write('*Depvar\n')
                    f.write('    176,\n')
                    f.write('*User Material, constants=2\n')
                    f.write('{}.,4.\n'.format(ngrain))
            else:
                if phase[i] == 1:
                    phase1_idx += 1
                    f.write('*Material, name=Ferrite_{}\n'.format(phase1_idx))
                    f.write('*Depvar\n')
                    f.write('    176,\n')
                    f.write('*User Material, constants=2\n')
                    f.write('{}.,3.\n'.format(phase1_idx))

        if self.phase_two_isotropic:
            f.write('**\n')
            f.write('*Material, name=Martensite\n')
            f.write('*Elastic\n')
            f.write('0.21, 0.3\n')
        f.close()

    def write_step_def(self) -> None:

        """simple function to write step definition
        variables should be introduced to give the user an option
        to modify amplidtude, and other parameters"""

        f = open(self.store_path + '/RVE_smooth.inp', 'a')
        f.write('**\n')
        f.write('*Amplitude, name=Amp-1\n')
        f.write('             0.,              0.,           10.,        10.,\n')
        f.write('**\n')
        f.write('**BOUNDARY CONDITIONS\n')
        f.write('**\n')
        f.write('** Name: H1 Type: Displacement/Rotation\n')
        f.write('*Boundary\n')
        f.write('H1, 1\n')
        f.write('H1, 2\n')
        f.write('** Name: V1 Type: Displacement/Rotation\n')
        f.write('*Boundary\n')
        f.write('V1, 1\n')
        f.write('V1, 2\n')
        f.write('V1, 3\n')
        f.write('** Name: V2 Type: Displacement/Rotation\n')
        f.write('*Boundary\n')
        f.write('V2, 2\n')
        f.write('V2, 3\n')
        f.write('**Name: V4 Type: Displacement/Rotatio\n')
        f.write('**\n')
        f.write('*Boundary\n')
        f.write('**V4, 1\n')
        f.write('**V4, 2\n')
        f.write('**V4, 3\n')
        f.write('** ----------------------------------------------------------------\n')
        f.write('**\n')
        f.write('** STEP: Step-1\n')
        f.write('**\n')
        f.write('*Step, name=Step-1, nlgeom=YES, inc=10000000, solver=ITERATIVE\n')
        f.write('*Static\n')
        f.write('0.001, 10., 1.05e-15, 0.25\n')
        f.write('**\n')
        f.write('** CONTROLS\n')
        f.write('**\n')
        f.write('*Controls, reset\n')
        f.write('*CONTROLS, PARAMETER=TIME INCREMENTATION\n')
        f.write('35, 50, 9, 50, 28, 5, 12, 45\n')
        f.write('**\n')
        f.write('*CONTROLS, PARAMETERS=LINE SEARCH\n')
        f.write('10\n')
        f.write('*SOLVER CONTROL\n')
        f.write('1e-5,200,\n')
        f.write('**\n')
        f.write('** BOUNDARY CONDITIONS\n')
        f.write('**\n')
        f.write('** Name: Load Type: Displacement/Rotation\n')
        f.write('*Boundary, amplitude=AMP-1\n')
        f.write('V4, 2, 2, 1\n')
        f.write('**\n')
        f.write('** OUTPUT REQUESTS\n')
        f.write('**\n')
        f.write('*Restart, write, frequency=0\n')
        f.write('**\n')
        f.write('** FIELD OUTPUT: F-Output-1\n')
        f.write('**\n')
        f.write('*Output, field, frequency=10\n')
        f.write('*Node Output\n')
        f.write('CF, COORD, RF, U\n')
        f.write('*Element Output, directions=YES\n')
        f.write('EVOL, LE, PE, PEEQ, PEMAG, S, SDV\n')
        f.write('*Contact Output\n')
        f.write('CDISP, CSTRESS\n')
        f.write('**\n')
        f.write('** HISTORY OUTPUT: H-Output-1\n')
        f.write('**\n')
        f.write('*Output, history, variable=PRESELECT\n')
        f.write('*End Step\n')
        f.close()

    def write_grain_data(self) -> None:
        f = open(self.store_path + '/graindata.inp', 'w+')
        f.write('!MMM Crystal Plasticity Input File\n')
        phase1_idx = 0
        numberofgrains = self.n_grains
        phase = [self.rve.loc[self.rve['GrainID'] == i].phaseID.values[0] for i in range(1, numberofgrains + 1)]
        grainsize = [np.cbrt(self.rve.loc[self.rve['GrainID'] == i].shape[0] *
                             self.bin_size**3*3/4/np.pi) for i in range(1, numberofgrains + 1)]

        for i in range(numberofgrains):
            ngrain = i+1
            if not self.phase_two_isotropic:
                """phi1 = int(np.random.rand() * 360)
                PHI = int(np.random.rand() * 360)
                phi2 = int(np.random.rand() * 360)"""
                phi1 = self.tex_phi1[i]
                PHI = self.tex_PHI[i]
                phi2 = self.tex_phi2[i]
                f.write('Grain: {}: {}: {}: {}: {}\n'.format(ngrain, phi1, PHI, phi2, grainsize[i]))
            else:
                if phase[i] == 1:
                    phase1_idx += 1
                    """phi1 = int(np.random.rand() * 360)
                    PHI = int(np.random.rand() * 360)
                    phi2 = int(np.random.rand() * 360)"""
                    phi1 = self.tex_phi1[i]
                    PHI = self.tex_PHI[i]
                    phi2 = self.tex_phi2[i]
                    f.write('Grain: {}: {}: {}: {}: {}\n'.format(phase1_idx, phi1, PHI, phi2, grainsize[i]))
        f.close()

    def plot_bot(self, rve_smooth_grid: pv.UnstructuredGrid, min_val: float, max_val: float, direction: int = 1,
                 storename: str = 'default', display=True) -> None:

        """first approach for a visualization helper
        should probably be moved to utilities"""

        # get cell centroids
        cells = rve_smooth_grid.cells.reshape(-1, 5)[:, 1:]
        cell_center = rve_smooth_grid.points[cells].mean(1)

        # extract cells below the 0 xy plane
        mask = np.where((cell_center[:, direction] >= min_val) & (cell_center[:, direction] <= max_val))
        cell_ind = np.asarray(mask)
        subgrid = rve_smooth_grid.extract_cells(cell_ind)

        # advanced plotting
        plotter = pv.Plotter()
        plotter.add_mesh(subgrid, 'lightgrey', lighting=True, show_edges=True, scalars='GrainID')
        plotter.remove_scalar_bar()
        plotter.add_bounding_box()
        if storename == 'default':
            plotter.show()
        elif storename != 'default' and display:
            plotter.show(screenshot=self.store_path + storename + '.png')
        else:
            plotter.show(screenshot=self.store_path + storename + '.png', auto_close=True)

    def mesh_and_build_abaqus_model(self) -> None:
        if self.gui:
            self.progress_obj.emit(0)
            self.infobox_obj.emit('starting mesher')
        GRID = self.gen_blocks()
        if self.gui:
            self.progress_obj.emit(25)
        GRID = self.gen_grains(GRID)
        smooth_mesh = self.smoothen_mesh(GRID)

        pbc_grid = smooth_mesh


        if self.roughness:
            # TODO: roghness einbauen
            #grid = self.apply_roughness(grid)
            pass

        f = open(self.store_path + '/RVE_smooth.inp', 'w+')
        f.write('*Heading\n')
        f.write('** Job name: Job-1 Model name: Job-1\n')
        f.write('** Generated by: DRAGen \n')
        f.write('** Date: {}\n'.format(datetime.datetime.now().strftime("%d.%m.%Y")))
        f.write('** Time: {}\n'.format(datetime.datetime.now().strftime("%H:%M:%S")))
        f.write('*Preprint, echo=NO, model=NO, history=NO, contact=NO\n')
        f.write('**\n')
        f.write('** PARTS\n')
        f.write('**\n')
        f.close()

        pv.save_meshio(self.store_path + '/rve-part.inp', smooth_mesh)
        f = open(self.store_path + '/rve-part.inp', 'r')
        lines = f.readlines()
        f.close()
        lines = [line.lower() for line in lines]
        startingLine = lines.index('*node\n')
        f = open(self.store_path + '/RVE_smooth.inp', 'a')
        f.write('*Part, name=PART-1\n')
        for line in lines[startingLine:]:
            if line.replace(" ", "") == "*element,type=c3d8rh\n":
                line = "*element,type=c3d8\n"
            if '*end' in line:
                line = line.replace('*end', '**\n')
            f.write(line)
        for i in range(self.n_grains):
            nGrain = i + 1
            cells = np.where(smooth_mesh.cell_data['GrainID'] == nGrain)[0]
            f.write('*Elset, elset=Set-{}\n'.format(nGrain))
            for j, cell in enumerate(cells + 1):
                if (j + 1) % 16 == 0:
                    f.write('\n')
                f.write(' {},'.format(cell))
            f.write('\n')

        phase1_idx = 0
        phase2_idx = 0
        for i in range(self.n_grains):
            nGrain = i + 1
            if self.rve.loc[GRID.cell_data['GrainID'] == nGrain].phaseID.values[0] == 1:
                phase1_idx += 1
                f.write('** Section: Section - {}\n'.format(nGrain))
                f.write('*Solid Section, elset=Set-{}, material=Ferrite_{}\n'.format(nGrain, phase1_idx))
            elif self.rve.loc[GRID.cell_data['GrainID'] == nGrain].phaseID.values[0] == 2:
                if not self.phase_two_isotropic:
                    phase2_idx += 1
                    f.write('** Section: Section - {}\n'.format(nGrain))
                    f.write('*Solid Section, elset=Set-{}, material=Martensite_{}\n'.format(nGrain, phase2_idx))
                else:
                    f.write('** Section: Section - {}\n'.format(nGrain))
                    f.write('*Solid Section, elset=Set-{}, material=Martensite\n'.format(nGrain))

        f.close()
        os.remove(self.store_path + '/rve-part.inp')
        x_max = max(GRID.points[:, 0])
        x_min = min(GRID.points[:, 0])
        y_max = max(GRID.points[:, 1])
        y_min = min(GRID.points[:, 1])
        z_max = max(GRID.points[:, 2])
        z_min = min(GRID.points[:, 2])

        grid_hull_df = pd.DataFrame(pbc_grid.points, columns=['x', 'y', 'z'])
        grid_hull_df = grid_hull_df.loc[(grid_hull_df['x'] == x_max) | (grid_hull_df['x'] == x_min) |
                                        (grid_hull_df['y'] == y_max) | (grid_hull_df['y'] == y_min) |
                                        (grid_hull_df['z'] == z_max) | (grid_hull_df['z'] == z_min)]

        self.make_assembly()         # Don't change the order
        self.pbc(GRID, grid_hull_df)      # of these four
        self.write_material_def()    # functions here
        self.write_step_def()        # it will lead to a faulty inputfile
        self.write_grain_data()

        if self.animation:
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(smooth_mesh, scalars='phaseID', scalar_bar_args={'title': 'Phase IDs'},
                             show_edges=True, interpolate_before_map=True)
            plotter.add_axes()
            plotter.show(interactive=True, auto_close=True, window_size=[800, 600],
                         screenshot=self.store_path+'/Figs/pyvista_smooth_Mesh_phases.png')
            plotter.close()

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(smooth_mesh, scalars='GrainID', scalar_bar_args={'title':'Grain IDs'},
                             show_edges=True, interpolate_before_map=True)
            plotter.add_axes()
            plotter.show(interactive=True, auto_close=True, window_size=[800, 600],
                         screenshot=self.store_path + '/Figs/pyvista_smooth_Mesh_grains.png')
            plotter.close()


if __name__ == '__main__':
    store_path = '.'
    box_size_x = 200
    periodic_rve_df = pd.read_csv('../periodic_rve_df.csv')
    grains_df = pd.read_csv('../grains_df.csv')
    rve = np.load('../RVE_Numpy.npy')

    print(max(np.where(rve > 0)[0]) - min(np.where(rve > 0)[0])+1)
    print(max(np.where(rve > 0)[1]) - min(np.where(rve > 0)[1])+1)
    print(max(np.where(rve > 0)[2]) - min(np.where(rve > 0)[2])+1)
    rve_shape = (max(np.where(rve > 0)[0]) - min(np.where(rve > 0)[0])+1,
                 max(np.where(rve > 0)[1]) - min(np.where(rve > 0)[1])+1,
                 max(np.where(rve > 0)[2]) - min(np.where(rve > 0)[2])+1)
    print(rve_shape)


    el_type = 'C3D8'
    #el_type = 'C3D4'
    mesher_obj = Mesher(box_size_x=box_size_x, box_size_y=None, box_size_z=100, rve_shape=rve_shape,
                        rve=periodic_rve_df, grains_df=grains_df, store_path=store_path,
                        phase_two_isotropic=True, animation=True,
                        infobox_obj=None, progress_obj=None, gui=False, element_type=el_type)
    mesher_obj.mesh_and_build_abaqus_model()