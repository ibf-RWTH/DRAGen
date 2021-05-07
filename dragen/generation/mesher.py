from typing import Tuple, Union, Any

import numpy as np
import pandas as pd
import pyvista as pv
import tetgen
import datetime
import os
import sys
import logging


class Mesher:

    def __init__(self, rve: pd.DataFrame, grains_df: pd.DataFrame, store_path,
                 phase_two_isotropic=True, animation=True, infobox_obj=None, progress_obj=None):
        self.rve = rve
        self.grains_df = grains_df
        self.store_path = store_path
        self.phase_two_isotropic = phase_two_isotropic
        self.animation = animation
        self.infobox_obj = infobox_obj
        self.progress_obj = progress_obj
        self.tex_phi1 = grains_df['phi1'].tolist()
        self.tex_PHI = grains_df['PHI'].to_list()
        self.tex_phi2 = grains_df['phi2'].tolist()
        self.x_max = int(max(rve.x))
        self.x_min = int(min(rve.x))
        self.y_max = int(max(rve.y))
        self.y_min = int(min(rve.y))
        self.z_max = int(max(rve.z))
        self.z_min = int(min(rve.z))
        self.n_grains = int(max(rve.GrainID))
        self.n_pts = int(rve.n_pts[0])
        self.bin_size = rve.box_size[0] / (self.n_pts+1) ## test
        self.logger = logging.getLogger("RVE-Gen")

    def gen_blocks(self) -> pv.UniformGrid:

        """this function generates a structured grid
        in py-vista according to the rve"""
        grid = pv.UniformGrid()

        # Set the grid dimensions: shape + 1 because we want to inject our values on
        #   the CELL data
        grid.dimensions = np.array((self.n_pts+1, self.n_pts+1, self.n_pts+1)) + 1

        # Edit the spatial reference
        grid.origin = (0, 0, 0)  # The bottom left corner of the data set
        # These are the cell sizes along each axis in the gui µm were entered here they are transforemed to mm
        grid.spacing = (self.bin_size/1000, self.bin_size/1000, self.bin_size/1000)
        return grid

    def gen_grains(self, grid: pv.UniformGrid) -> pv.UniformGrid:

        """the grainIDs are written on the cell_array"""

        rve = self.rve
        rve.sort_values(by=['x', 'y', 'z'], inplace=True)  # This sorting is important for some weird reason

        # Add the data values to the cell data
        grid.cell_arrays["GrainID"] = rve.GrainID  # Flatten the array!

        # Now plot the grid!
        if self.animation:
            grid.plot(show_edges=True, screenshot=self.store_path+'/Figs/pyvista_mesh.png', auto_close=True)
        return grid

    def convert_to_mesh(self, grid: pv.UniformGrid) -> tuple:

        """information about grainboundary elements of hex-mesh
        is extracted here and stored in pv.Polydata and
        in a pd.Dataframe"""

        numberOfGrains = self.n_grains
        grainboundary_df = pd.DataFrame()
        all_points = grid.points
        all_points_df = pd.DataFrame(all_points, columns=['x', 'y', 'z'], dtype=float)
        all_points_df.sort_values(by=['x', 'y', 'z'], inplace=True)

        for i in range(1,numberOfGrains + 1):
            sub_grid = grid.extract_cells(np.where(np.asarray(grid.cell_arrays.values())[0] == i))
            sub_surf = sub_grid.extract_surface()
            sub_surf.triangulate(inplace=True)

            # store all points of grain in DataFrame and reindex to indices from whole RVE
            p = sub_surf.points
            p_df = pd.DataFrame(p, columns=['x', 'y', 'z'], dtype=float)
            p_df.sort_values(by=['x', 'y', 'z'], inplace=True)

            compare_all = all_points_df.merge(p_df, on=['x', 'y', 'z'], how='left', indicator=True)

            compare_grain = compare_all.loc[compare_all['_merge'] == 'both'].copy()
            compare_grain.reset_index(inplace=True)
            compare_grain['grain_idx'] = p_df.index
            compare_grain.sort_values(by=['grain_idx'], inplace=True)

            # store all faces in Dataframe and reindex points to indices from whole RVE
            f = sub_surf.faces
            faces = np.reshape(f, (int(len(f) / 4), 4))

            f_df = pd.DataFrame(faces, columns=['npts', 'p1', 'p2', 'p3'])
            idx = np.asarray(compare_grain['index'])
            f_df['p1'] = [idx[j] for j in f_df['p1'].values]
            f_df['p2'] = [idx[j] for j in f_df['p2'].values]
            f_df['p3'] = [idx[j] for j in f_df['p3'].values]
            f_df['facelabel'] = str(i)

            grainboundary_df = pd.concat([grainboundary_df, f_df])

        # filter out duplicate triangles by sorting them and dropping duplicates
        sorted_tuple = [[grainboundary_df.p1.values[i],
                         grainboundary_df.p2.values[i],
                         grainboundary_df.p3.values[i]]
                        for i in range(len(grainboundary_df))]
        sorted_tuple = [sorted(item) for item in sorted_tuple]
        sorted_tuple = [tuple(item) for item in sorted_tuple]
        grainboundary_df.drop_duplicates(inplace=True)
        grainboundary_df['sorted_tris'] = sorted_tuple

        unique_grainboundary_df = grainboundary_df.drop_duplicates(subset=['sorted_tris'], keep='first')

        all_faces = unique_grainboundary_df.drop(['sorted_tris', 'facelabel'], axis=1)
        all_faces = np.asarray(all_faces, dtype='int32')
        all_faces = np.reshape(all_faces, (1, int(len(all_faces) * 4)))[0]
        boundaries = pv.PolyData(all_points, all_faces)

        return boundaries, grainboundary_df

    @staticmethod
    def gen_face_labels(tri_df: pd.DataFrame) -> np.ndarray:

        """all boundary triangles are investigated regarding to
        the grains they are connected to and the face labels
        are stored in a list"""

        tri_df.sort_values(by=['facelabel'], inplace=True)
        fl_df = tri_df[['sorted_tris', 'facelabel']]
        fl_df = fl_df.groupby(['sorted_tris'], sort=False)['facelabel'].apply(', '.join).reset_index()
        fl_df['LabelCount'] = fl_df['facelabel'].str.count(",") + 1
        fl_df.loc[fl_df['LabelCount'] == 1, 'facelabel'] += ',-1'
        fl_df['LabelCount'] = fl_df['facelabel'].str.count(",") + 1
        tri_df.drop_duplicates(subset='sorted_tris', inplace=True, keep='first')
        tri_df.reset_index(inplace=True, drop=True)
        tri_df['facelabel'] = fl_df['facelabel']
        facelabel = np.asarray(tri_df['facelabel'])
        facelabel = [item.split(',') for item in facelabel]
        facelabel = np.asarray([[int(fl) for fl in group] for group in facelabel])

        return facelabel

    @staticmethod
    def smooth(tri: pv.PolyData, rve: pv.UniformGrid, tri_df: pd.DataFrame, fl: list) -> pv.PolyData:

        """this function is smoothing the surface
        mesh of the grain boundaries """

        fl_df = pd.DataFrame(fl)
        pts = rve.points
        tri = tri.faces

        tri_df = tri_df.drop(['facelabel', 'sorted_tris'], axis=1)
        tri_sub = tri_df.copy()
        tri_sub.drop(fl_df.loc[(fl_df[0] == -1) | (fl_df[1] == -1)].index, inplace=True, axis='rows')
        tri_sub = tri_sub.to_numpy().astype('int32')

        ########## mesh and smooth blocks ########################
        surf = pv.PolyData(pts, tri_sub)
        smooth_boundaries = surf.smooth(n_iter=200)
        smooth_points = smooth_boundaries.points
        smooth_points = np.asarray(smooth_points)
        smooth_boundaries = pv.PolyData(smooth_points, tri_sub)  # kept here for debugging purposes
        smooth_rve = pv.PolyData(smooth_points, tri)
        return smooth_rve

    def build_abaqus_model(self, poly_data: pv.PolyData, rve: pv.UniformGrid,
                           fl: list, tri_df: pd.DataFrame = pd.DataFrame()) -> None:

        """building the abaqus model here so far only single phase supported
        for dual or multiple phase material_def needs to be adjusted"""

        fl_df = pd.DataFrame(fl)
        tri = tri_df.drop(['facelabel', 'sorted_tris'], axis=1)
        tri = np.asarray(tri)
        smooth_points = poly_data.points

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

        for i in range(self.n_grains):

            x_min = min(rve.x)
            y_min = min(rve.y)
            z_min = min(rve.z)
            x_max = max(rve.x)
            y_max = max(rve.y)
            z_max = max(rve.z)
            nGrain = i + 1

            tri_idx = fl_df.loc[(fl_df[0] == nGrain) | (fl_df[1] == nGrain)].index
            triGrain = tri[tri_idx, :]
            faces = triGrain.astype('int32')
            sub_surf = pv.PolyData(smooth_points, faces)

            tet = tetgen.TetGen(sub_surf)
            tet.tetrahedralize(order=1, mindihedral=10, minratio=1.5, supsteiner_level=1)
            sub_grid = tet.grid

            """
            This following code block is only needed if all grains are generated as independent parts
            and are merged together later. Or if cohesive contact definitions are defined.
            A first attempt for cohesive contact defs led to convergence issues which is why this route
            wasn't followed any further
            """
            #grain_hull_df = pd.DataFrame(sub_grid.points.tolist(), columns=['x', 'y', 'z'])
            #grain_hull_df = gridPointsDf.loc[(gridPointsDf['x'] == x_max) | (gridPointsDf['x'] == x_min) |
            #                                (gridPointsDf['y'] == y_max) | (gridPointsDf['y'] == y_min) |
            #                                (gridPointsDf['z'] == z_max) | (gridPointsDf['z'] == z_min)]
            #grain_hull_df['GrainID'] = nGrain
            #grid_hull_df = pd.concat([grid_hull_df, grain_hull_df])


            ncells = sub_grid.n_cells
            print(i, ncells)
            self.progress_obj.setValue(75+(100*(i+1)/self.n_grains/4))
            grainIDList = [i + 1]
            grainID_array = grainIDList * ncells
            sub_grid['GrainID'] = grainID_array
            if i == 0:
                grid = sub_grid
            else:
                if len(grid.cell_arrays.keys()) == 0:
                    # print(i, grid.cell_arrays.keys())
                    self.infobox_obj.add_text('uuups! I lost the grainID_key! please increase the resolution')
                    break
                grid = sub_grid.merge(grid)
            grain_vol = sub_grid.volume
            self.logger.info(str(grain_vol*10**9))
            self.grains_df.loc[self.grains_df['GrainID'] == i, 'final_conti_vol'] = grain_vol*10**9

        self.grains_df['final_conti_vol'].to_csv(self.store_path + '/Generation_Data/conti_output_vol.csv', index=False)

        print('ende', grid.cell_arrays.keys())
        #sys.exit()
        pv.save_meshio(self.store_path + '/rve-part.inp', grid)
        f = open(self.store_path + '/rve-part.inp', 'r')
        lines = f.readlines()
        f.close()
        startingLine = lines.index('*NODE\n')
        f = open(self.store_path + '/RVE_smooth.inp', 'a')
        f.write('*Part, name=PART-1\n')
        for line in lines[startingLine:]:
            f.write(line)
        for i in range(self.n_grains):
            nGrain = i + 1
            print('in for nGrain=', nGrain, grid.cell_arrays.keys())
            cells = np.where(grid.cell_arrays['GrainID'] == nGrain)[0]
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
            if self.rve.loc[rve['GrainID'] == nGrain].phaseID.values[0] == 1:
                phase1_idx += 1
                f.write('** Section: Section - {}\n'.format(nGrain))
                f.write('*Solid Section, elset=Set-{}, material=Ferrite_{}\n'.format(nGrain, phase1_idx))
            elif self.rve.loc[rve['GrainID'] == nGrain].phaseID.values[0] == 2:
                if not self.phase_two_isotropic:
                    phase2_idx += 1
                    f.write('** Section: Section - {}\n'.format(nGrain))
                    f.write('*Solid Section, elset=Set-{}, material=Martensite_{}\n'.format(nGrain, phase2_idx))
                else:
                    f.write('** Section: Section - {}\n'.format(nGrain))
                    f.write('*Solid Section, elset=Set-{}, material=Martensite\n'.format(nGrain))

        f.close()
        os.remove(self.store_path + '/rve-part.inp')

        '''this grid_hull_df needs to be removed if the upper block is used for the independent parts'''
        grid_hull_df = pd.DataFrame(grid.points.tolist(), columns=['x', 'y', 'z'])
        grid_hull_df = grid_hull_df.loc[(grid_hull_df['x'] == x_max) | (grid_hull_df['x'] == x_min) |
                                        (grid_hull_df['y'] == y_max) | (grid_hull_df['y'] == y_min) |
                                        (grid_hull_df['z'] == z_max) | (grid_hull_df['z'] == z_min)]

        self.make_assembly()         # Don't change the order
        self.pbc(rve, grid_hull_df)      # of these four
        self.write_material_def()    # functions here
        self.write_step_def()        # it will lead to a faulty inputfile
        self.write_grain_data()

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

    def pbc(self, rve: pv.UniformGrid, grid_hull_df: pd.DataFrame) -> None:

        """function to define the periodic boundary conditions
        if errors appear or equations are wrong check ppt presentation from ICAMS
        included in the docs folder called PBC_docs"""

        min_x = min(rve.x)
        min_y = min(rve.y)
        min_z = min(rve.z)
        max_x = max(rve.x)
        max_y = max(rve.y)
        max_z = max(rve.z)
        print(min_x, min_y, min_z)
        print(max_x, max_y, max_z)
        numberofgrains = self.n_grains
        ########## write Equation - sets ##########
        grid_hull_df = grid_hull_df.sort_values(by=['x', 'y', 'z'])
        grid_hull_df.index.rename('pointNumber', inplace=True)
        grid_hull_df = grid_hull_df.reset_index()
        grid_hull_df.index.rename('Eqn-Set', inplace=True)
        grid_hull_df = grid_hull_df.reset_index()
        #print(grid_hull_df.head())
        #print(min_x, max_x)
        ########## Define Corner Sets ###########
        corner_df = grid_hull_df.loc[((grid_hull_df['x'] == max_x) | (grid_hull_df['x'] == min_x)) &
                                     ((grid_hull_df['y'] == max_y) | (grid_hull_df['y'] == min_y)) &
                                     ((grid_hull_df['z'] == max_z) | (grid_hull_df['z'] == min_z))]

        V1_df = corner_df.loc[(corner_df['x'] == min_x) & (corner_df['y'] == min_y) & (corner_df['z'] == max_z)]
        V1 = V1_df['pointNumber'].values[0]
        V1Eqn = V1_df['Eqn-Set'].values[0]
        #print(V1_df)
        V2_df = corner_df.loc[(corner_df['x'] == max_x) & (corner_df['y'] == min_y) & (corner_df['z'] == max_z)]
        V2 = V2_df['pointNumber'].values[0]
        V2Eqn = V2_df['Eqn-Set'].values[0]
        #print(V2_df)
        V3_df = corner_df.loc[(corner_df['x'] == max_x) & (corner_df['y'] == max_y) & (corner_df['z'] == max_z)]
        V3 = V3_df['pointNumber'].values[0]
        V3Eqn = V3_df['Eqn-Set'].values[0]
        #print(V3_df)
        V4_df = corner_df.loc[(corner_df['x'] == min_x) & (corner_df['y'] == max_y) & (corner_df['z'] == max_z)]
        V4 = V4_df['pointNumber'].values[0]
        V4Eqn = V4_df['Eqn-Set'].values[0]
        #print(V4_df)
        H1_df = corner_df.loc[(corner_df['x'] == min_x) & (corner_df['y'] == min_y) & (corner_df['z'] == min_z)]
        H1 = H1_df['pointNumber'].values[0]
        H1Eqn = H1_df['Eqn-Set'].values[0]
        #print(H1_df)
        H2_df = corner_df.loc[(corner_df['x'] == max_x) & (corner_df['y'] == min_y) & (corner_df['z'] == min_z)]
        H2 = H2_df['pointNumber'].values[0]
        H2Eqn = H2_df['Eqn-Set'].values[0]
        #print(H2_df)
        H3_df = corner_df.loc[(corner_df['x'] == max_x) & (corner_df['y'] == max_y) & (corner_df['z'] == min_z)]
        H3 = H3_df['pointNumber'].values[0]
        H3Eqn = H3_df['Eqn-Set'].values[0]
        #print(H3_df)
        H4_df = corner_df.loc[(corner_df['x'] == min_x) & (corner_df['y'] == max_y) & (corner_df['z'] == min_z)]
        H4 = H4_df['pointNumber'].values[0]
        H4Eqn = H4_df['Eqn-Set'].values[0]
        #print(H4_df)
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

        self.logger.info('E_B1 ' + str(len(E_B1)))
        self.logger.info('E_B2 ' + str(len(E_B2)))
        self.logger.info('E_B3 ' + str(len(E_B3)))
        self.logger.info('E_B4 ' + str(len(E_B4)))
        self.logger.info('E_M1 ' + str(len(E_M1)))
        self.logger.info('E_M2 ' + str(len(E_M2)))
        self.logger.info('E_M3 ' + str(len(E_M3)))
        self.logger.info('E_M4 ' + str(len(E_M4)))
        self.logger.info('E_T1 ' + str(len(E_T1)))
        self.logger.info('E_T2 ' + str(len(E_T2)))
        self.logger.info('E_T3 ' + str(len(E_T3)))
        self.logger.info('E_T4 ' + str(len(E_T4)))
        self.logger.info('LeftSet ' + str(len(LeftSet)))
        self.logger.info('RightSet ' + str(len(RightSet)))
        self.logger.info('BottomSet ' + str(len(BottomSet)))
        self.logger.info('TopSet ' + str(len(TopSet)))
        self.logger.info('FrontSet ' + str(len(FrontSet)))
        self.logger.info('RearSet ' + str(len(RearSet)))

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
        ####################################################################

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
        self.progress_obj.setValue(0)
        self.infobox_obj.add_text('starting mesher')
        GRID = self.gen_blocks()
        self.progress_obj.setValue(25)
        GRID = self.gen_grains(GRID)
        grain_boundaries_poly_data, tri_df = self.convert_to_mesh(GRID)
        self.progress_obj.setValue(50)
        face_label = self.gen_face_labels(tri_df)
        smooth_grain_boundaries = self.smooth(grain_boundaries_poly_data, GRID, tri_df, face_label)
        self.progress_obj.setValue(75)
        self.build_abaqus_model(rve=GRID, poly_data=smooth_grain_boundaries, fl=face_label, tri_df=tri_df)
        self.progress_obj.setValue(100)

