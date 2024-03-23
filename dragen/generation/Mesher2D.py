import pyvista as pv
import numpy as np
import pandas as pd
import logging
import os
from math import isclose
import datetime
from dragen.utilities.Helpers import HelperFunctions
from dragen.utilities.InputInfo import RveInfo


class Mesher_2D(HelperFunctions):

    def __init__(self,rve_df: pd.DataFrame, grains_df: pd.DataFrame, store_path, phase_two_isotropic = True,
                 animation=True, infobox_obj=None, progress_obj=None):
        self.rve = rve_df
        self.grains_df = grains_df
        self.store_path = store_path
        self.phase_two_isotropic = phase_two_isotropic
        self.animation = animation
        self.infobox_obj = infobox_obj
        self.progress_obj = progress_obj
        self.tex_phi1 = grains_df['phi1'].tolist()
        self.tex_PHI = grains_df['PHI'].tolist()
        self.tex_phi2 = grains_df['phi2'].tolist()
        self.x_max = int(max(rve_df.x))
        self.x_min = int(min(rve_df.x))
        self.y_max = int(max(rve_df.y))
        self.y_min = int(min(rve_df.y))
        self.n_grains = int(max(rve_df.GrainID))
        self.n_pts = int(rve_df.n_pts[0])
        self.box_size = rve_df.box_size[0]
        self.bin_size = rve_df.box_size[0] / (self.n_pts+1) ## test
        self.logger = logging.getLogger("RVE-Gen")

        super().__init__()

    def polyline_from_points(self,points):
        poly = pv.PolyData()
        poly.points = points
        the_cell = np.arange(0, len(points), dtype=np.int_)
        the_cell = np.insert(the_cell, 0, len(points))
        poly.lines = the_cell
        return poly

    def gen_blocks(self):
        """this function generates a structured grid
                in py-vista according to the rve"""
        grid = pv.UniformGrid()


        # Set the grid dimensions: shape + 1 because we want to inject our values on
        #   the CELL data
        grid.dimensions = np.array((self.n_pts + 1, self.n_pts + 1, 1)) + 1

        # Edit the spatial reference
        grid.origin = (0, 0, 0)  # The bottom left corner of the data set

        # These are the cell sizes along each axis in the gui µm were entered here they are transforemed to mm
        grid.spacing = (self.bin_size / 1000, self.bin_size / 1000, self.bin_size / 1000)
        grid = grid.cast_to_unstructured_grid()
        return grid

    def gen_grains(self, grid):
        """the grainIDs are written on the cell_array"""

        rve = self.rve
        rve.sort_values(by=['x', 'y'], inplace=True)  # This sorting is important for some weird reason

        # Add the data values to the cell data
        grid.cell_data["GrainID"] = rve.GrainID
        grid.cell_data["phaseID"] = rve.phaseID

        # Now plot the grid!
        if self.animation:
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(grid, scalars='phaseID', stitle='Phase IDs',
                             show_edges=True, interpolate_before_map=True)
            plotter.add_axes()
            plotter.show(interactive=True, auto_close=True, window_size=[800, 600],
                         screenshot=self.store_path + '/Figs/pyvista_Hex_Mesh_phases.png')
            plotter.close()

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(grid, scalars='GrainID', stitle='Grain IDs',
                             show_edges=True, interpolate_before_map=True)
            plotter.add_axes()
            plotter.show(interactive=True, auto_close=True, window_size=[800, 600],
                         screenshot=self.store_path + '/Figs/pyvista_Hex_Mesh_grains.png')
            plotter.close()

        return grid

    def find_lines_with_node(self,index, lines):
        """Pass the index of the node in question.
        Returns the face indices of the faces with that node."""
        return [i for i, line in enumerate(lines) if index in line]

    def find_connected_vertices(self, index, lines):
        """Pass the index of the node in question.
        Returns the vertex indices of the vertices connected_points with that node."""
        cids = self.find_lines_with_node(index, lines)
        connected = np.unique(lines[cids].ravel())
        return np.delete(connected, np.argwhere(connected == index))

    def laplace_2D(self, lines, points, n_iter=10, alpha=0.1):

        check_line10 = None
        check_line11 = None

        p_new_N_minus_one = None
        p_0_N_minus_one = None
        move_flag = False
        label_set = list(set(np.asarray(lines.line_labels)))

        for i in range(n_iter):
            print(i)
            for label in label_set:

                current_lines = lines.loc[lines['line_labels'] == label]

                current_lines = [list(zip(current_lines.p1, current_lines.p2))]
                current_lines = np.asarray([list(item) for item in current_lines])

                current_point_set = set(current_lines.flatten())

                for p in current_point_set:

                    p_0 = points[p]
                    neighbors = self.find_connected_vertices(p, current_lines)

                    if len(neighbors) == 2:
                        help = [points[neighbors[j]].tolist() for j in range(len(neighbors))]
                        p_1 = np.array((np.array(help[0])+np.array(help[1])))/2
                        v = np.array((np.array(p_1)-np.array(p_0)))
                        p_new = p_0+v*alpha

                        if check_line10 is not None:
                            check_line2 = np.array(p_new_N_minus_one) - np.array(p_new)
                            check_line3 = np.array(p_0_N_minus_one) - np.array(p_new)
                            cross_prod_10 = np.cross(check_line10, check_line2)
                            cross_prod_11 = np.cross(check_line11, check_line2)
                            cross_prod_20 = np.cross(check_line10, check_line3)
                            cross_prod_21 = np.cross(check_line11, check_line3)
                            scalar1 = np.dot(cross_prod_10, cross_prod_11)

                            if (scalar1 > 0):
                                move_flag = True
                            else:
                                #print('false')
                                #print(p_new)
                                #print(p_0)

                                move_flag = False


                        if check_line10 is None:  # this condition is necessary
                            #print('this should only happen once')
                            points[p] = p_new                 # to make sure the new point doesnt cross a line
                        elif move_flag:
                            points[p] = p_new
                        #print(help[0])
                        p_new_N_minus_one = p_new
                        p_0_N_minus_one = p_0
                        check_line10 = np.array(help[0]) - np.array(p_new)
                        check_line11 = np.array(help[1]) - np.array(p_new)

        return points

    def gen_point_labels(self, pts_df: pd.DataFrame) -> pd.DataFrame:

        """all boundary triangles are investigated regarding to
        the grains they are connected to and the face labels
        are stored in a list"""

        pts_df.sort_values(by=['GrainID'], inplace=True)
        label_df = pts_df[['grid_index_total', 'GrainID']]
        label_df = label_df.groupby(['grid_index_total'], sort=False)['GrainID'].apply(', '.join).reset_index()
        label_df['grainCount'] = label_df['GrainID'].str.count(",") + 1
        label_df.loc[label_df['grainCount'] == 1, 'GrainID'] += ',-1'
        label_df['grainCount'] = label_df['GrainID'].str.count(",") + 1
        pts_df.drop_duplicates(subset='grid_index_total', inplace=True, keep='first')
        pts_df.reset_index(inplace=True, drop=True)
        pts_df['labels'] = label_df['GrainID']
        pts_df['grainCount'] = label_df['grainCount']


        return pts_df

    def gen_line_labels(self, line_df: pd.DataFrame) -> pd.DataFrame:

        """all boundary triangles are investigated regarding to
        the grains they are connected to and the face labels
        are stored in a list"""

        sorted_tuples = [list(zip(line_df.p1, line_df.p2))]
        sorted_tuples = [tuple(sorted(line)) for line in sorted_tuples]
        line_df['sorted_lines'] = sorted_tuples

        label_df = line_df[['sorted_lines', 'GrainID']]
        label_df = label_df.groupby(['sorted_lines'], sort=False)['GrainID'].apply(','.join).reset_index()
        label_df['grainCount'] = label_df['GrainID'].str.count(",") + 1

        label_df.loc[label_df['grainCount'] == 1, 'GrainID'] += ',-1'
        label_df['grainCount'] = label_df['GrainID'].str.count(",") + 1

        line_df.drop_duplicates(subset='sorted_lines', inplace=True, keep='first')
        line_df.reset_index(inplace=True, drop=True)
        line_df['line_labels'] = label_df['GrainID']
        line_df['grainCount'] = label_df['grainCount']
        #line_df.drop(line_df.loc[line_df['grainCount'] == 1].index, inplace=True)

        return line_df

    def smoothen_2D(self, grid):
        x_max = max(grid.points[:, 0])
        x_min = min(grid.points[:, 0])
        y_max = max(grid.points[:, 1])
        y_min = min(grid.points[:, 1])
        z_max = max(grid.points[:, 2])
        z_min = min(grid.points[:, 2])
        numberOfGrains = self.n_grains

        gid_list = list()
        pid_list = list()
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

        old_grid = grid.copy()
        for i in range(1, numberOfGrains + 1):
            phase = self.rve.loc[self.rve['GrainID'] == i].phaseID.values[0]
            grain_grid = old_grid.extract_cells(np.where(np.asarray(old_grid.cell_data.values())[0] == i))
            grain_surf = grain_grid.extract_surface()
            grain_surf_df = pd.DataFrame(data=grain_surf.points, columns=['x', 'y', 'z'])
            merged_pts_df = grain_surf_df.join(all_points_df_old.set_index(['x', 'y', 'z']), on=['x', 'y', 'z'])
            if RveInfo.smoothing_flag:
                n_iter=250
            else:
                n_iter=0
            grain_surf_smooth = grain_surf.smooth(n_iter=n_iter)
            smooth_pts_df = pd.DataFrame(data=grain_surf_smooth.points, columns=['x', 'y', 'z'])
            all_points_df.loc[merged_pts_df['ori_idx'], ['x', 'y', 'z']] = smooth_pts_df.values
            grain_vol = grain_grid.volume
            self.grains_df.loc[self.grains_df['GrainID'] == i - 1, 'meshed_conti_volume'] = grain_vol * 10 ** 9

        #self.grains_df[['GrainID', 'meshed_conti_volume', 'phaseID']]. \
        #    to_csv(self.store_path + '/Generation_Data/grain_data_output_conti.csv', index=False)

        all_points_df.loc[all_points_df_old['x_min'], 'x'] = x_min
        all_points_df.loc[all_points_df_old['y_min'], 'y'] = y_min
        all_points_df.loc[all_points_df_old['z_min'], 'z'] = z_min
        all_points_df.loc[all_points_df_old['x_max'], 'x'] = x_max
        all_points_df.loc[all_points_df_old['y_max'], 'y'] = y_max
        all_points_df.loc[all_points_df_old['z_max'], 'z'] = z_max

        grid.points = all_points_df[['x', 'y', 'z']].values

        return grid

    def run_mesher_2D(self):
        grid = self.gen_blocks()
        grid = self.gen_grains(grid)
        mesh = self.smoothen_2D(grid)
        return mesh

class BuildAbaqus2D:

    def __init__(self, pv_mesh, rve_df, grains_df: pd.DataFrame):

        self.mesh = pv_mesh
        self.rve_df = rve_df
        self.n_grains = int(max(pv_mesh.cell_data['GrainID']))


        self.tex_phi1 = grains_df['phi1'].tolist()
        self.tex_PHI = grains_df['PHI'].tolist()
        self.tex_phi2 = grains_df['phi2'].tolist()

    def build_abaqus_header(self) -> None:
        """us model here so far only single phase supported
        for dual or multiple phase material_def needs to be adjusted"""

        #fl_df = pd.DataFrame(fl)
        #tri = tri_df.drop(['facelabel', 'sorted_tris'], axis=1)
        #tri = np.asarray(tri)
        #smooth_points = poly_data.points

        f = open(RveInfo.store_path + '/DRAGen_RVE_2D.inp', 'w+')
        f.write('*Heading\n')
        f.write('** Job name: Job-1 Model name: Job-1\n')
        f.write('** Generated by: DRAGen \n')
        f.write('** Date: {}\n'.format(datetime.datetime.now().strftime("%d.%m.%Y")))
        f.write('** Time: {}\n'.format(datetime.datetime.now().strftime("%H:%M:%S")))
        f.write('*Preprint, echo=NO, model=NO, history=NO, contact=NO\n')
        f.write('**\n')
        f.write('** PARTS\n')
        f.write('**\n')
        f.write('*Part, name=Part-1\n')
        f.write('*End Part\n')
        f.write('**\n')
        f.close()

    def make_assembly(self) -> None:

        """simple function to write the assembly definition in the input file"""

        f = open(RveInfo.store_path + '/DRAGen_RVE_2D.inp', 'a')
        f.write('** ASSEMBLY\n')
        f.write('**\n')
        f.write('*Assembly, name=Assembly\n')
        f.write('**\n')
        f.write('*Instance, name=Part-1-1, part=Part-1\n')
        f.close()
        self.make_meshio_inp_file()
        self.generate_elementsets()
        self.assign_materials()
        f = open(RveInfo.store_path + '/DRAGen_RVE_2D.inp', 'a')
        f.write('*End Instance\n')
        if RveInfo.xfem_flag:
            f.write("*Enrichment, name=Crack-1, type=PROPAGATION CRACK, elset=Part-1-1.Set-XFEM, interaction=IntProp-1\n")
        f.write('**\n')
        if RveInfo.pbc_flag:
            f.write('*Include, Input=Nsets.inp\n')
            f.write('*Include, input=Edges.inp\n')
            f.write('*Include, input=Corners.inp\n')
            f.write('*Include, input=VerticeSets.inp\n')
        elif RveInfo.submodel_flag:
            f.write('*Include, Input=HullPointSet.inp\n')
        f.write('*End Assembly\n')
        f.write('** INCLUDE MATERIAL FILE **\n')
        f.write('*Include, input=Materials.inp\n')
        f.write('** INCLUDE STEP FILE **\n')
        f.write('*Include, input=Step.inp\n')
        f.close()

    def build_nodes_and_elements(self) -> None:
        faces = self.mesh.extract_surface().faces

        faces = faces.reshape((-1, 5))[:, 1:5]

        node_dict = {'x': self.mesh.points[:, 0], 'y': self.mesh.points[:, 1], 'z': self.mesh.points[:, 2]}
        elem_dict = {'p1': faces[:, 0], 'p2': faces[:, 1], 'p3': faces[:, 2], 'p4': faces[:, 3]}
        abaq_nodes_df = pd.DataFrame(data=node_dict)
        abaq_elem_df = pd.DataFrame(data=elem_dict)

        f = open(RveInfo.store_path + '/DRAGen_RVE_2D.inp', 'a')
        f.write('*Part, name=PART-1\n')
        f.write('*NODE\n')
        for i in range(len(abaq_nodes_df)):
            line = '{}, {}, {}, {}\n'.format(abaq_nodes_df.index[i]+1,
                                         abaq_nodes_df.x[i],
                                         abaq_nodes_df.y[i],
                                         abaq_nodes_df.z[i])
            f.write(line)

        f.write('*ELEMENT, TYPE=C3D8\n')
        for i in range(len(abaq_elem_df)):
            line = '{}, {}, {}, {}\n'.format(abaq_elem_df.index[i] + 1,
                                         abaq_elem_df.p1[i]+1,
                                         abaq_elem_df.p2[i]+1,
                                         abaq_elem_df.p3[i]+1,
                                         abaq_elem_df.p4[i]+1,
                                         abaq_elem_df.p5[i]+1) #add
            f.write(line)
        f.close()

    def submodelSet(self) -> None:
        grid_df = pd.DataFrame(self.mesh.points.tolist(), columns=['x', 'y', 'z'])
        max_x = max(grid_df.x)
        min_x = min(grid_df.x)
        max_y = max(grid_df.y)
        min_y = min(grid_df.y)
        max_z = max(grid_df.z)
        min_z = min(grid_df.z)
        grid_hull_df = grid_df.loc[(grid_df['x'] == max_x) | (grid_df['x'] == min_x) |
                                   (grid_df['y'] == max_y) | (grid_df['y'] == min_y) |
                                   (grid_df['z'] == max_z) | (grid_df['z'] == min_z)]
        OutPutFile = open(RveInfo.store_path + '/HullPointSet.inp', 'w+')
        grid_hull_df.sort_values(by=['x', 'y', 'z'], inplace=True)
        grid_hull_df.index.rename('pointNumber', inplace=True)
        grid_hull_df = grid_hull_df.reset_index()
        for i in grid_hull_df.index:
            OutPutFile.write('*Nset, nset=SET-Hull, instance=PART-1-1\n'.format(i + 1))
            OutPutFile.write(' {},\n'.format(int(grid_hull_df.loc[i]['pointNumber'] + 1)))
        OutPutFile.close()

    def make_meshio_inp_file(self):
        pv.save_meshio(f'{RveInfo.store_path}/rve-part.inp', self.mesh)
        f = open(f'{RveInfo.store_path}/rve-part.inp', 'r')
        lines = f.readlines()
        f.close()
        lines = [line.lower() for line in lines]
        startingLine = lines.index('*node\n')
        f = open(RveInfo.store_path + '/DRAGen_RVE_2D.inp', 'a')
        idx = [i for i, s in enumerate(lines) if '*element' in s.lower()][0]
        lines[idx] = '*ELEMENT, TYPE=C3D8\n'
        for line in lines[startingLine:]:
            f.write(line)
        f.close()
        os.remove(f'{RveInfo.store_path}/rve-part.inp')

    def generate_elementsets(self):
        f = open(RveInfo.store_path + '/DRAGen_RVE_2D.inp', 'a')
        f.write('**\n')
        for i in range(self.n_grains):
            nGrain = i + 1
            print('in for nGrain=', nGrain, self.mesh.cell_data.keys())
            cells = np.where(self.mesh.cell_data['GrainID'] == nGrain)[0]
            f.write('*Elset, elset=Set-{}\n'.format(nGrain))
            for j, cell in enumerate(cells + 1):
                if (j + 1) % 16 == 0:
                    f.write('\n')
                f.write(' {},'.format(cell))
            f.write('\n''**\n')
        if RveInfo.xfem_flag:
            f.write('*Elset, elset=Set-XFEM, instance=Part-1-1, generate\n')
            f.write(f'1,  {self.mesh.number_of_cells},      1\n')
        f.write('**\n')
        f.close()

    def assign_materials(self):
        f = open(RveInfo.store_path + '/DRAGen_RVE_2D.inp', 'a')
        phase1_idx = 0
        phase2_idx = 0
        for i in range(self.n_grains):
            nGrain = i + 1
            if self.rve_df.loc[self.rve_df['GrainID'] == nGrain].phaseID.values[0] == 1:
                phase1_idx += 1
                f.write('** Section: Section - {}\n'.format(nGrain))
                f.write('*Solid Section, elset=Set-{}, material=Ferrite_{}\n'.format(nGrain, phase1_idx))
            elif self.rve_df.loc[self.rve_df['GrainID'] == nGrain].phaseID.values[0] == 2:
                if not RveInfo.phase2iso_flag[2]:
                    phase2_idx += 1
                    f.write('** Section: Section - {}\n'.format(nGrain))
                    f.write('*Solid Section, elset=Set-{}, material=Martensite_{}\n'.format(nGrain, phase2_idx))
                else:
                    f.write('** Section: Section - {}\n'.format(nGrain))
                    f.write('*Solid Section, elset=Set-{}, material=Martensite\n'.format(nGrain))

        f.close()

    def pbc(self) -> None:

        """function to define the periodic boundary conditions
        if errors appear or equations are wrong check ppt presentation from ICAMS
        included in the docs folder called PBC_docs"""

        grid_df = pd.DataFrame(self.mesh.points.tolist(), columns=['x', 'y', 'z'])
        max_x = max(grid_df.x)
        min_x = min(grid_df.x)
        max_y = max(grid_df.y)
        min_y = min(grid_df.y)
        max_z = max(grid_df.z)
        min_z = min(grid_df.z)
        grid_hull_df = grid_df.loc[(grid_df['x'] == max_x) | (grid_df['x'] == min_x) |
                                   (grid_df['y'] == max_y) | (grid_df['y'] == min_y) |
                                   (grid_df['z'] == max_z) | (grid_df['z'] == min_z)]

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
        # print(V1_df)
        V2_df = corner_df.loc[(corner_df['x'] == max_x) & (corner_df['y'] == min_y) & (corner_df['z'] == max_z)]
        V2 = V2_df['pointNumber'].values[0]
        V2Eqn = V2_df['Eqn-Set'].values[0]
        # print(V2_df)
        V3_df = corner_df.loc[(corner_df['x'] == max_x) & (corner_df['y'] == max_y) & (corner_df['z'] == max_z)]
        V3 = V3_df['pointNumber'].values[0]
        V3Eqn = V3_df['Eqn-Set'].values[0]
        # print(V3_df)
        V4_df = corner_df.loc[(corner_df['x'] == min_x) & (corner_df['y'] == max_y) & (corner_df['z'] == max_z)]
        V4 = V4_df['pointNumber'].values[0]
        V4Eqn = V4_df['Eqn-Set'].values[0]
        # print(V4_df)
        H1_df = corner_df.loc[(corner_df['x'] == min_x) & (corner_df['y'] == min_y) & (corner_df['z'] == min_z)]
        H1 = H1_df['pointNumber'].values[0]
        H1Eqn = H1_df['Eqn-Set'].values[0]
        # print(H1_df)
        H2_df = corner_df.loc[(corner_df['x'] == max_x) & (corner_df['y'] == min_y) & (corner_df['z'] == min_z)]
        H2 = H2_df['pointNumber'].values[0]
        H2Eqn = H2_df['Eqn-Set'].values[0]
        # print(H2_df)
        H3_df = corner_df.loc[(corner_df['x'] == max_x) & (corner_df['y'] == max_y) & (corner_df['z'] == min_z)]
        H3 = H3_df['pointNumber'].values[0]
        H3Eqn = H3_df['Eqn-Set'].values[0]
        # print(H3_df)
        H4_df = corner_df.loc[(corner_df['x'] == min_x) & (corner_df['y'] == max_y) & (corner_df['z'] == min_z)]
        H4 = H4_df['pointNumber'].values[0]
        H4Eqn = H4_df['Eqn-Set'].values[0]
        # print(H4_df)
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


        OutPutFile = open(RveInfo.store_path + '/Nsets.inp', 'w')
        for i in grid_hull_df.index:
            OutPutFile.write('*Nset, nset=Eqn-Set-{}, instance=PART-1-1\n'.format(i + 1))
            OutPutFile.write(' {},\n'.format(int(grid_hull_df.loc[i]['pointNumber'] + 1)))
        OutPutFile.close()

        ############### Define Equations ###################################
        OutPutFile = open(RveInfo.store_path + '/LeftToRight.inp', 'w')

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

        OutPutFile = open(RveInfo.store_path + '/BottomToTop.inp', 'w')

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

        OutPutFile = open(RveInfo.store_path + '/FrontToRear.inp', 'w')

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

        OutPutFile = open(RveInfo.store_path + '/Edges.inp', 'w')

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

        OutPutFile = open(RveInfo.store_path + '/Corners.inp', 'w')

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

        OutPutFile = open(RveInfo.store_path + '/VerticeSets.inp', 'w')
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
        """simple function to write material definition in Input file
        needs to be adjusted for multiple phases"""
        phase1_idx = 0
        phase2_idx = 0
        phase3_idx = 0
        phase4_idx = 0
        phase5_idx = 0 #add
        numberofgrains = self.n_grains

        phase = [self.rve_df.loc[self.rve_df['GrainID'] == i].phaseID.values[0] for i in range(1, numberofgrains+1)]
        f = open(RveInfo.store_path + '/Materials.inp', 'w+')  # open in write mode to overwrite old files in case ther are any
        f.write('** MATERIALS\n')
        f.write('**\n')
        f.close()
        f = open(RveInfo.store_path + '/Materials.inp', 'a')

        for i in range(numberofgrains):
            ngrain = i+1
            if phase[i] == 1:
                phase1_idx += 1
                f.write('*Material, name=Ferrite_{}\n'.format(phase1_idx))
                f.write('*Depvar\n')
                f.write('    176,\n')
                f.write('*User Material, constants=2\n')
                f.write('{}.,3.\n'.format(phase1_idx))
                if RveInfo.xfem_flag:
                    f.write('*include, input=Ferrite_dmg.inp\n')
            elif phase[i] == 2:
                if not RveInfo.phase2iso_flag[2]:
                    phase2_idx += 1
                    f.write('*Material, name=Martensite_{}\n'.format(phase2_idx))
                    f.write('*Depvar\n')
                    f.write('    176,\n')
                    f.write('*User Material, constants=2\n')
                    f.write('{}.,4.\n'.format(ngrain))
                    if RveInfo.xfem_flag:
                        f.write('*include, input=Martensite_dmg.inp\n')
            elif phase[i] == 3:
                if not RveInfo.phase2iso_flag[2]:
                    phase3_idx += 1
                    f.write('*Material, name=Pearlite_{}\n'.format(phase3_idx))
                    f.write('*Depvar\n')
                    f.write('    176,\n')
                    f.write('*User Material, constants=2\n')
                    f.write('{}.,4.\n'.format(ngrain))
                    if RveInfo.xfem_flag:
                        f.write('*include, input=Pearlite_dmg.inp\n')
            elif phase[i] == 4:
                if not RveInfo.phase2iso_flag[3]:
                    phase4_idx += 1
                    f.write('*Material, name=Bainite_{}\n'.format(phase4_idx))
                    f.write('*Depvar\n')
                    f.write('    176,\n')
                    f.write('*User Material, constants=2\n')
                    f.write('{}.,4.\n'.format(ngrain))
                    if RveInfo.xfem_flag:
                        f.write('*include, input=Bainite_dmg.inp\n')
            elif phase[i] == 5: #add
                if not RveInfo.phase2iso_flag[3]:
                    phase5_idx += 1
                    f.write('*Material, name=Austenite_{}\n'.format(phase5_idx))
                    f.write('*Depvar\n')
                    f.write('    176,\n')
                    f.write('*User Material, constants=2\n')
                    f.write('{}.,2.\n'.format(ngrain))
                    if RveInfo.xfem_flag:
                        f.write('*include, input=Austenite_dmg.inp\n')

        if RveInfo.phase2iso_flag[2] and RveInfo.phase_ratio[2] > 0:
            f.write('**\n')
            f.write('*Material, name=Martensite\n')
            f.write('*Elastic\n')
            f.write('0.21, 0.3\n')
            f.write('**')
        if RveInfo.phase2iso_flag[3] and RveInfo.phase_ratio[3] > 0:
            f.write('**\n')
            f.write('*Material, name=Pearlite\n')
            f.write('*Elastic\n')
            f.write('0.21, 0.3\n')
            f.write('**')
        if RveInfo.phase2iso_flag[4] and RveInfo.phase_ratio[4] > 0:
            f.write('**\n')
            f.write('*Material, name=Bainite\n')
            f.write('*Elastic\n')
            f.write('0.21, 0.3\n')
            f.write('**')
        if RveInfo.phase2iso_flag[5] and RveInfo.phase_ratio[5] > 0: #add
            f.write('**\n')
            f.write('*Material, name=Austenite\n')
            f.write('*Elastic\n')
            f.write('0.21, 0.3\n')
            f.write('**')
        f.close()
        if RveInfo.xfem_flag and RveInfo.phase_ratio[1] > 0:
            f = open(RveInfo.store_path + '/Ferrite_dmg.inp', 'a')
            f.write('*Damage Initiation, Criterion=User, Failure Mechanisms=1, Properties=2 \n')
            f.write('** damage variable, max. element number \n')
            f.write('0.01, {} \n'.format(self.mesh.number_of_cells))
            f.write('*Damage Evolution, type=DISPLACEMENT, FAILURE INDEX=1 \n')
            f.write('1., \n')
            f.write('*Damage Stabilization \n')
            f.write('0.012 \n')
            f.close()

        if RveInfo.xfem_flag and RveInfo.phase_ratio[2] > 0:
            f = open(RveInfo.store_path + '/martensite_dmg.inp', 'a')
            f.write('*Damage Initiation, Criterion=User, Failure Mechanisms=1, Properties=2 \n')
            f.write('** damage variable, max. element number \n')
            f.write('0.01, {} \n'.format(self.mesh.number_of_cells))
            f.write('*Damage Evolution, type=DISPLACEMENT, FAILURE INDEX=1 \n')
            f.write('1., \n')
            f.write('*Damage Stabilization \n')
            f.write('0.012 \n')
            f.close()

        if RveInfo.xfem_flag and RveInfo.phase_ratio[3] > 0:
            f = open(RveInfo.store_path + '/Pearlite_dmg.inp', 'a')
            f.write('*Damage Initiation, Criterion=User, Failure Mechanisms=1, Properties=2 \n')
            f.write('** damage variable, max. element number \n')
            f.write('0.01, {} \n'.format(self.mesh.number_of_cells))
            f.write('*Damage Evolution, type=DISPLACEMENT, FAILURE INDEX=1 \n')
            f.write('1., \n')
            f.write('*Damage Stabilization \n')
            f.write('0.012 \n')
            f.close()

        if RveInfo.xfem_flag and RveInfo.phase_ratio[4] > 0:
            f = open(RveInfo.store_path + '/Bainite_dmg.inp', 'a')
            f.write('*Damage Initiation, Criterion=User, Failure Mechanisms=1, Properties=2 \n')
            f.write('** damage variable, max. element number \n')
            f.write('0.01, {} \n'.format(self.mesh.number_of_cells))
            f.write('*Damage Evolution, type=DISPLACEMENT, FAILURE INDEX=1 \n')
            f.write('1., \n')
            f.write('*Damage Stabilization \n')
            f.write('0.012 \n')
            f.close()
        if RveInfo.xfem_flag and RveInfo.phase_ratio[5] > 0: #add
            f = open(RveInfo.store_path + '/Austenite_dmg.inp', 'a')
            f.write('*Damage Initiation, Criterion=User, Failure Mechanisms=1, Properties=2 \n')
            f.write('** damage variable, max. element number \n')
            f.write('0.01, {} \n'.format(self.mesh.number_of_cells))
            f.write('*Damage Evolution, type=DISPLACEMENT, FAILURE INDEX=1 \n')
            f.write('1., \n')
            f.write('*Damage Stabilization \n')
            f.write('0.012 \n')
            f.close()

    def write_pbc_step_def(self) -> None:

        """simple function to write step definition
        variables should be introduced to give the user an option
        to modify amplidtude, and other parameters"""

        f = open(RveInfo.store_path + '/Step.inp', 'w+')
        f.write('**\n')
        f.write('**BOUNDARY CONDITIONS\n')
        f.write('**\n')
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
        if RveInfo.xfem_flag:
            f.write('** INTERACTION PROPERTIES\n')
            f.write('**\n')
            f.write('*Surface Interaction, name=IntProp-1\n')
            f.write('1.,\n')
            f.write('*Surface Behavior, pressure-overclosure=LINEAR\n')
            f.write('0.01,\n')
        f.write('** ----------------------------------------------------------------\n')
        f.write('**\n')
        f.write('** STEP: Step-1\n')
        f.write('**\n')
        f.write('*Step, name=Step-1, nlgeom=YES, inc=10000000, solver=iterative\n')
        f.write('*Static\n')
        f.write('1e-5, 600., 1e-15, 0.1\n')
        f.write('**\n')
        f.write('** CONTROLS\n')
        f.write('**\n')
        f.write('*Controls, reset\n')
        f.write('*Controls, analysis=discontinuous\n')
        f.write('*Controls, parameters=time incrementation\n')
        f.write(', , , , , , , 20, , ,20\n')
        f.write('**\n')
        f.write('*CONTROLS, PARAMETERS=LINE SEARCH\n')
        f.write('10\n')
        f.write('*SOLVER CONTROL\n')
        f.write('1e-5,200,\n')
        f.write('**\n')
        f.write('** BOUNDARY CONDITIONS\n')
        f.write('**\n')
        f.write('** Name: Load Type: Displacement/Rotation\n')
        f.write('*Boundary\n')
        f.write('V4, 2, 2, 0.01\n')
        f.write('**\n')
        if RveInfo.xfem_flag:
            f.write('** INTERACTIONS\n')
            f.write('**\n')
            f.write('*Enrichment Activation, name=Crack-1, activate=ON\n')
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

    def write_submodel_step_def(self) -> None:

        """simple function to write step definition
        variables should be introduced to give the user an option
        to modify amplidtude, and other parameters"""

        f = open(RveInfo.store_path + '/Step.inp', 'w+')
        f.write('**\n')
        f.write('** ----------------------------------------------------------------\n')
        f.write('**\n')
        f.write('** STEP: Step-1\n')
        f.write('**\n')
        f.write('*Step, name=Step-1, nlgeom=YES, inc=10000000, solver=ITERATIVE\n')
        f.write('*Static\n')
        f.write('** Step time needs to be adjusted to global .odb')
        f.write('0.001, 1., 1.05e-15, 0.25\n')
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
        f.write('** Name: Sub-BC-1 Type: Submodel\n')
        f.write('*Boundary, submodel, step=1\n')
        f.write('Set-Hull, 1, 1\n')
        f.write('Set-Hull, 2, 2\n')
        f.write('Set-Hull, 3, 3\n')
        f.write('**\n')
        if RveInfo.xfem_flag:
            f.write('*Include, input=Interactions.inp')
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
        f = open(RveInfo.store_path + '/graindata.inp', 'w+')
        f.write('!MMM Crystal Plasticity Input File\n')
        phase1_idx = 0
        numberofgrains = self.n_grains
        phase = [self.rve_df.loc[self.rve_df['GrainID'] == i].phaseID.values[0] for i in range(1, numberofgrains + 1)]
        grainsize = [np.cbrt(self.rve_df.loc[self.rve_df['GrainID'] == i].shape[0] *
                             RveInfo.bin_size**3*3/4/np.pi) for i in range(1, numberofgrains + 1)]

        for i in range(numberofgrains):
            ngrain = i+1
            if not RveInfo.phase2iso_flag[phase[i]]:
                phi1 = self.tex_phi1[i]
                PHI = self.tex_PHI[i]
                phi2 = self.tex_phi2[i]
                f.write('Grain: {}: {}: {}: {}: {}\n'.format(ngrain, phi1, PHI, phi2, grainsize[i]))
        f.close()

    def run(self):

        self.build_abaqus_header()
        self.make_assembly()



        if RveInfo.submodel_flag:
            self.submodelSet()
        elif RveInfo.pbc_flag:
            self.pbc()
        self.write_material_def()  # functions here
        if RveInfo.pbc_flag:
            self.write_pbc_step_def()  # it will lead to a faulty inputfile
        elif RveInfo.submodel_flag:
            self.write_submodel_step_def()
        self.write_grain_data()

        plotter = pv.Plotter()
        plotter.add_mesh(self.mesh, show_edges=True, scalars='GrainID', cmap='flag')
        plotter.show()
