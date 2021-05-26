import pyvista as pv
import numpy as np
import pandas as pd
import logging
from math import isclose
import datetime
from dragen.utilities.RVE_Utils import RVEUtils

import matplotlib.pyplot as plt
import sys


class Mesher_2D(RVEUtils):

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

        super().__init__(self.box_size, self.n_pts)

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
        grid.dimensions = np.array((self.n_pts + 1, self.n_pts + 1, 0)) + 1

        # Edit the spatial reference
        grid.origin = (0, 0, 0)  # The bottom left corner of the data set

        # These are the cell sizes along each axis in the gui µm were entered here they are transforemed to mm
        grid.spacing = (self.bin_size / 1000, self.bin_size / 1000, 0)

        return grid

    def gen_grains(self, grid):
        """the grainIDs are written on the cell_array"""

        rve = self.rve
        rve.sort_values(by=['x', 'y'], inplace=True)  # This sorting is important for some weird reason

        # Add the data values to the cell data
        grid.cell_arrays["GrainID"] = rve.GrainID
        grid.cell_arrays["phaseID"] = rve.phaseID

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

                current_lines = [*zip(current_lines.p1, current_lines.p2)]
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

        sorted_tuples = [(*zip(line_df.p1, line_df.p2))]
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

    def run_mesher_2D(self):
        mesh2d = self.gen_blocks()
        mesh2d = self.gen_grains(mesh2d)

        x_max = max(mesh2d.points[:, 0])
        y_max = max(mesh2d.points[:, 1])
        tri_mesh2d = mesh2d.triangulate()
        all_line_points = pd.DataFrame()
        all_lines = pd.DataFrame()
        for i in range(self.n_grains):
            submesh2d = tri_mesh2d.extract_cells(np.where(tri_mesh2d.cell_arrays.values()[0] == i+1))
            edges = submesh2d.extract_feature_edges()
            tri_mesh2d_df = pd.DataFrame(tri_mesh2d.points, columns=['x', 'y', 'z'])
            tri_mesh2d_df['grid_index_total'] = tri_mesh2d_df.index
            tri_mesh2d_df.sort_values(by=['x', 'y', 'z'], inplace=True)
            tri_mesh2d_df.reset_index(drop=True, inplace=True)

            edges_df = pd.DataFrame(edges.points, columns=['x', 'y', 'z'])
            edges_df['grid_index_edges'] = edges_df.index
            edges_df.sort_values(by=['x', 'y', 'z'], inplace=True)
            edges_df.reset_index(drop=True, inplace=True)

            compare_all = tri_mesh2d_df.merge(edges_df, on=['x', 'y', 'z'], how='left', indicator=True)
            compare_grain = compare_all.loc[compare_all['_merge'] == 'both'].copy()

            compare_grain.sort_values(by=['grid_index_edges'], inplace=True)
            idx = compare_grain['grid_index_total'].tolist()
            compare_grain['GrainID'] = str(i+1)
            all_line_points = pd.concat([all_line_points, compare_grain[['grid_index_total', 'GrainID']]])

            lines = edges.lines.reshape((-1, 3))[:, 1:3]
            compare_grain.sort_values(by='grid_index_edges')
            compare_grain.reset_index(drop=True, inplace=True)
            lines_df = pd.DataFrame()
            lines_df['p1'] = compare_grain.iloc[lines[:, 0]].grid_index_total.values
            lines_df['p2'] = compare_grain.iloc[lines[:, 1]].grid_index_total.values
            lines_df['GrainID'] = str(i+1)

            all_lines = pd.concat([all_lines, lines_df])
            #tri_mesh2d.points[idx] = smooth.points

        all_line_points.reset_index(drop=True, inplace=True)
        all_lines.reset_index(drop=True, inplace=True)

        all_line_points = self.gen_point_labels(all_line_points)
        all_lines = self.gen_line_labels(all_lines)
        """for i in range(len(all_lines)):
            #print(tri_mesh2d.points[all_lines.p1[0]])
            #sys.exit()
            plt.plot(tri_mesh2d.points[all_lines.p1[i]][0], tri_mesh2d.points[all_lines.p1[i]][1], 'o')
            plt.plot(tri_mesh2d.points[all_lines.p2[i]][0], tri_mesh2d.points[all_lines.p2[i]][1], 'o')
        plt.show()
        sys.exit()"""

        print(all_lines)

        line_labels = np.asarray(all_lines['line_labels'])
        line_labels = [item.split(',') for item in line_labels]
        line_labels = np.asarray([[int(ll) for ll in group] for group in line_labels])
        line_labels_tuples = [tuple(ll) for ll in line_labels]
        all_lines['line_labels'] = line_labels_tuples
        print(all_lines)
        print(line_labels)
        print(line_labels.shape)

        smooth = self.laplace_2D(all_lines, tri_mesh2d.points, alpha=0.25, n_iter=1)
        tri_mesh2d.points = smooth

        return tri_mesh2d

class BuildAbaqus2D:

    def __init__(self, pv_mesh, rve_df, grains_df: pd.DataFrame, store_path, phase_two_isotropic=True):
        self.mesh = pv_mesh
        self.rve_df = rve_df
        print(rve_df)

        self.store_path = store_path
        self.phase_two_isotropic = phase_two_isotropic
        self.n_grains = int(max(pv_mesh.cell_arrays['GrainID']))

        self.bin_size = rve_df.box_size[0] / (rve_df.n_pts[0]+1) ## test
        self.tex_phi1 = grains_df['phi1'].tolist()
        self.tex_PHI = grains_df['PHI'].tolist()
        self.tex_phi2 = grains_df['phi2'].tolist()


    #def build_abaqus_model(self, poly_data: pv.PolyData, rve: pv.UniformGrid,
    #                       fl: list, tri_df: pd.DataFrame = pd.DataFrame()) -> None:
    def build_abaqus_header(self) -> None:
        """us model here so far only single phase supported
        for dual or multiple phase material_def needs to be adjusted"""

        #fl_df = pd.DataFrame(fl)
        #tri = tri_df.drop(['facelabel', 'sorted_tris'], axis=1)
        #tri = np.asarray(tri)
        #smooth_points = poly_data.points

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
        f.write('*Include, input=Edges.inp\n')
        f.write('*Include, input=Corners.inp\n')
        f.write('*Include, input=VerticeSets.inp\n')
        f.write('*End Assembly\n')
        f.close()

    def build_nodes_and_elements(self) -> None:
        tris = self.mesh.extract_surface().faces
        tris = tris.reshape((-1, 4))[:, 1:4]

        node_dict = {'x': self.mesh.points[:, 0], 'y': self.mesh.points[:, 1]}
        elem_dict = {'p1': tris[:, 0], 'p2': tris[:, 1], 'p3': tris[:, 2]}
        abaq_nodes_df = pd.DataFrame(data=node_dict)
        abaq_elem_df = pd.DataFrame(data=elem_dict)

        f = open(self.store_path + '/RVE_smooth.inp', 'a')
        f.write('*Part, name=PART-1\n')
        f.write('*NODE\n')
        for i in range(len(abaq_nodes_df)):
            line = '{}, {}, {}\n'.format(abaq_nodes_df.index[i]+1,
                                         abaq_nodes_df.x[i],
                                         abaq_nodes_df.y[i])
            f.write(line)

        f.write('*ELEMENT, TYPE=CPE3\n')
        for i in range(len(abaq_elem_df)):
            line = '{}, {}, {}, {}\n'.format(abaq_elem_df.index[i] + 1,
                                         abaq_elem_df.p1[i]+1,
                                         abaq_elem_df.p2[i]+1,
                                         abaq_elem_df.p3[i]+1)
            f.write(line)
        f.close()

    def make_meshio_inp_file(self):
        pv.save_meshio(self.store_path + '/rve-part.inp',self.mesh)
        f = open(self.store_path + '/rve-part.inp', 'r')
        lines = f.readlines()
        f.close()
        startingLine = lines.index('*NODE\n')
        f = open(self.store_path + '/RVE_smooth.inp', 'a')
        f.write('*Part, name=PART-1\n')
        idx = [i for i, s in enumerate(lines) if '*element' in s.lower()][0]
        lines[idx] = '*ELEMENT, TYPE=CPE3\n'
        for line in lines[startingLine:]:
            f.write(line)
        f.close()


    def generate_elementsets(self):
        f = open(self.store_path + '/RVE_smooth.inp', 'a')
        for i in range(self.n_grains):
            nGrain = i + 1
            print('in for nGrain=', nGrain, self.mesh.cell_arrays.keys())
            cells = np.where(self.mesh.cell_arrays['GrainID'] == nGrain)[0]
            f.write('*Elset, elset=Set-{}\n'.format(nGrain))
            for j, cell in enumerate(cells + 1):
                if (j + 1) % 16 == 0:
                    f.write('\n')
                f.write(' {},'.format(cell))
            f.write('\n')
        f.close()

    def assign_materials(self):
        f = open(self.store_path + '/RVE_smooth.inp', 'a')
        phase1_idx = 0
        phase2_idx = 0
        for i in range(self.n_grains):
            nGrain = i + 1
            if self.rve_df.loc[self.rve_df['GrainID'] == nGrain].phaseID.values[0] == 1:
                phase1_idx += 1
                f.write('** Section: Section - {}\n'.format(nGrain))
                f.write('*Solid Section, elset=Set-{}, material=Ferrite_{}\n'.format(nGrain, phase1_idx))
            elif self.rve_df.loc[self.rve_df['GrainID'] == nGrain].phaseID.values[0] == 2:
                if not self.phase_two_isotropic:
                    phase2_idx += 1
                    f.write('** Section: Section - {}\n'.format(nGrain))
                    f.write('*Solid Section, elset=Set-{}, material=Martensite_{}\n'.format(nGrain, phase2_idx))
                else:
                    f.write('** Section: Section - {}\n'.format(nGrain))
                    f.write('*Solid Section, elset=Set-{}, material=Martensite\n'.format(nGrain))

        f.close()

    def pbc(self, grid_hull_df: pd.DataFrame) -> None:

        """function to define the periodic boundary conditions
        if errors appear or equations are wrong check ppt presentation from ICAMS
        included in the docs folder called PBC_docs"""

        min_x = min(grid_hull_df.x)
        min_y = min(grid_hull_df.y)

        max_x = max(grid_hull_df.x)
        max_y = max(grid_hull_df.y)

        print(min_x, min_y)
        print(max_x, max_y)
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
                                     ((grid_hull_df['y'] == max_y) | (grid_hull_df['y'] == min_y))]

        V1_df = corner_df.loc[(corner_df['x'] == min_x) & (corner_df['y'] == min_y)]
        V1 = V1_df['pointNumber'].values[0]
        V1Eqn = V1_df['Eqn-Set'].values[0]
        #print(V1_df)
        V2_df = corner_df.loc[(corner_df['x'] == max_x) & (corner_df['y'] == min_y)]
        V2 = V2_df['pointNumber'].values[0]
        V2Eqn = V2_df['Eqn-Set'].values[0]
        #print(V2_df)
        V3_df = corner_df.loc[(corner_df['x'] == max_x) & (corner_df['y'] == max_y)]
        V3 = V3_df['pointNumber'].values[0]
        V3Eqn = V3_df['Eqn-Set'].values[0]
        #print(V3_df)
        V4_df = corner_df.loc[(corner_df['x'] == min_x) & (corner_df['y'] == max_y)]
        V4 = V4_df['pointNumber'].values[0]
        V4Eqn = V4_df['Eqn-Set'].values[0]
        #print(V4_df)

        ############ Define Edge Sets ###############
        edges_df = grid_hull_df.loc[(((grid_hull_df['x'] == max_x) | (grid_hull_df['x'] == min_x)) &
                                     ((grid_hull_df['y'] != max_y) & (grid_hull_df['y'] != min_y))) |

                                    (((grid_hull_df['x'] != max_x) & (grid_hull_df['x'] != min_x)) &
                                     ((grid_hull_df['y'] == max_y) | (grid_hull_df['y'] == min_y)))]
        # edges_df.sort_values(by=['x', 'y', 'z'], inplace=True)

        # Top Edge
        Top = edges_df.loc[(edges_df['y'] == max_y)]['Eqn-Set'].to_list()

        # Right Edge
        Right = edges_df.loc[(edges_df['x'] == max_x)]['Eqn-Set'].to_list()

        # Bottom Edge
        Bottom = edges_df.loc[(edges_df['y'] == min_y)]['Eqn-Set'].to_list()

        # Left Edge
        Left = edges_df.loc[(edges_df['x'] == min_x)]['Eqn-Set'].to_list()

        #self.logger.info('E1 ' + str(len(Top)))
        #self.logger.info('E2 ' + str(len(Right)))
        #self.logger.info('E3 ' + str(len(Bottom)))
        #self.logger.info('E4 ' + str(len(Left)))


        OutPutFile = open(self.store_path + '/Nsets.inp', 'w')
        for i in grid_hull_df.index:
            OutPutFile.write('*Nset, nset=Eqn-Set-{}, instance=PART-1-1\n'.format(i + 1))
            OutPutFile.write(' {},\n'.format(int(grid_hull_df.loc[i]['pointNumber'] + 1)))
        OutPutFile.close()

        ############### Define Equations ###################################

        OutPutFile = open(self.store_path + '/Edges.inp', 'w')

        # Edges in x-y Plane
        # right top edge to left top edge
        OutPutFile.write('**** X-DIR \n')
        for i in range(len(Top)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(Top[i] + 1) + ',1, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(Bottom[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')

        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(Left)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(Left[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(Right[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')

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
        ####################################################################

    def write_material_def(self) -> None:

        """simple function to write material definition in Input file
        needs to be adjusted for multiple phases"""
        phase1_idx = 0
        phase2_idx = 0
        numberofgrains = self.n_grains

        phase = [self.rve_df.loc[self.rve_df['GrainID'] == i].phaseID.values[0] for i in range(1, numberofgrains+1)]
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
        phase = [self.rve_df.loc[self.rve_df['GrainID'] == i].phaseID.values[0] for i in range(1, numberofgrains + 1)]
        grainsize = [np.cbrt(self.rve_df.loc[self.rve_df['GrainID'] == i].shape[0] *
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

    def run(self):

        self.build_abaqus_header()
        #self.build_nodes_and_elements()
        self.make_meshio_inp_file()
        self.generate_elementsets()
        self.assign_materials()
        self.make_assembly()
        '''this grid_hull_df needs to be removed if the upper block is used for the independent parts'''
        grid_hull_df = pd.DataFrame(self.mesh.points.tolist(), columns=['x', 'y', 'z'])
        max_x = max(grid_hull_df.x)
        min_x = min(grid_hull_df.x)
        max_y = max(grid_hull_df.y)
        min_y = min(grid_hull_df.y)
        grid_hull_df = grid_hull_df.loc[(grid_hull_df['x'] == max_x) | (grid_hull_df['x'] == min_x) |
                                        (grid_hull_df['y'] == max_y) | (grid_hull_df['y'] == min_y)]
        self.pbc(grid_hull_df)
        self.write_material_def()  # functions here
        self.write_step_def()  # it will lead to a faulty inputfile
        self.write_grain_data()

        plotter = pv.Plotter()
        plotter.add_mesh(self.mesh, show_edges=True, scalars='GrainID', cmap='flag')
        plotter.show()

if __name__ == '__main__':
    mesh_path = '../mesh.msh'
    grains_df_path = '../grains_df.csv'
    rve_df_path = '../test_rve.csv'

    mesh = pv.read_meshio(mesh_path)
    grains_df = pd.read_csv(grains_df_path)
    rve_df = pd.read_csv(rve_df_path)

    mesher_obj = Mesher_2D(rve_df, grains_df,store_path='./', animation=False)
    smooth_mesh= mesher_obj.run_mesher_2D()

    p = pv.Plotter()
    p.add_mesh(smooth_mesh, scalars='GrainID', show_edges=True)
    p.show()

    #abaqus_obj = BuildAbaqus2D(mesh, rve_df_path, store_path='./')
    #abaqus_obj.build_abaqus_header()
    #abaqus_obj.build_nodes_and_elements()
    #abaqus_obj.generate_elementsets()
    #abaqus_obj.assign_materials()
    #abaqus_obj.make_assembly()
    '''this grid_hull_df needs to be removed if the upper block is used for the independent parts'''
    #grid_hull_df = pd.DataFrame(mesh.points.tolist(), columns=['x', 'y', 'z'])
    #max_x = max(grid_hull_df.x)
    #min_x = min(grid_hull_df.x)
    #max_y = max(grid_hull_df.y)
    #min_y = min(grid_hull_df.y)
    #grid_hull_df = grid_hull_df.loc[(grid_hull_df['x'] == max_x) | (grid_hull_df['x'] == min_x) |
    #                                (grid_hull_df['y'] == max_y) | (grid_hull_df['y'] == min_y)]
    #abaqus_obj.pbc(grid_hull_df)
    #abaqus_obj.write_material_def()  # functions here
    #abaqus_obj.write_step_def()  # it will lead to a faulty inputfile
    #abaqus_obj.write_grain_data()

    #plotter = pv.Plotter()
    #plotter.add_mesh(mesh, show_edges=True, scalars='GrainID', cmap='flag')
    #plotter.show()