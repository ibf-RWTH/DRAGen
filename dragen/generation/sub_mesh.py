# _*_ coding: utf-8 _*_
"""
Time:     2021/8/1 16:37
Author:   Linghao Kong
Version:  V 0.1
File:     sub_mesh
Describe: Write during the internship at IEHK RWTH"""
import pandas as pd
from dragen.generation.mesher import Mesher
import pyvista as pv
import numpy as np
import tetgen
import datetime
import logging
class Sub_Mesher(Mesher):

    def __init__(self, rve: pd.DataFrame, grains_df: pd.DataFrame, store_path,
                 phase_two_isotropic=True, animation=True, infobox_obj=None, progress_obj=None, gui=True, elem='C3D4'):
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
        self.n_packets = int(max(rve.packet_id))  # add
        self.n_blocks = int(max(rve.block_id))  # add
        self.n_pts = int(rve['n_pts'].iloc[0])
        self.bin_size = rve['box_size'].iloc[0] / (self.n_pts + 1)  ## test
        self.logger = logging.getLogger("RVE-Gen")
        self.gui = gui
        self.elem = elem
        self.n_materials = 0

        super().__init__(rve,grains_df,store_path,phase_two_isotropic, animation, infobox_obj, progress_obj, gui, elem)

    def gen_substruct(self):
        grid = self.gen_blocks()
        grid = self.gen_grains(grid)

        rve = self.rve
        rve.sort_values(by=['x', 'y', 'z'], inplace=True)
        grid.cell_arrays['packet_id'] = rve.packet_id  # assign nodes with values in dict way
        grid.cell_arrays['block_id'] = rve.block_id

        if self.animation:
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(grid, scalars='packet_id', stitle='block_ids',
                             show_edges=True,
                             interpolate_before_map=True)  # different packet_id attribute of cell_arrays given different color
            plotter.add_axes()
            plotter.show(interactive=True, auto_close=True, window_size=[800, 600],
                         screenshot='./Figs/pyvista_Hex_Mesh_blocks.png')
            plotter.close()

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(grid, scalars='block_id', stitle='block_ids',
                             show_edges=True,
                             interpolate_before_map=True)  # different block_id attribute of cell_arrays given different color
            plotter.add_axes()
            plotter.show(interactive=True, auto_close=True, window_size=[800, 600],
                         screenshot='./Figs/pyvista_Hex_Mesh_blocks.png')
            plotter.close()

        return grid


    def subs_to_mesh(self, grid):
        all_points = grid.points
        all_points_df = pd.DataFrame(all_points, columns=['x', 'y', 'z'], dtype=float)
        all_points_df.sort_values(by=['x', 'y', 'z'], inplace=True)

        grainboundary_df = pd.DataFrame()
        for i in range(1, self.n_blocks + 1):
            sub_grid = grid.extract_cells(np.where(np.asarray(grid.cell_arrays.values())[3] == i))
            sub_surf = sub_grid.extract_surface()
            sub_surf.triangulate(inplace=True)

            p = sub_surf.points
            p_df = pd.DataFrame(p, columns=['x', 'y', 'z'], dtype=float)
            p_df.sort_values(by=['x', 'y', 'z'], inplace=True)
            compare_all = all_points_df.merge(p_df, on=['x', 'y', 'z'], how='left', indicator=True)
            compare_grain = compare_all.loc[compare_all['_merge'] == 'both'].copy()  # find the corresponding data fast

            compare_grain.reset_index(inplace=True)
            compare_grain['block_idx'] = p_df.index
            compare_grain.sort_values(by=['block_idx'], inplace=True)

            faces = sub_surf.faces
            faces = np.reshape(faces, (int(len(faces) / 4), 4))  # number of nodes, nodes idx
            f_df = pd.DataFrame(faces, columns=['nnds', 'p1', 'p2', 'p3'])
            idx = np.asarray(compare_grain['index'])  # reindex from whole rve
            f_df['p1'] = [idx[j] for j in f_df['p1'].values]
            f_df['p2'] = [idx[j] for j in f_df['p2'].values]
            f_df['p3'] = [idx[j] for j in f_df['p3'].values]
            f_df['facelabel'] = str(i)

            grainboundary_df = pd.concat([grainboundary_df, f_df])

        sorted_tuple = [[grainboundary_df.p1.values[i],
                         grainboundary_df.p2.values[i],
                         grainboundary_df.p3.values[i]]
                        for i in range(len(grainboundary_df))]

        sorted_tuple = [sorted(item) for item in sorted_tuple]  # sorted nodes in each face
        sorted_tuple = [tuple(item) for item in sorted_tuple]

        grainboundary_df['sorted_tris'] = sorted_tuple
        unique_grainboundary_df = grainboundary_df.drop_duplicates(subset=['sorted_tris'], keep='first')

        all_faces = unique_grainboundary_df.drop(['sorted_tris', 'facelabel'], axis=1)
        all_faces = np.array(all_faces, dtype='int32')
        all_faces = np.reshape(all_faces, (1, int(len(all_faces) * 4)))[0]  # reform in shape of polydata
        boundaries = pv.PolyData(all_points, all_faces)

        return boundaries, grainboundary_df


    def build_abaqus_model(self, poly_data: pv.PolyData, rve: pv.UniformGrid,
                          fl: list, tri_df: pd.DataFrame = pd.DataFrame()) -> None:
        """building the mesh_practice model here so far only single phase supported
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

        x_min = min(rve.x)
        y_min = min(rve.y)
        z_min = min(rve.z)
        x_max = max(rve.x)
        y_max = max(rve.y)
        z_max = max(rve.z)

        bid_list = []
        pid_list = []
        gid_list = []

        for i in range(self.n_blocks):

            bid = i + 1
            tri_idx = fl_df.loc[(fl_df[0] == bid) | (fl_df[1] == bid)].index
            triGrain = tri[tri_idx, :]
            faces = triGrain.astype('int32')
            sub_surf = pv.PolyData(smooth_points, faces)

            tet = tetgen.TetGen(sub_surf)

            if self.elem == 'C3D4':
                tet.tetrahedralize(order=1, mindihedral=10, minratio=1.5, supsteiner_level=1)
            elif self.elem == 'C3D10':
                tet.tetrahedralize(order=2, mindihedral=10, minratio=1.5, supsteiner_level=1)
            sub_grid = tet.grid  # grid 1


            """
            This following code block is only needed if all grains are generated as independent parts
            and are merged together later. Or if cohesive contact definitions are defined.
            A first attempt for cohesive contact defs led to convergence issues which is why this route
            wasn't followed any further
            """
            # grain_hull_df = pd.DataFrame(sub_grid.points.tolist(), columns=['x', 'y', 'z'])
            # grain_hull_df = gridPointsDf.loc[(gridPointsDf['x'] == x_max) | (gridPointsDf['x'] == x_min) |
            #                                (gridPointsDf['y'] == y_max) | (gridPointsDf['y'] == y_min) |
            #                                (gridPointsDf['z'] == z_max) | (gridPointsDf['z'] == z_min)]
            # grain_hull_df['GrainID'] = nGrain
            # grid_hull_df = pd.concat([grid_hull_df, grain_hull_df])

            ncells = sub_grid.n_cells
            print(i, ncells)
            blockIDList = [i + 1]
            packetIDList = [self.bid_to_pid(i + 1)]
            grainIDList = [self.bid_to_gid(i + 1)]
            blockID_array = blockIDList * ncells
            packetID_array = packetIDList * ncells
            grainID_array = grainIDList * ncells
            bid_list.extend(blockID_array)
            pid_list.extend(packetID_array)
            gid_list.extend(grainID_array)
            if i == 0:
                grid = sub_grid  # grid2
            else:
                grid = sub_grid.merge(grid)  # grid3
            grain_vol = sub_grid.volume
                # self.logger.info(str(grain_vol*10**9))
            self.grains_df.loc[self.grains_df['GrainID'] == i, 'meshed_conti_volume'] = grain_vol * 10 ** 9

        self.grains_df['phaseID'] -= 1
        self.grains_df.to_csv(self.store_path + '/Generation_Data/grain_data_output_conti.csv', index=False)

        grid.cell_arrays['block_id'] = bid_list
        grid.cell_arrays['packet_id'] = pid_list
        grid.cell_arrays['GrainID'] = gid_list

        pv.save_meshio(self.store_path + '/rve-part.inp', grid)

        with open(self.store_path + '/rve-part.inp', 'r') as f:
            lines = f.readlines()

        startingLine = lines.index('*NODE\n')

        with open(self.store_path + '/RVE_smooth.inp', 'a') as f:

            f.write('*Part, name=PART-1\n')

            for line in lines[startingLine:]:
                f.write(line)

            for i in range(self.n_blocks):

                nBlock = i + 1
                cells = np.where(grid.cell_arrays['block_id'] == nBlock)[0]

                f.write('*Elset, elset=Set-Block{}\n'.format(nBlock))
                for j, cell in enumerate(cells + 1):
                    if (j + 1) % 16 == 0:
                        f.write('\n')
                    f.write(' {},'.format(cell))
                f.write('\n')

            for i in range(self.n_packets):

                nPacket = i + 1
                cells = np.where(grid.cell_arrays['packet_id'] == nPacket)[0]

                f.write('*Elset, elset=Set-Packet{}\n'.format(nPacket))
                for j, cell in enumerate(cells + 1):
                    if (j + 1) % 16 == 0:
                        f.write('\n')
                    f.write(' {},'.format(cell))
                f.write('\n')

            for i in range(self.n_grains):

                nGrain = i + 1
                cells = np.where(grid.cell_arrays['GrainID'] == nGrain)[0]

                f.write('*Elset, elset=Set-Grain{}\n'.format(nGrain))
                for j, cell in enumerate(cells + 1):
                    if (j + 1) % 16 == 0:
                        f.write('\n')
                    f.write(' {},'.format(cell))
                f.write('\n')

            for i in range(self.n_blocks):
                nBlock = i + 1
                if self.rve.loc[rve['block_id'] == nBlock].phaseID.values[0] == 1:

                    f.write('** Section: Section - Block{}\n'.format(nBlock))
                    f.write('*Solid Section, elset=Set-Block{}, material=Material_{}\n'.format(nBlock, nBlock))
                elif self.rve.loc[rve['block_id'] == nBlock].phaseID.values[0] == 2:
                    if not self.phase_two_isotropic:

                        f.write('** Section: Section - Block{}\n'.format(nBlock))
                        f.write('*Solid Section, elset=Set-Block{}, material=Material_{}\n'.format(nBlock, nBlock))
                    else:
                        f.write('** Section: Section - Block{}\n'.format(nBlock))
                        f.write('*Solid Section, elset=Set-Block{}, material=Material\n'.format(nBlock))

        grid_hull_df = pd.DataFrame(grid.points.tolist(), columns=['x', 'y', 'z'])
        grid_hull_df = grid_hull_df.loc[(grid_hull_df['x'] == x_max) | (grid_hull_df['x'] == x_min) |
                                        (grid_hull_df['y'] == y_max) | (grid_hull_df['y'] == y_min) |
                                        (grid_hull_df['z'] == z_max) | (grid_hull_df['z'] == z_min)]

        self.make_assembly()  # Don't change the order
        self.pbc(rve, grid_hull_df)  # of these four
        self.write_material()
        self.write_step_def()  # it will lead to a faulty inputfile
        self.write_block_data()



    def write_material(self):
        # phase = [self.rve.loc[self.rve['GrainID'] == i].phaseID.values[0] for i in range(1, numberofsub + 1)]
        with open(self.store_path + '/RVE_smooth.inp', 'a') as f:
            f.write('**\n')
            f.write('** MATERIALS\n')
            f.write('**\n')
            for i in range(self.n_blocks):
                f.write('*Material, name=Material_{}\n'.format(i + 1))
                f.write('*Depvar\n')
                f.write('    176,\n')
                f.write('*User Material, constants=2\n')
                f.write('{}.,3.\n'.format(i + 1))


    def write_block_data(self):
        f = open(self.store_path + '/graindata.inp', 'w+')
        f.write('!MMM Crystal Plasticity Input File\n')
        phase1_idx = 0
        blocks = self.rve.groupby('block_id').head(1)

        bt = blocks.apply(lambda block:block['block_thickness'],axis=1)
        bid = blocks.apply(lambda block: block['block_id'], axis=1)

        bid_to_bt = dict(zip(bid,bt))
        phase = [self.rve.loc[self.rve['block_id'] == i].phaseID.values[0] for i in range(1, self.n_blocks + 1)]
        grainsize = [bid_to_bt[i] for i in range(1, self.n_blocks + 1)]

        if 'phi1'in self.rve.head(0) and 'PHI' in self.rve.head(0) and 'phi2' in self.rve.head(0):
            for i in range(self.n_blocks):
                if not self.phase_two_isotropic:
                    phi1 = self.rve[self.rve['block_id'] == i+1]['phi1'].values[0]
                    PHI = self.rve[self.rve['block_id'] == i+1]['PHI'].values[0]
                    phi2 = self.rve[self.rve['block_id'] == i+1]['phi2'].values[0]
                    f.write('Material: {}: {}: {}: {}: {}\n'.format(i + 1, phi1, PHI, phi2, grainsize[i]))

                else:

                    if phase[i] == 1:
                        phase1_idx += 1
                        """phi1 = int(np.random.rand() * 360)
                        PHI = int(np.random.rand() * 360)
                        phi2 = int(np.random.rand() * 360)"""
                        phi1 = self.rve[self.rve['block_id'] == i + 1]['phi1'].values[0]
                        PHI = self.rve[self.rve['block_id'] == i + 1]['PHI'].values[0]
                        phi2 = self.rve[self.rve['block_id'] == i + 1]['phi2'].values[0]
                        f.write('Material: {}: {}: {}: {}: {}\n'.format(phase1_idx, phi1, PHI, phi2, grainsize[i]))
        else:
            angles = blocks.apply(lambda block: self.comp_angle(self.grains_df, block), axis=1)
            bid_to_angles = dict(zip(bid, angles))
            for i in range(self.n_blocks):
                angle = bid_to_angles[i + 1]
                if not self.phase_two_isotropic:

                    phi1 = angle[0]
                    PHI = angle[1]
                    phi2 = angle[2]
                    f.write('Material: {}: {}: {}: {}: {}\n'.format(i + 1, phi1, PHI, phi2, grainsize[i]))

                else:

                    if phase[i] == 1:
                        phase1_idx += 1
                        """phi1 = int(np.random.rand() * 360)
                        PHI = int(np.random.rand() * 360)
                        phi2 = int(np.random.rand() * 360)"""
                        phi1 = angle[0]
                        PHI = angle[1]
                        phi2 = angle[2]
                        f.write('Material: {}: {}: {}: {}: {}\n'.format(phase1_idx, phi1, PHI, phi2, grainsize[i]))

        f.close()

        # with open(self.store_path +'/neper_ori.txt','a+') as f:
        #     for i in range(self.n_blocks):
        #         angle = bid_to_angles[i + 1]
        #         phi1 = angle[0]
        #         PHI = angle[1]
        #         phi2 = angle[2]
        #         f.write('Material: {}: {}: {}: {}: \n'.format(i + 1, phi1, PHI, phi2))


    def bid_to_pid(self, bid):
        pid_list = self.rve.loc[self.rve['block_id'] == bid, 'packet_id']

        return int(pid_list.iloc[0])


    def bid_to_gid(self, bid):
        gid_list = self.rve.loc[self.rve['block_id'] == bid, 'GrainID']

        return int(gid_list.iloc[0])


    def comp_angle(self, grain_data, point_data):
        T_list = [np.array(0) for i in range(24)]

        T_list[0] = np.array([[0.742, 0.667, 0.075],
                              [0.650, 0.742, 0.167],
                              [0.167, 0.075, 0.983]])

        T_list[1] = np.array([[0.075, 0.667, -0.742],
                              [-0.167, 0.742, 0.650],
                              [0.983, 0.075, 0.167]])

        T_list[2] = np.array([[-0.667, -0.075, 0.742, ],
                              [0.742, -0.167, 0.650],
                              [0.075, 0.983, 0.167]])

        T_list[3] = np.array([[0.667, -0.742, 0.075],
                              [0.742, 0.650, -0.167],
                              [0.075, 0.167, 0.983]])

        T_list[4] = np.array([[-0.075, 0.742, -0.667],
                              [-0.167, 0.650, 0.742],
                              [0.983, 0.167, 0.075]])

        T_list[5] = np.array([[-0.742, 0.075, 0.667],
                              [0.650, -0.167, 0.742],
                              [0.167, 0.983, 0.075]])

        T_list[6] = np.array([[-0.075, 0.667, 0.742],
                              [-0.167, -0.742, 0.650],
                              [0.983, -0.075, 0.167]])

        T_list[7] = np.array([[-0.742, -0.667, 0.075],
                              [0.650, -0.742, -0.167],
                              [0.167, -0.075, 0.983]])

        T_list[8] = np.array([[0.742, 0.075, -0.667],
                              [0.650, 0.167, 0.742],
                              [0.167, -0.983, 0.075]])

        T_list[9] = np.array([[0.075, 0.742, 0.667],
                              [-0.167, -0.650, 0.742],
                              [0.983, -0.167, 0.075]])

        T_list[10] = np.array([[-0.667, -0.742, -0.075],
                               [0.742, -0.650, -0.167],
                               [0.075, -0.167, 0.983]])

        T_list[11] = np.array([[0.667, -0.075, -0.742],
                               [0.742, 0.167, 0.650],
                               [0.075, -0.983, 0.167]])

        T_list[12] = np.array([[0.667, 0.742, -0.075],
                               [-0.742, 0.650, -0.167],
                               [-0.075, 0.167, 0.983]])

        T_list[13] = np.array([[-0.667, 0.075, -0.742],
                               [-0.742, -0.167, 0.650],
                               [-0.075, 0.983, 0.167]])

        T_list[14] = np.array([[0.075, -0.667, 0.742],
                               [0.167, 0.742, 0.650],
                               [-0.983, 0.075, 0.167]])

        T_list[15] = np.array([[0.742, 0.667, 0.075],
                               [-0.650, 0.742, -0.167],
                               [-0.167, 0.075, 0.983]])

        T_list[16] = np.array([[-0.742, 0.075, -0.667],
                               [-0.650, -0.167, 0.742],
                               [-0.167, 0.983, 0.075]])

        T_list[17] = np.array([[-0.075, -0.742, 0.667],
                               [0.167, 0.650, 0.742],
                               [-0.983, 0.167, 0.075]])

        T_list[18] = np.array([[0.742, -0.075, 0.667],
                               [0.650, -0.167, -0.742],
                               [0.167, 0.983, -0.075]])

        T_list[19] = np.array([[0.075, -0.742, -0.667],
                               [-0.167, 0.650, -0.742],
                               [0.983, 0.167, -0.075]])

        T_list[20] = np.array([[-0.667, 0.742, 0.075],
                               [0.742, 0.650, 0.167],
                               [0.075, 0.167, -0.983]])

        T_list[21] = np.array([[0.667, 0.075, 0.742],
                               [0.742, -0.167, -0.650],
                               [0.075, 0.983, -0.167]])

        T_list[22] = np.array([[-0.075, -0.667, -0.742],
                               [-0.167, 0.742, -0.650],
                               [0.983, 0.075, -0.167]])

        T_list[23] = np.array([[-0.742, 0.667, -0.075],
                               [0.650, 0.742, 0.167],
                               [0.167, 0.075, -0.983]])

        gid = point_data['GrainID']

        if point_data['phaseID'] == 1:
            return grain_data.loc[grain_data['GrainID'] == gid, 'phi1'].values[0], \
                   grain_data.loc[grain_data['GrainID'] == gid, 'PHI'].values[0], \
                   grain_data.loc[grain_data['GrainID'] == gid, 'phi2'].values[0]

        if point_data['phaseID'] == 2:

            try:
                i = int(str(point_data['block_orientation']).lstrip('V')) - 1

            except:
                i = np.random.randint(24)  # needs modification

            T = T_list[i]
            phi1 = grain_data.loc[grain_data['GrainID'] == gid, 'phi1'].values[0]
            PHI = grain_data.loc[grain_data['GrainID'] == gid, 'PHI'].values[0]
            phi2 = grain_data.loc[grain_data['GrainID'] == gid, 'phi2'].values[0]

            R1 = np.array([[np.cos(np.deg2rad(phi1)), -np.sin(np.deg2rad(phi1)), 0],
                           [np.sin(np.deg2rad(phi1)), np.cos(np.deg2rad(phi1)), 0],
                           [0, 0, 1]])

            R2 = np.array([[1, 0, 0],
                           [0, np.cos(np.deg2rad(PHI)), -np.sin(np.deg2rad(PHI))],
                           [0, np.sin(np.deg2rad(PHI)), np.cos(np.deg2rad(PHI))]])

            R3 = np.array([[np.cos(np.deg2rad(phi2)), -np.sin(np.deg2rad(phi2)), 0],
                           [np.sin(np.deg2rad(phi2)), np.cos(np.deg2rad(phi2)), 0],
                           [0, 0, 1]])

            result = np.dot(R3, R2)
            R = np.matrix(np.dot(result, R1))

            RB = T * R
            N, n, n1, n2 = 1, 1, 1, 1

            if RB[2, 2] > 1:
                N = 1 / RB[2, 2]

            if RB[2, 2] < -1:
                N = -1 / RB[2, 2]

            RB[2, 2] = N * RB[2, 2]
            PHIB = np.degrees(np.arccos(RB[2, 2]))
            sin_PHIB = np.sin(np.deg2rad(PHIB))

            if RB[2, 0] / sin_PHIB > 1 or RB[2, 0] / sin_PHIB < -1:
                n1 = sin_PHIB / RB[2, 0]

            if RB[0, 2] / sin_PHIB > 1 or RB[0, 2] / sin_PHIB < -1:
                n2 = sin_PHIB / RB[0, 2]

            if abs(n1) > abs(n2):

                n = n2

            else:

                n = n1

            # recalculate after scaling
            RB = N * n * RB
            PHIB = np.degrees(np.arccos(RB[2, 2]))
            if PHIB < 0:
                PHIB = PHIB + 360
            sin_PHIB = np.sin(np.deg2rad(PHIB))
            phi1B = np.degrees(np.arcsin(RB[2, 0] / sin_PHIB))
            if phi1B < 0:
                phi1B = phi1B + 360
            phi2B = np.degrees(np.arcsin(RB[0, 2] / sin_PHIB))
            if phi2B < 0:
                phi2B = phi2B + 360

            return phi1B, PHIB, phi2B

    # def find_material(self):
    #     rve = self.rve
    #     phase_groups = rve.groupby('phaseID')
    #     martensite_group = phase_groups.get_group(2)
    #     bid_list = martensite_group.loc[martensite_group['block_orientation'].isnull(), 'block_id']
    #     # fill in nan
    #     for bid in bid_list:
    #         i = 1
    #         bv = martensite_group.loc[martensite_group['block_id'] == bid, 'block_orientation']
    #         while (bv.isnull()).any():
    #             try:
    #
    #                 bv = martensite_group.loc[martensite_group['block_id'] == bid - 2 * i, 'block_orientation']
    #
    #             except:
    #
    #                 bv = martensite_group.loc[martensite_group['block_id'] == bid + 2 * i, 'block_orientation']
    #
    #             i += 1
    #
    #         rve.loc[rve['block_id'] == bid, 'block_orientation'] = bv.iloc[0]
    #
    #     groups1 = rve[rve['phaseID'] == 2].groupby(['GrainID', 'block_orientation'])
    #     kl = list(groups1.groups.keys())
    #     bid_list = []
    #     mid_list = []
    #     n = 0
    #     for key in kl:
    #         n += 1
    #         group = groups1.get_group(key)
    #         group_bid_list = list(set(group['block_id']))
    #         group_mid_list = [n for i in range(len(group_bid_list))]
    #
    #         bid_list.extend(group_bid_list)
    #         mid_list.extend(group_mid_list)
    #
    #     groups2 = rve[rve['phaseID'] == 1].groupby('GrainID')
    #     kl = list(groups2.groups.keys())
    #
    #     for key in kl:
    #         n += 1
    #         group = groups2.get_group(key)
    #         group_bid_list = list(set(group['block_id']))
    #         group_mid_list = [n for i in range(len(group_bid_list))]
    #
    #         bid_list.extend(group_bid_list)
    #         mid_list.extend(group_mid_list)
    #
    #     bid_to_mid = dict(zip(bid_list, mid_list))
    #
    #     return bid_to_mid, n

    def mesh_and_build_abaqus_model(self) -> None:
        if self.gui:
            self.progress_obj.emit(0)
            self.infobox_obj.emit('starting mesher')

        if self.gui:
            self.progress_obj.emit(25)
        GRID = self.gen_substruct()
        grain_boundaries_poly_data, tri_df = self.subs_to_mesh(GRID)
        if self.gui:
            self.progress_obj.emit(50)
        face_label = self.gen_face_labels(tri_df)
        smooth_grain_boundaries = self.smooth(grain_boundaries_poly_data, GRID, tri_df, face_label)
        if self.gui:
            self.progress_obj.emit(75)
        self.build_abaqus_model(rve=GRID, poly_data=smooth_grain_boundaries, fl=face_label, tri_df=tri_df)
        if self.gui:
            self.progress_obj.emit(100)

if __name__ == "__main__":
    rve = pd.read_csv('F:/pycharm/2nd_mini_thesis/mesh_practice/neper_model/neper_rve.csv')
    grains_df = pd.read_csv('F:/pycharm/2nd_mini_thesis/mesh_practice/final_input/01.08.2021/grains.csv')

    mesh = Sub_Mesher(rve, grains_df, 'F:/pycharm/2nd_mini_thesis/mesh_practice/neper_model/input2', phase_two_isotropic=False, animation=False, gui=False)
    mesh.mesh_and_build_abaqus_model()