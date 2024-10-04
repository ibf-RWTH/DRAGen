"""
Time:     2022/02/21
Authors:   Linghao Kong, Manuel Henrich
Version:  V 1.0
File:     SubMesher
Describe: Modification of Submesher V 0.1"""
import sys

import matplotlib.pyplot as plt
import pandas as pd
from dragen.generation.Mesher3D import AbaqusMesher
from dragen.utilities.InputInfo import RveInfo
from dragen.utilities.Helpers import HelperFunctions
import pyvista as pv
import numpy as np
import datetime
import os
import tetgen


class SubMesher(AbaqusMesher):
    """
    SubMesher inherits all the functions from AbaqusMesher and therefore also the functions from PvGridGeneration.py
    So, no need to take care of inheritance of MeshingHelper
    self.gen_blocks() is already available.
    """

    def __init__(self, rve_shape: tuple, rve: pd.DataFrame, subs_df: pd.DataFrame):
        super().__init__(rve_shape, rve, subs_df)

        self.rve = rve
        self.subs_df = subs_df
        self.rve.sort_values(by=['block_id'],inplace=True)
        sampled_blocks = self.rve.groupby('block_id').first()
        self.phi1 = sampled_blocks['phi1'].tolist()
        self.PHI = sampled_blocks['PHI'].tolist()
        self.phi2 = sampled_blocks['phi2'].tolist()
        self.bt_list = sampled_blocks['block_thickness'].to_list()
        self.x_max = int(max(rve.x))
        self.x_min = int(min(rve.x))
        self.y_max = int(max(rve.y))
        self.y_min = int(min(rve.y))
        self.z_max = int(max(rve.z))
        self.z_min = int(min(rve.z))
        self.n_blocks = int(max(rve.block_id))
        self.n_packets = int(max(rve.packet_id))
        self.n_grains = int(max(rve.GrainID))

        self.idnum = {1: 'GrainID', 2: 'packet_id', 3: 'block_id'}
        self.subsnum = {1: 'Grain', 2: 'Packet', 3: 'Block'}

    def gen_subs(self):
        if RveInfo.gui_flag:
            RveInfo.progress_obj.emit(0)
            RveInfo.infobox_obj.emit('starting mesher')
        grid = self.gen_blocks()
        if RveInfo.gui_flag:
            RveInfo.progress_obj.emit(25)
        grid = self.gen_grains(grid)

        self.rve.sort_values(by=['z','y','x'], inplace=True)
        grid.cell_data['packet_id'] = self.rve['packet_id'].to_numpy()
        grid.cell_data['block_id'] = self.rve['block_id'].to_numpy()
        print("block id are ")
        bids = list(set(grid.cell_data['block_id']))
        bids = sorted(bids)
        bid_dict = dict()
        for bid in bids:
            print(bid)
            bid_dict[bid] = grid.cell_data
        #sys.exit()
        if RveInfo.anim_flag:
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(grid, scalars='packet_id',
                             show_edges=True, interpolate_before_map=True)
            plotter.add_axes()
            plotter.show(interactive=True, auto_close=True, window_size=[800, 600],
                         screenshot=RveInfo.store_path + '/Figs/pyvista_Hex_Mesh_packets.png')
            plotter.close()

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(grid, scalars='block_id',
                             show_edges=True, interpolate_before_map=True)
            plotter.add_axes()
            plotter.show(interactive=True, auto_close=True, window_size=[800, 600],
                         screenshot=RveInfo.store_path + '/Figs/pyvista_Hex_Mesh_blocks.png')
            plotter.close()

        return grid

    def write_substruct_material_def(self) -> None:
        numberofblocks = self.n_blocks

        phase = [self.rve.loc[self.rve['block_id'] == i].phaseID.values[0] for i in range(1, numberofblocks + 1)]
        f = open(RveInfo.store_path + '/Materials.inp',
                 'w+')  # open in write mode to overwrite old files in case ther are any
        f.write('** MATERIALS\n')
        f.write('**\n')
        if RveInfo.phase2iso_flag[1] and RveInfo.phase_ratio[1] > 0:
            f.write('**\n')
            f.write('*Include, Input=Ferrite.inp\n')
            ff = open(RveInfo.store_path + '/Ferrite.inp', 'w+') 
            ff.write('** MATERIALS\n')
            ff.write('**\n')
            ff.write('*Material, name=Ferrite\n')
            ff.write('*Elastic\n')
            ff.write('210000, 0.3')
            ff.close()
        if RveInfo.phase2iso_flag[2] and RveInfo.phase_ratio[2] > 0:
            f.write('**\n')
            f.write('*Include, Input=Martensite.inp\n')
            ff = open(RveInfo.store_path + '/Martensite.inp', 'w+') 
            ff.write('** MATERIALS\n')
            ff.write('**\n')
            ff.write('*Material, name=Martensite\n')
            ff.write('*Elastic\n')
            ff.write('210000, 0.3')
            ff.close()
        if RveInfo.phase2iso_flag[3] and RveInfo.phase_ratio[3] > 0:
            f.write('**\n')
            f.write('*Include, Input=Pearlite.inp\n')
            ff = open(RveInfo.store_path + '/Pearlite.inp', 'w+') 
            ff.write('** MATERIALS\n')
            ff.write('**\n')
            ff.write('*Material, name=Pearlite\n')
            ff.write('*Elastic\n')
            ff.write('210000, 0.3')
            ff.close()
        if RveInfo.phase2iso_flag[4] and RveInfo.phase_ratio[4] > 0:
            f.write('**\n')
            f.write('*Include, Input=Bainite.inp\n')
            ff = open(RveInfo.store_path + '/Bainite.inp', 'w+') 
            ff.write('** MATERIALS\n')
            ff.write('**\n')
            ff.write('*Material, name=Bainite\n')
            ff.write('*Elastic\n')
            ff.write('210000, 0.3')
            ff.close()
        if RveInfo.phase2iso_flag[5] and RveInfo.phase_ratio[5] > 0:
            f.write('**\n')
            f.write('*Include, Input=Austenite.inp\n')
            ff = open(RveInfo.store_path + '/Austenite.inp', 'w+') 
            ff.write('** MATERIALS\n')
            ff.write('**\n')
            ff.write('*Material, name=Austenite\n')
            ff.write('*Elastic\n')
            ff.write('210000, 0.3')
            ff.close()
        if RveInfo.phase_ratio[6] > 0:
            f.write('**\n')
            f.write('*Include, Input=Inclusions.inp\n')
            ff = open(RveInfo.store_path + '/Inclusions.inp', 'w+') 
            ff.write('** MATERIALS\n')
            ff.write('**\n')
            ff.write('*Material, name=Inclusions\n')
            ff.write('*Elastic\n')
            ff.write('210000, 0.3')
            ff.close()
            # add inclusion
        f.close()
        
        for i in range(numberofblocks):
            HelperFunctions.write_material_helper(i, phase, self.grains_df)

    def write_block_data(self) -> None:
        f = open(RveInfo.store_path + '/graindata.inp', 'w+')
        f.write('!MMM Crystal Plasticity Input File\n')
        phase1_idx = 0
        numberofblocks = self.n_blocks
        phase = [self.rve.loc[self.rve['block_id'] == i].phaseID.values[0] for i in range(1, numberofblocks + 1)]
        #grainsize = [self.rve.loc[self.rve['block_id']==i,'block_thickness'] for i in range(1, numberofblocks + 1)]
        #only correct for pure martensite
        #for non martensite phase grainsize should be the diamater of grains or something, or define block thickness of other
        #phase as its grain size in the future
        for i in range(numberofblocks):
            nblock = i + 1
            if not RveInfo.phase2iso_flag:  #[]?(12-03-2024)
                """phi1 = int(np.random.rand() * 360)
                PHI = int(np.random.rand() * 360)
                phi2 = int(np.random.rand() * 360)"""
                phi1 = self.phi1[i]
                PHI = self.PHI[i]
                phi2 = self.phi2[i]
                f.write('Grain: {}: {}: {}: {}: {}\n'.format(nblock, phi1, PHI, phi2, self.bt_list[i]))
            else:
                if phase[i] == 1:
                    phase1_idx += 1
                    """phi1 = int(np.random.rand() * 360)
                    PHI = int(np.random.rand() * 360)
                    phi2 = int(np.random.rand() * 360)"""
                    phi1 = self.phi1[i]
                    PHI = self.PHI[i]
                    phi2 = self.phi2[i]
                    f.write('Grain: {}: {}: {}: {}: {}\n'.format(phase1_idx, phi1, PHI, phi2, self.bt_list[i]))
        f.close()

    def bid_to_pid(self, bid):
        pid_list = self.rve.loc[self.rve['block_id'] == bid, 'packet_id']

        return int(pid_list.iloc[0])

    def bid_to_gid(self, bid):
        gid_list = self.rve.loc[self.rve['block_id'] == bid, 'GrainID']

        return int(gid_list.iloc[0])

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
        numberOfBlocks = self.n_blocks

        bid_list = list()
        gid_list = list()
        pid_list = list()
        pak_id_list = list()

        ######################################
        if RveInfo.element_type != 'C3D8' and RveInfo.element_type != 'HEX8':
            old_grid = grid.copy()
            grid_tet = pv.UnstructuredGrid()
            for i in range(1, numberOfBlocks):
                phase = self.rve.loc[self.rve['block_id'] == i].phaseID.values[0]
                print(f"celldata:{old_grid.cell_data['block_id']}")

                print("print here block id {}".format(i),np.where(np.asarray(old_grid.cell_data['block_id'] == i)))
                grain_grid_tet = old_grid.extract_cells(np.asarray(old_grid.cell_data['block_id'] == i))
                grain_surf_tet = grain_grid_tet.extract_surface(pass_pointid=True, pass_cellid=True)
                grain_surf_tet.triangulate(inplace=True)
                if not grain_surf_tet.is_all_triangles:
                    grain_surf_tet.plot()
                    plt.show()
                tet = tetgen.TetGen(grain_surf_tet)
                if RveInfo.element_type == 'C3D4':
                    tet.tetrahedralize(order=1, mindihedral=10, minratio=1.5, supsteiner_level=0, steinerleft=0)
                elif RveInfo.element_type == 'C3D10':
                    tet.tetrahedralize(order=2, mindihedral=10, minratio=1.5, supsteiner_level=0, steinerleft=0)
                tet_grain_grid = tet.grid
                ncells = tet_grain_grid.n_cells

                if RveInfo.gui_flag:
                    RveInfo.progress_obj.emit(75+(100*(i+1)/self.n_blocks/4))

                blockIDList = [i]
                packetIDList = [self.bid_to_pid(i)]
                grainIDList = [self.bid_to_gid(i)]
                blockID_array = blockIDList * ncells
                packetID_array = packetIDList * ncells
                grainID_array = grainIDList * ncells
                bid_list.extend(blockID_array)
                pak_id_list.extend(packetID_array)
                gid_list.extend(grainID_array)

                phaseIDList = [phase]
                phaseID_array = phaseIDList * ncells
                pid_list.extend(phaseID_array)
                if i == 0:
                    grid_tet = tet_grain_grid
                else:
                    grid_tet = tet_grain_grid.merge(grid_tet, merge_points=True)

            grid_tet.cell_data["block_id"] = np.asarray(bid_list)
            grid_tet.cell_data["packet_id"] = np.asarray(pak_id_list)
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

        old_grid = grid.copy()  # copy doesn't copy the dynamically assigned new property...
        for i in range(1, numberOfBlocks + 1):
            phase = self.rve.loc[self.rve['block_id'] == i].phaseID.values[0]
            grain_grid = old_grid.extract_cells(np.where(old_grid.cell_data['block_id']==i))
            grain_surf = grain_grid.extract_surface()
            grain_surf_df = pd.DataFrame(data=grain_surf.points, columns=['x', 'y', 'z'])
            merged_pts_df = grain_surf_df.join(all_points_df_old.set_index(['x', 'y', 'z']), on=['x', 'y', 'z'])
            grain_surf_smooth = grain_surf.smooth(n_iter=200)
            smooth_pts_df = pd.DataFrame(data=grain_surf_smooth.points, columns=['x', 'y', 'z'])
            all_points_df.loc[merged_pts_df['ori_idx'], ['x', 'y', 'z']] = smooth_pts_df.values

        for i in range(1,self.n_grains+1):
            grain_grid = old_grid.extract_cells(np.where(old_grid.cell_data['GrainID'] == i))
            grain_vol = grain_grid.volume
            self.grains_df.loc[self.grains_df['GrainID'] == i-1, 'meshed_conti_volume'] = grain_vol * 10 ** 9

        self.grains_df[['GrainID','meshed_conti_volume', 'phaseID']].\
        to_csv(RveInfo.store_path + '/Generation_Data/grain_data_output_conti.csv', index=False)

        all_points_df.loc[all_points_df_old['x_min'], 'x'] = x_min
        all_points_df.loc[all_points_df_old['y_min'], 'y'] = y_min
        all_points_df.loc[all_points_df_old['z_min'], 'z'] = z_min
        all_points_df.loc[all_points_df_old['x_max'], 'x'] = x_max
        all_points_df.loc[all_points_df_old['y_max'], 'y'] = y_max
        all_points_df.loc[all_points_df_old['z_max'], 'z'] = z_max

        grid.points = all_points_df[['x', 'y', 'z']].values
        return grid

    def run(self) -> None:

        GRID = self.gen_subs()
        smooth_mesh = self.smoothen_mesh(GRID)

        pbc_grid = smooth_mesh


        if RveInfo.roughness_flag:
            # TODO: roghness einbauen
            # grid = self.apply_roughness(grid)
            pass

        f = open(RveInfo.store_path + '/DRAGen_RVE.inp', 'w+')
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

        pv.save_meshio(RveInfo.store_path + '/rve-part.inp', smooth_mesh)
        f = open(RveInfo.store_path + '/rve-part.inp', 'r')
        lines = f.readlines()
        f.close()
        lines = [line.lower() for line in lines]
        startingLine = lines.index('*node\n')
        f = open(RveInfo.store_path + '/DRAGen_RVE.inp', 'a')
        f.write('*Part, name=PART-1\n')
        for line in lines[startingLine:]:
            if line.replace(" ", "") == "*element,type=c3d8rh\n":
                line = "*element,type=c3d8\n"
            if '*end' in line:
                line = line.replace('*end', '**\n')
            f.write(line)
        n = 1
        for nSubs in [self.n_grains,self.n_packets,self.n_blocks]:
            print('current sub is {},number is {}'.format(self.subsnum[n],nSubs))
            for i in range(nSubs):
                nSub = i + 1
                cells = np.where(smooth_mesh.cell_data[self.idnum[n]] == nSub)[0]
                f.write('*Elset, elset=Set-{}{}\n'.format(self.subsnum[n],nSub))
                for j, cell in enumerate(cells + 1):
                    if (j + 1) % 16 == 0:
                        f.write('\n')
                    f.write(' {},'.format(cell))
                f.write('\n')

            n += 1

        phase1_idx = 0
        phase2_idx = 0
        phase3_idx = 0
        phase4_idx = 0
        phase5_idx = 0
        for i in range(self.n_blocks):
            nBlock = i + 1
            if self.rve.loc[GRID.cell_data['block_id'] == nBlock].phaseID.values[0] == 1:
                phase1_idx += 1
                f.write('** Section: Section - {}\n'.format(nBlock))
                f.write('*Solid Section, elset=Set-Block{}, material=Ferrite_{}\n'.format(nBlock, phase1_idx))
            elif self.rve.loc[GRID.cell_data['block_id'] == nBlock].phaseID.values[0] == 2:
                if not RveInfo.phase2iso_flag[2]:
                    phase2_idx += 1
                    f.write('** Section: Section - {}\n'.format(nBlock))
                    f.write('*Solid Section, elset=Set-Block{}, material=Martensite_{}\n'.format(nBlock, phase2_idx))
                else:
                    f.write('** Section: Section - {}\n'.format(nBlock))
                    f.write('*Solid Section, elset=Set-Block{}, material=Martensite\n'.format(nBlock))
            elif self.rve.loc[GRID.cell_data['block_id'] == nBlock].phaseID.values[0] == 3:
                if not RveInfo.phase2iso_flag[3]:
                    phase3_idx += 1
                    f.write('** Section: Section - {}\n'.format(nBlock))
                    f.write(
                        '*Solid Section, elset=Set-Block{}, material=Pearlite_{}\n'.format(nBlock, phase3_idx))
                else:
                    f.write('** Section: Section - {}\n'.format(nBlock))
                    f.write('*Solid Section, elset=Set-Block{}, material=Pearlite\n'.format(nBlock))

            elif self.rve.loc[GRID.cell_data['block_id'] == nBlock].phaseID.values[0] == 4:
                if not RveInfo.phase2iso_flag[4]:
                    phase4_idx += 1
                    f.write('** Section: Section - {}\n'.format(nBlock))
                    f.write('*Solid Section, elset=Set-Block{}, material=Bainite_{}\n'.format(nBlock, phase4_idx))
                else:
                    f.write('** Section: Section - {}\n'.format(nBlock))
                    f.write('*Solid Section, elset=Set-Block{}, material=Bainite\n'.format(nBlock))
            elif self.rve.loc[GRID.cell_data['block_id'] == nBlock].phaseID.values[0] == 5: 
                if not RveInfo.phase2iso_flag[5]:
                    phase5_idx += 1
                    f.write('** Section: Section - {}\n'.format(nBlock))
                    f.write('*Solid Section, elset=Set-Block{}, material=Austenite_{}\n'.format(nBlock, phase5_idx))
                else:
                    f.write('** Section: Section - {}\n'.format(nBlock))
                    f.write('*Solid Section, elset=Set-Block{}, material=Austenite\n'.format(nBlock))


        f.close()
        os.remove(RveInfo.store_path + '/rve-part.inp')
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

        self.make_assembly()  # Don't change the order
        #self.pbc(GRID, grid_hull_df)  # of these four
        #self.write_substruct_material_def()  # functions here
        if RveInfo.gui_flag:
            RveInfo.progress_obj.emit(50)
        if RveInfo.submodel_flag:
            self.submodelSet(grid_hull_df)
        if RveInfo.pbc_flag:
            self.pbc(GRID, grid_hull_df)
        self.write_substruct_material_def()
        if RveInfo.submodel_flag:
            self.write_submodel_step_def()
        elif RveInfo.pbc_flag:
            print('pbcs were assumed')
            self.write_pbc_step_def()
        #if RveInfo.pbc_flag:
        #    self.write_pbc_step_def()  # it will lead to a faulty inputfile
        if RveInfo.gui_flag:
            RveInfo.progress_obj.emit(75)
        self.write_block_data()
        if RveInfo.gui_flag:
            RveInfo.progress_obj.emit(100)