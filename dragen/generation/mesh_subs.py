# _*_ coding: utf-8 _*_
"""
Time:     2021/10/25 17:04
Author:   Linghao Kong
Version:  V 0.1
File:     SubMesher
Describe: Write during the internship at IEHK RWTH"""
import pandas as pd
from dragen.generation.mesher import Mesher
import pyvista as pv
import numpy as np
import datetime
import logging
import os
class SubMesher(Mesher):
    def __init__(self,box_size_x: int, box_size_y: int = None, box_size_z: int = None,  rve: pd.DataFrame = None,
                 subs_df: pd.DataFrame = None, store_path: str = None,
                 phase_two_isotropic=True, animation=True, infobox_obj=None,
                 progress_obj=None, gui=True, element_type='C3D4'):
        self.box_size_x = box_size_x
        self.box_size_y = box_size_y
        self.box_size_z = box_size_z
        self.rve = rve
        self.subs_df = subs_df
        self.store_path = store_path
        self.phase_two_isotropic = phase_two_isotropic
        self.animation = animation
        self.infobox_obj = infobox_obj
        self.progress_obj = progress_obj
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
        self.n_pts_x = int(rve.n_pts) + 1
        self.bin_size = rve.box_size / (self.n_pts_x)  ## test
        if self.box_size_y is not None:
            self.n_pts_y = int(self.box_size_y / self.bin_size)
        else:
            self.box_size_y = self.box_size_x
            self.n_pts_y = self.n_pts_x

        if self.box_size_z is not None:
            self.n_pts_z = int(self.box_size_z / self.bin_size)
        else:
            self.box_size_z = self.box_size_x
            self.n_pts_z = self.n_pts_x

        self.logger = logging.getLogger("RVE-Gen")
        self.gui = gui
        self.element_type = element_type
        self.roughness = True
        self.idnum = {1:'GrainID',2:'packet_id',3:'block_id'}
        self.subsnum = {1:'Grain',2:'Packet',3:'Block'}
        super().__init__(box_size_x=box_size_x,box_size_y=box_size_y,box_size_z=box_size_z,rve=rve,
                         grains_df=subs_df,store_path=store_path,phase_two_isotropic=phase_two_isotropic,animation=animation,
                         infobox_obj=infobox_obj,progress_obj=progress_obj,gui=gui,element_type=element_type)

    def gen_subs(self):
        if self.gui:
            self.progress_obj.emit(0)
            self.infobox_obj.emit('starting mesher')
        grid = self.gen_blocks()
        if self.gui:
            self.progress_obj.emit(25)
        grid = self.gen_grains(grid)

        grid.cell_data['packet_id'] = self.rve['packet_id'].to_numpy()
        grid.cell_data['block_id'] = self.rve['block_id'].to_numpy()

        if self.animation:
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(grid, scalars='packet_id', scalar_bar_args={'title': 'Packet IDs'},
                             show_edges=True, interpolate_before_map=True)
            plotter.add_axes()
            plotter.show(interactive=True, auto_close=True, window_size=[800, 600],
                         screenshot=self.store_path + '/Figs/pyvista_Hex_Mesh_packets.png')
            plotter.close()

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(grid, scalars='block_id', scalar_bar_args={'title': 'Block IDs'},
                             show_edges=True, interpolate_before_map=True)
            plotter.add_axes()
            plotter.show(interactive=True, auto_close=True, window_size=[800, 600],
                         screenshot=self.store_path + '/Figs/pyvista_Hex_Mesh_blocks.png')
            plotter.close()

        return grid

    def write_material_def(self) -> None:
        phase1_idx = 0
        phase2_idx = 0
        numberofblocks = self.n_blocks

        phase = [self.rve.loc[self.rve['block_id'] == i].phaseID.values[0] for i in range(1, numberofblocks + 1)]
        f = open(self.store_path + '/RVE_smooth.inp', 'a')

        f.write('**\n')
        f.write('** MATERIALS\n')
        f.write('**\n')
        for i in range(numberofblocks):
            nblock = i + 1
            if not self.phase_two_isotropic:
                if phase[i] == 1:
                    phase1_idx += 1
                    f.write('*Material, name=Ferrite_{}\n'.format(phase1_idx))
                    f.write('*Depvar\n')
                    f.write('    176,\n')
                    f.write('*User Material, constants=2\n')
                    f.write('{}.,3.\n'.format(nblock))
                elif phase[i] == 2:
                    phase2_idx += 1
                    f.write('*Material, name=Martensite_{}\n'.format(phase2_idx))
                    f.write('*Depvar\n')
                    f.write('    176,\n')
                    f.write('*User Material, constants=2\n')
                    f.write('{}.,4.\n'.format(nblock))
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

    def write_block_data(self) -> None:
        f = open(self.store_path + '/graindata.inp', 'w+')
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
            if not self.phase_two_isotropic:
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
                    phi1 = self.tex_phi1[i]
                    PHI = self.tex_PHI[i]
                    phi2 = self.tex_phi2[i]
                    f.write('Grain: {}: {}: {}: {}: {}\n'.format(phase1_idx, phi1, PHI, phi2, self.bt_list[i]))
        f.close()

    def mesh_and_build_abaqus_model(self) -> None:

        GRID = self.gen_subs()
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
        startingLine = lines.index('*NODE\n')
        f = open(self.store_path + '/RVE_smooth.inp', 'a')
        f.write('*Part, name=PART-1\n')
        for line in lines[startingLine:]:
            f.write(line)

        n = 1
        for nSubs in [self.n_grains,self.n_packets,self.n_blocks]:
            print('current sub is {},number is {}'.format(self.subsnum[n],nSubs))
            for i in range(nSubs):
                nSub = i + 1
                # print('in for nBlock=', nBlock, smooth_mesh.cell_data.keys())
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
        for i in range(self.n_blocks):
            nBlock = i + 1
            if self.rve.loc[GRID.cell_data['block_id'] == nBlock].phaseID.values[0] == 1:
                phase1_idx += 1
                f.write('** Section: Section - {}\n'.format(nBlock))
                f.write('*Solid Section, elset=Set-Block{}, material=Ferrite_{}\n'.format(nBlock, phase1_idx))
            elif self.rve.loc[GRID.cell_data['block_id'] == nBlock].phaseID.values[0] == 2:
                if not self.phase_two_isotropic:
                    phase2_idx += 1
                    f.write('** Section: Section - {}\n'.format(nBlock))
                    f.write('*Solid Section, elset=Set-Block{}, material=Martensite_{}\n'.format(nBlock, phase2_idx))
                else:
                    f.write('** Section: Section - {}\n'.format(nBlock))
                    f.write('*Solid Section, elset=Set-Block{}, material=Martensite\n'.format(nBlock))

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

        self.make_assembly()  # Don't change the order
        self.pbc(GRID, grid_hull_df)  # of these four
        self.write_material_def()  # functions here
        self.write_step_def()  # it will lead to a faulty inputfile
        self.write_block_data()