"""Abaqus + substructure.

Case_029/030 cover substructure with DAMASK/MOOSE but are 100% ferrite, so no packet or block is
ever generated there -- they only prove the pipeline is inert when nothing is transformable. This
scenario is the counterpart: half the volume is martensite, so packets, blocks and the per-block
Abaqus deck (substructure.inp / SubstructureMaterials.inp / SubstructureSections.inp) are really
produced. Block thicknesses come from the example EBSD distribution (subs_file_flag).
"""
from dragen.run import Run
#Model details
dimension = 3
box_size = 25
box_size_y = None  # if this is None it will be set to the main box_size value
box_size_z = None  # for sheet rve set z to None and y to different value than x the other way round is buggy
resolution = 1
number_of_rves = 1
smoothing_flag = False

# Banding Parameters:
number_of_bands = 0
band_filling = 1
band_orientation = 'xy'
lower_band_bound = 2
upper_band_bound = 4
root = r'./'
shrink_factor = 0.4

#Inclusion Setting
inclusion_flag = False
inclusion_ratio = 0.05
slope_offset = 0

#Files:
Martensite = r'./ExampleInput/Martensite/TrainedData_Martensite.pkl'
Austenite = r'./ExampleInput/Ferrite/TrainedData_Ferrite.pkl'

# Substructure params
subs_flag = True
t_mu = 1.0
subs_file_flag = True
subs_file = './ExampleInput/Substructure/example_block_inp.csv'

#Texture Type
moose_flag = False
abaqus_flag = True
damask_flag = False
#Texture Setting
pbc_flag = True
submodel_flag = False
phase2iso_flag = {1:True, 2:False, 3:True, 4:True, 5:True}
x_fem_flag = False
calibration_rve_flag = False
element_type = 'HEX8'
anim_flag = False

#Choosing active files
files = {1: None, 2: Martensite, 3: None, 4: None, 5: Austenite, 6: None, 7: None}
phase_ratio = {1: 0, 2: 0.5, 3: 0, 4: 0, 5: 0.5, 6: 0, 7: 0}
phases = ['Ferrite', 'Martensite', 'Pearlite', 'Bainite', 'Austenite', 'Inclusions', 'Bands']


Run(dimension=dimension, box_size=box_size, box_size_y=box_size_y, box_size_z=box_size_z, resolution=resolution,
    number_of_rves=number_of_rves, slope_offset=slope_offset, abaqus_flag=abaqus_flag, damask_flag=damask_flag,
    moose_flag=moose_flag, calibration_rve_flag=calibration_rve_flag, element_type=element_type, pbc_flag=pbc_flag,
    submodel_flag=submodel_flag,
    phase2iso_flag=phase2iso_flag, smoothing_flag=smoothing_flag, xfem_flag=x_fem_flag, gui_flag=False,
    anim_flag=anim_flag,
    root=root, info_box_obj=None, progress_obj=None, phase_ratio=phase_ratio,
    file_dict=files, phases=phases, number_of_bands=number_of_bands, upper_band_bound=upper_band_bound,
    lower_band_bound=lower_band_bound, band_orientation=band_orientation, band_filling=band_filling,
    subs_flag=subs_flag, subs_file_flag=subs_file_flag, subs_file=subs_file, t_mu=t_mu).run()
