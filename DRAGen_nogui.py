import numpy as np
from sympy import false

from dragen.run import Run
#Model details
dimension = 3
box_size = 30
box_size_y = None  # if this is None it will be set to the main box_size value
box_size_z = None # for sheet rve set z to None and y to different value than x the other way round is buggy
resolution = 2
number_of_rves = 1
smoothing_flag = False

# Banding Parameters:
# If you want to add banding, change the number_of_bands to 1 or higher has to be integer 
number_of_bands = 0
band_filling = 0.5
band_orientation = 'xz'
lower_band_bound = 4.99
upper_band_bound = 5
root = r'./'
shrink_factor = 0.35

# Inclusion Setting
# To add make inclusions_flag = True
inclusion_flag = False
inclusion_ratio = 0.05
slope_offset = 0

# Substructure params (only used when subs_flag is True, 3D only)
subs_flag = True
# Block thickness: either a mean value in micrometres...
t_mu = 1.0
# ...or sampled from a measured distribution (needs a 'block_thickness' column)
subs_file_flag = False
subs_file = './ExampleInput/Substructure/example_block_inp.csv'
# Phases that get packets/blocks: 2 Martensite, 3 Pearlite, 4 Bainite
subs_transformable_phase_ids = [2, 3, 4]
# Percentile clip on the measured block thickness distribution (only used with subs_file_flag)
subs_lower_percentile = 5
subs_upper_percentile = 95
# Target cells per packet, and the packet size below which a packet is kept as a single block
subs_min_packet_cells = 100
subs_min_block_cells = 10
# Packets/blocks smaller than this get merged into a neighbour
subs_min_cells_per_packet = 5
subs_min_cells_per_block = 5
# 'KS' -> the 24 ideal Kurdjumov-Sachs variants
# 'experimental' -> transformations measured from a parent/child EBSD pair. Needs
#   subs_child_orientation_file (grain_id, phi1, PHI, phi2) and a parent orientation csv; the
#   parent defaults to the phase input file of the first transformable phase.
subs_orientation_mode = 'KS'
subs_parent_orientation_file = None
subs_child_orientation_file = None

#Texture Type
moose_flag = False
abaqus_flag = False
damask_flag = True
#Texture Setting
pbc_flag = True
submodel_flag = False
phase2iso_flag = {1:True, 2:False, 3:True, 4:True, 5:True}
x_fem_flag = False
calibration_rve_flag = False
element_type = 'HEX8'
anim_flag = False

#Files:
# r'Y:\03_Projekte\DFG\TRR_B05\02_Experimente\80_TrainedGANs\DP800GAN_STAND20220709\TrainedData_1.pkl'
Ferrite= r'Y:\03_Projekte\DFG\TRR_B05\02_Experimente\80_TrainedGANs\DP800GAN_STAND20220709\Ferrite_RDxBN.csv'
Martensite = r'Y:\03_Projekte\DFG\TRR_B05\02_Experimente\80_TrainedGANs\DP800GAN_STAND20220709\Martensite_RDxBN.csv'
#Pearlite = r'./ExampleInput/Pearlite/TrainedData_Pearlite.pkl'
Bainite = r'./ExampleInput/Ferrite/Homogeneous_Grainsize.csv'
Austenite = r'./ExampleInput/Austenite/TrainedData_Austenite.pkl'

#Choosing active files

# If the phase ratio is > 0, a file has to be provided
#pr_bands = number_of_bands * np.mean([lower_band_bound, upper_band_bound]) * box_size * box_size / (box_size**3)
#pr_ferrite = 1 - 1*pr_bands
#pr_martensite = 0.35 - 0.5*pr_bands

phase_ratio = {1: 0.3, 2: 0, 3: 0, 4: 0.7, 5: 0, 6: 0, 7: 0}
files = {1: Ferrite, 2: None, 3: None, 4: Austenite, 5: None, 6: None, 7: None}
phases = ['Ferrite', 'Martensite', 'Pearlite', 'Bainite', 'Austenite', 'Inclusions', 'Bands']

"test git"
'''
specific number is fixed for each phase. 1->ferrite, 2->martensite so far. The order of input files should also have the 
same order as phases. file1->ferrite, file2->martensite. The substructures will only be generated in martensite.

Number 6 specifies the inclusions and number 7 the Band phase. Either .csv or .pkl
'''

Run(dimension=dimension, box_size=box_size, box_size_y=box_size_y, box_size_z=box_size_z, resolution=resolution,
    number_of_rves=number_of_rves, slope_offset=slope_offset, abaqus_flag=abaqus_flag, damask_flag=damask_flag,
    moose_flag=moose_flag, calibration_rve_flag=calibration_rve_flag, element_type=element_type, pbc_flag=pbc_flag, submodel_flag=submodel_flag,
    phase2iso_flag=phase2iso_flag, smoothing_flag=smoothing_flag, xfem_flag=x_fem_flag, gui_flag=False, anim_flag=anim_flag,
    root=root, info_box_obj=None, progress_obj=None, phase_ratio=phase_ratio,
    file_dict=files, phases=phases, number_of_bands=number_of_bands, upper_band_bound=upper_band_bound,
    lower_band_bound=lower_band_bound, band_orientation=band_orientation, band_filling=band_filling,
    subs_flag=subs_flag, subs_file_flag=subs_file_flag, subs_file=subs_file, t_mu=t_mu,
    subs_transformable_phase_ids=subs_transformable_phase_ids,
    subs_lower_percentile=subs_lower_percentile, subs_upper_percentile=subs_upper_percentile,
    subs_min_packet_cells=subs_min_packet_cells, subs_min_block_cells=subs_min_block_cells,
    subs_min_cells_per_packet=subs_min_cells_per_packet,
    subs_min_cells_per_block=subs_min_cells_per_block,
    subs_orientation_mode=subs_orientation_mode,
    subs_parent_orientation_file=subs_parent_orientation_file,
    subs_child_orientation_file=subs_child_orientation_file).run()

