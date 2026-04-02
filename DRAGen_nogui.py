import numpy as np

from dragen.run import Run
#Model details
dimension = 3
box_size = 30
box_size_y = None  # if this is None it will be set to the main box_size value
box_size_z = None  # for sheet rve set z to None and y to different value than x the other way round is buggy
resolution = 2
number_of_rves = 3
smoothing_flag = False

# Banding Parameters:
# If you want to add banding, change the number_of_bands to 1 or higher has to be integer 
number_of_bands = 1
band_filling = 1
band_orientation = 'xy'
lower_band_bound = 2
upper_band_bound = 4
visualization_flag = False #plotting images to figs
root = r'./'
shrink_factor = 0.4

# Inclusion Setting
# To add make inclusions_flag = True
inclusion_flag = False
inclusion_ratio = 0.05
slope_offset = 0

# Substructure params
subs_flag = False
equiv_d = 5
p_sigma = 0.1
t_mu = 1.0
b_sigma = 0.1
subs_file_flag = False
subs_file = './ExampleInput/Substructure/example_block_inp.csv'
circularity = 1
decreasing_factor = 0.95
plot = False
plt_name = 'substructure_plot.png'
save = True
filename = 'substructure_plot.png'
orientation_relationship = 'KS'
upper = 2.5
lower = 1.5

#Texture Type
moose_flag = False
abaqus_flag = True
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
Ferrite =  r'./ExampleInput/Ferrite/TrainedData_Ferrite.pkl'
Martensite = r'./ExampleInput/Martensite/TrainedData_Martensite.pkl'
#Pearlite = r'./ExampleInput/Pearlite/TrainedData_Pearlite.pkl'
#Bainite = r'./ExampleInput/Bainite/TrainedData_Bainite.pkl'
#Austenite = r'./ExampleInput/Austenite/TrainedData_Austenite.pkl'
Austenite = r'./ExampleInput/Ferrite/TrainedData_Ferrite.pkl' #r'./ExampleInput/Ferrite/Homogeneous_Grainsize.csv'

#Choosing active files

# If the phase ratio is > 0, a file has to be provided
pr_bands = number_of_bands * np.mean([lower_band_bound, upper_band_bound]) * box_size * box_size / (box_size**3)
pr_ferrite = 0.8 - 0.5*pr_bands
pr_austenite = 0.2 - 0.5*pr_bands

phase_ratio = {1: pr_ferrite, 2: 0, 3: 0, 4: 0, 5: pr_austenite, 6: 0, 7: pr_bands}

files = {1: Ferrite, 2: None, 3: Martensite, 4: None, 5: Austenite, 6: None, 7: Martensite}

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
    visualization_flag=visualization_flag, root=root, info_box_obj=None, progress_obj=None, phase_ratio=phase_ratio,
    file_dict=files, phases=phases, number_of_bands=number_of_bands, upper_band_bound=upper_band_bound,
    lower_band_bound=lower_band_bound, band_orientation=band_orientation, band_filling=band_filling,
    subs_flag=subs_flag, subs_file_flag=subs_file_flag,
    subs_file=subs_file, equiv_d=equiv_d, p_sigma=p_sigma, t_mu=t_mu, b_sigma=b_sigma,
    decreasing_factor=decreasing_factor, lower=lower, upper=upper, circularity=circularity, plt_name=plt_name,
    save=save, plot=plot, filename=filename, orientation_relationship=orientation_relationship).run()

