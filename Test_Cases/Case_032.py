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
# If you want to add banding, change the number_of_bands to 1 or higher has to be integer 
number_of_bands = 1
band_filling = 1
band_orientation = 'xy'
lower_band_bound = 2
upper_band_bound = 4
visualization_flag = False #plotting images to figs
root = r'./'
shrink_factor = 0.4

#Inclusion Setting
# To add make inclusions_flag = True
inclusion_flag = False
inclusion_ratio = 0.05
slope_offset = 0


#Files:
Ferrite = r'./ExampleInput/Ferrite/TrainedData_Ferrite.pkl'
#Martensite = r'./ExampleInput/Martensite/TrainedData_Martensite.pkl'
#Pearlite = r'./ExampleInput/Pearlite/TrainedData_Pearlite.pkl'   
#Bainite = r'./ExampleInput/Bainite/TrainedData_Bainite.pkl' 
#Austenite = r'./ExampleInput/Austenite/TrainedData_Austenite.pkl'

#PAGs

#Blocks

#Inclusions
#Inclusion = r'./ExampleInput/Inclusions/TrainedData_Inclusion.pkl'


#Bands Files File 6

#Bands = r'./ExampleInput/Banding/TrainedData_Band.pkl'
Bands = r'./ExampleInput/Martensite/TrainedData_Martensite.pkl'


# test pearlite phase
# Substructure params
subs_flag = False
equiv_d = 5
p_sigma = 0.1
t_mu = 1.0
b_sigma = 0.1
subs_file_flag = False
subs_file = r'./ExampleInput/Substructure/example_block_inp.csv'

#Texture Type
moose_flag = False
abaqus_flag = False
damask_flag = True
#Texture Setting
pbc_flag = True
submodel_flag = False
phase2iso_flag = {1:True, 2:True, 3:True, 4:True, 5:True}
x_fem_flag = False
element_type = 'HEX8'
anim_flag = False

#Choosing active files
files = {1: Ferrite, 2: None, 3: None, 4: None, 5:None, 6: None, 7: Bands}  # ['Ferrite', 'Martensite', 'Pearlite', 'Bainite', 'Inclusion', 'Banding']
# Change the file name to 'None' if its empty
phase_ratio = {1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.15}
phases = ['Ferrite', 'Martensite', 'Pearlite', 'Bainite', 'Austenite', 'Inclusions', 'Bands']

#Band thickness
upper = None
lower = None

circularity = 1
decreasing_factor = 0.95

#Plot and save settings
plot = False
plt_name = 'substructure_plot.png'

save = True
filename = 'substructure_plot.png'
orientation_relationship = 'KS'

"test git"
'''
specific number is fixed for each phase. 1->ferrite, 2->martensite so far. The order of input files should also have the 
same order as phases. file1->ferrite, file2->martensite. The substructures will only be generated in martensite.

Number 5 specifies the inclusions and number 6 the Band phase. Either .csv or .pkl
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

