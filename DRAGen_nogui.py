from dragen.run import Run
dimension = 3
box_size = 15
box_size_y = None  # if this is None it will be set to the main box_size value
box_size_z = None  # for sheet rve set z to None and y to different value than x the other way round is buggy
resolution = 2
number_of_rves = 1
smoothing_flag = False
# Banding Params
number_of_bands = 0
band_filling = 0.6
band_orientation = 'xy'
lower_band_bound = 2
upper_band_bound = 5
visualization_flag = False
root = r'./'
shrink_factor = 0.4
# Substructure params
equiv_d = 5
p_sigma = 0.1
t_mu = 1.0
b_sigma = 0.1
inclusion_flag = False
inclusion_ratio = 0.01
slope_offset = 0
# Example Files
#
#
###NO30
#file1 = r'F:/OCAS/NO30_RVE_data/Data_processed.csv'
#file2 = r"D:\2nd mini-thesis\dragen\ExampleInput\example_pag_inp.csv"
#DP800
file1 = r'./ExampleInput/TrainedData_Ferrite.pkl'
file2 = r'./ExampleInput/TrainedData_Martensite.pkl'

#Bainite
#PAGs

#Blocks

#Inclusions
#file5 = r'E:\Sciebo\IEHK\Publications\ComputationalSci\DRAGen\matdata\Inclusions/Inclusions_DRAGen_Input.csv'

#Bands
#file6 = r'E:\Sciebo\IEHK\Publications\ComputationalSci\DRAGen\matdata\DP800/TrainedData_Martensite.pkl'
#file6 = r'E:\Sciebo\IEHK\Publications\ComputationalSci\DRAGen\matdata\DP800/TrainedData_Ferrite.pkl'

# test pearlite phase
subs_flag = False
subs_file = './ExampleInput/example_block_inp.csv'
subs_file_flag = False
gan_flag = False
moose_flag = False
abaqus_flag = True
pbc_flag = True
submodel_flag = False
damask_flag = False
phase2iso_flag = True
x_fem_flag = True
element_type = 'HEX8'
anim_flag = False

files = {1: file1, 2: file2, 3: None, 4: None, 5: None, 6: None}  # , 2: file2, 6: file6}
phase_ratio = {1: 0.8, 2: 0.2, 3: 0, 4: 0, 5: 0, 6: 0}
phases = ['Ferrite', 'Martensite'] #, 'Bands']

upper = None
lower = None
circularity = 1
decreasing_factor = 0.95
plt_name = 'substructure_plot.png'
save = True
plot = False
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
    moose_flag=moose_flag, element_type=element_type, pbc_flag=pbc_flag, submodel_flag=submodel_flag,
    phase2iso_flag=phase2iso_flag, smoothing_flag=smoothing_flag, xfem_flag=x_fem_flag, gui_flag=False, anim_flag=anim_flag,
    visualization_flag=visualization_flag, root=root, info_box_obj=None, progress_obj=None, phase_ratio=phase_ratio,
    file_dict=files, phases=phases, number_of_bands=number_of_bands, upper_band_bound=upper_band_bound,
    lower_band_bound=lower_band_bound, band_orientation=band_orientation, band_filling=band_filling,
    subs_flag=subs_flag, subs_file_flag=subs_file_flag,
    subs_file=subs_file, equiv_d=equiv_d, p_sigma=p_sigma, t_mu=t_mu, b_sigma=b_sigma,
    decreasing_factor=decreasing_factor, lower=lower, upper=upper, circularity=circularity, plt_name=plt_name,
    save=save, plot=plot, filename=filename, orientation_relationship=orientation_relationship).run()

