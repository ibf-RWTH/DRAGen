from dragen.run import Run
dimension = 3
box_size = 600
box_size_y = None  # if this is None it will be set to the main box_size value
box_size_z = 300  # for sheet rve set z to None and y to different value than x the other way round is buggy
resolution = 0.08
number_of_rves = 100
number_of_bands = 0
band_filling = 1.2
lower_band_bound = 2
upper_band_bound = 5
visualization_flag = True
root = '../'
shrink_factor = 0.4
# Example Files
equiv_d = 5
p_sigma = 0.1
t_mu = 1.0
b_sigma = 0.1
inclusion_flag = False
inclusion_ratio = 0.01
# Example Files
#
#
file1 = r'C:/temp/ElectricsteelData/Data_processed.csv'
#file1 = r'./ExampleInput/TrainedData_2.pkl'
file2 = r'./ExampleInput/martensit.csv'
file3 = r'./ExampleInput/pearlite_21_grains.csv'
file6 = r'./ExampleInput/TrainedData_6.pkl'


# test pearlite phase
subs_flag = False
subs_file = './ExampleInput/example_block_inp.csv'
subs_file_flag = True
gui_flag = False
gan_flag = False
moose_flag = False
abaqus_flag = True
pbc_flag = False
submodel_flag = True
damask_flag = False
phase2iso_flag = True
element_type = 'HEX8'
anim_flag = False
exe_flag = False

files = {1: file1}#, 2: file2}
phase_ratio = {1: 1}#, 2: 0.2}  # Pass for bands
phases = ['Ferrite']#, 'Martensite']

'''
specific number is fixed for each phase. 1->ferrite, 2->martensite so far. The order of input files should also have the 
same order as phases. file1->ferrite, file2->martensite. The substructures will only be generated in martensite.

Number 5 specifies the inclusions and number 6 the Band phase. Either .csv or .pkl
'''

Run(box_size, element_type=element_type, box_size_y=box_size_y, box_size_z=box_size_z, resolution=resolution,
    number_of_rves=number_of_rves,
    number_of_bands=number_of_bands, dimension=dimension,
    visualization_flag=visualization_flag, file_dict=files, equiv_d=equiv_d, p_sigma=p_sigma, t_mu=t_mu,
    b_sigma=b_sigma,
    phase_ratio=phase_ratio, root=root, shrink_factor=shrink_factor, gui_flag=gui_flag,
    gan_flag=gan_flag, pbc_flag=pbc_flag, submodel_flag=submodel_flag, phase2iso_flag=phase2iso_flag,
    info_box_obj=None, progress_obj=None, subs_file_flag=subs_file_flag, subs_file=subs_file, phases=phases,
    subs_flag=subs_flag, moose_flag=moose_flag, abaqus_flag=abaqus_flag, damask_flag=damask_flag,
    anim_flag=anim_flag, exe_flag=exe_flag, inclusion_flag=inclusion_flag,
    inclusion_ratio=inclusion_ratio, band_filling=band_filling, lower_band_bound=lower_band_bound,
    upper_band_bound=upper_band_bound).run()