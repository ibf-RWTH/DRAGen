from dragen.run import Run
dimension = 2
box_size = 64
box_size_y = None  # if this is None it will be set to the main box_size value
box_size_z = None  # for sheet rve set z to None and y to different value than x the other way round is buggy
resolution = 2
number_of_rves = 1
number_of_bands = 0
band_filling = 1.2
lower_band_bound = 2
upper_band_bound = 5
visualization_flag = True
store_path = '../'
shrink_factor = 0.4
# Example Files
equiv_d = 5
p_sigma = 0.1
t_mu = 1.0
b_sigma = 0.1
inclusion_flag = False
inclusion_ratio = 0.01
# Example Files
# file1 = r'C:\Venvs\dragen\ExampleInput\ferrite_54_grains_processed.csv'
file1 = r'./ExampleInput/TrainedData_2.pkl'
file2 = r'./ExampleInput/martensit.csv'
file3 = r'./ExampleInput/pearlite_21_grains.csv'
file6 = r'./ExampleInput/TrainedData_6.pkl'


# test pearlite phase
subs_flag = False
subs_file = './ExampleInput/example_block_inp.csv'
subs_file_flag = True
gui_flag = False
gan_flag = False
moose_flag = True
abaqus_flag = True
damask_flag = True
phase2iso_flag = True
element_type = 'HEX8'
anim_flag = False
exe_flag = False
files = {1: file1, 2: file2}
phase_ratio = {1: 0.8, 2: 0.2}  # Pass for bands
phases = ['Ferrite', 'Martensite']

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
    phase_ratio=phase_ratio, store_path=store_path, shrink_factor=shrink_factor, gui_flag=gui_flag,
    gan_flag=gan_flag,
    info_box_obj=None, progress_obj=None, subs_file_flag=subs_file_flag, subs_file=subs_file, phases=phases,
    subs_flag=subs_flag, moose_flag=moose_flag, abaqus_flag=abaqus_flag, damask_flag=damask_flag,
    anim_flag=anim_flag, exe_flag=exe_flag, inclusion_flag=inclusion_flag,
    inclusion_ratio=inclusion_ratio, band_filling=band_filling, lower_band_bound=lower_band_bound,
    upper_band_bound=upper_band_bound).run()
