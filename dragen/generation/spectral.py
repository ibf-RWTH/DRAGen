"""
Functions for writing output fo Spectral Solver: 3 Files:
    .geom - Geometrie
    .load - load and stress state conditions
    .config - Material and phases
"""
import numpy as np
import pandas as pd
import pyvista as pv
import damask


def make_config(store_path, n_grains, grains_df, band=True) -> None:
    """
    Five Parts:
        homogenization
        crystallite
        phase
        microstructure
        texture
    For demonstration:
        bcc ferrite -> from Orkun
        bcc martensite -> from Damask
    :return: None
    """
    with open(store_path + '/' + 'material.config', 'w') as mat:
        # Homogenization
        mat.writelines('#--------------------# \n')
        mat.writelines('<homogenization> \n')
        mat.writelines('#--------------------# \n')
        mat.writelines('\n')
        mat.writelines('[SX] \n')
        mat.writelines('mech\tNone\n')
        mat.writelines('\n')

        # Crystallite
        mat.writelines('#--------------------# \n')
        mat.writelines('<crystallite> \n')
        mat.writelines('#--------------------# \n')
        mat.writelines('[almostAll] \n')
        mat.writelines('(output) phase \n')
        mat.writelines('(output) volume \n')
        mat.writelines('(output) texture \n')
        mat.writelines('(output) orientation \n')
        mat.writelines('(output) grainrotation \n')
        mat.writelines('(output) f \n')
        mat.writelines('(output) fe \n')
        mat.writelines('(output) fp \n')
        mat.writelines('(output) p \n')
        mat.writelines('(output) lp \n')
        mat.writelines('(output) fe \n')
        mat.writelines('\n')

        # Phases
        mat.writelines('#--------------------# \n')
        mat.writelines('<phase> \n')
        mat.writelines('#--------------------# \n')

        # No1 - Ferrite
        mat.writelines('\n[BCC - Ferrite] \n')
        mat.writelines('elasticity hooke \n')
        mat.writelines('plasticity phenopowerlaw \n\n')

        output_list = 'resistance_slip shearrate_slip resolvedstress_slip accumulatedshear_slip'.split(' ')
        for out in output_list:
            mat.writelines('(output)  {}\n'.format(out))
        mat.writelines('\n')

        param_list = 'Nslip 12, Ntwin 0, c11 231e9, c12 134.7e9, c44 116.4e9, gdot0_slip 0.0001,' \
                     ' n_slip 20, tau0_slip 89.65e6, tausat_slip 819.69e6, a_slip 1.47, h0_slipslip 523.67e6, ' \
                     'interaction_slipslip 1 1 1.4 1.4 1.4 1.4, atol_resistance 1'.split(', ')

        mat.writelines('lattice_structure bcc\n')
        for p in param_list:
            mat.writelines(p + '\n')

        # No2 - Martensite
        mat.writelines('\n[BCC - Martensite] \n')
        mat.writelines('elasticity hooke \n')
        mat.writelines('plasticity phenopowerlaw \n\n')

        output_list = 'resistance_slip shearrate_slip resolvedstress_slip accumulatedshear_slip'.split(' ')
        for out in output_list:
            mat.writelines('(output)  {}\n'.format(out))
        mat.writelines('\n')

        param_list = 'Nslip 12, Ntwin 0, c11 106.75e9, c12 60.41e9, c44 28.34e9, gdot0_slip 0.001,' \
                     ' n_slip 20, tau0_slip 31e6, tausat_slip 63e6, a_slip 2.25, h0_slipslip 75e6, ' \
                     'interaction_slipslip 1 1 1.4 1.4 1.4 1.4, atol_resistance 1'.split(', ')

        mat.writelines('lattice_structure bcc\n')
        for p in param_list:
            mat.writelines(p + '\n')

        # Microstructure
        mat.writelines('\n#--------------------# \n')
        mat.writelines('<microstructure> \n')
        mat.writelines('#--------------------# \n')

        for v in range(n_grains):
            if v + 1 < 10:
                GrainNo = '[Grain00' + str(v + 1) + ']'
                texture = 'texture 00' + str(v + 1)
            elif v + 1 >= 100:
                GrainNo = '[Grain' + str(v + 1) + ']'
                texture = 'texture ' + str(v + 1)
            else:
                GrainNo = '[Grain0' + str(v + 1) + ']'
                texture = 'texture 0' + str(v + 1)
            mat.writelines(GrainNo + '\n' + 'crystallite 1\n')
            phase = int(grains_df['phaseID'][v])
            mat.writelines('(constituent)  phase {}   '.format(phase) + texture + '   fraction 1.0\n')

        if band == True:
            v_martensite = n_grains
            if v_martensite + 1 < 10:
                GrainNo = '[Grain00' + str(v_martensite + 1) + ']'
                texture = 'texture 00' + str(v_martensite + 1)
            elif v_martensite + 1 >= 100:
                GrainNo = '[Grain' + str(v_martensite + 1) + ']'
                texture = 'texture ' + str(v_martensite + 1)
            else:
                GrainNo = '[Grain0' + str(v_martensite + 1) + ']'
                texture = 'texture 0' + str(v_martensite + 1)
            mat.writelines(GrainNo + '\n' + 'crystallite 1\n')
            mat.writelines('(constituent)  phase 2   ' + texture + '   fraction 1.0\n')
        else:
            pass

        # Texture
        mat.writelines('\n#--------------------# \n')
        mat.writelines('<texture> \n')
        mat.writelines('#--------------------# \n')
        if band:
            for v in range(n_grains + 1):
                if v + 1 < 10:
                    GrainNo = '[Grain00' + str(v + 1) + ']'
                elif v + 1 >= 100:
                    GrainNo = '[Grain' + str(v + 1) + ']'
                else:
                    GrainNo = '[Grain0' + str(v + 1) + ']'
                angles = np.random.random(size=2) * 360
                angles2 = np.random.random(size=2) * 180
                mat.writelines(GrainNo + '\n')
                mat.writelines('(gauss)  phi1 {}    Phi {}    phi2 {}   scatter 0.0   fraction 1.0\n'
                               .format(angles[0], angles2[0], angles[1]))
        else:
            for v in range(n_grains):
                if v + 1 < 10:
                    GrainNo = '[Grain00' + str(v + 1) + ']'
                elif v + 1 >= 100:
                    GrainNo = '[Grain' + str(v + 1) + ']'
                else:
                    GrainNo = '[Grain0' + str(v + 1) + ']'
                angles = np.random.random(size=3) * 360
                angles2 = np.random.random(size=2) * 180
                mat.writelines(GrainNo + '\n')
                mat.writelines('(gauss)  phi1 {}    Phi {}    phi2 {}   scatter 0.0   fraction 1.0\n'
                               .format(angles[0], angles2[1], angles[2]))


def make_geom(rve, grid_size, spacing, store_path) -> None:
    # Start with the plot
    start = int(rve.__len__() / 4)
    stop = int(rve.__len__() / 4 + rve.__len__() / 2)
    real_rve = rve[start:stop, start:stop, start:stop]
    x = np.linspace(0, grid_size * spacing, num=real_rve.__len__() + 1)
    y = np.linspace(0, grid_size * spacing, num=real_rve.__len__() + 1)
    z = np.linspace(0, grid_size * spacing, num=real_rve.__len__() + 1)
    xx, yy, zz = np.meshgrid(x, y, z)
    grid = pv.StructuredGrid(xx, yy, zz)
    grid.cell_arrays['grains'] = real_rve.flatten()
    plotter = pv.Plotter(window_size=(800, 600))
    plotter.add_mesh(grid)
    plotter.add_axes()
    plotter.show(screenshot=store_path + '/rve.png')

    homogenization = 1
    with open(store_path + '/' + 'RVE.geom', 'w') as geom:
        geom.writelines('4 header\n')  # Kommt drauf an wie viele Header es gibt
        geom.writelines('grid\ta {0}\tb {0}\tc {0}\n'.format(grid_size))
        geom.writelines('size\tx {0}\ty {0}\tz {0}\n'.format(spacing / 1000))
        geom.writelines('microstructures {}\n'.format(rve.max()))
        geom.writelines('homogenization\t{}\n'.format(homogenization))

        # Write the values
        start = int(rve.__len__() / 4)
        stop = int(rve.__len__() / 4 + rve.__len__() / 2)
        real_rve = rve[start:stop, start:stop, start:stop]
        s = real_rve.__len__()
        print(s)
        for i in range(s):
            for j in range(s):
                str1 = ''
                for v in real_rve[j, i, :]:
                    v = int(v)
                    if v / 10 < 1:  # Einstellig
                        str1 = str1 + '   {}'.format(v)
                    elif v / 10 >= 10:
                        str1 = str1 + ' {}'.format(v)
                    else:
                        str1 = str1 + '  {}'.format(v)

                str1 = str1.strip()
                geom.writelines(str1 + '\n')


def make_load(store_path) -> None:
    # Einfachster Load case
    # 1 Last am Ende
    with open(store_path + '/' + 'loadX.load', 'w') as load:
        load.writelines(
            'fdot * 0 0  0 2.0e-2 0  0 0 *  stress  0 * *   * * *   * * 0  time 10  incs 100')  # rot 0.70710678 -0.70710678 0.0  0.70710678 0.70710678 0.0  0.0 0.0 1.0')


def make_load_from_defgrad(file_path, store_path, single=True):
    """
    Writes the defgrad for Damask-Spectral .load file
    Two possibilities:
        1.) Write the deformation gradient RATE for each extracted element from Abaqus
            (so 100 incs in Abaqus -> 100 Damask load cases (single=False)
            Currently leading to odd results!!!!
        2.) Use the resulting defgrad (last increment) as single load case (single=True)
    """
    defgrad_dataframe = pd.read_csv(file_path, sep='\t', header=0)

    # Step 1: Correct the F11 - F33 cols
    defgrad_dataframe['F11_new'] = defgrad_dataframe['F11'] - 1
    defgrad_dataframe['F22_new'] = defgrad_dataframe['F22'] - 1
    defgrad_dataframe['F33_new'] = defgrad_dataframe['F33'] - 1
    defgrad_dataframe['F11_new'].iloc[0] = 0
    defgrad_dataframe['F22_new'].iloc[0] = 0
    defgrad_dataframe['F33_new'].iloc[0] = 0

    # Step 2: Get the STEP-values
    F11_step = defgrad_dataframe['F11_new'].diff().tolist()
    F22_step = defgrad_dataframe['F22_new'].diff().tolist()
    F33_step = defgrad_dataframe['F33_new'].diff().tolist()
    F12_step = defgrad_dataframe['F12'].diff().tolist()
    F23_step = defgrad_dataframe['F23'].diff().tolist()
    F31_step = defgrad_dataframe['F31'].diff().tolist()
    F21_step = defgrad_dataframe['F21'].diff().tolist()
    F32_step = defgrad_dataframe['F32'].diff().tolist()
    F13_step = defgrad_dataframe['F13'].diff().tolist()
    time_step = defgrad_dataframe['time'].diff().tolist()

    # Step 3: Write the data
    if not single:
        print('WARNING: Load case currently results in odd results in DAMASK!!')
        n_frames = defgrad_dataframe.__len__()
        with open(store_path + '/loadpath.load', 'w') as loadpath:
            for i in range(1, n_frames):
                loadpath.writelines('Fdot {} {} {} {} {} {} {} {} {} stress * * * * 0 * * * 0 time {} incs 1\n'.format(
                    F11_step[i], F12_step[i], F13_step[i],
                    F21_step[i], '*', F23_step[i],
                    F31_step[i], F32_step[i], '*',
                    1  # Time has to be zero to ensure valid sum of defgrad
                ))
    else:
        F11 = defgrad_dataframe['F11_new'].iloc[-1]
        F12 = defgrad_dataframe['F12'].iloc[-1]
        F13 = defgrad_dataframe['F13'].iloc[-1]
        F21 = defgrad_dataframe['F21'].iloc[-1]
        F23 = defgrad_dataframe['F23'].iloc[-1]
        F31 = defgrad_dataframe['F31'].iloc[-1]
        F32 = defgrad_dataframe['F32'].iloc[-1]
        with open(store_path + '/loadpath.load', 'w') as loadpath:
            loadpath.writelines('Fdot {} {} {} {} {} {} {} {} {} stress * * * * 0 * * * 0 time {} incs 100\n'.format(
                F11, F12, F13,
                F21, '*', F23,
                F31, F32, '*',
                1
            ))


"""
INPUT FOR DAMASK 3 from here on!
"""


def write_material(store_path: str, grains: list) -> None:
    matdata = damask.ConfigMaterial()

    # Homog
    matdata['homogenization']['SX'] = {'N_constitutents': 1, 'mechanical': {'type': 'pass'}}

    # Phase
    ferrite = {'lattice': 'cI',
               'elastic': {'type': 'Hooke', 'C_11': 10, 'C_12': 10, 'C_44': 10},
               'plastic': {'type': 'phenopowerlaw',
                           'N_sl': [12],
                           'a_sl': 2.0,
                           'dot_gamma_0_sl': 0.001,
                           'h0_sl_sl': 10,
                           'h_sl_sl': [1, 1, 1.4, 1.4, 1.4, 1.4],
                           'n_sl': 20,
                           'xi_0_sl': 10,
                           'xi_inf_sl': 10}
               }
    matdata['phase']['Ferrite'] = ferrite
    matdata['phase']['Martensite'] = ferrite

    # Material
    for p in grains:
        if p == 1:
            matdata = matdata.material_add(phase=['Ferrite'], O=damask.Rotation.from_random(1),
                                           homogenization='SX')
        else:
            matdata = matdata.material_add(phase=['Martensite'], O=damask.Rotation.from_random(1),
                                           homogenization='SX')

    matdata.save(store_path + '/Material.yaml')


def write_load(store_path: str) -> None:
    load_case = damask.Config(solver={'mechanical': 'spectral_basic'},
                                      loadstep=[])

    F = [1e-2, 0, 0, 0, 'x', 0, 0, 0, 'x']
    P = ['x' if i != 'x' else 0 for i in F]

    load_case['loadstep'].append({'boundary_conditions': {},
                                  'discretization': {'t': 10., 'N': 100}, 'f_out': 4})
    load_case['loadstep'][0]['boundary_conditions']['mechanical'] = \
        {'P': [P[0:3], P[3:6], P[6:9]],
         'F': [F[0:3], F[3:6], F[6:9]]}

    load_case.save(store_path + '/load.yaml')


def write_grid(store_path: str, rve: np.ndarray, spacing: float) -> None:
    start = int(rve.__len__() / 4)
    stop = int(rve.__len__() / 4 + rve.__len__() / 2)
    real_rve = rve[start:stop, start:stop, start:stop]
    # Create the spatial reference
    grid = pv.UniformGrid()

    # Set the grid dimensions: shape + 1 because we want to inject our values on
    #   the CELL data
    grid.dimensions = np.array(real_rve.shape) + 1

    # Edit the spatial reference
    grid.origin = (0, 0, 0)  # The bottom left corner of the data set
    grid.spacing = (spacing, spacing, spacing)  # These are the cell sizes along each axis

    # Add the data values to the cell data
    grid.cell_arrays["material"] = real_rve.flatten(order="C")  # Flatten the array in C-Style

    # Now save the grid
    grid.plot(screenshot=store_path + '/rve.png')
    grid.save(store_path + '/grid.vti')


if __name__ == '__main__':
    # Test plotting function
    rve = np.load('../../ExampleInput/RVE_Numpy.npy')

    write_grid(store_path='../../OutputData', rve=rve)
