"""
INPUT FOR DAMASK 3 from here on!
"""
import damask
import numpy as np
import pyvista as pv


def write_material(store_path: str, grains: list) -> None:
    matdata = damask.ConfigMaterial()

    # Homog
    matdata['homogenization']['SX'] = {'N_constituents': 1, 'mechanical': {'type': 'pass'}}

    # Phase
    ferrite = {'lattice': 'cI',
               'mechanical':
                   {'output': ['F', 'P'],
                    'elastic': {'type': 'Hooke', 'C_11': 233.3e9, 'C_12': 135.5e9, 'C_44': 128e9},
                    'plastic': {'type': 'phenopowerlaw',
                                'N_sl': [12, 12],
                                'a_sl': 4.5,
                                'atol_xi': 1,
                                'dot_gamma_0_sl': 0.001,
                                'h_0_sl-sl': 625e6,
                                'h_sl-sl': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                'n_sl': 5,
                                'xi_0_sl': [150e6, 150e6],
                                'xi_inf_sl': [400e6, 400e6]
                                }
                    }
               }

    martensite = {'lattice': 'cI',
                  'mechanical':
                      {'output': ['F', 'P'],
                       'elastic': {'type': 'Hooke', 'C_11': 417.4e9, 'C_12': 242.4e9, 'C_44': 211.1e9},
                       'plastic': {'type': 'isotropic',
                                   'xi_0': 250000000,
                                   'xi_inf': 750000000,
                                   'dot_gamma_0': 0.001,
                                   'n': 30,
                                   'M': 3,
                                   'h_0': 2500000000,
                                   'a': 1.25,
                                   'dilatation': False
                                   }
                       }
                  }

    inclusion = {'lattice': 'cI',
                 'mechanical':
                     {'output': ['F', 'P'],
                      'elastic': {'type': 'Hooke', 'C_11': 417.4e10, 'C_12': 242.4e10, 'C_44': 211.1e10},
                      }
                 }

    matdata['phase']['Ferrite'] = ferrite
    matdata['phase']['Martensite'] = martensite
    if 3 in grains:
        matdata['phase']['ThirdPhase'] = inclusion

    # Material
    print(grains)
    for p in grains:
        if p == 1:
            matdata = matdata.material_add(phase=['Ferrite'], O=damask.Rotation.from_random(1),
                                           homogenization='SX')
        elif p == 2:
            matdata = matdata.material_add(phase=['Martensite'], O=damask.Rotation.from_random(1),
                                           homogenization='SX')
        elif p == 3:
            matdata = matdata.material_add(phase=['ThirdPhase'], O=damask.Rotation.from_random(1),
                                           homogenization='SX')

    print('Anzahl materialien in Materials.yaml: ', grains.__len__())
    matdata.save(store_path + '/material.yaml')


def write_load(store_path: str) -> None:
    load_case = damask.Config(solver={'mechanical': 'spectral_polarization'},
                              loadstep=[])

    F = ['x', 0, 0, 0, 'x', 0, 0, 0, 2e-2]
    P = ['x' if i != 'x' else 0 for i in F]

    load_case['loadstep'].append({'boundary_conditions': {},
                                  'discretization': {'t': 10., 'N': 100}, 'f_out': 4})
    load_case['loadstep'][0]['boundary_conditions']['mechanical'] = \
        {'P': [P[0:3], P[3:6], P[6:9]],
         'dot_F': [F[0:3], F[3:6], F[6:9]]}

    load_case.save(store_path + '/load.yaml')


def write_grid(store_path: str, rve: np.ndarray, spacing: float, grains: list) -> None:
    start = int(rve.__len__() / 4)
    stop = int(rve.__len__() / 4 + rve.__len__() / 2)
    real_rve = rve[start:stop, start:stop, start:stop]
    print(np.asarray(np.unique(real_rve, return_counts=True)).T)
    real_rve = real_rve - 1  # Grid.vti starts at zero
    print(np.asarray(np.unique(real_rve, return_counts=True)).T)

    grid = damask.Grid(material=real_rve, size=[spacing, spacing, spacing])
    print('Anzahl Materialien im Grid', grid.N_materials)
    print(grid)
    grid.save(fname=store_path + '/grid.vti', compress=True)

    # Only for visualization
    pv.set_plot_theme('document')
    grid = pv.UniformGrid()

    # Set the grid dimensions: shape + 1 because we want to inject our values on
    #   the CELL data
    grid.dimensions = np.array(real_rve.shape) + 1

    # Edit the spatial reference
    grid.origin = (0, 0, 0)  # The bottom left corner of the data set
    grid.spacing = (spacing, spacing, spacing)  # These are the cell sizes along each axis

    # Add the data values to the cell data
    x = np.linspace(0, 10, int(rve.__len__() / 2) + 1, endpoint=True)
    y = np.linspace(0, 10, int(rve.__len__() / 2) + 1, endpoint=True)
    z = np.linspace(0, 10, int(rve.__len__() / 2) + 1, endpoint=True)
    xx, yy, zz = np.meshgrid(x, y, z)
    grid = pv.StructuredGrid(xx, yy, zz)

    grid.cell_arrays["material"] = real_rve.flatten(order="C")  # Flatten the array in C-Style
