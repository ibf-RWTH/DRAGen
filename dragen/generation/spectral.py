"""
INPUT FOR DAMASK 3 from here on!
"""
import damask
import numpy as np
import pandas as pd
import pyvista as pv
import os
import yaml

from dragen.postprocessing import visualization as viz
from dragen.utilities.InputInfo import RveInfo


def write_material(store_path: str, grains: list, angles: pd.DataFrame) -> None:
    matdata = damask.ConfigMaterial()

    # Homog
    matdata['homogenization']['SX'] = {'N_constituents': 1, 'mechanical': {'type': 'pass'}}

    # https://doi.org/10.1016/j.commatsci.2022.111903
    ferrite = {'lattice': 'cI',
               'mechanical':
                   {'output': ['F', 'P', 'F_p'],
                    'elastic': {'type': 'Hooke', 'C_11': 233.3e9, 'C_12': 135.5e9, 'C_44': 128e9},
                    'plastic': {'type': 'phenopowerlaw',
                                'N_sl': [12, 12],
                                'a_sl': [4.5, 4.5],
                                'dot_gamma_0_sl': [0.001, 0.001],
                                'h_0_sl-sl': [625e6, 625e6],
                                'h_sl-sl': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                'n_sl': [5, 5],
                                'xi_0_sl': [150e6, 150e6],
                                'xi_inf_sl': [400e6, 400e6]
                                }
                    }
               }

    # As an example, we include dislotwin code here:
    # https://damask-multiphysics.org/documentation/file_formats/materialpoint/phase/index.html#dislotwin-twip-trip
    austenite = {'lattice': 'cF',
 'mechanical':
     {'output': ['F', 'P', 'F_p'],
      'elastic': {'type': 'Hooke', 'C_11': 233.3e9, 'C_12': 135.5e9, 'C_44': 128e9},
      'plastic': {'type': 'dislotwin',
                  'output': ['rho_mob', 'rho_dip', 'gamma_sl', 'Lambda_sl', 'tau_pass', 'f_tw', 'Lambda_tw', 'f_tr'],
                  # Glide
                  'N_sl': [12],
                  'f_edge': [1.0],
                  'b_sl': [2.56e-10],        # a/sqrt(2)
                  'Q_sl': [3.5e-19],
                  'p_sl': [0.325],
                  'q_sl': [1.55],
                  'B': [0.001],
                  'i_sl': [30.0],
                  'v_0': [1.4e+3],
                  'tau_0': [5.5e+8],         # adjusted
                  'D_a': 2.0,
                  'Q_cl': 3.0e-19,
                  'rho_mob_0': [5.0e+10],
                  'rho_dip_0': [5.0e+10],
                  'h_sl-sl': [0.122, 0.122, 0.625, 0.07, 0.137, 0.137, 0.122],
                  # Twin
                  'N_tw': [12],
                  'b_tw': [1.47e-10],        # a_cF/sqrt(6)
                  'L_tw': 1.91e-7,           # 1300 *b_tw
                  'i_tw': 10.0,
                  't_tw': [5.0e-8],
                  'p_tw': [7],               # A, adjusted
                  'h_tw-tw': [1.0, 1.0],
                  'h_sl-tw': [1.0, 1.0, 1.0],
                  # Transformation
                  'N_tr': [12],
                  'b_tr': [1.47e-10],        # a_cF/sqrt(6)
                  'L_tr': 2.21e-7,           # 1500 *b_tr
                  'i_tr': 10.0,              # adjusted
                  't_tr': [1.0e-7],
                  'p_tr': [4],               # B, adjusted
                  'V_mol': 7.09e-6,
                  'c/a_hP': 1.633,
                  'Delta_G': 1.2055e+2,
                  'Delta_G,T': 2.5515,
                  'Delta_G,T^2': 1.4952e-3,
                  'h_tr-tr': [1.0, 1.0],
                  'h_sl-tr': [1.5, 1.5, 1.5],
                  # Twin & Transformation
                  'T_ref': 293.15,
                  'Gamma_sf': 2.833e-2,
                  'Gamma_sf,T': 1.214e-4,
                  'Gamma_sf,T^2': 1.473e-7,
                  'x_c': 1.0e-9,
                  'V_cs': 1.67e-29,
                  # Slip & Twin & Transformation
                  'D': 5.0e-5
                  }
      }
 }

    # https://doi.org/10.1016/j.actamat.2025.121321
    martensite = {'lattice': 'cI',
                  'mechanical':
                      {'output': ['F', 'P', 'F_p'],
                       'elastic': {'type': 'Hooke', 'C_11': 417.4e9, 'C_12': 242.4e9, 'C_44': 211.1e9},
                       'plastic': {'type': 'isotropic',
                                   'xi_0': 1e9,
                                   'xi_inf': 2e9,
                                   'dot_gamma_0': 0.001,
                                   'n': 20,
                                   'M': 3,
                                   'h_0': 5e9,
                                   'a': 3.25,
                                   'dilatation': False
                                   }
                       }
                  }

    # Bainite also isotropic - random parameters
    bainite = {'lattice': 'cI',
                  'mechanical':
                      {'output': ['F', 'P', 'F_p'],
                       'elastic': {'type': 'Hooke', 'C_11': 417.4e9, 'C_12': 242.4e9, 'C_44': 211.1e9},
                       'plastic': {'type': 'isotropic',
                                   'xi_0': 0.5e9,
                                   'xi_inf': 0.5e9,
                                   'dot_gamma_0': 0.001,
                                   'n': 20,
                                   'M': 3,
                                   'h_0': 5e9,
                                   'a': 1.25,
                                   'dilatation': False
                                   }
                       }
                  }

    # Pearlite from unpublished TRR188 data
    pearlite = {'lattice': 'cI',
                  'mechanical':
                      {'output': ['F', 'P', 'F_p'],
                       'elastic': {'type': 'Hooke', 'C_11': 417.4e9, 'C_12': 242.4e9, 'C_44': 211.1e9},
                       'plastic': {'type': 'isotropic',
                                   'xi_0': 150e6,
                                   'xi_inf': 600e6,
                                   'dot_gamma_0': 0.001,
                                   'n': 20,
                                   'M': 3,
                                   'h_0': 8e9,
                                   'a': 1.25,
                                   'dilatation': False
                                   }
                       }
                  }

    # Purely elastic, very stiff
    inclusion = {'lattice': 'cI',
                 'mechanical':
                     {'output': ['F', 'P', 'F_p'],
                      'elastic': {'type': 'Hooke', 'C_11': 417.4e10, 'C_12': 242.4e10, 'C_44': 211.1e10},
                      }
                 }

    # Dummy phase, should be replaced with martensite or pearlite
    band = {'lattice': 'cI',
                 'mechanical':
                     {'output': ['F', 'P', 'F_p'],
                      'elastic': {'type': 'Hooke', 'C_11': 417.4e10, 'C_12': 242.4e10, 'C_44': 211.1e10},
                      }
                 }

    if 1 in grains:
        matdata['phase']['Ferrite'] = ferrite
    if 2 in grains:
        matdata['phase']['Martensite'] = martensite
    if 3 in grains:
        matdata['phase']['Pearlite'] = pearlite
    if 4 in grains:
        matdata['phase']['Bainite'] = bainite
    if 5 in grains:
        matdata['phase']['Austenite'] = austenite
    if 6 in grains:
        matdata['phase']['ThirdPhase'] = inclusion
    if 7 in grains:
        matdata['phase']['Band'] = band

    # Material
    i = 0
    for p in grains:
        if p == 1:
            o = damask.Rotation.from_Euler_angles(angles.loc[i].to_numpy(), degrees=True)
            matdata = matdata.material_add(phase=['Ferrite'], O=o, homogenization='SX')
        elif p == 2:
            matdata = matdata.material_add(phase=['Martensite'], O=damask.Rotation.from_random(1),
                                           homogenization='SX')
        elif p == 3:
            matdata = matdata.material_add(phase=['Pearlite'], O=damask.Rotation.from_random(1),
                                           homogenization='SX')
        elif p == 4:
            matdata = matdata.material_add(phase=['Bainite'], O=damask.Rotation.from_random(1),
                                           homogenization='SX')
        elif p == 5:
            o = damask.Rotation.from_Euler_angles(angles.loc[i].to_numpy(), degrees=True)
            matdata = matdata.material_add(phase=['Austenite'], O=o,
                                           homogenization='SX')
        elif p == 6:
            matdata = matdata.material_add(phase=['ThirdPhase'], O=damask.Rotation.from_random(1),
                                           homogenization='SX')
        elif p == 7:
            matdata = matdata.material_add(phase=['Band'], O=damask.Rotation.from_random(1),
                                           homogenization='SX')
        i += 1

    print('Anzahl materialien in Materials.yaml: ', grains.__len__())
    matdata.save(store_path + '/material.yaml')


def write_load(store_path: str) -> None:
    load_case = damask.LoadcaseGrid(solver={'mechanical': 'spectral_polarization'}, loadstep=[])

    F = [0.001, 0, 0, 0, 'x', 0, 0, 0, 'x']
    P = ['x' if i != 'x' else 0 for i in F]

    load_case['loadstep'].append({'boundary_conditions': {},
                                  'discretization': {'t': 1, 'N': 10}, 'f_out': 1})
    load_case['loadstep'][0]['boundary_conditions']['mechanical'] = \
        {'P': [P[0:3], P[3:6], P[6:9]],
         'dot_F': [F[0:3], F[3:6], F[6:9]]}

    load_case.save(store_path + '/load.yaml')


def write_grid(store_path: str, rve: np.ndarray, spacing: float) -> None:
    l = np.array([0, 0, 1])
    if rve.dtype != np.int64:
        rve = rve.astype('int64')
    rve = rve - 1
    if RveInfo.box_size_y is None and RveInfo.box_size_z is None:
        grid = damask.GeomGrid(material=rve, size=[RveInfo.box_size, RveInfo.box_size, RveInfo.box_size])
    elif RveInfo.box_size_z is None and RveInfo.box_size_y is not None:
        grid = damask.GeomGrid(material=rve, size=[RveInfo.box_size, RveInfo.box_size_y, RveInfo.box_size])
    elif RveInfo.box_size_y is None and RveInfo.box_size_z is not None:
        grid = damask.GeomGrid(material=rve, size=[RveInfo.box_size, RveInfo.box_size, RveInfo.box_size_z])
    else:
        grid = damask.GeomGrid(material=rve, size=[RveInfo.box_size, RveInfo.box_size_y, RveInfo.box_size_z])

    print('Anzahl Materialien im Grid', grid.N_materials)

    grid.save(fname=store_path + '/grid.vti', compress=True)

    grid2 = pv.read(os.path.join(store_path, r'grid.vti'))
    
    grid2['phi'] = [1 for _ in range(rve.shape[0]*rve.shape[1]*rve.shape[2])]
 
    with open(os.path.join(store_path, r'material.yaml'), 'r') as ym:
        ym = yaml.safe_load(ym)
    
    mat = ym['material']
    phase_list = list()
    for m in mat:
        phase = m['constituents'][0]['phase']
        if 'Ferrite' in phase:
            phase_list.append(1)
        elif 'Martensite' in phase:
            phase_list.append(2)
        elif 'Pearlite' in phase:
            phase_list.append(3)
        elif 'Bainite' in phase:
            phase_list.append(4)
        elif 'Austenite' in phase:
            phase_list.append(5)
        elif 'Band' in phase:
            phase_list.append(6)
        else:
            phase_list.append(7)

    material_ID = grid2['material'].flatten()

    phases = list(ym['phase'].keys())
    info = []

    for m in ym['material']:
        c = m['constituents'][0]
        phase = c['phase']
        info.append({'phase': phase,
                     'phaseID': phases.index(phase),
                     'lattice': ym['phase'][phase]['lattice'],
                     'O': c['O'],
                     })

    IPF = np.ones((len(material_ID), 3), np.uint8)
    for i, data in enumerate(info):
        IPF[np.where(material_ID == i)] = \
            np.uint8(damask.Orientation(data['O'], lattice=data['lattice']).IPF_color(l) * 255)

    grid2[f'IPF_{l}'] = IPF

    phase_array = np.zeros(grid2['material'].__len__())
    for k in range(phase_list.__len__()):
        if phase_list[k] == 1:
            points = grid2['material'].flatten(order='F') == k
            phase_array[points] = 1
        elif phase_list[k] == 2:
            points = grid2['material'].flatten(order='F') == k
            phase_array[points] = 2
        elif phase_list[k] == 3:
            points = grid2['material'].flatten(order='F') == k
            phase_array[points] = 3
        elif phase_list[k] == 4:
            points = grid2['material'].flatten(order='F') == k
            phase_array[points] = 4
        elif phase_list[k] == 5:
            points = grid2['material'].flatten(order='F') == k
            phase_array[points] = 5
        elif phase_list[k] == 6:
            points = grid2['material'].flatten(order='F') == k
            phase_array[points] = 6
        else:
            points = grid2['material'].flatten(order='F') == k
            phase_array[points] = 7
    
    grid2['phases'] = phase_array

    # Placeholder: Add Subs-Code here

    viz.plot_srve(grid2, store_path)
    
    grid2.save(os.path.join(store_path, r'grid.vti'))