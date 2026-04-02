"""
Visualization routines for the final generated sRVE
--> Routines work with the DAMASK-Output!

Author: Niklas Fehlemann

Phases:
1: IPF
2: Black
3: Brown
4: Grey
5: IPF with edges
6: Black with edges
7: white

"""
import pyvista as pv
import numpy as np

def plot_srve(grid, path):
    # --- Farb- und Transparenz-Konfiguration pro Phase ---
    # Solid-Color Phasen: RGB-Werte 0–255
    PHASE_SOLID_COLORS = {
        2: (0,   0,   0  ),   # Schwarz
        3: (101, 67,  33 ),   # Braun (saddlebrown)
        4: (128, 128, 128),   # Grau
        6: (255, 255, 255),   # Weiß
        7: (0,   0,   0  ),   # Schwarz (transparent)
    }

    # --- Plotter aufsetzen ---
    plotter = pv.Plotter(window_size=[1024, 1024], off_screen=True)

    # --- Erst alle opaken Phasen ---
    for phase_id in [1, 2, 3, 4, 7]:
        indices = np.where(grid['phases'] == phase_id)[0]
        if len(indices) == 0:
            continue
        sub_grid = grid.extract_cells(indices)

        if phase_id == 1:
            plotter.add_mesh(sub_grid, scalars='IPF_[0 0 1]', rgb=True,
                             opacity=1.0, show_scalar_bar=False)
        else:
            color = [c / 255.0 for c in PHASE_SOLID_COLORS[phase_id]]
            plotter.add_mesh(sub_grid, color=color, opacity=1.0,
                             show_scalar_bar=False)

    # --- Dann Phasen mit Voxelgrenzen ---
    for phase_id in [5, 6]:
        indices = np.where(grid['phases'] == phase_id)[0]
        if len(indices) == 0:
            continue

        sub_grid = grid.extract_cells(indices)

        if phase_id == 5:
            rgb = sub_grid['IPF_[0 0 1]'].copy().astype(float)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            blend_factor = 0.5
            rgb_blended = rgb * (1 - blend_factor) + np.ones_like(rgb) * blend_factor
            rgb_blended = np.clip(rgb_blended, 0.0, 1.0)
            sub_grid['IPF_P5'] = rgb_blended

            # Flächen rendern
            plotter.add_mesh(sub_grid, scalars='IPF_P5', rgb=True,
                             show_scalar_bar=False)
            # Voxelgrenzen drüber
            voxel_edges = sub_grid.extract_all_edges()
            plotter.add_mesh(voxel_edges, color='white', line_width=0.5,
                             show_scalar_bar=False)

        elif phase_id == 6:
            # Flächen rendern
            plotter.add_mesh(sub_grid, color=[0.2, 0.2, 0.2],
                             show_scalar_bar=False)
            # Voxelgrenzen drüber
            voxel_edges = sub_grid.extract_all_edges()
            plotter.add_mesh(voxel_edges, color='white', line_width=0.5,
                             show_scalar_bar=False)

    # Kanten, Achsen, Kamera wie gehabt ...
    grid_edges = grid.extract_feature_edges(45)
    plotter.add_mesh(grid_edges, color='grey', line_width=1)
    plotter.add_axes(line_width=5, color='black', xlabel='RD', ylabel='ND', zlabel='TD')
    camera = plotter.camera
    camera.roll = 0
    camera.up = (0, 1, 0)
    plotter.camera = camera
    plotter.show(screenshot=path + f'/SRVE_IPF_001.png', window_size=[1024, 1024])