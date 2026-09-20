"""Screenshots of the labelled RVE: one per ID level plus a side-by-side comparison.

Label IDs are plotted as random RGB rather than through a colormap, because neighbouring packets
and blocks have consecutive IDs and any continuous colormap makes them indistinguishable.
"""
import numpy as np
import pyvista as pv

from dragen.substructure.config import SubsConfig
from dragen.utilities.InputInfo import RveInfo

BASE_SCALARS = [('phaseID', 'phase_id.png', 0),
                ('GrainID', 'grain_id.png', 1),
                ('PacketID', 'packet_id.png', 2),
                ('BlockID', 'block_id.png', 3)]


def _assign_random_rgb(mesh: pv.DataSet, label_name: str, rgb_name: str, seed: int) -> np.ndarray:
    if label_name not in mesh.cell_data:
        raise KeyError(f'Missing cell_data array: {label_name}')

    labels = np.asarray(mesh.cell_data[label_name]).astype(np.int64)
    unique_labels = np.unique(labels)

    rng = np.random.default_rng(seed)
    colors = rng.integers(40, 256, size=(len(unique_labels), 3), dtype=np.uint8)

    mesh.cell_data[rgb_name] = colors[np.searchsorted(unique_labels, labels)]

    return labels


def _save_plot(cfg: SubsConfig, mesh: pv.DataSet, scalars: str, filename: str, seed: int) -> None:
    save_path = cfg.path(filename)
    mesh_copy = mesh.copy(deep=True)

    rgb_name = scalars + '_RGB'
    labels = _assign_random_rgb(mesh_copy, scalars, rgb_name, seed)

    plotter = pv.Plotter(off_screen=True)
    plotter.set_background('white')

    if scalars == 'VariantID':
        # 24 KS variants map cleanly onto a categorical colormap, and the legend is meaningful.
        plotter.add_mesh(mesh_copy, scalars=scalars, cmap='tab20', categories=True,
                         show_scalar_bar=True, scalar_bar_args={'title': scalars},
                         show_edges=False)
    else:
        # Invisible categorical layer purely to get a scalar bar, RGB layer for the actual colors.
        plotter.add_mesh(mesh_copy, scalars=labels, cmap='tab20', categories=True,
                         show_scalar_bar=True, opacity=0.0, scalar_bar_args={'title': scalars})
        plotter.add_mesh(mesh_copy, scalars=rgb_name, rgb=True, show_edges=False)

    plotter.view_isometric()
    plotter.screenshot(save_path)
    plotter.close()

    RveInfo.LOGGER.info(f'Saved: {save_path}')


def _save_combined_plot(cfg: SubsConfig, mesh: pv.DataSet, scalar_list, seeds,
                        filename: str) -> None:
    save_path = cfg.path(filename)

    plotter = pv.Plotter(off_screen=True, shape=(1, len(scalar_list)), window_size=(2400, 700))
    plotter.set_background('white')

    for i, scalars in enumerate(scalar_list):
        if scalars not in mesh.cell_data:
            raise KeyError(f'Missing cell_data array: {scalars}')

        plotter.subplot(0, i)

        mesh_copy = mesh.copy(deep=True)
        rgb_name = scalars + '_RGB'
        _assign_random_rgb(mesh_copy, scalars, rgb_name, seeds[i])

        plotter.add_mesh(mesh_copy, scalars=rgb_name, rgb=True, show_edges=False)
        plotter.add_text(scalars, position='upper_left', font_size=14, color='black')
        plotter.view_isometric()

    plotter.link_views()
    plotter.screenshot(save_path)
    plotter.close()

    RveInfo.LOGGER.info(f'Saved combined plot: {save_path}')


def save_substructure_plots(cfg: SubsConfig, mesh: pv.DataSet) -> None:
    experimental = 'ExperimentalTemplateGrainID' in mesh.cell_data

    scalars = list(BASE_SCALARS)
    if experimental:
        # In experimental mode the template grain, not the KS variant, is what is worth seeing.
        scalars.append(('ExperimentalTemplateGrainID', 'experimental_template_grain_id.png', 5))
    else:
        scalars.append(('VariantID', 'variant_id.png', 4))

    for name, filename, seed in scalars:
        _save_plot(cfg, mesh, name, filename, seed)

    combined = [name for name, _, _ in BASE_SCALARS]
    combined_seeds = [seed for _, _, seed in BASE_SCALARS]

    if not experimental:
        combined.append('VariantID')
        combined_seeds.append(4)

    _save_combined_plot(cfg, mesh, combined, combined_seeds,
                        'combined_phase_grain_packet_block.png')
