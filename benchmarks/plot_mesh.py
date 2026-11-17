"""Pictures of the meshing steps: staircase surface next to the smoothed surface (M3).

Example:
    venv/Scripts/python benchmarks/plot_mesh.py

Writes to benchmarks/results/figures/:
    mesh_two_contact_patches.png   inner grain boundaries of the corpus case, staircase | smoothed
    mesh_ring_triple_line.png      the same for the cube on the boundary plane (ring triple line)
    mesh_rve_grain.png             one grain of the generated RVE lognormal 30^3 with its lines
    mesh_kinks.png                 kink angles between neighbouring triangles, staircase vs smoothed
Surfaces are coloured by grain, junction lines are black, anchors red.

and VTK MultiBlocks for ParaView to benchmarks/results/mesh/model/<name>_staircase.vtm and
<name>_smoothed.vtm (blocks 'surfaces', 'curves', 'points'; cell data surface, outer, other,
order, imprinted) for two_contact_patches, ring_triple_line, periodic_imprint and the RVE.
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

import pyvista as pv  # noqa: E402

from dragen.config import RveConfig  # noqa: E402
from dragen.mesh import corpus, quality  # noqa: E402
from dragen.mesh.cleanup import ManifoldCleaner  # noqa: E402
from dragen.mesh.model import build_model  # noqa: E402
from dragen.mesh.smoothing import smooth  # noqa: E402
from dragen.mesh.surface import is_box_face  # noqa: E402
from dragen.mesh.topology import build_topology  # noqa: E402

OUT = os.path.join(REPO_ROOT, 'benchmarks', 'results', 'figures')
VTK_OUT = os.path.join(REPO_ROOT, 'benchmarks', 'results', 'mesh', 'model')
pv.OFF_SCREEN = True


def prepare(labels, spacing, periodic):
    labels, _ = ManifoldCleaner(periodic).run(labels)
    model = build_model(build_topology(labels, spacing, periodic))
    return model, smooth(model, spacing).points


def add(plotter, model, points, keep, lines, title):
    tris = model.oriented[keep]
    faces = np.column_stack([np.full(len(tris), 3), tris]).ravel()
    surf = pv.PolyData(points, faces=faces)
    outer = np.array([s.regions[0] for s in model.surfaces])[model.topology.patch[keep]]
    surf.cell_data['grain'] = outer
    plotter.add_mesh(surf, scalars='grain', cmap='tab20', show_scalar_bar=False, show_edges=True,
                     edge_color='#555555', line_width=0.3, smooth_shading=False)
    for line in lines:
        plotter.add_mesh(pv.lines_from_points(points[line.vertices]), color='black', line_width=3)
    anchors = np.unique(np.concatenate([[ln.vertices[0], ln.vertices[-1]] for ln in lines])) if lines else []
    if len(anchors):
        plotter.add_mesh(pv.PolyData(points[anchors]), color='red', point_size=9,
                         render_points_as_spheres=True)
    plotter.add_text(title, font_size=11)


def side_by_side(model, smoothed, keep, lines, path, camera=None):
    p = pv.Plotter(shape=(1, 2), window_size=(1800, 850), off_screen=True)
    for col, (pts, title) in enumerate(((model.mesh.points, 'Stufenoberflaeche (Voxel)'),
                                         (smoothed, 'geglaettet (M3)'))):
        p.subplot(0, col)
        add(p, model, pts, keep, lines, title)
        if camera:
            p.camera_position = camera
        else:
            p.view_isometric()
    p.link_views()
    p.screenshot(path)
    p.close()
    print(path)


def inner_only(model):
    regions = np.array([s.regions for s in model.surfaces])
    return ~is_box_face(regions[model.topology.patch, 1])


def export_vtk(model, smoothed, name):
    os.makedirs(VTK_OUT, exist_ok=True)
    for tag, pts in (('staircase', model.mesh.points), ('smoothed', smoothed)):
        blocks = model.to_pyvista()
        for key in ('surfaces', 'curves'):
            blocks[key].points = pts
        blocks['points'].points = pts[model.points]
        path = os.path.join(VTK_OUT, f'{name}_{tag}.vtm')
        blocks.save(path)
        print(path)


def corpus_pictures():
    for case in (corpus.two_contact_patches(), corpus.ring_triple_line(), corpus.periodic_imprint()):
        model, x = prepare(case.labels, (1, 1, 1), case.periodic)
        export_vtk(model, x, case.name)
        if case.name == 'periodic_imprint':
            continue
        keep = inner_only(model)
        lines = [ln for ln in model.curves if all(r > 0 for r in ln.regions)]
        side_by_side(model, x, keep, lines, os.path.join(OUT, f'mesh_{case.name}.png'))


def rve_pictures(tag='lognormal_box15_res2_seed1'):
    src = os.path.join(REPO_ROOT, 'benchmarks', 'results', 'rework', tag)
    labels = np.load(os.path.join(src, 'labels.npz'))['labels']
    box = RveConfig.load(os.path.join(src, 'config.yaml')).box
    model, x = prepare(labels, box.spacing, box.periodic)
    export_vtk(model, x, tag)
    # a grain that does not touch the box, as large as possible
    regions = np.array([s.regions for s in model.surfaces])
    touches_box = {int(g) for g, o in regions if is_box_face(o)}
    sizes = np.bincount(labels.ravel())
    candidates = [g for g in np.argsort(-sizes) if g > 0 and g not in touches_box]
    grain = int(candidates[0]) if candidates else int(np.argmax(sizes[1:]) + 1)
    shell = model.shells[grain][:, 0]
    keep = np.isin(model.topology.patch, shell)
    lines = [ln for ln in model.curves if grain in ln.regions]
    side_by_side(model, x, keep, lines, os.path.join(OUT, 'mesh_rve_grain.png'))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    k0 = quality.surface_kinks(model.mesh.points, model.oriented, model.topology.patch)
    k = quality.surface_kinks(x, model.oriented, model.topology.patch)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 100, 51)
    ax.hist(k0, bins=bins, alpha=0.6, label=f'Stufen (Anteil 90°: {np.mean(k0 >= 89.9):.0%})')
    ax.hist(k, bins=bins, alpha=0.6, label=f'geglaettet (Anteil 90°: {np.mean(k >= 89.9):.1%})')
    ax.set_yscale('log')
    ax.set_xlabel('Knick zwischen benachbarten Dreiecken einer Flaeche [°]')
    ax.set_ylabel('Anzahl Kanten')
    ax.set_title(f'{tag} ({labels.shape[0]}³ Voxel)')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT, 'mesh_kinks.png')
    fig.savefig(path, dpi=150)
    print(path)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    corpus_pictures()
    rve_pictures()
