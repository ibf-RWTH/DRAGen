"""Clean-up, topology (M1), surface model (M2) and smoothing (M3) on the RVEs of the reworked
generator.

Reads every benchmarks/results/rework/<tag>/labels.npz (written by rework_benchmark.py), removes
edge and point self-contacts (dragen.mesh.cleanup), builds the topology (dragen.mesh.topology)
and the discrete surface model (dragen.mesh.model) and checks them: every grain surface a closed
2-manifold, every grain still one body, Euler check equal to 1, no anchor inside a patch, and the
model checks (every curve used, closed shells, consistent orientation, exact periodic links),
then smooths the model and judges the smoothed surface with dragen.mesh.quality (volumes,
crossings, folds, kinks, anchors, box faces, periodic partners).

Example:
    venv/Scripts/python benchmarks/topology_benchmark.py                       # all stored RVEs
    venv/Scripts/python benchmarks/topology_benchmark.py --tag lognormal_box30_res2_seed1

Results: benchmarks/results/mesh/topology/<tag>/result.json and summary.csv.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from rework_benchmark import code_state  # noqa: E402
from dragen.analysis.metrics import components_per_grain  # noqa: E402
from dragen.config import RveConfig  # noqa: E402
from dragen.mesh.cleanup import ManifoldCleaner  # noqa: E402
from dragen.mesh import quality  # noqa: E402
from dragen.mesh.model import build_model  # noqa: E402
from dragen.mesh.smoothing import smooth  # noqa: E402
from dragen.mesh.quality import grain_surfaces  # noqa: E402
from dragen.mesh.topology import build_topology  # noqa: E402

SOURCE_DIR = os.path.join(REPO_ROOT, 'benchmarks', 'results', 'rework')
RESULT_DIR = os.path.join(REPO_ROOT, 'benchmarks', 'results', 'mesh', 'topology')


def run(tag: str) -> dict:
    src = os.path.join(SOURCE_DIR, tag)
    labels = np.load(os.path.join(src, 'labels.npz'))['labels']
    box = RveConfig.load(os.path.join(src, 'config.yaml')).box
    periodic, spacing = tuple(box.periodic), tuple(box.spacing)
    out = {'tag': tag, 'code': code_state(), 'shape': list(labels.shape), 'spacing': list(spacing),
           'periodic': list(periodic), 'n_grains': int(labels.max()), 'seconds': {}}

    t = time.perf_counter()
    cleaned, stats = ManifoldCleaner(periodic).run(labels)
    out['seconds']['cleanup'] = time.perf_counter() - t
    counts = np.bincount(labels.ravel())
    rel = [abs(v) / counts[g] for g, v in stats.voxels_gained.items()]
    out['cleanup'] = {'critical_before': stats.critical_before, 'critical_after': stats.critical_after,
                      'changed_voxels': stats.changed_voxels,
                      'changed_fraction': stats.changed_voxels / labels.size,
                      'grains_changed': len(stats.voxels_gained),
                      'max_relative_grain_change': float(max(rel)) if rel else 0.0,
                      'passes': stats.passes}

    t = time.perf_counter()
    topo = build_topology(cleaned, spacing, periodic)
    out['seconds']['topology'] = time.perf_counter() - t
    t = time.perf_counter()
    out['topology'] = topo.summary()
    out['seconds']['checks'] = time.perf_counter() - t
    orders = np.array([ln.order for ln in topo.lines if not ln.imprinted])
    lengths = np.array([len(ln.vertices) - 1 for ln in topo.lines])
    out['topology']['line_order_counts'] = {int(k): int(v) for k, v in zip(*np.unique(orders, return_counts=True))}
    out['topology']['line_edges_median'] = float(np.median(lengths)) if len(lengths) else 0.0
    out['topology']['mesh_triangles'] = int(len(topo.mesh.triangles))

    t = time.perf_counter()
    model = build_model(topo)
    out['seconds']['model'] = time.perf_counter() - t
    t = time.perf_counter()
    out['model'] = model.check()
    out['seconds']['model_checks'] = time.perf_counter() - t

    t = time.perf_counter()
    result = smooth(model, spacing)
    out['seconds']['smoothing'] = time.perf_counter() - t
    t = time.perf_counter()
    out['smoothing'] = smoothing_report(model, result, labels.shape, spacing, periodic)
    out['seconds']['smoothing_checks'] = time.perf_counter() - t

    surfaces = grain_surfaces(topo.mesh)
    out['checks'] = {
        'grains_not_closed_manifold': int(np.count_nonzero(~surfaces['closed_manifold'])),
        'grains_with_several_bodies': int(np.count_nonzero(components_per_grain(cleaned, periodic)[1:] > 1)),
        'grains_lost': int(np.count_nonzero(np.bincount(cleaned.ravel(), minlength=len(counts))[1:] == 0)),
        'euler_check': out['topology']['euler_check'],
        'isolated_anchors': out['topology']['isolated_anchors'],
    }
    return out


def smoothing_report(model, result, shape, spacing, periodic) -> dict:
    mesh, x = model.mesh, result.points
    tris = mesh.triangles
    kink0 = quality.surface_kinks(mesh.points, model.oriented, model.topology.patch)
    kink = quality.surface_kinks(x, model.oriented, model.topology.patch)
    angle = quality.triangle_quality(x, tris)['min_angle']
    box = np.array(shape) * np.array(spacing)
    box_tris = tris[(mesh.grains < 0).any(axis=1)]
    per = quality.periodic_mismatch(x, box_tris, (0, 0, 0), box, [a for a in range(3) if periodic[a]])
    locked = model.box_lock
    rep = dict(result.stats)
    rep.update({
        'self_intersections': quality.self_intersections(x, tris),
        'folded_triangles': quality.folded_triangles(mesh.points, x, model.oriented),
        'kink_deg': {'staircase_median': float(np.median(kink0)), 'median': float(np.median(kink)),
                     'p90': float(np.percentile(kink, 90)), 'p99': float(np.percentile(kink, 99)),
                     'share_ge_90_staircase': float(np.mean(kink0 >= 89.9)),
                     'share_ge_90': float(np.mean(kink >= 89.9))},
        'min_angle_deg': {'min': float(angle.min()), 'p1': float(np.percentile(angle, 1)),
                          'median': float(np.median(angle))},
        'anchors_moved': quality.anchor_displacement(mesh.points, x, model.points)['moved'],
        'locked_coordinate_max_change': float(np.abs((x - mesh.points)[locked]).max()) if locked.any() else 0.0,
        'periodic_unmatched_nodes': int(sum(v['unmatched_nodes'] for v in per.values())),
        'periodic_unmatched_facets': int(sum(v['unmatched_facets'] for v in per.values())),
    })
    return rep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', nargs='*', default=None)
    args = parser.parse_args()
    tags = args.tag or sorted(os.path.basename(os.path.dirname(p))
                              for p in glob.glob(os.path.join(SOURCE_DIR, '*', 'labels.npz')))
    rows = []
    for tag in tags:
        result = run(tag)
        out_dir = os.path.join(RESULT_DIR, tag)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'result.json'), 'w') as f:
            json.dump(result, f, indent=2)
        row = {'tag': tag, 'voxels': int(np.prod(result['shape'])), 'grains': result['n_grains']}
        row.update({f'cleanup_{k}': v for k, v in result['cleanup'].items()})
        row.update({k: v for k, v in result['topology'].items() if not isinstance(v, dict)})
        row.update({f'model_{k}': v for k, v in result['model'].items()})
        sm = result['smoothing']
        row.update({'smooth_volume_max_abs': sm['volume_final']['max_abs'],
                    'smooth_volume_before_correction_max_abs': sm['volume_after_smoothing']['max_abs'],
                    'smooth_reverted_grains': sm['reverted_grains'],
                    'smooth_crossings_before_untangle': sm['untangling']['crossings_before'],
                    'smooth_untangled_vertices': sm['untangling']['vertices'],
                    'smooth_self_intersections': sm['self_intersections'],
                    'smooth_folded': sm['folded_triangles'],
                    'smooth_kink_median': sm['kink_deg']['median'], 'smooth_kink_p90': sm['kink_deg']['p90'],
                    'smooth_share_ge_90': sm['kink_deg']['share_ge_90'],
                    'staircase_share_ge_90': sm['kink_deg']['share_ge_90_staircase'],
                    'smooth_min_angle_p1': sm['min_angle_deg']['p1'],
                    'smooth_min_angle_min': sm['min_angle_deg']['min'],
                    'smooth_anchors_moved': sm['anchors_moved'],
                    'smooth_locked_change': sm['locked_coordinate_max_change'],
                    'smooth_periodic_unmatched_nodes': sm['periodic_unmatched_nodes'],
                    'smooth_periodic_unmatched_facets': sm['periodic_unmatched_facets'],
                    'smooth_displacement_max': sm['displacement_voxels']['max'],
                    'smooth_guarded_steps': sm['guarded_vertex_steps']})
        row.update({f'check_{k}': v for k, v in result['checks'].items()})
        row.update({f'seconds_{k}': v for k, v in result['seconds'].items()})
        rows.append(row)
        print(row, flush=True)
    if args.tag is None:
        import pandas as pd
        pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, 'summary.csv'), index=False)


if __name__ == '__main__':
    main()
