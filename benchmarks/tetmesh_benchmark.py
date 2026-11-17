"""Volume meshing (M4, M4a) of the RVEs of the reworked generator.

For every benchmarks/results/rework/<tag>/labels.npz the whole pipeline of dragen.mesh.mesher:
clean-up, topology, surface model, smoothing, adaptive remeshing of the grain boundaries (skipped
with --no-remesh, the state of commit 7d1c47d) and the gmsh tet mesh with feedback, judged with
dragen.mesh.tetmesh.check: inverted elements, gaps and overused faces, midside nodes, element
quality (scaled Jacobian, dihedral angles, radius ratio), element volume per grain against the
voxel volume and against the volume enclosed by the final surface, grain boundary faces of the tet
mesh against the surface triangles, periodic node pairs (corner and midside nodes) and facets on
opposite box faces; for the remeshed surface also dragen.mesh.remeshing.check.

Example:
    venv/Scripts/python benchmarks/tetmesh_benchmark.py                        # all stored RVEs
    venv/Scripts/python benchmarks/tetmesh_benchmark.py --tag lognormal_box30_res2_seed1 --variant smoothed --no-remesh

Results: benchmarks/results/mesh/tetmesh/<variant>/<tag>/result.json and <variant>/summary.csv;
<variant> is 'remeshed' unless --variant is given ('default' holds the run on 7d1c47d).
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
from dragen.config import RveConfig  # noqa: E402
from dragen.mesh import remeshing  # noqa: E402
from dragen.mesh.mesher import MeshConfig, mesh_rve  # noqa: E402
from dragen.mesh.remeshing import RemeshConfig, RemeshResult  # noqa: E402
from dragen.mesh.tetmesh import TetMeshConfig, check  # noqa: E402

SOURCE_DIR = os.path.join(REPO_ROOT, 'benchmarks', 'results', 'rework')
RESULT_DIR = os.path.join(REPO_ROOT, 'benchmarks', 'results', 'mesh', 'tetmesh')


def run(tag: str, config: MeshConfig) -> dict:
    src = os.path.join(SOURCE_DIR, tag)
    labels = np.load(os.path.join(src, 'labels.npz'))['labels']
    box = RveConfig.load(os.path.join(src, 'config.yaml')).box
    periodic, spacing = tuple(box.periodic), tuple(box.spacing)
    out = {'tag': tag, 'code': code_state(), 'shape': list(labels.shape),
           'config': {'tetmesh': vars(config.tetmesh),
                      'remeshing': vars(config.remeshing) if config.remeshing else None,
                      'min_dihedral': config.min_dihedral, 'feedback': list(config.feedback)},
           'spacing': list(spacing), 'periodic': list(periodic), 'n_grains': int(labels.max())}
    t = time.perf_counter()
    result = mesh_rve(labels, spacing, periodic, config)
    out['seconds'] = dict(result.stats['seconds'], total=time.perf_counter() - t)
    out['seconds'].update({f'tetmesh_{k}': v for k, v in result.tets.stats.pop('seconds').items()})
    out['tetmesh'] = result.tets.stats
    out['feedback'] = result.stats['feedback']
    out['warnings'] = result.stats['warnings']
    out['feedback_best_round'] = result.stats['feedback_best_round']
    out['coarse_grain_share'] = result.stats.get('coarse_grain_share')
    out['staircase_triangles'] = int(len(result.staircase.oriented))
    if config.remeshing is not None:
        rem = result.stats['remeshing']
        out['remeshing'] = {k: rem[k] for k in ('passes', 'refused', 'volume_before_correction', 'volume_final',
                                                 'triangles_before', 'triangles_after')}
        out['remeshing']['check'] = remeshing.check(result.staircase, RemeshResult(result.surface, result.points),
                                                    spacing)
    t = time.perf_counter()
    out['quality'] = check(result.surface, result.points, result.tets.mesh, labels.shape, spacing, periodic,
                           result.smoothed.grain_volume_target)
    out['seconds']['checks'] = time.perf_counter() - t
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', nargs='*', default=None)
    parser.add_argument('--variant', default='remeshed')
    parser.add_argument('--no-remesh', action='store_true')
    parser.add_argument('--growth', type=float, default=TetMeshConfig.growth)
    parser.add_argument('--max-size', type=float, default=None)
    parser.add_argument('--optimize', default=TetMeshConfig.optimize)
    parser.add_argument('--algorithm', type=int, default=TetMeshConfig.algorithm)
    args = parser.parse_args()
    config = MeshConfig(tetmesh=TetMeshConfig(growth=args.growth, max_size=args.max_size, optimize=args.optimize,
                                              algorithm=args.algorithm),
                        remeshing=None if args.no_remesh else RemeshConfig())
    tags = args.tag or sorted(os.path.basename(os.path.dirname(p))
                              for p in glob.glob(os.path.join(SOURCE_DIR, '*', 'labels.npz')))
    rows = []
    for tag in tags:
        try:
            result = run(tag, config)
        except Exception as err:                       # keep going, record the failure
            out_dir = os.path.join(RESULT_DIR, args.variant, tag)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, 'result.json'), 'w') as f:
                json.dump({'tag': tag, 'code': code_state(), 'error': repr(err)}, f, indent=2)
            rows.append({'tag': tag, 'error': repr(err)})
            print(json.dumps(rows[-1]), flush=True)
            continue
        out_dir = os.path.join(RESULT_DIR, args.variant, tag)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'result.json'), 'w') as f:
            json.dump(result, f, indent=2)
        q = result['quality']
        fb = result['feedback']
        row = {'tag': tag, 'voxels': int(np.prod(result['shape'])), 'grains': result['n_grains'],
               'staircase_triangles': result['staircase_triangles'], 'triangles': fb[result['feedback_best_round']]['triangles'],
               'feedback_rounds': len(fb) - 1, 'feedback_best_round': result.get('feedback_best_round'), 'below_5deg_first': fb[0]['below_min_dihedral'],
               'below_5deg': fb[result['feedback_best_round']]['below_min_dihedral'],
               'volume_vs_surface_max_abs': q['volume_vs_surface']['max_abs'],
               'bodies': result['tetmesh']['bodies'], 'cavities': result['tetmesh']['cavities'],
               'elements': q['n_elements'], 'nodes': q['n_points'], 'elements_per_voxel': q['elements_per_voxel'],
               'inverted': q['inverted_elements'], 'open_inner_faces': q['open_inner_faces'],
               'overused_faces': q['overused_faces'], 'midside_mismatch': q.get('midside_mismatch', 0),
               'several_bodies': q['grains_with_several_bodies'],
               'volume_max_abs': q['volume_deviation']['max_abs'],
               'surface_nodes_moved': q['surface_nodes_moved'],
               'interface_missing': q['interface_faces_missing'], 'interface_extra': q['interface_faces_extra'],
               'periodic_unmatched_nodes': q['periodic_unmatched_nodes'],
               'periodic_unmatched_facets': q['periodic_unmatched_facets'],
               'dihedral_min': q['dihedral_min_deg']['min'], 'dihedral_min_p1': q['dihedral_min_deg']['p1'],
               'dihedral_max': q['dihedral_max_deg']['max'],
               'scaled_jacobian_min': q['scaled_jacobian']['min'], 'scaled_jacobian_p1': q['scaled_jacobian']['p1'],
               'radius_ratio_p1': q['radius_ratio']['p1'],
               'longest_edge_median': q['longest_edge_voxels']['median'],
               'longest_edge_max': q['longest_edge_voxels']['max']}
        row.update({f's_{k}': round(v, 2) for k, v in result['seconds'].items()})
        rows.append(row)
        print(json.dumps(row), flush=True)
    import pandas as pd
    path = os.path.join(RESULT_DIR, args.variant, 'summary.csv')
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old[~old['tag'].isin(df['tag'])], df]).sort_values('tag')
    df.to_csv(path, index=False)


if __name__ == '__main__':
    main()
