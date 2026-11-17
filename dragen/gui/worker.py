"""Worker process of the browser front end: one generation or meshing job per call.

    python -m dragen.gui.worker generate <job dir>
    python -m dragen.gui.worker mesh <job dir>
    python -m dragen.gui.worker export <job dir>

A generation job reads <job dir>/config.yaml and writes, like benchmarks/rework_benchmark.py,
labels.npz, grains.csv (row k belongs to label k + 1) and result.json (report, acceptance
violations, bands, inclusions, tessellation statistics, timings). A meshing job reads the run
named in <job dir>/job.json ('parent') and writes mesh.json (statistics and quality checks) and
mesh.npz (tet10 nodes, elements, grain per element). Progress goes to stdout, which the server
captures in log.txt.

An export job reads <job dir>/options.json (run directory, optional mesh job directory, formats
and their settings), writes the files of dragen.export into <job dir>/export/, zips them into
export.zip and lists them in export.json.
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np


def _log(message: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def _plain(value):
    """JSON-safe copy (numpy scalars and arrays, tuples)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def generate_job(job_dir: Path) -> None:
    from dragen.analysis import metrics
    from dragen.analysis.acceptance import feature_violations, violations
    from dragen.analysis.features import band_report, inclusion_report
    from dragen.config import RveConfig
    from dragen.pipeline import expected_phase_fractions, generate

    config = RveConfig.load(job_dir / 'config.yaml')
    _log(f'box {config.box.size} um, {config.box.resolution:g} voxels/um -> {config.box.shape} voxels, '
         f'seed {config.seed}')
    for p in config.phases:
        _log(f'phase {p.phase.name.lower()}: {p.fraction:g} from {p.source}')
    if config.bands.count:
        _log(f'{config.bands.count} band(s) of {config.bands.phase.name.lower()}, filling {config.bands.filling:g}')
    if config.inclusions.fraction:
        _log(f'inclusions: {config.inclusions.fraction:g} of the box from {config.inclusions.source}')

    _log('generating (sampling, RSA, tessellation, repair, inclusions) ...')
    start = time.perf_counter()
    result = generate(config)
    total = time.perf_counter() - start
    for stage, seconds in result.seconds.items():
        _log(f'  {stage:<13s} {seconds:8.2f} s')
    _log(f'{len(result.grains)} grains in {total:.1f} s')

    _log('evaluating ...')
    target = np.concatenate([[0.0], result.grains.volume])
    phase = np.concatenate([[0], result.grains.phase])
    rep = metrics.report(result.labels, target, phase, result.domain.voxel_volume, config.box.periodic)
    target_ratio = expected_phase_fractions(config, result.bands, result.domain)
    bands = band_report(result.labels, result.grains, result.bands, result.domain, result.band_grains)
    requested = None if result.inclusions is None else result.inclusions.placed + result.inclusions.not_placed
    inclusions = inclusion_report(result.labels, result.grains, result.domain, requested)
    stats = result.tessellation
    n = result.labels.size
    out = {
        'shape': list(result.labels.shape), 'n_grains': len(result.grains),
        'spacing': list(result.domain.spacing),
        'seconds': {**{k: round(v, 3) for k, v in result.seconds.items()}, 'total': round(total, 3)},
        'report': rep.to_dict(),
        'target_phase_ratio': target_ratio,
        'acceptance_violations': violations(rep, target_ratio) + feature_violations(
            bands, inclusions, config.bands.filling, config.inclusions.fraction),
        'bands': bands,
        'inclusions': inclusions,
        'rsa_attempts': result.rsa_attempts,
        'repaired_voxels': result.repaired_voxels,
        'tessellation': {
            'epochs': stats.epochs,
            'fraction_after_rsa': round(stats.filled_after_rsa / n, 4),
            'fraction_from_regular_growth': round(stats.filled_regular / n, 4),
            'fraction_from_reset_phase': round(stats.filled_reset / n, 4),
            'fraction_from_dilation': round(stats.filled_by_dilation / n, 4),
            'filled_per_epoch': [round(f / n, 4) for f in stats.filled_per_epoch],
        },
        'config': config.to_dict(),
    }
    np.savez_compressed(job_dir / 'labels.npz', labels=result.labels)
    result.grains.to_frame().to_csv(job_dir / 'grains.csv', index=False)
    (job_dir / 'result.json').write_text(json.dumps(_plain(out), indent=2), encoding='utf-8')
    viol = out['acceptance_violations']
    _log('accepted: all criteria met' if not viol else f'{len(viol)} acceptance criteria violated')
    for v in viol:
        _log(f'  - {v}')


def mesh_job(job_dir: Path) -> None:
    from dragen.config import RveConfig
    from dragen.mesh.mesher import MeshConfig, mesh_rve
    from dragen.mesh.tetmesh import TetMeshConfig, check

    job = json.loads((job_dir / 'job.json').read_text(encoding='utf-8'))
    run_dir = job_dir.parent / job['parent']
    config = RveConfig.load(run_dir / 'config.yaml')
    labels = np.load(run_dir / 'labels.npz')['labels']
    box = config.box
    _log(f'meshing run {job["parent"]}: {labels.shape} voxels, {int(labels.max())} grains')
    _log('clean-up, topology, smoothing, adaptive remeshing, gmsh (C3D10) ...')
    start = time.perf_counter()
    result = mesh_rve(labels, box.spacing, box.periodic, MeshConfig(tetmesh=TetMeshConfig(verbosity=0)))
    total = time.perf_counter() - start
    for stage, seconds in result.stats.get('seconds', {}).items():
        _log(f'  {stage:<13s} {seconds:8.2f} s')
    _log('checking ...')
    quality = check(result.surface, result.points, result.tets.mesh, labels.shape, box.spacing, box.periodic,
                    result.smoothed.grain_volume_target)
    mesh = result.tets.mesh
    out = {'parent': job['parent'], 'seconds': {**result.stats.get('seconds', {}), 'total': total},
           'feedback': result.stats.get('feedback', []), 'elements': int(len(mesh.cells)),
           'nodes': int(len(mesh.points)), 'cell_type': mesh.cell_type,
           'surface_triangles': int(len(result.surface.oriented)),
           'staircase_triangles': int(len(result.staircase.oriented)), 'quality': quality}
    # the final grain boundary surface for the 3D view: triangles oriented out of surf_regions[:, 0]
    # (always a grain), surf_regions[:, 1] the other side (negative = box face)
    surface = result.surface
    regions = np.array([s.regions for s in surface.surfaces], dtype=np.int64)[surface.topology.patch]
    np.savez_compressed(job_dir / 'mesh.npz', points=mesh.points, cells=mesh.cells, grain=mesh.grain,
                        surf_points=np.asarray(result.points, dtype=np.float32),
                        surf_triangles=np.asarray(surface.oriented, dtype=np.uint32),
                        surf_regions=regions.astype(np.int32))
    (job_dir / 'mesh.json').write_text(json.dumps(_plain(out), indent=2), encoding='utf-8')
    _log(f'{out["elements"]} {mesh.cell_type} elements, {out["nodes"]} nodes in {total:.1f} s')


EXPORT_FORMATS = ('damask', 'abaqus', 'exodus', 'vtu', 'gmsh', 'grains')


def export_job(job_dir: Path) -> None:
    import zipfile

    import pandas as pd

    from dragen import export
    from dragen.api import result_from_generation, tet_mesh_from_arrays
    from dragen.config import RveConfig
    from dragen.core.grains import GrainTable

    options = json.loads((job_dir / 'options.json').read_text(encoding='utf-8'))
    run_dir = Path(options['run_dir'])
    config = RveConfig.load(run_dir / 'config.yaml')
    labels = np.load(run_dir / 'labels.npz')['labels']
    grains = GrainTable.from_frame(pd.read_csv(run_dir / 'grains.csv'))
    tet = None
    if options.get('mesh_dir'):
        mesh = np.load(Path(options['mesh_dir']) / 'mesh.npz')
        tet = tet_mesh_from_arrays(mesh['points'], mesh['cells'], mesh['grain'], config.box)
        _log(f'mesh: {len(tet.cells)} {tet.cell_type} elements from {Path(options["mesh_dir"]).name}')
    else:
        _log(f'mesh: voxels ({labels.size} hexahedra)')
    result = result_from_generation(config, labels, grains, tet=tet,
                                    metadata={'gui_run': run_dir.name, 'gui_mesh': options.get('mesh_dir')})
    substructure = None
    if options.get('substructure'):
        from dragen.substructure import SubstructureConfig, apply_substructure, generate_substructure
        sub = generate_substructure(result, SubstructureConfig.from_dict(options['substructure']))
        result = apply_substructure(result, sub)
        substructure = sub.stats
        _log(f'substructure: {sub.stats["packets"]} packets, {sub.stats["blocks"]} blocks -> '
             f'{result.n_grains} regions')
    out_dir = job_dir / 'export'
    files, seconds = [], {}
    for fmt in options['formats']:
        if fmt not in EXPORT_FORMATS:
            raise ValueError(f'unknown export format {fmt!r}')
        _log(f'{fmt} ...')
        start = time.perf_counter()
        if fmt == 'damask':
            written = export.export_damask(result, out_dir / 'damask')
        elif fmt == 'abaqus':
            written = export.export_abaqus(result, out_dir / 'abaqus' / 'rve',
                                           length_unit=options.get('length_unit', 'm'),
                                           element_type=options.get('element_type') or None,
                                           material_per=options.get('material_per', 'grain'))
        elif fmt == 'grains':
            written = export.export_grains(result, out_dir / 'grains')
        else:
            written = getattr(export, f'export_{fmt}')(result, out_dir / 'rve')
        seconds[fmt] = round(time.perf_counter() - start, 3)
        written = written.values() if isinstance(written, dict) else [written]
        for path in written:
            files.append({'format': fmt, 'name': Path(path).relative_to(out_dir).as_posix(),
                          'bytes': Path(path).stat().st_size})
        _log(f'  {fmt}: {seconds[fmt]:.2f} s')
    (out_dir / 'metadata.json').write_text(json.dumps(_plain(result.metadata), indent=2), encoding='utf-8')
    with zipfile.ZipFile(job_dir / 'export.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out_dir.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(out_dir).as_posix())
    summary = {'parent': job_dir.name, 'run': run_dir.name, 'mesh': 'tet' if tet is not None else 'voxel',
               'cell_type': tet.cell_type if tet is not None else 'hexahedron', 'options': options,
               'files': files, 'seconds': seconds, 'substructure': substructure, 'zip_bytes': (job_dir / 'export.zip').stat().st_size}
    (job_dir / 'export.json').write_text(json.dumps(_plain(summary), indent=2), encoding='utf-8')
    _log(f'{len(files)} files, export.zip {summary["zip_bytes"] / 2 ** 20:.1f} MB')


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in ('generate', 'mesh', 'export'):
        print(__doc__)
        return 2
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
                        datefmt='%H:%M:%S', stream=sys.stdout)
    job_dir = Path(argv[1])
    {'generate': generate_job, 'mesh': mesh_job, 'export': export_job}[argv[0]](job_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
