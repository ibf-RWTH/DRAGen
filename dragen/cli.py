"""Command line interface of DRAGen (console script `dragen`, or `python -m dragen`).

    dragen template [config.yaml]              write an example configuration
    dragen generate config.yaml -o out/        generate (and optionally mesh and export) RVEs
    dragen export out/rve_000 -f abaqus,exodus export a saved RVE again
    dragen gui [--port 8765]                   browser front end (DRAGen Studio)

The configuration is the YAML of dragen.config.RveConfig (micrometres, as in the GUI presets)
with optional sections for the command line:

    mesh:   {kind: voxel | tet, order: 2}
    export: {formats: [damask, abaqus, exodus, vtu, gmsh, grains],
             length_unit: m, material_per: grain, element_type: null}
    substructure: {block_thickness: 1.0, ...}      # dragen.substructure.SubstructureConfig

Command line options override these sections. Every RVE goes into <out>/rve_<k>/: rve.npz and
metadata.json (dragen.api.RVEResult.save), config.yaml and the export files. With a substructure
the RVE with one region per block is saved in <out>/rve_<k>/substructure/ and exported instead. n_rves > 1 gives
consecutive seeds starting at `seed`.
"""
import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import yaml

log = logging.getLogger('dragen.cli')

FORMATS = ('damask', 'abaqus', 'exodus', 'vtu', 'gmsh', 'grains')
CLI_SECTIONS = ('mesh', 'export', 'substructure')


def load_config(path: Path):
    """(RveConfig, mesh section, export section, substructure section) from a configuration file."""
    from dragen.config import RveConfig
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    mesh = data.pop('mesh', None) or {}
    export = data.pop('export', None) or {}
    substructure = data.pop('substructure', None) or {}
    return RveConfig.from_dict(data), dict(mesh), dict(export), dict(substructure)


def template() -> str:
    """Example configuration: one ferrite phase from the example CSV (grains of radius 10 um)."""
    from dragen.config import BoxConfig, PhaseConfig, RveConfig
    config = RveConfig(box=BoxConfig(size=(50.0, 50.0, 50.0), resolution=1.0),
                       phases=(PhaseConfig('ferrite', 'ExampleInput/Ferrite/Homogeneous_Grainsize.csv', 1.0),),
                       seed=1)
    text = ('# DRAGen RVE configuration (lengths in micrometres). Grain sources: CSV with a[, b, c,\n'
            '# alpha, phi1, PHI, phi2] or a trained GAN (.pkl). Phases: ferrite, martensite, pearlite,\n'
            '# bainite, austenite. Bands: set count > 0 and a source; inclusions: fraction > 0.\n')
    text += yaml.safe_dump(config.to_dict(), sort_keys=False)
    text += yaml.safe_dump({'mesh': {'kind': 'voxel', 'order': 2},
                            'export': {'formats': ['grains', 'vtu'], 'length_unit': 'm',
                                       'material_per': 'grain', 'element_type': None}}, sort_keys=False)
    return text


def export_result(result, out_dir: Path, formats: List[str], length_unit: str = 'm',
                  material_per: str = 'grain', element_type: Optional[str] = None) -> Dict[str, List[Path]]:
    """Write the chosen formats of an RVEResult into out_dir; returns the files per format."""
    from dragen import export
    unknown = [f for f in formats if f not in FORMATS]
    if unknown:
        raise ValueError(f'unknown export formats {unknown}, choose from {list(FORMATS)}')
    out: Dict[str, List[Path]] = {}
    for fmt in formats:
        start = time.perf_counter()
        if fmt == 'damask':
            written = export.export_damask(result, out_dir / 'damask')
        elif fmt == 'abaqus':
            written = export.export_abaqus(result, out_dir / 'abaqus' / 'rve', length_unit=length_unit,
                                           element_type=element_type, material_per=material_per)
        elif fmt == 'grains':
            written = export.export_grains(result, out_dir / 'grains')
        else:
            written = getattr(export, f'export_{fmt}')(result, out_dir / 'rve')
        out[fmt] = list(written.values()) if isinstance(written, dict) else [written]
        log.info('  %-7s %6.2f s  %s', fmt, time.perf_counter() - start, ', '.join(p.name for p in out[fmt]))
    return out


def generate(config, out: Path, mesh: dict, export: dict, substructure: Optional[dict] = None) -> List[Path]:
    """Generate config.n_rves RVEs into out/rve_<k>; returns their directories."""
    from dragen.api import _tet_mesh, result_from_generation
    from dragen.pipeline import generate as run_pipeline
    from dragen.substructure import SubstructureConfig, apply_substructure, generate_substructure

    subs_config = SubstructureConfig.from_dict(substructure) if substructure else None

    kind = mesh.get('kind', 'voxel')
    if kind not in ('voxel', 'tet'):
        raise ValueError(f"mesh kind must be 'voxel' or 'tet', got {kind!r}")
    dirs = []
    for k in range(config.n_rves):
        seed = None if config.seed is None else config.seed + k
        one = replace(config, seed=seed, n_rves=1)
        rve_dir = Path(out) / f'rve_{k:03d}'
        rve_dir.mkdir(parents=True, exist_ok=True)
        one.save(rve_dir / 'config.yaml')
        log.info('RVE %d/%d: %s voxels, seed %s -> %s', k + 1, config.n_rves, one.box.shape, seed, rve_dir)
        start = time.perf_counter()
        result = run_pipeline(one)
        seconds = dict(result.seconds)
        tet, meta = None, {}
        if kind == 'tet':
            t = time.perf_counter()
            tet, meta = _tet_mesh(result.labels, one.box, int(mesh.get('order', 2)))
            seconds['tet_mesh'] = time.perf_counter() - t
            log.info('  tet mesh: %d elements in %.1f s', len(tet.cells), seconds['tet_mesh'])
        seconds['total'] = time.perf_counter() - start
        rve = result_from_generation(one, result.labels, result.grains, tet=tet,
                                     metadata={'config': one.to_dict(), 'seconds': seconds, **meta})
        rve.save(rve_dir)
        log.info('  %d grains in %.1f s', rve.n_grains, seconds['total'])
        if subs_config is not None:
            sub = generate_substructure(rve, replace(subs_config, seed=subs_config.seed + k))
            rve = apply_substructure(rve, sub)
            rve.save(rve_dir / 'substructure')
            log.info('  substructure: %d packets, %d blocks -> %d regions', sub.stats['packets'],
                     sub.stats['blocks'], rve.n_grains)
        if export.get('formats'):
            export_result(rve, rve_dir, list(export['formats']), export.get('length_unit', 'm'),
                          export.get('material_per', 'grain'), export.get('element_type'))
        dirs.append(rve_dir)
    return dirs


def _formats(text: Optional[str]) -> Optional[List[str]]:
    if text is None:
        return None
    return ['damask', 'abaqus', 'exodus', 'vtu', 'gmsh', 'grains'] if text == 'all' else \
        [f.strip() for f in text.split(',') if f.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='dragen', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-q', '--quiet', action='store_true', help='only warnings and errors')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('template', help='write an example configuration')
    p.add_argument('path', nargs='?', help='file to write (default: print)')

    def export_options(p):
        p.add_argument('-f', '--formats', help=f'comma separated: {",".join(FORMATS)} or all')
        p.add_argument('--length-unit', choices=['m', 'mm', 'um'], help='Abaqus length unit')
        p.add_argument('--material-per', choices=['grain', 'phase'], help='Abaqus materials per grain or phase')
        p.add_argument('--element-type', help='Abaqus element type (default C3D8 / C3D4 / C3D10)')

    p = sub.add_parser('generate', help='generate RVEs from a configuration')
    p.add_argument('config', type=Path)
    p.add_argument('-o', '--out', type=Path, required=True, help='output directory')
    p.add_argument('--seed', type=int, help='overrides the seed of the configuration')
    p.add_argument('-n', '--n-rves', type=int, help='overrides n_rves')
    p.add_argument('--mesh', choices=['voxel', 'tet'], help='voxel labels only, or also a tetrahedral mesh')
    p.add_argument('--order', type=int, choices=[1, 2], help='tetrahedra with 4 or 10 nodes')
    export_options(p)

    p = sub.add_parser('export', help='export saved RVEs (directories with rve.npz)')
    p.add_argument('rves', type=Path, nargs='+')
    p.add_argument('-o', '--out', type=Path, help='output directory (default: the RVE directory)')
    p.add_argument('--mesh', choices=['auto', 'voxel', 'tet'], default='auto')
    export_options(p)

    p = sub.add_parser('gui', help='browser front end')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--workdir', default=str(Path.home() / 'DRAGen_runs'))
    p.add_argument('--parallel', type=int, default=2)
    p.add_argument('--no-browser', action='store_true')

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format='%(message)s', stream=sys.stderr)

    try:
        if args.command == 'template':
            text = template()
            if args.path:
                Path(args.path).write_text(text, encoding='utf-8')
                log.info('wrote %s', args.path)
            else:
                sys.stdout.write(text)
        elif args.command == 'generate':
            config, mesh, export, substructure = load_config(args.config)
            if args.seed is not None:
                config = replace(config, seed=args.seed)
            if args.n_rves is not None:
                config = replace(config, n_rves=args.n_rves)
            if args.mesh:
                mesh['kind'] = args.mesh
            if args.order:
                mesh['order'] = args.order
            _override(export, args)
            generate(config, args.out, mesh, export, substructure)
        elif args.command == 'export':
            from dragen.api import RVEResult
            export = {}
            _override(export, args)
            if not export.get('formats'):
                parser.error('export needs --formats')
            for rve_dir in args.rves:
                result = RVEResult.load(rve_dir)
                if args.mesh == 'voxel':
                    result.tet_mesh = None
                elif args.mesh == 'tet' and result.tet_mesh is None:
                    raise ValueError(f'{rve_dir} has no tetrahedral mesh')
                target = (args.out / rve_dir.name) if args.out and len(args.rves) > 1 else (args.out or rve_dir)
                log.info('%s -> %s', rve_dir, target)
                export_result(result, Path(target), export['formats'], export.get('length_unit', 'm'),
                              export.get('material_per', 'grain'), export.get('element_type'))
        elif args.command == 'gui':
            import webbrowser
            from dragen.gui.server import serve
            server = serve(args.host, args.port, args.workdir, max_parallel=args.parallel)
            url = f'http://{args.host}:{server.server_address[1]}/'
            print(f'DRAGen Studio on {url} (work directory {Path(args.workdir).resolve()}), Ctrl+C to stop')
            if not args.no_browser:
                webbrowser.open(url)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
    except (ValueError, FileNotFoundError, KeyError, TypeError) as err:
        log.error('error: %s', err)
        return 1
    return 0


def _override(export: dict, args) -> None:
    formats = _formats(args.formats)
    if formats is not None:
        export['formats'] = formats
    for key in ('length_unit', 'material_per', 'element_type'):
        value = getattr(args, key)
        if value is not None:
            export[key] = value


if __name__ == '__main__':
    sys.exit(main())
