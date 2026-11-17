"""HTTP server of the browser front end: static single-page app plus a JSON API.

    GET    /api/info                       defaults, phases, unsupported features, mesh available
    GET    /api/inputs                     grain sources in ExampleInput and benchmarks/inputs
    POST   /api/validate                   {config} -> {ok, errors, warnings, config, shape, voxels}
    GET    /api/presets                    names
    GET    /api/presets/<name>             config
    POST   /api/presets                    {name, config}
    DELETE /api/presets/<name>
    GET    /api/jobs                       all jobs, newest first
    POST   /api/jobs                       {name, config} -> one generation job per RVE
    GET    /api/jobs/<id>                  job, result, config, grain meta, mesh result
    GET    /api/jobs/<id>/log?offset=n     log lines from line n
    GET    /api/jobs/<id>/slice?axis=a&index=i
    GET    /api/jobs/<id>/histogram
    GET    /api/jobs/<id>/surface?source=staircase|smoothed&step=auto|n&mesh=<mesh job>   binary, see dragen.gui.surface
    GET    /api/jobs/<id>/files/<name>     config.yaml, result.json, grains.csv, labels.npz, mesh.*
    POST   /api/jobs/<id>/cancel
    POST   /api/jobs/<id>/mesh             start a meshing job for a finished generation run
    POST   /api/jobs/<id>/export           {formats, mesh, length_unit, material_per, element_type,
                                            substructure: SubstructureConfig dict or null}
                                           -> export job (files of dragen.export, export.zip)
    DELETE /api/jobs/<id>
"""
import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

from dragen.config import Phase
from dragen.gui import runs
from dragen.gui.jobs import DONE, JobManager

STATIC = Path(__file__).resolve().parent / 'static'
DOWNLOADS = ('config.yaml', 'result.json', 'grains.csv', 'labels.npz', 'log.txt', 'mesh.json', 'mesh.npz',
             'export.zip', 'export.json')
EXPORT_LABELS = {'damask': 'DAMASK grid (grid.vti, material.yaml)',
                 'abaqus': 'Abaqus (rve.inp, graindata.inp)',
                 'exodus': 'Exodus II for MOOSE (rve.e)',
                 'vtu': 'VTK unstructured grid (rve.vtu)',
                 'gmsh': 'Gmsh MSH 2.2 (rve.msh)',
                 'grains': 'Grain table (grains.csv)'}


def export_unavailable() -> dict:
    """Per export format the reason it cannot run here (missing package); absent = available."""
    out = {}
    try:
        import damask  # noqa: F401
    except Exception as err:
        out['damask'] = f"{type(err).__name__}: {err} (install the extra 'damask')"
    try:
        import meshio  # noqa: F401
    except Exception as err:
        for fmt in ('vtu', 'gmsh'):
            out[fmt] = f'{type(err).__name__}: {err}'
    return out


def mesh_available() -> Optional[str]:
    """None if dragen.mesh.mesher and gmsh import, else the reason."""
    try:
        import gmsh  # noqa: F401
        from dragen.mesh.mesher import mesh_rve  # noqa: F401
    except Exception as err:   # the meshing API is experimental; the GUI must work without it
        return f'{type(err).__name__}: {err}'
    return None


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class Handler(BaseHTTPRequestHandler):
    server: 'GuiServer'
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):     # keep the console quiet
        if self.server.verbose:
            super().log_message(fmt, *args)

    # ------------------------------------------------------------------------------------- #

    def do_GET(self):
        self._dispatch('GET')

    def do_POST(self):
        self._dispatch('POST')

    def do_DELETE(self):
        self._dispatch('DELETE')

    def _dispatch(self, method: str):
        url = urlparse(self.path)
        parts = [unquote(p) for p in url.path.strip('/').split('/') if p]
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        try:
            if parts[:1] == ['api']:
                body = self._body() if method == 'POST' else {}
                result = self._api(method, parts[1:], query, body)
                if isinstance(result, tuple) and isinstance(result[0], bytes):   # binary payload
                    self._send_bytes(result[0])
                elif isinstance(result, tuple):       # a file download
                    self._send_file(*result)
                else:
                    self._send_json(result)
            elif method == 'GET':
                self._static(parts)
            else:
                raise ApiError(HTTPStatus.NOT_FOUND, 'not found')
        except ApiError as err:
            self._send_json({'error': str(err)}, err.status)
        except KeyError as err:
            self._send_json({'error': f'not found: {err.args[0] if err.args else ""}'}, HTTPStatus.NOT_FOUND)
        except (ValueError, TypeError) as err:
            self._send_json({'error': str(err)}, HTTPStatus.BAD_REQUEST)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as err:   # report instead of dropping the connection
            self._send_json({'error': f'{type(err).__name__}: {err}'}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _api(self, method, parts, query, body):
        jobs = self.server.jobs
        workdir = jobs.workdir
        route = (method, *parts[:1])
        if route == ('GET', 'info'):
            return {'workdir': str(workdir), 'defaults': runs.default_config(),
                    'phases': [p.name.lower() for p in Phase if p not in (Phase.INCLUSIONS, Phase.BANDS)],
                    'band_phases': [p.name.lower() for p in Phase if p != Phase.INCLUSIONS],
                    'not_supported': runs.NOT_SUPPORTED, 'mesh_unavailable': self.server.mesh_unavailable,
                    'export_formats': EXPORT_LABELS, 'export_unavailable': self.server.export_unavailable,
                    'max_parallel': jobs.max_parallel}
        if route == ('GET', 'inputs'):
            return {'inputs': runs.input_files()}
        if route == ('POST', 'validate'):
            return runs.validate(body.get('config', {}))
        if parts[:1] == ['presets']:
            return self._presets(method, parts[1:], body, workdir)
        if parts[:1] == ['jobs']:
            return self._jobs(method, parts[1:], query, body)
        raise ApiError(HTTPStatus.NOT_FOUND, f'unknown endpoint /api/{"/".join(parts)}')

    def _presets(self, method, parts, body, workdir):
        if method == 'GET' and not parts:
            return {'presets': runs.list_presets(workdir)}
        if method == 'GET' and len(parts) == 1:
            return {'name': parts[0], 'config': runs.load_preset(workdir, parts[0])}
        if method == 'POST' and not parts:
            check = runs.validate(body.get('config', {}))
            if not check['ok']:
                raise ApiError(HTTPStatus.BAD_REQUEST, '; '.join(check['errors']))
            return {'name': runs.save_preset(workdir, body.get('name', ''), check['config'])}
        if method == 'DELETE' and len(parts) == 1:
            runs.delete_preset(workdir, parts[0])
            return {'deleted': parts[0]}
        raise ApiError(HTTPStatus.NOT_FOUND, 'unknown preset endpoint')

    def _jobs(self, method, parts, query, body):
        jobs = self.server.jobs
        if not parts:
            if method == 'GET':
                return {'jobs': [j.to_dict() for j in jobs.list()]}
            if method == 'POST':
                check = runs.validate(body.get('config', {}))
                if not check['ok']:
                    raise ApiError(HTTPStatus.BAD_REQUEST, '; '.join(check['errors']))
                configs = runs.split_rves(check['config'])
                name = (body.get('name') or '').strip()
                created = []
                for k, cfg in enumerate(configs):
                    label = name or f'RVE seed {cfg["seed"]}'
                    if len(configs) > 1 and name:
                        label = f'{name} #{k + 1}'
                    created.append(jobs.submit('generate', label, config=cfg).to_dict())
                return {'jobs': created}
            raise ApiError(HTTPStatus.METHOD_NOT_ALLOWED, 'method not allowed')
        job = jobs.get(parts[0])
        data = runs.RunData(jobs.path(job.id))
        action = parts[1] if len(parts) > 1 else None
        if method == 'GET' and action is None:
            out = {'job': job.to_dict(), 'config': data.config(), 'result': data.result(), 'mesh': data.mesh(),
                   'export': data.export()}
            if job.kind == 'generate' and job.status == DONE:
                out['grains'] = data.grain_meta()
                out['spacing'] = list(data.spacing())
            children = [j for j in jobs.list() if j.parent == job.id]
            out['meshes'] = [j.to_dict() for j in children if j.kind == 'mesh']
            out['exports'] = [j.to_dict() for j in children if j.kind == 'export']
            out['surfaces'] = self._surface_sources(job)
            return out
        if method == 'GET' and action == 'surface':
            if job.kind != 'generate' or job.status != DONE:
                raise ApiError(HTTPStatus.CONFLICT, 'the run has no label field yet')
            source = query.get('source', 'staircase')
            mesh_dir = None
            if source == 'smoothed':
                meshes = [s for s in self._surface_sources(job) if s['source'] == 'smoothed']
                wanted = query.get('mesh')
                hit = [s for s in meshes if s['mesh'] == wanted] if wanted else meshes[:1]
                if not hit:
                    raise ApiError(HTTPStatus.NOT_FOUND, 'no meshed surface for this run')
                mesh_dir = jobs.path(hit[0]['mesh'])
            elif source != 'staircase':
                raise ApiError(HTTPStatus.BAD_REQUEST, f'unknown surface source {source!r}')
            step = int(query['step']) if query.get('step', 'auto') not in ('', 'auto') else None
            return data.surface(source, step, mesh_dir), None
        if method == 'GET' and action == 'log':
            offset = int(query.get('offset', 0))
            lines = jobs.log(job.id, offset)
            return {'offset': offset, 'lines': lines}
        if method == 'GET' and action in ('slice', 'histogram'):
            if job.kind != 'generate' or job.status != DONE:
                raise ApiError(HTTPStatus.CONFLICT, 'the run has no label field yet')
            if action == 'slice':
                return data.slice(int(query.get('axis', 2)), int(query.get('index', -1)))
            return data.histogram(int(query.get('bins', 30)))
        if method == 'GET' and action == 'files' and len(parts) == 3:
            name = parts[2]
            if name not in DOWNLOADS or not (data.path / name).is_file():
                raise ApiError(HTTPStatus.NOT_FOUND, f'no file {name}')
            return data.path / name, f'{job.id}_{name}'
        if method == 'POST' and action == 'cancel':
            return {'job': jobs.cancel(job.id).to_dict()}
        if method == 'POST' and action == 'mesh':
            if self.server.mesh_unavailable:
                raise ApiError(HTTPStatus.CONFLICT, f'meshing is not available: {self.server.mesh_unavailable}')
            if job.kind != 'generate' or job.status != DONE:
                raise ApiError(HTTPStatus.CONFLICT, 'only finished generation runs can be meshed')
            if job.imported:
                raise ApiError(HTTPStatus.CONFLICT, 'imported runs cannot be meshed from the GUI; '
                                                    'their directory is not a job directory')
            mesh = jobs.submit('mesh', f'mesh of {job.name}', parent=job.id)
            return {'job': mesh.to_dict()}
        if method == 'POST' and action == 'export':
            if job.kind != 'generate' or job.status != DONE:
                raise ApiError(HTTPStatus.CONFLICT, 'only finished generation runs can be exported')
            options = self._export_options(job, body)
            exp = jobs.submit('export', f'export of {job.name}', parent=job.id, options=options)
            return {'job': exp.to_dict()}
        if method == 'DELETE' and action is None:
            jobs.delete(job.id)
            return {'deleted': job.id}
        raise ApiError(HTTPStatus.NOT_FOUND, 'unknown job endpoint')

    def _export_options(self, job, body: dict) -> dict:
        """Checked options of an export job (absolute paths, so imported runs work as well)."""
        jobs = self.server.jobs
        formats = body.get('formats') or []
        unknown = [f for f in formats if f not in EXPORT_LABELS]
        if not formats or unknown:
            raise ApiError(HTTPStatus.BAD_REQUEST, f'choose export formats from {list(EXPORT_LABELS)}'
                                                   + (f'; unknown: {unknown}' if unknown else ''))
        missing = {f: r for f, r in self.server.export_unavailable.items() if f in formats}
        if missing:
            raise ApiError(HTTPStatus.CONFLICT, '; '.join(f'{f}: {r}' for f, r in missing.items()))
        mesh_dir = None
        mesh_id = body.get('mesh') or None
        if mesh_id not in (None, 'voxel'):
            try:
                mesh = jobs.get(mesh_id)
            except KeyError:
                raise ApiError(HTTPStatus.NOT_FOUND, f'no mesh job {mesh_id}')
            if mesh.kind != 'mesh' or mesh.parent != job.id or mesh.status != DONE:
                raise ApiError(HTTPStatus.CONFLICT, 'the mesh must be a finished meshing job of this run')
            mesh_dir = str(jobs.path(mesh.id).resolve())
        unit = body.get('length_unit', 'm')
        if unit not in ('m', 'mm', 'um'):
            raise ApiError(HTTPStatus.BAD_REQUEST, f'length_unit must be m, mm or um, got {unit!r}')
        per = body.get('material_per', 'grain')
        if per not in ('grain', 'phase'):
            raise ApiError(HTTPStatus.BAD_REQUEST, f'material_per must be grain or phase, got {per!r}')
        element = str(body.get('element_type') or '').strip().upper() or None
        if element is not None and not element.replace('_', '').isalnum():
            raise ApiError(HTTPStatus.BAD_REQUEST, f'invalid element type {element!r}')
        substructure = body.get('substructure') or None
        if substructure is not None:
            from dragen.substructure import SubstructureConfig
            try:
                substructure = SubstructureConfig.from_dict(substructure).to_dict()
            except (ValueError, TypeError, FileNotFoundError) as err:
                raise ApiError(HTTPStatus.BAD_REQUEST, f'substructure: {err}')
        run_dir = jobs.path(job.id)
        return {'run_dir': str(run_dir.resolve()), 'mesh_dir': mesh_dir, 'formats': list(dict.fromkeys(formats)),
                'length_unit': unit, 'material_per': per, 'element_type': element, 'substructure': substructure}

    def _surface_sources(self, job) -> list:
        """Staircase for every finished run, plus the smoothed surface of each finished meshing
        job of the run whose mesh.npz holds the surface (newest first)."""
        if job.kind != 'generate' or job.status != DONE:
            return []
        out = [{'source': 'staircase', 'label': 'Staircase (voxel faces)'}]
        import numpy as np
        for mesh in self.server.jobs.list():
            if mesh.parent != job.id or mesh.status != DONE:
                continue
            path = self.server.jobs.path(mesh.id) / 'mesh.npz'
            try:
                with np.load(path) as npz:
                    has = 'surf_points' in npz.files
            except OSError:
                has = False
            if has:
                out.append({'source': 'smoothed', 'mesh': mesh.id, 'label': f'Smoothed surface ({mesh.name})'})
        return out

    # ------------------------------------------------------------------------------------- #

    def _body(self) -> dict:
        length = int(self.headers.get('Content-Length') or 0)
        if not length:
            return {}
        try:
            data = json.loads(self.rfile.read(length).decode('utf-8'))
        except json.JSONDecodeError as err:
            raise ApiError(HTTPStatus.BAD_REQUEST, f'invalid JSON: {err}')
        if not isinstance(data, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, 'expected a JSON object')
        return data

    def _send_json(self, data, status=HTTPStatus.OK):
        payload = json.dumps(data, allow_nan=False, default=_json_default).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(payload)

    def _send_bytes(self, data: bytes):
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, download_name: str):
        data = path.read_bytes()
        kind = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Content-Disposition', f'attachment; filename="{download_name}"')
        self.end_headers()
        self.wfile.write(data)

    def _static(self, parts):
        rel = '/'.join(parts) or 'index.html'
        path = (STATIC / rel).resolve()
        if STATIC not in path.parents and path != STATIC or not path.is_file():
            path = STATIC / 'index.html'                 # single-page app
        data = path.read_bytes()
        kind = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        if kind.startswith('text/') or kind in ('application/javascript',):
            kind += '; charset=utf-8'
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(data)


def _json_default(value):
    import numpy as np
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'not JSON serialisable: {type(value).__name__}')


class GuiServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, workdir, max_parallel: int = 2, verbose: bool = False):
        self.jobs = JobManager(workdir, max_parallel=max_parallel)
        self.mesh_unavailable = mesh_available()
        self.export_unavailable = export_unavailable()
        self.verbose = verbose
        super().__init__(address, Handler)

    def server_close(self):
        self.jobs.shutdown()
        super().server_close()


def serve(host: str = '127.0.0.1', port: int = 8765, workdir='.', max_parallel: int = 2,
          verbose: bool = False) -> GuiServer:
    """Create the server; call serve_forever() on it (or run it in a thread)."""
    return GuiServer((host, port), workdir, max_parallel=max_parallel, verbose=verbose)


def run_in_thread(server: GuiServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, name='dragen-gui', daemon=True)
    thread.start()
    return thread
