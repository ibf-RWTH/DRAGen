"""Browser front end (dragen.gui): JSON API, job queue with worker processes, run results."""
import base64
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'benchmarks'))

from dragen.gui import runs  # noqa: E402
from dragen.gui.server import run_in_thread, serve  # noqa: E402


@pytest.fixture(scope='module')
def server(tmp_path_factory):
    srv = serve('127.0.0.1', 0, tmp_path_factory.mktemp('gui_runs'), max_parallel=1)
    run_in_thread(srv)
    yield srv
    srv.shutdown()
    srv.server_close()


def call(server, path, method='GET', body=None, raw=False):
    url = f'http://127.0.0.1:{server.server_address[1]}{path}'
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            payload = res.read()
            return res.status, payload if raw else json.loads(payload)
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


@pytest.fixture(scope='module')
def tiny_config():
    import common
    config = runs.default_config()
    config['box'] = {'size': [10.0, 10.0, 10.0], 'resolution': 1.0, 'periodic': [True, True, True]}
    config['phases'] = [{'phase': 'ferrite', 'source': common.lognormal_input(), 'fraction': 1.0}]
    config['seed'] = 3
    return config


def test_static_app_and_info(server):
    status, page = call(server, '/', raw=True)
    assert status == 200 and b'DRAGen' in page
    status, js = call(server, '/app.js', raw=True)
    assert status == 200 and b'function' in js
    status, info = call(server, '/api/info')
    assert status == 200
    assert 'ferrite' in info['phases'] and info['defaults']['box']['size']
    assert any('2D' in f['feature'] for f in info['not_supported'])


def test_validation_errors_are_readable(server, tiny_config):
    status, res = call(server, '/api/validate', 'POST', {'config': tiny_config})
    assert status == 200 and res['ok'], res
    assert res['shape'] == [10, 10, 10] and res['voxels'] == 1000

    bad = json.loads(json.dumps(tiny_config))
    bad['phases'][0]['fraction'] = 0.7
    status, res = call(server, '/api/validate', 'POST', {'config': bad})
    assert not res['ok'] and 'add up to 1' in res['errors'][0]

    bad = json.loads(json.dumps(tiny_config))
    bad['phases'][0]['source'] = 'does/not/exist.csv'
    status, res = call(server, '/api/validate', 'POST', {'config': bad})
    assert not res['ok'] and 'file not found' in res['errors'][0]

    bad = json.loads(json.dumps(tiny_config))
    bad['box']['wobble'] = 1
    status, res = call(server, '/api/validate', 'POST', {'config': bad})
    assert not res['ok'] and res['errors'] == ['unknown setting: wobble']


def test_presets_round_trip(server, tiny_config):
    status, res = call(server, '/api/presets', 'POST', {'name': 'tiny ferrite', 'config': tiny_config})
    assert status == 200 and res['name'] == 'tiny_ferrite'
    assert 'tiny_ferrite' in call(server, '/api/presets')[1]['presets']
    status, res = call(server, '/api/presets/tiny_ferrite')
    assert res['config']['box']['size'] == [10.0, 10.0, 10.0]
    assert call(server, '/api/presets/tiny_ferrite', 'DELETE')[0] == 200
    assert call(server, '/api/presets/tiny_ferrite')[0] == 404


def test_split_rves_uses_consecutive_seeds(tiny_config):
    configs = runs.split_rves(dict(tiny_config, n_rves=3, seed=7))
    assert [c['seed'] for c in configs] == [7, 8, 9] and all(c['n_rves'] == 1 for c in configs)


def test_generation_job(server, tiny_config):
    status, res = call(server, '/api/jobs', 'POST', {'name': 'tiny', 'config': tiny_config})
    assert status == 200 and len(res['jobs']) == 1
    job_id = res['jobs'][0]['id']
    deadline = time.time() + 240
    while True:
        job = call(server, f'/api/jobs/{job_id}')[1]['job']
        if job['status'] not in ('queued', 'running'):
            break
        assert time.time() < deadline, 'job did not finish'
        time.sleep(0.5)
    log = call(server, f'/api/jobs/{job_id}/log')[1]['lines']
    assert job['status'] == 'done', '\n'.join(log)
    assert any('grains in' in line for line in log)

    detail = call(server, f'/api/jobs/{job_id}')[1]
    result = detail['result']
    assert result['shape'] == [10, 10, 10] and result['n_grains'] > 0
    assert result['report']['fill_ratio'] == 1.0
    assert len(detail['grains']['phase']) == result['n_grains'] + 1

    status, sl = call(server, f'/api/jobs/{job_id}/slice?axis=2&index=4')
    assert status == 200 and (sl['rows'], sl['cols'], sl['index']) == (10, 10, 4)
    labels = np.frombuffer(base64.b64decode(sl['data']), dtype='<i4').reshape(10, 10)
    stored = np.load(os.path.join(server.jobs.path(job_id), 'labels.npz'))['labels']
    assert np.array_equal(labels, stored[:, :, 4])

    hist = call(server, f'/api/jobs/{job_id}/histogram')[1]
    assert sum(hist['actual']) == result['n_grains'] and len(hist['edges']) == len(hist['actual']) + 1

    status, raw = call(server, f'/api/jobs/{job_id}/files/config.yaml', raw=True)
    assert status == 200 and b'resolution' in raw
    assert call(server, f'/api/jobs/{job_id}/files/secret.txt')[0] == 404

    # a restarted manager finds the run again
    from dragen.gui.jobs import JobManager
    again = JobManager(server.jobs.workdir, max_parallel=0)
    assert again.get(job_id).status == 'done'
    again.shutdown()


def test_cancel_and_delete_queued_job(server, tiny_config):
    # max_parallel = 1 and a long first job keep the second one queued
    big = dict(tiny_config, box={'size': [40.0, 40.0, 40.0], 'resolution': 2.0, 'periodic': [True] * 3})
    first = call(server, '/api/jobs', 'POST', {'config': big})[1]['jobs'][0]['id']
    second = call(server, '/api/jobs', 'POST', {'config': tiny_config})[1]['jobs'][0]['id']
    assert call(server, f'/api/jobs/{second}/cancel', 'POST')[1]['job']['status'] == 'cancelled'
    assert call(server, f'/api/jobs/{first}/cancel', 'POST')[1]['job']['status'] == 'cancelled'
    assert call(server, f'/api/jobs/{second}', 'DELETE')[0] == 200
    assert call(server, f'/api/jobs/{second}')[0] == 404
    assert not os.path.exists(server.jobs.path(second))


def test_mesh_requires_finished_run(server, tiny_config):
    job_id = call(server, '/api/jobs', 'POST', {'config': tiny_config})[1]['jobs'][0]['id']
    call(server, f'/api/jobs/{job_id}/cancel', 'POST')
    status, res = call(server, f'/api/jobs/{job_id}/mesh', 'POST')
    assert status == 409 and 'error' in res


def test_imported_runs_are_listed(tmp_path):
    from dragen.gui.jobs import JobManager
    run = tmp_path / 'lognormal_box10_res1_seed1'
    run.mkdir()
    np.savez_compressed(run / 'labels.npz', labels=np.ones((4, 4, 4), dtype=np.int32))
    (run / 'result.json').write_text('{}', encoding='utf-8')
    manager = JobManager(tmp_path, max_parallel=0)
    job = manager.get(run.name)
    assert job.imported and job.status == 'done'
    with pytest.raises(ValueError):
        manager.delete(run.name)
    manager.shutdown()


# --------------------------------------------------------------------------------------- #
# 3D view

def parse_surface(payload: bytes):
    assert payload[:4] == b'DRGS'
    hlen = struct.unpack('<I', payload[4:8])[0]
    header = json.loads(payload[8:8 + hlen])
    off = 8 + hlen
    v, t = header['n_points'], header['n_triangles']
    points = np.frombuffer(payload, '<f4', v * 3, off).reshape(v, 3)
    off += v * 12
    tris = np.frombuffer(payload, '<u4', t * 3, off).reshape(t, 3)
    off += t * 12
    regions = np.frombuffer(payload, '<i4', t * 2, off).reshape(t, 2)
    off += t * 8
    mask = np.frombuffer(payload, 'u1', t, off)
    off += t + (-t % 4)
    assert off == len(payload)
    return header, points, tris, regions, mask


def triangle_area(points, tris):
    p = points[tris].astype(float)
    return np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) / 2


def test_greedy_merge_keeps_the_voxel_faces():
    from dragen.gui.surface import staircase_surface
    from dragen.mesh.surface import staircase_mesh
    rng = np.random.default_rng(0)
    labels = rng.integers(1, 5, (7, 6, 5)).astype(np.int32)
    labels[1:5, 1:4, 1:4] = 9
    surf = staircase_surface(labels, (0.5, 0.5, 0.5))
    ref = staircase_mesh(labels, (0.5, 0.5, 0.5))
    assert surf.meta['face_area_voxels'] == surf.meta['voxel_faces'] == len(ref.triangles) // 2
    assert len(surf.triangles) < len(ref.triangles)
    # same area per (front grain, back region) as the voxel faces
    def area_per_pair(points, tris, regions):
        area = triangle_area(points, tris)
        key = regions[:, 0] * 100 + regions[:, 1]
        return {int(k): area[key == k].sum() for k in np.unique(key)}

    merged = area_per_pair(surf.points, surf.triangles, surf.regions)
    voxels = area_per_pair(ref.points, ref.triangles, np.where(ref.grains < 0, -1, ref.grains))
    assert merged.keys() == voxels.keys()
    assert all(abs(merged[k] - voxels[k]) < 1e-9 for k in voxels)
    # the front grain lies behind every face
    p = surf.points[surf.triangles].astype(float)
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.linalg.norm(n, axis=1)[:, None]
    voxel = np.floor((p.mean(axis=1) - 0.1 * n) / 0.5).astype(int)
    inside = np.all((voxel >= 0) & (voxel < labels.shape), axis=1)
    assert np.array_equal(labels[tuple(voxel[inside].T)], surf.regions[inside, 0])


def test_surface_endpoint(server, tiny_config):
    job_id = call(server, '/api/jobs', 'POST', {'config': tiny_config})[1]['jobs'][0]['id']
    deadline = time.time() + 240
    while call(server, f'/api/jobs/{job_id}')[1]['job']['status'] in ('queued', 'running'):
        assert time.time() < deadline
        time.sleep(0.5)
    detail = call(server, f'/api/jobs/{job_id}')[1]
    assert detail['surfaces'] == [{'source': 'staircase', 'label': 'Staircase (voxel faces)'}]
    assert detail['spacing'] == [1.0, 1.0, 1.0]
    n_grains = detail['result']['n_grains']
    assert len(detail['grains']['actual_volume']) == n_grains + 1
    assert sum(detail['grains']['actual_volume']) == pytest.approx(1000.0)

    status, payload = call(server, f'/api/jobs/{job_id}/surface?source=staircase&step=1', raw=True)
    assert status == 200
    header, points, tris, regions, mask = parse_surface(payload)
    assert header['source'] == 'staircase' and header['step'] == 1
    assert tris.max() < len(points)
    assert regions[:, 0].min() >= 1 and regions[:, 0].max() <= n_grains
    assert set(np.unique(regions[:, 1][regions[:, 1] < 1])) <= {-1}
    assert regions[:, 1].max() <= n_grains
    # greedy merge: total area equals the voxel face area (spacing 1 um)
    assert triangle_area(points, tris).sum() == pytest.approx(header['voxel_faces'])
    assert set(np.unique(mask)) <= {2, 4}
    # all points inside the box
    assert points.min() >= 0 and points.max() <= 10

    status, payload = call(server, f'/api/jobs/{job_id}/surface?step=2', raw=True)
    header2 = parse_surface(payload)[0]
    assert status == 200 and header2['step'] == 2 and header2['shape'] == [5, 5, 5]

    assert call(server, f'/api/jobs/{job_id}/surface?source=smoothed')[0] == 404
    assert call(server, f'/api/jobs/{job_id}/surface?source=nonsense')[0] == 400


def test_smoothed_surface_from_mesh_npz(tmp_path):
    from dragen.gui.surface import smoothed_surface
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    tris = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32)
    regions = np.array([[1, -5], [1, -3], [1, 2], [1, 2]], dtype=np.int32)
    np.savez_compressed(tmp_path / 'mesh.npz', points=points, cells=np.zeros((0, 10)), grain=np.zeros(0),
                        surf_points=points, surf_triangles=tris, surf_regions=regions)
    surf = smoothed_surface(tmp_path / 'mesh.npz', (1, 1, 1))
    header, p, t, r, _ = parse_surface(surf.to_bytes())
    assert header['source'] == 'smoothed' and len(t) == 4
    assert r[:, 1].tolist() == [-1, -1, 2, 2]
    np.savez_compressed(tmp_path / 'old.npz', points=points)
    assert smoothed_surface(tmp_path / 'old.npz', (1, 1, 1)) is None


def _wait(server, job_id, seconds=240):
    deadline = time.time() + seconds
    while True:
        job = call(server, f'/api/jobs/{job_id}')[1]['job']
        if job['status'] not in ('queued', 'running'):
            return job
        assert time.time() < deadline, 'job did not finish'
        time.sleep(0.5)


def test_export_job(server, tiny_config):
    import zipfile
    info = call(server, '/api/info')[1]
    assert set(info['export_formats']) == {'damask', 'abaqus', 'exodus', 'vtu', 'gmsh', 'grains'}
    assert not any('Export' in f['feature'] for f in info['not_supported'])
    run = call(server, '/api/jobs', 'POST', {'name': 'to export', 'config': tiny_config})[1]['jobs'][0]['id']
    assert _wait(server, run)['status'] == 'done'

    assert call(server, f'/api/jobs/{run}/export', 'POST', {'formats': []})[0] == 400
    assert call(server, f'/api/jobs/{run}/export', 'POST', {'formats': ['stl']})[0] == 400
    assert call(server, f'/api/jobs/{run}/export', 'POST', {'formats': ['vtu'], 'length_unit': 'inch'})[0] == 400
    assert call(server, f'/api/jobs/{run}/export', 'POST', {'formats': ['vtu'], 'mesh': 'nope'})[0] == 404

    formats = ['abaqus', 'exodus', 'vtu', 'grains'] + ([] if 'damask' in info['export_unavailable'] else ['damask'])
    status, res = call(server, f'/api/jobs/{run}/export', 'POST',
                       {'formats': formats, 'mesh': 'voxel', 'length_unit': 'um', 'material_per': 'phase'})
    assert status == 200 and res['job']['kind'] == 'export' and res['job']['parent'] == run
    exp = res['job']['id']
    job = _wait(server, exp)
    assert job['status'] == 'done', '\n'.join(call(server, f'/api/jobs/{exp}/log')[1]['lines'])
    detail = call(server, f'/api/jobs/{exp}')[1]
    summary = detail['export']
    assert summary['mesh'] == 'voxel' and summary['cell_type'] == 'hexahedron'
    assert {f['format'] for f in summary['files']} == set(formats)
    assert call(server, f'/api/jobs/{run}')[1]['exports'][0]['id'] == exp

    status, raw = call(server, f'/api/jobs/{exp}/files/export.zip', raw=True)
    assert status == 200
    path = os.path.join(server.jobs.path(exp), 'downloaded.zip')
    with open(path, 'wb') as f:
        f.write(raw)
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        assert {'abaqus/rve.inp', 'abaqus/graindata.inp', 'rve.e', 'rve.vtu', 'grains.csv', 'metadata.json'} <= names
        inp = z.read('abaqus/rve.inp').decode()
    labels = np.load(os.path.join(server.jobs.path(run), 'labels.npz'))['labels']
    assert f'*Element, type=C3D8' in inp and 'material=Ferrite\n' in inp
    assert '10, 10, 10\n' in inp or ', 10, 10, 10' in inp                  # last node at 10 um
    assert inp.count('*Elset, elset=GRAIN_') == int(labels.max())


def test_export_with_substructure(server, tiny_config):
    config = dict(tiny_config, phases=[dict(tiny_config['phases'][0], fraction=0.5),
                                       dict(tiny_config['phases'][0], phase='martensite', fraction=0.5)])
    run = call(server, '/api/jobs', 'POST', {'config': config})[1]['jobs'][0]['id']
    assert _wait(server, run)['status'] == 'done'
    assert call(server, f'/api/jobs/{run}/export', 'POST',
                {'formats': ['grains'], 'substructure': {'block_thickness': -1}})[0] == 400
    exp = call(server, f'/api/jobs/{run}/export', 'POST',
               {'formats': ['grains'], 'substructure': {'block_thickness': 1.0, 'min_packet_voxels': 20}})[1]['job']['id']
    assert _wait(server, exp)['status'] == 'done'
    summary = call(server, f'/api/jobs/{exp}')[1]['export']
    assert summary['substructure']['blocks'] > 0
    assert not any('Substructure' in f['feature'] for f in call(server, '/api/info')[1]['not_supported'])
