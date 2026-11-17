"""Command line interface dragen.cli (console script `dragen`)."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from dragen.api import RVEResult
from dragen.cli import load_config, main, template


@pytest.fixture(scope='module')
def config_file(tmp_path_factory):
    d = tmp_path_factory.mktemp('cli')
    rng = np.random.default_rng(0)
    a = rng.uniform(1.5, 2.5, 2000)
    pd.DataFrame({'a': a, 'b': a, 'c': a}).to_csv(d / 'grains.csv', index=False)
    data = yaml.safe_load(template().split('mesh:')[0])
    data['box'] = {'size': [8.0, 8.0, 8.0], 'resolution': 2.0, 'periodic': [True, True, True]}
    data['phases'] = [{'phase': 'ferrite', 'source': str(d / 'grains.csv'), 'fraction': 1.0}]
    data['seed'] = 5
    data['export'] = {'formats': ['grains'], 'length_unit': 'um'}
    path = d / 'config.yaml'
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    return path


def test_template_is_a_valid_configuration(tmp_path):
    assert main(['-q', 'template', str(tmp_path / 't.yaml')]) == 0
    config, mesh, export, substructure = load_config(tmp_path / 't.yaml')
    assert substructure == {}
    assert config.box.shape == (50, 50, 50) and mesh['kind'] == 'voxel' and 'grains' in export['formats']


def test_generate_several_rves_with_exports(config_file, tmp_path):
    out = tmp_path / 'out'
    assert main(['-q', 'generate', str(config_file), '-o', str(out), '-n', '2', '-f', 'abaqus,grains',
                 '--material-per', 'phase']) == 0
    dirs = sorted(out.iterdir())
    assert [d.name for d in dirs] == ['rve_000', 'rve_001']
    first, second = (RVEResult.load(d) for d in dirs)
    assert first.metadata['seed'] == 5 and second.metadata['seed'] == 6
    assert not np.array_equal(first.grain_ids, second.grain_ids)
    assert first.grain_ids.shape == (16, 16, 16) and first.box_size_m == (8e-6,) * 3
    grains = pd.read_csv(dirs[0] / 'grains.csv')
    assert len(grains) == first.n_grains
    inp = (dirs[0] / 'abaqus' / 'rve.inp').read_text()
    assert 'material=Ferrite\n' in inp and 'lengths in um' in inp       # section from the config file
    assert (dirs[0] / 'config.yaml').exists()


def test_same_seed_gives_same_rve(config_file, tmp_path):
    for name in ('a', 'b'):
        assert main(['-q', 'generate', str(config_file), '-o', str(tmp_path / name)]) == 0
    a, b = (RVEResult.load(tmp_path / n / 'rve_000') for n in 'ab')
    assert np.array_equal(a.grain_ids, b.grain_ids) and np.allclose(a.orientations, b.orientations)


def test_export_command_and_save_load_round_trip(config_file, tmp_path):
    assert main(['-q', 'generate', str(config_file), '-o', str(tmp_path / 'g')]) == 0
    rve = tmp_path / 'g' / 'rve_000'
    r = RVEResult.load(rve)
    assert r.band_ids.dtype == np.int32 and r.radii_m.shape == (r.n_grains, 3)
    assert main(['-q', 'export', str(rve), '-f', 'vtu,exodus', '-o', str(tmp_path / 'x')]) == 0
    assert (tmp_path / 'x' / 'rve.vtu').exists() and (tmp_path / 'x' / 'rve.e').exists()
    assert main(['-q', 'export', str(rve), '-f', 'vtu', '--mesh', 'tet']) == 1     # no tet mesh
    assert main(['-q', 'export', str(rve), '-f', 'stl']) == 1


def test_tet_mesh_from_the_command_line(config_file, tmp_path):
    assert main(['-q', 'generate', str(config_file), '-o', str(tmp_path), '--mesh', 'tet', '--order', '1',
                 '-f', 'exodus']) == 0
    r = RVEResult.load(tmp_path / 'rve_000')
    assert r.tet_mesh is not None and r.tet_mesh.cells.shape[1] == 4 and set(r.tet_mesh.periodic_pairs) == {0, 1, 2}


def test_module_entry_point():
    out = subprocess.run([sys.executable, '-m', 'dragen', '--help'], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and 'generate' in out.stdout and 'export' in out.stdout


def test_substructure_from_the_configuration(config_file, tmp_path):
    data = yaml.safe_load(config_file.read_text())
    data['phases'] = [{'phase': 'ferrite', 'source': data['phases'][0]['source'], 'fraction': 0.5},
                      {'phase': 'martensite', 'source': data['phases'][0]['source'], 'fraction': 0.5}]
    data['substructure'] = {'block_thickness': 0.5, 'min_packet_voxels': 50}
    path = tmp_path / 'subs.yaml'
    path.write_text(yaml.safe_dump(data))
    assert main(['-q', 'generate', str(path), '-o', str(tmp_path / 'o')]) == 0
    rve = RVEResult.load(tmp_path / 'o' / 'rve_000')
    blocks = RVEResult.load(tmp_path / 'o' / 'rve_000' / 'substructure')
    assert blocks.n_grains > rve.n_grains
    assert np.array_equal(blocks.parent_grain_ids[blocks.grain_ids - 1], rve.grain_ids)
    table = pd.read_csv(tmp_path / 'o' / 'rve_000' / 'grains.csv')            # exported: the blocks
    assert len(table) == blocks.n_grains and 'packet_id' in table


def test_2d_rve_with_triangles(config_file, tmp_path):
    data = yaml.safe_load(config_file.read_text())
    data['box'] = {'size': [10.0, 8.0], 'resolution': 2.0, 'periodic': [True, True]}
    data['mesh'] = {'kind': 'tet', 'order': 2}
    path = tmp_path / 'c2d.yaml'
    path.write_text(yaml.safe_dump(data))
    assert main(['-q', 'generate', str(path), '-o', str(tmp_path / 'o'), '-f', 'abaqus,exodus,grains']) == 0
    r = RVEResult.load(tmp_path / 'o' / 'rve_000')
    assert r.grain_ids.shape == (20, 16) and r.tet_mesh.cell_type == 'triangle6'
    assert '*Element, type=CPE6' in (tmp_path / 'o' / 'rve_000' / 'abaqus' / 'rve.inp').read_text()
