"""Data behind the API: config validation, input files, presets, run results, slices, histograms."""
import base64
import re
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from dragen.config import Phase, RveConfig
from dragen.gui.jobs import REPO_ROOT

INPUT_DIRS = (REPO_ROOT / 'ExampleInput', REPO_ROOT / 'benchmarks' / 'inputs')

NOT_SUPPORTED = [
    {'feature': 'Slope offset', 'planned': 'E2', 'note': 'not implemented in the new code (V25)'},
    {'feature': '2D RVEs', 'planned': 'later',
     'note': 'library (RVESpec with two box sizes) and CLI support 2D with triangle meshes; the GUI form and views are 3D only'},
    {'feature': 'Progress per epoch, animations', 'planned': 'E2',
     'note': 'the generator reports stage timings only'},
]


def default_config() -> dict:
    lognormal = REPO_ROOT / 'benchmarks' / 'inputs' / 'lognormal_ferrite.csv'
    ferrite = lognormal if lognormal.exists() else REPO_ROOT / 'ExampleInput' / 'Ferrite' / 'Homogeneous_Grainsize.csv'
    return {
        'box': {'size': [20.0, 20.0, 20.0], 'resolution': 2.0, 'periodic': [True, True, True]},
        'phases': [{'phase': 'ferrite', 'source': str(ferrite), 'fraction': 1.0}],
        'rsa': {'shrink_volume_fraction': 0.3, 'max_overlap': 0.05, 'max_attempts_per_grain': 10000},
        'tessellation': {'connected_growth': True, 'kernel_margin': 2.0, 'growth_exponent': 0.3},
        'bands': {'count': 0, 'phase': 'martensite', 'source': None, 'thickness': [0.0, 0.0],
                  'normal_axis': 2, 'filling': 0.5},
        'inclusions': {'source': None, 'fraction': 0.0},
        'seed': 1, 'n_rves': 1,
    }


def validate(data: dict) -> dict:
    """Build an RveConfig from the form data; return it normalised, or readable errors."""
    errors: List[str] = []
    config = None
    try:
        config = RveConfig.from_dict(data)
    except (ValueError, TypeError, KeyError) as err:
        errors.append(_readable(err))
    if config is not None:
        for label, source in _sources(config):
            if not Path(source).is_file():
                errors.append(f'{label}: file not found: {source}')
            elif not source.lower().endswith(('.csv', '.pkl')):
                errors.append(f'{label}: grain sources must be .csv or .pkl')
    out = {'ok': not errors, 'errors': errors}
    if config is not None:
        out['config'] = config.to_dict()
        out['shape'] = list(config.box.shape)
        out['voxels'] = int(np.prod(config.box.shape))
        out['spacing'] = list(config.box.spacing)
        out['warnings'] = _warnings(config)
    return out


def _sources(config: RveConfig) -> List[Tuple[str, str]]:
    out = [(f'phase {p.phase.name.lower()}', p.source) for p in config.phases]
    if config.bands.count:
        out.append(('bands', config.bands.source))
    if config.inclusions.fraction > 0:
        out.append(('inclusions', config.inclusions.source))
    return out


def _warnings(config: RveConfig) -> List[str]:
    out = []
    voxels = int(np.prod(config.box.shape))
    if voxels > 128 ** 3:
        out.append(f'{voxels:,} voxels: generation may take minutes and several GB of memory')
    if config.box.resolution < 2:
        out.append('below 2 voxels/um the volume criteria fail for fine-grained inputs (process notes O09)')
    if config.n_rves > 1:
        out.append(f'{config.n_rves} RVEs: one job per RVE, seeds {config.seed} ... '
                   f'{(config.seed or 0) + config.n_rves - 1}' if config.seed is not None
                   else f'{config.n_rves} RVEs: one job per RVE with random seeds')
    if any(s.lower().endswith('.pkl') for _, s in _sources(config)):
        out.append('GAN sources load torch in the worker; the first RVE takes longer')
    return out


def _readable(err: Exception) -> str:
    text = str(err)
    if isinstance(err, TypeError):
        match = re.search(r"unexpected keyword argument '(\w+)'", text)
        if match:
            return f'unknown setting: {match.group(1)}'
        match = re.search(r"missing \d+ required positional argument[s]?: (.*)", text)
        if match:
            return f'missing setting: {match.group(1)}'
    if isinstance(err, KeyError):
        return f'missing section: {text}'
    return text


def split_rves(config: dict) -> List[dict]:
    """One config per RVE: n_rves copies with consecutive seeds (random seeds without a seed)."""
    n = int(config.get('n_rves', 1))
    if n <= 1:
        return [dict(config, n_rves=1)]
    seed = config.get('seed')
    if seed is None:
        seeds = [int(s) for s in np.random.default_rng().integers(0, 2 ** 31, n)]
    else:
        seeds = [int(seed) + i for i in range(n)]
    return [dict(config, seed=s, n_rves=1) for s in seeds]


def input_files() -> List[dict]:
    """Grain sources found in the example and benchmark input folders, with a guessed phase."""
    out = []
    phases = [p.name.lower() for p in Phase]
    for root in INPUT_DIRS:
        if not root.exists():
            continue
        for path in sorted(root.rglob('*')):
            if path.suffix.lower() not in ('.csv', '.pkl') or not path.is_file():
                continue
            text = str(path.relative_to(REPO_ROOT)).lower()
            guess = next((p for p in phases if p in text), None)
            if 'band' in text:
                guess = 'bands'
            if 'inclusion' in text or 'einschl' in text:
                guess = 'inclusions'
            kind = 'GAN' if path.suffix.lower() == '.pkl' else 'CSV'
            if kind == 'CSV' and not _looks_like_grain_csv(path):
                continue
            out.append({'path': str(path), 'name': str(path.relative_to(REPO_ROOT)), 'kind': kind, 'phase': guess})
    return out


def _looks_like_grain_csv(path: Path) -> bool:
    try:
        header = pd.read_csv(path, nrows=0).columns
    except Exception:
        return False
    return 'a' in header


# ------------------------------------------------------------------------------------------ #
# presets

def _preset_dir(workdir: Path) -> Path:
    path = Path(workdir) / '_presets'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _preset_name(name: str) -> str:
    clean = re.sub(r'[^A-Za-z0-9_.-]+', '_', name.strip()).strip('._')
    if not clean:
        raise ValueError('preset name is empty')
    return clean


def list_presets(workdir: Path) -> List[str]:
    return sorted(p.stem for p in _preset_dir(workdir).glob('*.yaml'))


def save_preset(workdir: Path, name: str, config: dict) -> str:
    name = _preset_name(name)
    RveConfig.from_dict(config).save(_preset_dir(workdir) / f'{name}.yaml')
    return name


def load_preset(workdir: Path, name: str) -> dict:
    path = _preset_dir(workdir) / f'{_preset_name(name)}.yaml'
    if not path.exists():
        raise KeyError(name)
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def delete_preset(workdir: Path, name: str) -> None:
    path = _preset_dir(workdir) / f'{_preset_name(name)}.yaml'
    if not path.exists():
        raise KeyError(name)
    path.unlink()


# ------------------------------------------------------------------------------------------ #
# results

class RunData:
    """Read access to one run directory; labels are cached by the server."""

    _labels_cache: 'OrderedDict[str, np.ndarray]' = OrderedDict()
    _surface_cache: 'OrderedDict[tuple, bytes]' = OrderedDict()

    def __init__(self, path: Path):
        self.path = Path(path)

    def result(self) -> Optional[dict]:
        import json
        path = self.path / 'result.json'
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding='utf-8'))

    def config(self) -> Optional[dict]:
        path = self.path / 'config.yaml'
        return yaml.safe_load(path.read_text(encoding='utf-8')) if path.exists() else None

    def labels(self) -> np.ndarray:
        key = str(self.path / 'labels.npz')
        cache = RunData._labels_cache
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        labels = np.load(key)['labels']
        cache[key] = labels
        while len(cache) > 3:
            cache.popitem(last=False)
        return labels

    def grains(self) -> Optional[pd.DataFrame]:
        path = self.path / 'grains.csv'
        return pd.read_csv(path) if path.exists() else None

    def grain_meta(self) -> dict:
        """Phase, target volume and voxel volume per label (index = label, 0 unused), for
        colouring and hover."""
        grains = self.grains()
        if grains is None:
            return {'phase': [], 'volume': [], 'band': [], 'actual_volume': []}
        n = len(grains) + 1
        counts = np.bincount(self.labels().ravel().clip(0), minlength=n)[:n]
        voxel = float(np.prod(self.spacing()))
        return {'phase': [0] + grains['phaseID'].astype(int).tolist(),
                'volume': [0.0] + grains['volume'].round(4).tolist(),
                'actual_volume': (counts * voxel).round(4).tolist(),
                'band': [-1] + grains['band'].astype(int).tolist() if 'band' in grains else []}

    def spacing(self):
        result = self.result() or {}
        if result.get('spacing'):
            return result['spacing']
        return list(RveConfig.from_dict(self.config()).box.spacing)

    def surface(self, source: str = 'staircase', step: Optional[int] = None, mesh_dir: Optional[Path] = None) -> bytes:
        """Binary grain boundary surface for the 3D view (see dragen.gui.surface), cached."""
        from dragen.gui import surface as surf
        if source == 'smoothed':
            if mesh_dir is None:
                raise ValueError('no meshed surface for this run')
            key = (str(mesh_dir), 'smoothed', 1)
        else:
            step = surf.auto_step(self.labels().shape) if not step else int(step)
            if step < 1:
                raise ValueError('step must be >= 1')
            key = (str(self.path), 'staircase', step)
        cache = RunData._surface_cache
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        if source == 'smoothed':
            size = np.array(self.labels().shape) * np.array(self.spacing())
            s = surf.smoothed_surface(Path(mesh_dir) / 'mesh.npz', size)
            if s is None:
                raise ValueError('the mesh of this run stored no surface (meshed with an older GUI version)')
        else:
            s = surf.staircase_surface(self.labels(), self.spacing(), step)
        data = s.to_bytes()
        cache[key] = data
        while len(cache) > 4:
            cache.popitem(last=False)
        return data

    def slice(self, axis: int, index: int) -> dict:
        """One slice of the label field as base64 little-endian int32, row-major (rows, cols).

        axis 2 (z) shows x down and y across, axis 0 (x) y down and z across, axis 1 (y) x down
        and z across."""
        labels = self.labels()
        if not 0 <= axis < labels.ndim:
            raise ValueError(f'axis must be in [0, {labels.ndim})')
        n = labels.shape[axis]
        index = int(np.clip(index, 0, n - 1))
        cut = np.take(labels, index, axis=axis)
        data = np.ascontiguousarray(cut, dtype='<i4')
        return {'axis': axis, 'index': index, 'count': n, 'rows': int(cut.shape[0]), 'cols': int(cut.shape[1]),
                'data': base64.b64encode(data.tobytes()).decode('ascii')}

    def histogram(self, bins: int = 30) -> dict:
        """Equivalent diameters of the grains in the RVE (voxel volume) and of their targets
        (sampled ellipsoids), on common bins."""
        grains = self.grains()
        labels = self.labels()
        voxel = float(np.prod(self.spacing()))
        counts = np.bincount(labels[labels > 0].ravel())[1:]
        dims = labels.ndim

        def diameter(volume):
            volume = np.asarray(volume, dtype=float)
            return (6 * volume / np.pi) ** (1 / 3) if dims == 3 else 2 * np.sqrt(volume / np.pi)
        actual = diameter(counts[counts > 0] * voxel)
        target = diameter(grains['volume'].to_numpy()) if grains is not None else np.zeros(0)
        both = np.concatenate([actual, target]) if len(target) else actual
        if not len(both):
            return {'edges': [], 'actual': [], 'target': []}
        edges = np.linspace(0, float(both.max()) * 1.02, bins + 1)
        return {'edges': edges.round(4).tolist(),
                'actual': np.histogram(actual, edges)[0].tolist(),
                'target': np.histogram(target, edges)[0].tolist() if len(target) else [],
                'unit': 'um', 'quantity': 'equivalent diameter'}

    def export(self) -> Optional[dict]:
        import json
        path = self.path / 'export.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

    def mesh(self) -> Optional[dict]:
        import json
        path = self.path / 'mesh.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
