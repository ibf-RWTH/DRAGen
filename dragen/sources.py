"""Grain sources: read measured or synthetic grain data and sample the grains of one RVE.

A CSV source holds one row per measured grain with column a (semi-axis, micrometres) and the
optional columns b, c (default a), alpha (default 0) and phi1, PHI, phi2 (default random texture).

A GAN source draws a fresh set of synthetic grains from a trained WGAN (.pkl, see
InputGenerator/) for every RVE, as the legacy generator did.
"""
import contextlib
import io
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from dragen.config.phases import Phase
from dragen.core.grains import EULER, GrainTable


def random_euler(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random orientations as Bunge Euler angles in degrees, phi1 and phi2 in [0, 360)."""
    euler = Rotation.random(n, random_state=rng).as_euler('ZXZ', degrees=True)
    euler[:, [0, 2]] %= 360
    return euler


def open_source(path: str, phase: Phase, ndim: int = 3) -> 'GrainSource':
    if path.lower().endswith('.csv'):
        return CsvSource(path, phase, ndim)
    if path.lower().endswith('.pkl'):
        return GanSource(path, phase, ndim)
    raise ValueError(f'{path}: grain sources must be .csv or .pkl')


class GrainSource:
    """Sampling shared by all sources; subclasses provide the grain data via _grain_data()."""

    def __init__(self, path: str, phase: Phase, ndim: int = 3):
        self.path = path
        self.phase = Phase.parse(phase)
        self.ndim = ndim

    def _grain_data(self, rng: np.random.Generator) -> pd.DataFrame:
        """Columns a, b, c, alpha and optionally phi1, PHI, phi2."""
        raise NotImplementedError

    def sample(self, volume: float, rng: np.random.Generator, min_radius: float = 0.0,
               max_radius: Optional[float] = None, tolerance: float = 0.05,
               max_extent: Optional[Tuple[int, float]] = None) -> GrainTable:
        """Draw grains without replacement until their total volume lies in [volume, (1 + tolerance) * volume].

        If at the end no remaining grain fits into that band, the grain that overshoots least is
        added, so the total reaches the volume. Grains with a radius below min_radius (they would
        not cover one voxel) or a above max_radius (larger than half the box) are skipped, as in
        the legacy sampler. max_extent = (axis, length) also skips grains whose ellipsoid is longer
        than length along that axis (band grains must fit into the band thickness). Raises if the
        source does not hold enough volume.
        """
        data = self._grain_data(rng)
        radii = data[['a', 'b', 'c'][:self.ndim]].to_numpy()
        keep = radii.min(axis=1) >= min_radius
        if max_radius is not None:
            keep &= data['a'].to_numpy() < max_radius
        if max_extent is not None:
            keep &= self._extent(data, max_extent[0]) <= max_extent[1]
        candidates = rng.permutation(np.flatnonzero(keep))
        vol = self._volume(radii[candidates])

        chosen = []
        total = 0.0
        upper = (1 + tolerance) * volume
        skipped = []
        for i, v in zip(candidates, vol):
            if total >= volume:
                break
            if total + v > upper:
                skipped.append((v, i))  # would overshoot the tolerance band: try the next one
                continue
            chosen.append(i)
            total += v
        if total < volume and skipped:
            # no grain fits the band any more: take the one that overshoots least
            v, i = min(skipped)
            chosen.append(i)
            total += v
        if total < volume:
            raise ValueError(f'{self.path}: not enough grains for a volume of {volume:.1f} '
                             f'(reached {total:.1f})')
        return self._table(data, np.asarray(chosen), rng)

    def _extent(self, data: pd.DataFrame, axis: int) -> np.ndarray:
        """Length of each ellipsoid along an axis (alpha rotates about the last axis)."""
        if axis == self.ndim - 1 and self.ndim == 3:
            return 2 * data['c'].to_numpy()
        t = np.deg2rad(data['alpha'].to_numpy())
        a, b = data['a'].to_numpy(), data['b'].to_numpy()
        if axis == 0:
            return 2 * np.sqrt((a * np.cos(t)) ** 2 + (b * np.sin(t)) ** 2)
        return 2 * np.sqrt((a * np.sin(t)) ** 2 + (b * np.cos(t)) ** 2)

    def _volume(self, radii: np.ndarray) -> np.ndarray:
        return (4 / 3 if self.ndim == 3 else 1.0) * np.pi * np.prod(radii, axis=1)

    def _table(self, data: pd.DataFrame, rows: np.ndarray, rng: np.random.Generator) -> GrainTable:
        data = data.iloc[rows]
        if all(c in data for c in EULER):
            euler = data[EULER].to_numpy()
        else:
            euler = random_euler(len(rows), rng)
        return GrainTable(radii=data[['a', 'b', 'c'][:self.ndim]].to_numpy(),
                          alpha=data['alpha'].to_numpy(), euler=euler,
                          phase=np.full(len(rows), int(self.phase)))


class CsvSource(GrainSource):

    def __init__(self, path: str, phase: Phase, ndim: int = 3):
        super().__init__(path, phase, ndim)
        self._data = self._read(path)

    def __len__(self) -> int:
        return len(self._data)

    def _grain_data(self, rng: np.random.Generator) -> pd.DataFrame:
        return self._data

    @staticmethod
    def _read(path: str) -> pd.DataFrame:
        data = pd.read_csv(path)
        if 'a' not in data or data['a'].count() == 0:
            raise ValueError(f"{path}: column 'a' is required")
        out = pd.DataFrame({'a': data['a'].astype(float)})
        for axis in ('b', 'c'):
            out[axis] = data[axis].astype(float) if axis in data and data[axis].count() else out['a']
        out['alpha'] = data['alpha'].astype(float) if 'alpha' in data and data['alpha'].count() else 0.0
        if all(c in data and data[c].count() for c in EULER):
            out[EULER] = data[EULER].astype(float)
        return out.dropna(subset=['a', 'b', 'c', 'alpha']).reset_index(drop=True)


@contextlib.contextmanager
def _stdout_to_log():
    """The GAN code prints progress; send it to the log at DEBUG level instead of stdout."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield
    text = buffer.getvalue().strip()
    if text:
        logging.getLogger(__name__).debug('GAN: %s', text)


class GanSource(GrainSource):
    """Synthetic grains from a trained WGAN.

    The generator yields per grain the section area, aspect ratio and slope of a 2D ellipse fit
    (and optionally phi1, PHI, phi2). They are turned into ellipsoids as in the legacy
    Helpers.read_input_gan, see ellipses_to_ellipsoids.
    """

    def __init__(self, path: str, phase: Phase, ndim: int = 3, batch_size: int = 10000):
        super().__init__(path, phase, ndim)
        self.batch_size = batch_size
        self._gan = None

    def _load(self):
        if self._gan is None:
            from dragen._extras import headless_matplotlib, require
            require('torch', 'ml')
            headless_matplotlib()                           # the GAN code imports pyplot
            from InputGenerator.C_WGAN_GP import WGANCGP   # pulls in torch, so only on demand
            with _stdout_to_log():
                gan = WGANCGP(df_list=[], storepath='.', num_features=3, gen_iters=500000)
                gan.load_trained_states(single=False, file_list=[self.path])
            self._gan = gan
        return self._gan

    def _grain_data(self, rng: np.random.Generator) -> pd.DataFrame:
        import torch
        gan = self._load()          # loading builds networks and draws from torch's RNG: seed after it
        torch.manual_seed(int(rng.integers(2 ** 31)))
        with _stdout_to_log():
            raw = gan.sample_batch(label=0, size=self.batch_size).dropna(axis=0)
        return ellipses_to_ellipsoids(raw)


def ellipses_to_ellipsoids(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert WGAN output (area, aspect ratio, slope[, phi1, PHI, phi2]) into grain columns.

    As in the legacy Helpers.read_input_gan: major semi-axis sqrt(area * AR / pi), minor
    area / (pi * major), c = major, alpha = slope, and for slopes in (45, 135] the two in-plane
    axes are swapped. (The legacy swap uses a chained pandas assignment that works up to
    pandas 2.x and would silently stop working under copy-on-write.)
    """
    area, ratio, slope = (raw.iloc[:, i].to_numpy(float) for i in range(3))
    major = np.sqrt(area * ratio / np.pi)
    minor = area / (np.pi * major)
    swap = (slope > 45) & (slope <= 135)
    a = np.where(swap, minor, major)
    b = np.where(swap, major, minor)
    out = pd.DataFrame({'a': a, 'b': b, 'c': a, 'alpha': slope})
    if raw.shape[1] > 3:
        out[EULER] = raw.iloc[:, 3:6].to_numpy(float)
    return out.dropna().reset_index(drop=True)
