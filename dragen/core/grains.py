"""Grain table: one row per grain, stored as columns of numpy arrays."""
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np
import pandas as pd

EULER = ['phi1', 'PHI', 'phi2']


@dataclass(frozen=True)
class GrainTable:
    """Grains of one RVE.

    radii: (G, ndim) unshrunk semi-axes a, b[, c] in micrometres.
    alpha: (G,) rotation about the last axis in degrees.
    euler: (G, 3) Bunge Euler angles in degrees.
    phase: (G,) phase ID.
    band: (G,) index of the band a grain was seeded in, -1 for matrix grains and inclusions.
    centers: (G, ndim) centre voxel per grain, set by the RSA; None before.
    """

    radii: np.ndarray
    alpha: np.ndarray
    euler: np.ndarray
    phase: np.ndarray
    band: Optional[np.ndarray] = None
    centers: Optional[np.ndarray] = field(default=None)

    def __post_init__(self):
        n = len(self.radii)
        if self.band is None:
            object.__setattr__(self, 'band', np.full(n, -1, dtype=int))
        if not (len(self.alpha) == len(self.euler) == len(self.phase) == len(self.band) == n):
            raise ValueError('all grain columns need the same length')
        if self.radii.ndim != 2 or self.radii.shape[1] not in (2, 3):
            raise ValueError(f'radii must be (G, 2) or (G, 3), got {self.radii.shape}')

    def __len__(self) -> int:
        return len(self.radii)

    @property
    def ndim(self) -> int:
        return self.radii.shape[1]

    @property
    def volume(self) -> np.ndarray:
        """Continuous volume (area in 2D) of the unshrunk ellipsoids."""
        if self.ndim == 3:
            return 4 / 3 * np.pi * np.prod(self.radii, axis=1)
        return np.pi * np.prod(self.radii, axis=1)

    def take(self, index) -> 'GrainTable':
        return GrainTable(radii=self.radii[index], alpha=self.alpha[index], euler=self.euler[index],
                          phase=self.phase[index], band=self.band[index],
                          centers=None if self.centers is None else self.centers[index])

    def with_centers(self, centers: np.ndarray) -> 'GrainTable':
        return replace(self, centers=np.asarray(centers))

    def with_band(self, band: int) -> 'GrainTable':
        return replace(self, band=np.full(len(self), band, dtype=int))

    @staticmethod
    def concat(tables: Sequence['GrainTable']) -> 'GrainTable':
        tables = [t for t in tables if len(t)]
        with_centers = [t.centers is not None for t in tables]
        if any(with_centers) and not all(with_centers):
            raise ValueError('either all or none of the tables need centres')
        return GrainTable(radii=np.concatenate([t.radii for t in tables]),
                          alpha=np.concatenate([t.alpha for t in tables]),
                          euler=np.concatenate([t.euler for t in tables]),
                          phase=np.concatenate([t.phase for t in tables]),
                          band=np.concatenate([t.band for t in tables]),
                          centers=np.concatenate([t.centers for t in tables]) if all(with_centers) else None)

    def to_frame(self) -> pd.DataFrame:
        """Grain i of the table is grain ID i + 1 in the label field."""
        axes = ['a', 'b', 'c'][:self.ndim]
        df = pd.DataFrame(self.radii, columns=axes)
        df['alpha'] = self.alpha
        df[EULER] = self.euler
        df['phaseID'] = self.phase
        df['band'] = self.band
        df['volume'] = self.volume
        if self.centers is not None:
            df[[f'center_{a}' for a in 'xyz'[:self.ndim]]] = self.centers
        df.insert(0, 'GrainID', np.arange(1, len(self) + 1))
        return df

    @staticmethod
    def from_frame(df: pd.DataFrame) -> 'GrainTable':
        """Inverse of to_frame (row order = grain order; volume and GrainID are ignored)."""
        axes = [a for a in ('a', 'b', 'c') if a in df]
        centers = [f'center_{a}' for a in 'xyz'[:len(axes)]]
        return GrainTable(radii=df[axes].to_numpy(dtype=float), alpha=df['alpha'].to_numpy(dtype=float),
                          euler=df[EULER].to_numpy(dtype=float), phase=df['phaseID'].to_numpy(dtype=int),
                          band=df['band'].to_numpy(dtype=int) if 'band' in df else None,
                          centers=df[centers].to_numpy() if all(c in df for c in centers) else None)
