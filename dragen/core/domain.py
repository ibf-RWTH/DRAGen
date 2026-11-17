"""The voxel grid an RVE lives on, with periodicity handled per axis.

Every stage addresses voxels through this class instead of building full coordinate grids or
rolling whole arrays: a shape (ellipsoid, band, ...) is a set of integer offsets around a centre,
and `wrap` maps centre + offsets onto flat voxel indices. On a periodic axis indices wrap around
(modulo), on a non-periodic axis everything outside the box is dropped. Works for 2D and 3D.
"""
import math
from typing import Sequence, Tuple

import numpy as np

from dragen.config.rve import BoxConfig


class VoxelDomain:

    def __init__(self, shape: Sequence[int], spacing: Sequence[float], periodic: Sequence[bool]):
        if not len(shape) == len(spacing) == len(periodic):
            raise ValueError('shape, spacing and periodic need the same number of axes')
        self.shape: Tuple[int, ...] = tuple(int(n) for n in shape)
        self.spacing: Tuple[float, ...] = tuple(float(s) for s in spacing)
        self.periodic: Tuple[bool, ...] = tuple(bool(p) for p in periodic)
        self._shape = np.asarray(self.shape, dtype=np.int64)
        self._periodic = np.asarray(self.periodic)

    @classmethod
    def from_box(cls, box: BoxConfig) -> 'VoxelDomain':
        return cls(box.shape, box.spacing, box.periodic)

    def __repr__(self) -> str:
        return f'VoxelDomain(shape={self.shape}, spacing={self.spacing}, periodic={self.periodic})'

    # ------------------------------------------------------------------ #

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def n_voxels(self) -> int:
        return math.prod(self.shape)

    @property
    def voxel_volume(self) -> float:
        return math.prod(self.spacing)

    @property
    def size(self) -> Tuple[float, ...]:
        """Physical edge length per axis."""
        return tuple(n * s for n, s in zip(self.shape, self.spacing))

    def new_labels(self, dtype=np.int32) -> np.ndarray:
        """An empty label field (0 = unassigned)."""
        return np.zeros(self.shape, dtype=dtype)

    # ------------------------------------------------------------------ #

    def wrap(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map integer voxel coordinates, possibly outside the box, onto flat indices.

        points: (M, ndim) integer array.
        Returns (flat, inside): `inside` is a boolean mask over the M points telling which ones
        exist in the domain, `flat` holds the flat indices of exactly those points, in order.
        """
        points = np.asarray(points, dtype=np.int64).reshape(-1, self.ndim)
        wrapped = np.where(self._periodic, points % self._shape, points)
        inside = np.all((wrapped >= 0) & (wrapped < self._shape), axis=1)
        flat = np.ravel_multi_index(wrapped[inside].T, self.shape)
        return flat, inside

    def place(self, center: Sequence[int], offsets: np.ndarray) -> np.ndarray:
        """Flat indices of a shape given as offsets around an integer centre voxel.

        Offsets that fall outside a non-periodic axis are dropped.
        """
        flat, _ = self.wrap(np.asarray(center, dtype=np.int64) + np.asarray(offsets, dtype=np.int64))
        return flat

    def unravel(self, flat: np.ndarray) -> np.ndarray:
        """(M, ndim) integer coordinates of flat indices."""
        return np.stack(np.unravel_index(np.asarray(flat, dtype=np.int64), self.shape), axis=1)

    def face_neighbors(self, flat: np.ndarray) -> np.ndarray:
        """Flat indices of the 2*ndim face neighbours of each voxel.

        Returns (M, 2*ndim), ordered -axis0, +axis0, -axis1, ...; -1 where the neighbour lies
        beyond a non-periodic boundary.
        """
        coords = self.unravel(flat)
        out = np.full((len(coords), 2 * self.ndim), -1, dtype=np.int64)
        for axis in range(self.ndim):
            for k, step in enumerate((-1, 1)):
                shifted = coords.copy()
                shifted[:, axis] += step
                idx, inside = self.wrap(shifted)
                out[inside, 2 * axis + k] = idx
        return out

    def voxel_centers(self, flat: np.ndarray) -> np.ndarray:
        """Physical coordinates of voxel centres, (M, ndim)."""
        return (self.unravel(flat) + 0.5) * np.asarray(self.spacing)
