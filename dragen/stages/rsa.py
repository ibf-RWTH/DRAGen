"""Discrete random sequential addition of shrunk ellipsoid seeds.

Grains are placed in the order given (largest first). For each grain a centre is drawn uniformly
among the free voxels; the seed (epoch-0 part of its kernel) is accepted if at most
`max_overlap` of it lies on occupied voxels and the part that is actually written stays one
connected body. Occupied voxels are never overwritten.

With a region (e.g. a band slab) the centre is drawn among the free voxels of the region, and at
most `max_outside` of the seed may lie outside it.
"""
from typing import List, Optional

import numpy as np
from scipy import ndimage

from dragen.config.rve import RsaConfig
from dragen.core.domain import VoxelDomain
from dragen.core.kernel import EllipsoidKernel


class RsaError(RuntimeError):
    pass


class RsaPlacer:

    def __init__(self, domain: VoxelDomain, config: RsaConfig, rng: np.random.Generator):
        self.domain = domain
        self.config = config
        self.rng = rng
        self.attempts = 0

    def place(self, labels: np.ndarray, kernels: List[EllipsoidKernel], first_id: int = 1,
              region: Optional[np.ndarray] = None, max_outside: float = 0.1) -> np.ndarray:
        """Place one seed per kernel; grain i gets label first_id + i. Returns (G, ndim) centres.

        region: optional flat boolean mask the seeds have to lie in.
        """
        flat_labels = labels.reshape(-1)
        pool = None if region is None else np.flatnonzero(region)
        centers = np.empty((len(kernels), self.domain.ndim), dtype=np.int64)
        for i, kernel in enumerate(kernels):
            centers[i] = self._place_one(flat_labels, kernel.seed, first_id + i, region, pool, max_outside)
        return centers

    def _place_one(self, flat_labels, seed, grain_id, region, pool, max_outside) -> np.ndarray:
        for _ in range(self.config.max_attempts_per_grain):
            self.attempts += 1
            center = self._random_free_voxel(flat_labels, pool)
            flat, inside = self.domain.wrap(center + seed)
            if len(flat) > 1:
                # a seed wider than half a periodic box wraps onto itself
                first = np.sort(np.unique(flat, return_index=True)[1])
                flat, inside = flat[first], np.flatnonzero(inside)[first]
            occupied = flat_labels[flat] != 0
            if occupied.mean() > self.config.max_overlap:
                continue
            if region is not None and 1 - region[flat].mean() > max_outside:
                continue
            if occupied.any() and not _connected(seed[inside][~occupied]):
                continue
            flat_labels[flat[~occupied]] = grain_id
            return center
        raise RsaError(f'grain {grain_id}: no valid position after {self.config.max_attempts_per_grain} '
                       'attempts; lower shrink_volume_fraction or raise max_overlap')

    def _random_free_voxel(self, flat_labels: np.ndarray, pool: Optional[np.ndarray] = None) -> np.ndarray:
        """Uniformly random free voxel, of the whole box or of `pool` (flat indices)."""
        while True:
            if pool is None:
                draws = self.rng.integers(0, flat_labels.size, size=64)
            else:
                draws = pool[self.rng.integers(0, pool.size, size=64)]
            free = draws[flat_labels[draws] == 0]
            if free.size:
                return self.domain.unravel(free[:1])[0]
            if not np.any(flat_labels[pool] == 0 if pool is not None else flat_labels == 0):
                raise RsaError('no free voxel left for the next seed')


def _connected(offsets: np.ndarray) -> bool:
    """True if the voxels at these integer offsets form one face-connected body."""
    local = offsets - offsets.min(axis=0)
    mask = np.zeros(local.max(axis=0) + 1, dtype=bool)
    mask[tuple(local.T)] = True
    _, n = ndimage.label(mask)
    return n == 1
