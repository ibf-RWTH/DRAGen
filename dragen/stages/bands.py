"""Bands of a second phase (Henrich et al. 2023, Heliyon, section 3.3.4).

A band is a slab of the box normal to one axis. Its thickness is drawn uniformly between the
configured bounds, its centre uniformly along the normal axis; bands must not overlap. Band grains
are sampled until they hold `filling` times the slab volume, their seeds are placed inside the
slab by the RSA, and from then on they grow like every other grain.
"""
from dataclasses import dataclass
from typing import List

import numpy as np

from dragen.config.rve import BandConfig
from dragen.core.domain import VoxelDomain


@dataclass(frozen=True)
class Band:
    axis: int
    center: float       # micrometres along `axis`
    thickness: float    # micrometres

    def distance(self, coordinate: np.ndarray, length: float, periodic: bool) -> np.ndarray:
        d = np.abs(coordinate - self.center)
        return np.minimum(d, length - d) if periodic else d

    def mask(self, domain: VoxelDomain) -> np.ndarray:
        """Flat boolean mask of the voxels whose centre lies inside the slab."""
        n = domain.shape[self.axis]
        centres = (np.arange(n) + 0.5) * domain.spacing[self.axis]
        inside_1d = self.distance(centres, domain.size[self.axis], domain.periodic[self.axis]) <= self.thickness / 2
        shape = [1] * domain.ndim
        shape[self.axis] = n
        return np.broadcast_to(inside_1d.reshape(shape), domain.shape).ravel()

    def overlaps(self, other: 'Band', length: float, periodic: bool) -> bool:
        gap = self.distance(np.asarray(other.center), length, periodic)
        return bool(gap < (self.thickness + other.thickness) / 2)


class BandError(RuntimeError):
    pass


def place_bands(config: BandConfig, domain: VoxelDomain, rng: np.random.Generator,
                max_attempts: int = 1000) -> List[Band]:
    axis = config.normal_axis
    length = domain.size[axis]
    periodic = domain.periodic[axis]
    lo, hi = config.thickness
    thickness = rng.uniform(lo, hi, size=config.count)
    if thickness.sum() > length:
        raise BandError(f'{config.count} bands of {thickness.sum():.2f} um do not fit into {length:.2f} um')
    bands: List[Band] = []
    for t in thickness:
        for _ in range(max_attempts):
            if periodic:
                center = rng.uniform(0, length)
            else:
                center = rng.uniform(t / 2, length - t / 2)   # a band ends at a free surface, not beyond it
            band = Band(axis=axis, center=float(center), thickness=float(t))
            if not any(band.overlaps(b, length, periodic) for b in bands):
                bands.append(band)
                break
        else:
            raise BandError(f'no position for band {len(bands) + 1} of {t:.2f} um without overlap')
    return bands
