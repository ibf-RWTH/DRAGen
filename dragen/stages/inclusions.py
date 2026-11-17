"""Non-growing inclusions or pores, placed after tessellation (Henrich et al. 2023, Heliyon, 3.3.3).

Every inclusion is a full-size ellipsoid placed at a random voxel. It is accepted only if the
inclusion and a one-voxel shell around it lie inside a single grain, so it never touches a grain
boundary and its host stays one body. The shell contains all 26 neighbours (faces, edges and
corners): a shell of face neighbours only is not face-connected along the edges of the inclusion,
and a host voxel there could lose its last face connection to the rest of the grain.

Centres are drawn only among voxels whose distance to the nearest grain boundary is at least the
smallest semi-axis of the inclusion plus the shell (a necessary condition, from a distance
transform recomputed after every placement); the exact test above decides. Inclusions are placed largest first; placed ones get
consecutive IDs, the others are only counted.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy import ndimage

from dragen.core.domain import VoxelDomain
from dragen.core.kernel import EllipsoidKernel


@dataclass
class InclusionStats:
    placed: int = 0
    not_placed: int = 0
    attempts: int = 0
    hosts: List[int] = field(default_factory=list)


class InclusionPlacer:

    def __init__(self, domain: VoxelDomain, rng: np.random.Generator, max_attempts: int = 2000):
        self.domain = domain
        self.rng = rng
        self.max_attempts = max_attempts
        self.stats = InclusionStats()

    def place(self, labels: np.ndarray, kernels: List[EllipsoidKernel], first_id: int):
        """Place the inclusions; the k-th placed one gets label first_id + k.

        Returns (placed, centers): indices of the placed kernels and their (P, ndim) centres.
        """
        flat_labels = labels.reshape(-1)
        placed, centers = [], []
        distance = self._boundary_distance(labels)
        for i, kernel in enumerate(kernels):
            reach = np.abs(kernel.offsets).max(axis=0)
            reach = reach[reach > 0].min() if np.any(reach > 0) else 0
            candidates = np.flatnonzero(distance.ravel() >= reach + 1)
            result = None
            if candidates.size:
                result = self._place_one(flat_labels, kernel.offsets, first_id + len(placed), first_id, candidates)
            if result is None:
                self.stats.not_placed += 1
            else:
                placed.append(i)
                centers.append(result)
                distance = self._boundary_distance(labels)
        return np.asarray(placed, dtype=int), np.asarray(centers, dtype=np.int64).reshape(-1, self.domain.ndim)

    def _place_one(self, flat_labels, offsets, inclusion_id, first_inclusion_id, candidates):
        for _ in range(self.max_attempts):
            self.stats.attempts += 1
            center = self.domain.unravel(candidates[self.rng.integers(0, candidates.size, size=1)])[0]
            host = flat_labels[np.ravel_multi_index(tuple(center), self.domain.shape)]
            if host <= 0 or host >= first_inclusion_id:
                continue
            flat, _ = self.domain.wrap(center + offsets)
            flat = np.unique(flat)
            if np.any(flat_labels[flat] != host):
                continue
            shell = self._shell(flat)
            if np.any(flat_labels[shell] != host):
                continue
            flat_labels[flat] = inclusion_id
            self.stats.placed += 1
            self.stats.hosts.append(int(host))
            return center
        return None

    def _boundary_distance(self, labels: np.ndarray) -> np.ndarray:
        """Distance in voxels from every voxel to the nearest voxel of another label (periodic)."""
        pad = [(8, 8) if p else (0, 0) for p in self.domain.periodic]
        padded = np.pad(labels, pad, mode='wrap')
        boundary = np.zeros(padded.shape, dtype=bool)
        for axis in range(padded.ndim):
            diff = np.diff(padded, axis=axis) != 0
            lo = [slice(None)] * padded.ndim
            hi = [slice(None)] * padded.ndim
            lo[axis] = slice(None, -1)
            hi[axis] = slice(1, None)
            boundary[tuple(lo)] |= diff
            boundary[tuple(hi)] |= diff
        distance = ndimage.distance_transform_edt(~boundary)
        inner = tuple(slice(a, distance.shape[i] - b) for i, (a, b) in enumerate(pad))
        return distance[inner]

    def _shell(self, flat: np.ndarray) -> np.ndarray:
        """Voxels within one step (faces, edges, corners) of `flat`, excluding `flat` itself."""
        offsets = np.stack(np.meshgrid(*[[-1, 0, 1]] * self.domain.ndim, indexing='ij'), axis=-1)
        offsets = offsets.reshape(-1, self.domain.ndim)
        offsets = offsets[np.any(offsets != 0, axis=1)]
        coords = self.domain.unravel(flat)
        shell, _ = self.domain.wrap((coords[:, None, :] + offsets[None]).reshape(-1, self.domain.ndim))
        return np.setdiff1d(np.unique(shell), flat)
