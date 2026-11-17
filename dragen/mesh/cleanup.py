"""Voxel clean-up before meshing: no grain may touch itself only along an edge or at a point.

Such contacts make the staircase surface of the grain non-manifold (process notes 7.8: most grains
of a generated RVE have them). The test is local. Around every voxel corner sit eight voxels
(the octants of a 2x2x2 block); for every grain in the block, its octants and the remaining
octants must each be face-connected within the block. A grain that holds two diagonal voxels of
a block face while the other two belong to something else violates this (edge contact), and so
does a grain that holds two opposite corners only (point contact). If no block violates it, the
surface of every grain is a closed 2-manifold. Outside the box counts as "something else" on
every axis, matching the box mesh, in which the box faces are regions of their own.

Repair is greedy: for each violating block, try giving one of its voxels to a face neighbour's
grain, and take the change that removes the most violations in the eight blocks around that
voxel. Where every single change adds as many violations as it removes (e.g. a grain touching
itself at a point inside a crowded junction), pairs of changes are tried. A voxel is only taken
from a grain if the grain keeps at least one voxel and the grain's voxels around it stay
face-connected (checked in the 3x3x3 neighbourhood, wrapped on periodic axes), so no grain falls
apart.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

_OCTANTS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
_BIT = 1 << np.arange(8)
_OCTANT_NEIGHBOURS = [[b for b in range(8) if bin(a ^ b).count('1') == 1] for a in range(8)]
_FACE_STEPS = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]])
OUTSIDE = -1


def _face_connected(mask: int) -> bool:
    members = [a for a in range(8) if mask >> a & 1]
    seen, stack = {members[0]}, [members[0]]
    while stack:
        a = stack.pop()
        for b in _OCTANT_NEIGHBOURS[a]:
            if mask >> b & 1 and b not in seen:
                seen.add(b)
                stack.append(b)
    return len(seen) == len(members)


def _critical_table() -> np.ndarray:
    table = np.zeros(256, dtype=bool)
    for mask in range(1, 255):
        table[mask] = not (_face_connected(mask) and _face_connected(255 - mask))
    return table


CRITICAL = _critical_table()
"""CRITICAL[mask] is True when the octants in mask (bit 4i + 2j + k) or the others are not
face-connected within the block."""


def _blocks(padded: np.ndarray, lo: Sequence[int], hi: Sequence[int]) -> np.ndarray:
    """Octant labels of the blocks whose corners lie in [lo, hi) (corner index = padded index of
    octant 0 + 1); returns (..., 8)."""
    return np.stack([padded[lo[0] + i:hi[0] + i, lo[1] + j:hi[1] + j, lo[2] + k:hi[2] + k]
                     for i, j, k in _OCTANTS], axis=-1)


def _critical_pairs(octants: np.ndarray) -> np.ndarray:
    """Boolean (..., 8): octant k's grain violates the criterion in its block."""
    masks = np.zeros(octants.shape, dtype=np.int64)
    for k in range(8):
        masks += (octants == octants[..., k:k + 1]) * _BIT[k]
    return CRITICAL[masks] & (octants > 0)


def critical_blocks(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Corner index (M, 3) and grain (M,) of every violating (block, grain) pair."""
    padded = np.pad(np.asarray(labels), 1, constant_values=OUTSIDE)
    n = np.array(labels.shape) + 1
    octants = _blocks(padded, (0, 0, 0), n)
    crit = _critical_pairs(octants)
    corner, k = np.nonzero(crit.reshape(-1, 8))
    pairs = np.unique(np.column_stack([corner, octants.reshape(-1, 8)[corner, k]]), axis=0)
    return np.column_stack(np.unravel_index(pairs[:, 0], tuple(n))), pairs[:, 1]


@dataclass
class CleanupStats:
    critical_before: int = 0
    critical_after: int = 0
    changed_voxels: int = 0
    passes: int = 0
    voxels_gained: Dict[int, int] = field(default_factory=dict)
    """Net voxel change per grain (only grains that changed)."""


class ManifoldCleaner:
    """Removes edge and point self-contacts of grains by reassigning single voxels."""

    def __init__(self, periodic: Sequence[bool], max_passes: int = 20):
        self.periodic = tuple(bool(p) for p in periodic)
        self.max_passes = max_passes

    def run(self, labels: np.ndarray) -> Tuple[np.ndarray, CleanupStats]:
        original = np.asarray(labels)
        stats = CleanupStats()
        padded = np.pad(original, 1, constant_values=OUTSIDE)
        counts = np.bincount(original.ravel())
        corners, _ = critical_blocks(original)
        stats.critical_before = len(corners)
        for _ in range(self.max_passes):
            if not len(corners):
                break
            stats.passes += 1
            changed = False
            for corner in np.unique(corners, axis=0):
                if self._local_critical(padded, corner, corner + 1) == 0:
                    continue
                moves = self._best_move(padded, counts, corner) or self._best_pair(padded, counts, corner)
                for move in moves or ():
                    self._apply(padded, counts, move)
                    changed = True
            corners, _ = critical_blocks(padded[1:-1, 1:-1, 1:-1])
            if not changed:
                break
        out = padded[1:-1, 1:-1, 1:-1].copy()
        stats.critical_after = len(corners)
        diff = out != original
        stats.changed_voxels = int(np.count_nonzero(diff))
        gained = np.bincount(out[diff], minlength=len(counts)) - np.bincount(original[diff], minlength=len(counts))
        stats.voxels_gained = {int(g): int(v) for g, v in enumerate(gained) if v}
        if stats.critical_after:
            logging.getLogger(__name__).warning('%d edge or point self-contacts could not be removed',
                                                stats.critical_after)
        return out, stats

    # ------------------------------------------------------------------------------------- #

    @staticmethod
    def _local_critical(padded, lo, hi) -> int:
        """Violating (block, grain) pairs among the blocks with corners in [lo, hi)."""
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.array(padded.shape) - 1)
        octants = _blocks(padded, lo, hi)
        crit = _critical_pairs(octants).reshape(-1, 8)
        oct8 = octants.reshape(-1, 8)
        return int(sum(len(set(oct8[b][crit[b]])) for b in np.flatnonzero(crit.any(axis=1))))

    @staticmethod
    def _critical_corners(padded, lo, hi) -> np.ndarray:
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.array(padded.shape) - 1)
        crit = _critical_pairs(_blocks(padded, lo, hi)).any(axis=-1)
        return np.argwhere(crit) + lo

    @staticmethod
    def _apply(padded, counts, move):
        (x, y, z), donor, receiver = move
        padded[x + 1, y + 1, z + 1] = receiver
        counts[donor] -= 1
        counts[receiver] += 1

    def _moves(self, padded, counts, corner):
        """Every allowed reassignment of a voxel of the block, with the number of violations it
        removes in the eight blocks around the voxel (negative if it adds some)."""
        shape = np.array(padded.shape) - 2
        for offset in _OCTANTS:
            voxel = corner - 1 + offset
            if np.any(voxel < 0) or np.any(voxel >= shape):
                continue
            x, y, z = voxel
            donor = padded[x + 1, y + 1, z + 1]
            if counts[donor] <= 1 or not self._can_give(padded, voxel, donor):
                continue
            receivers = {padded[tuple(voxel + 1 + s)] for s in _FACE_STEPS}
            receivers = sorted(r for r in receivers if r > 0 and r != donor)
            if not receivers:
                continue
            before = self._local_critical(padded, voxel, voxel + 2)
            for receiver in receivers:
                padded[x + 1, y + 1, z + 1] = receiver
                after = self._local_critical(padded, voxel, voxel + 2)
                padded[x + 1, y + 1, z + 1] = donor
                yield voxel, donor, receiver, before - after

    def _best_move(self, padded, counts, corner):
        best = None
        for voxel, donor, receiver, gain in self._moves(padded, counts, corner):
            if gain <= 0:
                continue
            # most violations removed first, then take from the larger grain
            key = (-gain, -counts[donor], counts[receiver], int(receiver), tuple(voxel))
            if best is None or key < best[0]:
                best = (key, [(voxel, donor, receiver)])
        return None if best is None else best[1]

    def _best_pair(self, padded, counts, corner):
        """Two reassignments that together remove violations where no single one does."""
        best = None
        for voxel, donor, receiver, gain in list(self._moves(padded, counts, corner)):
            first = (voxel, donor, receiver)
            self._apply(padded, counts, first)
            for second_corner in self._critical_corners(padded, voxel, voxel + 2):
                for v2, d2, r2, g2 in list(self._moves(padded, counts, second_corner)):
                    total = gain + g2
                    if total <= 0:
                        continue
                    key = (-total, -counts[donor] - counts[d2], tuple(voxel), tuple(v2), int(r2))
                    if best is None or key < best[0]:
                        best = (key, [first, (v2, d2, r2)])
            self._apply(padded, counts, (voxel, receiver, donor))
        return None if best is None else best[1]

    def _can_give(self, padded, voxel, donor) -> bool:
        """Donor stays face-connected around the voxel when the voxel leaves it."""
        shape = np.array(padded.shape) - 2
        offsets = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
        coords = voxel + offsets
        inside = np.ones(len(coords), dtype=bool)
        for a in range(3):
            if self.periodic[a]:
                coords[:, a] %= shape[a]
            else:
                inside &= (coords[:, a] >= 0) & (coords[:, a] < shape[a])
        member = np.zeros(27, dtype=bool)
        member[inside] = padded[tuple((coords[inside] + 1).T)] == donor
        member[13] = False                                   # the voxel itself leaves
        face = [4, 10, 12, 14, 16, 22]                       # face neighbours of the centre
        starts = [f for f in face if member[f]]
        if not starts:
            return False
        seen, stack = {starts[0]}, [starts[0]]
        while stack:
            a = stack.pop()
            for b in _cube_neighbours(a):
                if member[b] and b not in seen:
                    seen.add(b)
                    stack.append(b)
        return all(f in seen for f in starts)


def _cube_neighbours(a: int) -> List[int]:
    i, j, k = a // 9, a // 3 % 3, a % 3
    out = []
    for d in range(3):
        for step in (-1, 1):
            c = [i, j, k]
            c[d] += step
            if 0 <= c[d] <= 2:
                out.append(9 * c[0] + 3 * c[1] + c[2])
    return out
