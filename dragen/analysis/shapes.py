"""Shape descriptors per grain, computed on its largest connected body.

solidity      voxel volume / volume of the convex hull of the voxel cubes. 1 for a convex body
              (e.g. a Voronoi cell), smaller for grains with concave, interlocking boundaries.
elongation_2  middle / longest principal axis (from the second moment of the voxel centres)
elongation_3  shortest / longest principal axis
slope         angle of the longest principal axis in the x-y plane, degrees in [0, 180),
              comparable to the input alpha

Grains that cross a periodic boundary are unwrapped first: on every periodic axis the body is
cut at the largest gap in its occupied coordinates.
"""
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from dragen.analysis.metrics import component_labels

_CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])


def _unwrap(coords: np.ndarray, shape: Sequence[int], periodic: Sequence[bool]) -> np.ndarray:
    coords = coords.copy()
    for axis, (n, per) in enumerate(zip(shape, periodic)):
        c = coords[:, axis]
        if not per or c.min() > 0 or c.max() < n - 1:
            continue
        u = np.unique(c)
        gaps = np.diff(np.concatenate([u, [u[0] + n]]))
        start = u[(np.argmax(gaps) + 1) % len(u)]
        coords[:, axis] = (c - start) % n
    return coords


def _shape_of(coords: np.ndarray, spacing: np.ndarray) -> dict:
    centres = (coords + 0.5) * spacing
    cov = np.cov(centres.T) if len(coords) > 1 else np.zeros((3, 3))
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.clip(eigval, 0, None)
    axes = np.sqrt(eigval[::-1] + spacing.mean() ** 2 / 12)   # a voxel has variance s^2/12 per axis
    major = eigvec[:, -1]
    slope = np.degrees(np.arctan2(major[1], major[0])) % 180
    corners = np.unique((coords[:, None, :] + _CORNERS[None]).reshape(-1, 3), axis=0) * spacing
    try:
        hull = ConvexHull(corners).volume
    except Exception:                       # degenerate (all corners coplanar cannot happen for cubes)
        hull = np.nan
    volume = len(coords) * np.prod(spacing)
    return {'volume': volume, 'solidity': volume / hull,
            'elongation_2': axes[1] / axes[0], 'elongation_3': axes[2] / axes[0], 'slope': slope}


def grain_shapes(labels: np.ndarray, spacing: Sequence[float], periodic: Sequence[bool],
                 min_voxels: int = 8) -> pd.DataFrame:
    """One row per grain ID (largest body only); grains with fewer than min_voxels are skipped."""
    if labels.ndim != 3:
        raise ValueError('grain_shapes needs a 3D label field')
    spacing = np.asarray(spacing, dtype=float)
    flat = labels.ravel()
    comp = component_labels(labels, periodic)
    idx = np.flatnonzero(flat > 0)
    # largest component per grain
    pairs, sizes = np.unique(np.stack([flat[idx], comp[idx]]), axis=1, return_counts=True)
    order = np.lexsort((-sizes, pairs[0]))
    first = np.ones(order.size, dtype=bool)
    first[1:] = pairs[0, order[1:]] != pairs[0, order[:-1]]
    main_comp = dict(zip(pairs[0, order[first]].tolist(), pairs[1, order[first]].tolist()))

    keep = idx[np.array([main_comp[g] for g in flat[idx]]) == comp[idx]]
    keep = keep[np.argsort(flat[keep], kind='stable')]
    grain_of = flat[keep]
    bounds = np.flatnonzero(np.diff(grain_of)) + 1
    rows = []
    for chunk in np.split(keep, bounds):
        if len(chunk) < min_voxels:
            continue
        coords = _unwrap(np.stack(np.unravel_index(chunk, labels.shape), axis=1), labels.shape, periodic)
        rows.append({'grain': int(flat[chunk[0]]), **_shape_of(coords, spacing)})
    return pd.DataFrame(rows)
