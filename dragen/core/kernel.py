"""Ellipsoid kernels: the voxels a grain covers, as offsets around its centre voxel.

A kernel is computed once per grain and covers everything the grain can reach during
tessellation. Every offset carries the epoch at which it enters the growing ellipsoid, so the
RSA uses the epoch-0 part (the shrunk seed) and the tessellation only advances a pointer through
offsets sorted by entry epoch instead of re-evaluating the ellipsoid on the whole grid.

Growth law (same as the legacy DiscreteTesselation3D.grow): per epoch every semi-axis grows by
    a_k = a_0 * (1 + k * g_a),   g_a = bin_size / a_max
where a_0 is the shrunk radius of this grain and a_max the largest shrunk radius of all grains;
b and c likewise with their own maxima. The axes therefore scale at different rates, but the
ellipsoid at epoch k always contains the one at epoch k - 1, so the entry epoch of a voxel is the
smallest k with f_k(offset) <= 1 and can be found by bisection.
"""
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class EllipsoidKernel:
    offsets: np.ndarray        # (M, ndim) int16, sorted by entry epoch, centre voxel first
    entry_epoch: np.ndarray    # (M,) int32, non-decreasing
    seed_size: int             # number of offsets with entry epoch 0
    target_voxels: int         # voxels of the unshrunk ellipsoid (discrete target volume)

    @property
    def size(self) -> int:
        return len(self.entry_epoch)

    @property
    def seed(self) -> np.ndarray:
        return self.offsets[:self.seed_size]


def _rotated_quadratic(offsets_phys: np.ndarray, radii: np.ndarray, alpha_deg: float) -> np.ndarray:
    """Normalised ellipsoid value of physical offsets, rotated by alpha about the last axis (z in 3D).

    radii: (ndim,) for one ellipsoid or (M, ndim) for one ellipsoid per offset.
    """
    t = np.deg2rad(alpha_deg)
    x, y = offsets_phys[:, 0], offsets_phys[:, 1]
    u = x * np.cos(t) + y * np.sin(t)
    v = -x * np.sin(t) + y * np.cos(t)
    f = (u / radii[..., 0]) ** 2 + (v / radii[..., 1]) ** 2
    if offsets_phys.shape[1] == 3:
        f = f + (offsets_phys[:, 2] / radii[..., 2]) ** 2
    return f


def build_kernel(radii: Sequence[float], alpha: float, spacing: Sequence[float], shrink: float,
                 growth_rate: Sequence[float], max_epoch: int) -> EllipsoidKernel:
    """Kernel of one grain.

    radii: unshrunk semi-axes (a, b[, c]) in micrometres.
    alpha: rotation in degrees about the last axis.
    spacing: voxel edge lengths.
    shrink: linear shrink factor of the seed (radii * shrink at epoch 0).
    growth_rate: (g_a, g_b[, g_c]) relative growth per epoch, see module docstring.
    max_epoch: last epoch the kernel has to cover.
    """
    radii = np.asarray(radii, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    rate = np.asarray(growth_rate, dtype=float)
    ndim = len(spacing)
    r0 = radii * shrink

    # Bounding box of the largest ellipsoid; rotation mixes the first two axes, so their reach is
    # bounded by the larger of the two radii.
    r_max = r0 * (1 + max_epoch * rate)
    reach = r_max.copy()
    reach[:2] = r_max[:2].max()
    half = np.ceil(reach / spacing).astype(int)
    axes = [np.arange(-h, h + 1, dtype=np.int16) for h in half]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, ndim)
    phys = grid * spacing

    inside = _rotated_quadratic(phys, r_max, alpha) <= 1
    inside[np.all(grid == 0, axis=1)] = True        # the centre voxel always belongs to the grain
    grid, phys = grid[inside], phys[inside]

    # smallest k in [0, max_epoch] with f_k <= 1, by bisection (f_k decreases in k)
    lo = np.zeros(len(grid), dtype=np.int32)
    hi = np.full(len(grid), max_epoch, dtype=np.int32)
    while np.any(lo < hi):
        mid = (lo + hi) // 2
        r_mid = r0 * (1 + mid[:, None] * rate)
        ok = _rotated_quadratic(phys, r_mid, alpha) <= 1
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid + 1)
    epoch = lo
    centre = np.all(grid == 0, axis=1)
    epoch[centre] = 0

    # A thin, rotated seed can consist of voxels that only touch along edges. The seed is written
    # as a whole by the RSA, so keep only its face-connected part around the centre; the rest
    # enters one epoch later and is reached through face neighbours during growth.
    seed = np.flatnonzero(epoch == 0)
    local = grid[seed] - grid[seed].min(axis=0)
    mask = np.zeros(local.max(axis=0) + 1, dtype=bool)
    mask[tuple(local.T)] = True
    components, _ = ndimage.label(mask)
    seed_component = components[tuple(local.T)]
    detached = seed[seed_component != seed_component[centre[seed]][0]]
    epoch[detached] = min(1, max_epoch)
    if max_epoch == 0:
        keep = np.ones(len(grid), dtype=bool)
        keep[detached] = False
        grid, phys, epoch = grid[keep], phys[keep], epoch[keep]

    order = np.lexsort((_rotated_quadratic(phys, r0, alpha), epoch))
    grid, epoch = grid[order], epoch[order]
    target = int(np.count_nonzero(_rotated_quadratic(grid * spacing, radii, alpha) <= 1))
    return EllipsoidKernel(offsets=grid.astype(np.int16), entry_epoch=epoch,
                           seed_size=int(np.searchsorted(epoch, 0, side='right')),
                           target_voxels=max(target, 1))

