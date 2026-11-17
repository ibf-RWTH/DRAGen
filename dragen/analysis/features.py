"""Metrics for bands and inclusions."""
from typing import List

import numpy as np

from dragen.analysis.metrics import component_labels
from dragen.config.phases import Phase
from dragen.core.domain import VoxelDomain
from dragen.core.grains import GrainTable


def band_report(labels: np.ndarray, grains: GrainTable, bands: List, domain: VoxelDomain,
                merged_grains: List[int] = None) -> List[dict]:
    """Per band: how much of its slab the band grain fills, how much of it lies outside, and in
    how many face-connected pieces it lies (across periodic boundaries).

    filling: band voxels inside the slab / slab voxels (the band filling of the paper).
    outside_fraction: band voxels outside the slab / all voxels of the band.
    pieces: connected pieces of the band grain (ID k + 1); 1 is required (one grain, one body).
    merged_grains: RveResult.band_grains (grains merged into the band and band grains left
    detached from it as grains of their own).
    """
    flat = labels.ravel()
    components = component_labels(labels, domain.periodic) if bands else None
    band_of_label = np.concatenate([[-1], grains.band])
    owner = np.where(flat > 0, band_of_label[np.clip(flat, 0, len(grains))], -1)
    out = []
    for k, band in enumerate(bands):
        slab = band.mask(domain)
        mine = owner == k
        inside = np.count_nonzero(mine & slab)
        total = np.count_nonzero(mine)
        # the band grain itself (ID k + 1); detached band grains are grains of their own
        pieces = len(np.unique(components[flat == k + 1])) if total else 0
        out.append({'band': k, 'center': band.center, 'thickness': band.thickness,
                    'slab_voxels': int(slab.sum()), 'filling': inside / slab.sum(),
                    'outside_fraction': (total - inside) / total if total else 0.0,
                    'grains': int(np.count_nonzero(grains.band == k)),
                    'merged_grains': None if merged_grains is None else int(merged_grains[k]['merged']),
                    'detached_grains': None if merged_grains is None else int(merged_grains[k]['detached']),
                    'pieces': int(pieces)})
    return out


def inclusion_report(labels: np.ndarray, grains: GrainTable, domain: VoxelDomain, requested: int = None) -> dict:
    """Placed inclusions, their volume fraction, and whether any touches a grain boundary.

    requested: number of sampled inclusions (placed + not placed); defaults to the placed ones.
    expected_placed_fraction: continuous ellipsoid volume of the placed inclusions / box volume.

    touching_boundary counts inclusions whose face neighbours belong to more than one other label
    (they would sit on a grain boundary) or to another inclusion.
    """
    flat = labels.ravel()
    ids = np.flatnonzero(grains.phase == Phase.INCLUSIONS) + 1
    if ids.size == 0:
        return {'inclusions': int(requested or 0), 'placed': 0, 'volume_fraction': 0.0,
                'expected_placed_fraction': 0.0, 'touching_boundary': 0}
    counts = np.bincount(flat[flat > 0], minlength=ids.max() + 1)[ids]
    placed = ids[counts > 0]
    touching = 0
    for i in placed:
        vox = np.flatnonzero(flat == i)
        nb = domain.face_neighbors(vox).ravel()
        nb = flat[nb[nb >= 0]]
        others = np.unique(nb[nb != i])
        if len(others) != 1 or np.isin(others, ids).any():
            touching += 1
    expected = grains.volume[placed - 1].sum() / (flat.size * domain.voxel_volume)
    return {'inclusions': int(requested if requested is not None else ids.size), 'placed': int(placed.size),
            'volume_fraction': float(counts.sum() / flat.size), 'expected_placed_fraction': float(expected),
            'touching_boundary': touching}
