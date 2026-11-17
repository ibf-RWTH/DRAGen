"""Martensite substructure on the voxel field of an RVEResult: parent grain -> packets -> blocks
-> orientations, and an RVEResult in which every block is a region of its own.

Stages (as in the legacy pyvista pipeline, now on voxels, numpy/scipy only):
    packets   every grain of a substructured phase is split by k-means on its voxel centres into
              min(4, voxels // min_packet_voxels + 1) packets; coordinates are unwrapped across
              periodic boundaries first, so a grain cut by the box is clustered as one body
    merge     packets below min_voxels_per_packet join the neighbouring packet of the same grain
              with the most contacts (26-neighbourhood), repeated until none is left or no merge
              is possible
    blocks    every packet is cut by parallel planes normal to a random direction: constant
              thickness, or thicknesses drawn from a measured distribution
    merge     blocks below min_voxels_per_block join the neighbouring block of the same packet
              with the most contacts; small blocks are no targets (one pass)
    orient    KS variants or measured transformations (dragen.substructure.orientation)
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

from dragen.substructure import orientation as ori
from dragen.substructure.config import THICKNESS_COLUMN, SubstructureConfig

log = logging.getLogger(__name__)

MAX_PACKETS_PER_GRAIN = 4
_NEIGHBOURS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                        if (i, j, k) != (0, 0, 0)])


@dataclass
class Substructure:
    """packet_ids, block_ids: per voxel, 1-based, -1 outside substructured grains.
    Per packet (index p - 1): packet_grain, packet_class (KS habit plane class, -1 otherwise).
    Per block (index b - 1): block_packet, block_grain, block_euler (Bunge, degrees),
    block_variant (KS variant 0..23 or measured relation index), block_thickness_um."""

    packet_ids: np.ndarray
    block_ids: np.ndarray
    packet_grain: np.ndarray
    packet_class: np.ndarray
    block_packet: np.ndarray
    block_grain: np.ndarray
    block_euler: np.ndarray
    block_variant: np.ndarray
    block_thickness_um: np.ndarray
    config: SubstructureConfig
    stats: Dict = field(default_factory=dict)


def generate_substructure(result, config: SubstructureConfig) -> Substructure:
    """Packets, blocks and block orientations for the grains of config.phases."""
    labels = np.asarray(result.grain_ids)
    if labels.ndim != 3:
        raise ValueError('the substructure needs a 3D RVE')
    rng = np.random.default_rng(config.seed)
    spacing_um = np.asarray(result.voxel_size_m) * 1e6
    coords = _unwrapped(labels, result.periodic)                     # (N, 3) voxel units
    centres_um = (coords + 0.5) * spacing_um
    flat = labels.ravel()
    target = np.isin(result.phase_ids, config.phase_ids)
    grains = np.flatnonzero(target) + 1
    voxels_of = _voxels_by_label(flat, len(result.phase_ids))
    stats = {'substructured_grains': int(len(grains))}

    # packets
    packet = np.full(flat.size, -1, dtype=np.int64)
    n = 0
    for g in grains:
        idx = voxels_of[g]
        k = min(MAX_PACKETS_PER_GRAIN, max(1, len(idx) // config.min_packet_voxels + 1))
        local = _kmeans(centres_um[idx], k, rng) if k > 1 else np.zeros(len(idx), int)
        _, local = np.unique(local, return_inverse=True)
        packet[idx] = local + n + 1
        n += int(local.max()) + 1
    stats['packets_raw'] = n
    packet, merged = _merge_small(packet.reshape(labels.shape), labels, result.periodic,
                                  config.min_voxels_per_packet, config.max_merge_iterations, small_targets=True)
    packet = _renumber(packet.ravel())
    stats['packets_merged'] = merged

    # blocks
    pool = _thickness_pool(config) if config.mode == 'file' else None
    block = np.full(flat.size, -1, dtype=np.int64)
    thickness_of = []
    n = 0
    voxels_of_packet = _voxels_by_label(np.maximum(packet, 0), int(packet.max(initial=0)))
    for p in range(1, int(packet.max(initial=0)) + 1):
        idx = voxels_of_packet[p]
        if len(idx) > config.min_block_voxels:
            normal = rng.normal(size=3)
            normal /= np.linalg.norm(normal)
            d = centres_um[idx] @ normal
            if pool is None:
                local = np.floor((d - d.min()) / config.block_thickness).astype(int)
                sampled = np.full(int(local.max()) + 1, config.block_thickness)
            else:
                local, sampled = _variable_slices(d, pool, rng)
        else:
            local = np.zeros(len(idx), int)
            sampled = np.array([config.block_thickness if pool is None else float(rng.choice(pool))])
        used, local = np.unique(local, return_inverse=True)
        block[idx] = local + n + 1
        thickness_of.extend(sampled[used].tolist())
        n += len(used)
    stats['blocks_raw'] = n
    thickness_of = np.asarray(thickness_of, dtype=float)
    block_shaped, merged = _merge_small(block.reshape(labels.shape), packet.reshape(labels.shape), result.periodic,
                                        config.min_voxels_per_block, 1, small_targets=False)
    stats['blocks_merged'] = merged
    old = block_shaped.ravel()
    block = _renumber(old)
    n_blocks = int(block.max(initial=0))
    first = np.zeros(n_blocks + 1, dtype=np.int64)          # one voxel per block for its data
    first[block[block > 0]] = np.flatnonzero(block > 0)
    first = first[1:]
    block_packet = packet[first]
    block_grain = flat[first]
    block_thickness = thickness_of[old[first] - 1] if n_blocks else np.zeros(0)
    counts = np.bincount(block[block > 0], minlength=n_blocks + 1)[1:]
    stats['blocks_below_min'] = int(np.count_nonzero(counts < config.min_voxels_per_block))

    # orientations
    n_packets = int(packet.max(initial=0))
    packet_grain = np.zeros(n_packets, dtype=np.int64)
    if n_packets:
        packet_grain[packet[packet > 0] - 1] = flat[packet > 0]
    euler = np.asarray(result.euler_bunge_deg, dtype=float)
    if config.orientation_mode == 'KS':
        block_euler, variant, packet_class = ori.ks_orientations(euler, block_grain - 1, block_packet - 1, n_packets, rng)
    else:
        library = ori.experimental_library(config.parent_orientation_file, config.child_orientation_file)
        block_euler, _, variant = ori.experimental_orientations(euler, block_grain - 1, library, rng)
        packet_class = np.full(n_packets, -1)
    stats.update(packets=n_packets, blocks=n_blocks)
    log.info('substructure: %d grains, %d packets, %d blocks', len(grains), n_packets, n_blocks)
    shape = labels.shape
    return Substructure(packet_ids=packet.reshape(shape).astype(np.int32), block_ids=block.reshape(shape).astype(np.int32),
                        packet_grain=packet_grain.astype(np.int32), packet_class=np.asarray(packet_class, dtype=np.int32),
                        block_packet=block_packet.astype(np.int32), block_grain=block_grain.astype(np.int32),
                        block_euler=block_euler, block_variant=np.asarray(variant, dtype=np.int32),
                        block_thickness_um=block_thickness, config=config, stats=stats)


def apply_substructure(result, sub: Substructure):
    """RVEResult with one region per block and per grain without substructure. Region IDs follow
    the parent grains (grain 1 or its blocks first). parent_grain_ids and packet_ids map regions
    to their grain and packet (-1: no substructure). A tetrahedral mesh keeps its elements; each
    element gets the region of the voxel at its centroid (a neighbouring voxel of the same grain
    if that voxel belongs to another grain)."""
    from dragen.api import RVEResult, TetMesh, euler_to_quaternion

    G = result.n_grains
    n_blocks = len(sub.block_grain)
    # region order: per grain, the grain itself or its blocks
    has_blocks = np.zeros(G + 1, dtype=bool)
    has_blocks[sub.block_grain] = True
    key_grain = np.concatenate([np.flatnonzero(~has_blocks[1:]) + 1, sub.block_grain])
    key_block = np.concatenate([np.zeros(G - int(has_blocks[1:].sum()), int), np.arange(1, n_blocks + 1)])
    order = np.lexsort((key_block, key_grain))
    key_grain, key_block = key_grain[order], key_block[order]
    region_of_grain = np.zeros(G + 1, dtype=np.int64)
    region_of_block = np.zeros(n_blocks + 1, dtype=np.int64)
    regions = np.arange(1, len(order) + 1)
    plain = key_block == 0
    region_of_grain[key_grain[plain]] = regions[plain]
    region_of_block[key_block[~plain]] = regions[~plain]

    labels = np.asarray(result.grain_ids)
    blocks = sub.block_ids
    voxel_region = np.where(blocks > 0, region_of_block[np.maximum(blocks, 0)], region_of_grain[labels]).astype(np.int32)
    euler = np.where(plain[:, None], result.euler_bunge_deg[key_grain - 1],
                     sub.block_euler[np.maximum(key_block, 1) - 1] if n_blocks else 0.0)
    packets = np.where(plain, -1, sub.block_packet[np.maximum(key_block, 1) - 1] if n_blocks else -1)

    tet = None
    if result.tet_mesh is not None:
        tet = TetMesh(points=result.tet_mesh.points, cells=result.tet_mesh.cells,
                      grain_ids=_element_regions(result, voxel_region, key_grain, region_of_grain),
                      periodic_pairs=result.tet_mesh.periodic_pairs)
    meta = dict(result.metadata, substructure={'config': sub.config.to_dict(), **sub.stats})
    meta['n_grains'] = int(len(order))
    return RVEResult(grain_ids=voxel_region, orientations=euler_to_quaternion(euler), euler_bunge_deg=euler,
                     phase_ids=result.phase_ids[key_grain - 1], phase_names=dict(result.phase_names),
                     box_size_m=result.box_size_m, periodic=result.periodic,
                     band_ids=result.band_ids[key_grain - 1], tet_mesh=tet, metadata=meta,
                     parent_grain_ids=key_grain.astype(np.int32), packet_ids=np.asarray(packets, dtype=np.int32))


# --------------------------------------------------------------------------- helpers

def _voxels_by_label(flat: np.ndarray, n: int):
    """List index l -> flat voxel indices with label l (0 ... n)."""
    order = np.argsort(flat, kind='stable')
    bounds = np.searchsorted(flat[order], np.arange(n + 2))
    return [order[bounds[l]:bounds[l + 1]] for l in range(n + 1)]


def _unwrapped(labels: np.ndarray, periodic) -> np.ndarray:
    """Voxel index coordinates (N, 3), shifted by box lengths so that every grain cut by a
    periodic boundary is one contiguous body: pieces of a grain (face-connected without
    periodicity) are joined along a spanning tree of their contacts across the box faces."""
    from dragen.analysis.metrics import component_labels
    shape = np.array(labels.shape)
    piece = component_labels(labels, (False,) * 3)
    _, piece = np.unique(piece, return_inverse=True)
    n_pieces = int(piece.max()) + 1
    piece = piece.reshape(labels.shape)
    rows, cols, shifts = [], [], []
    for axis in range(3):
        if not periodic[axis]:
            continue
        lo = np.take(labels, 0, axis=axis).ravel()
        hi = np.take(labels, -1, axis=axis).ravel()
        same = lo == hi
        p_lo = np.take(piece, 0, axis=axis).ravel()[same]
        p_hi = np.take(piece, -1, axis=axis).ravel()[same]
        pairs = np.unique(np.column_stack([p_hi, p_lo]), axis=0)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        shift = np.zeros(3, dtype=np.int64)
        shift[axis] = shape[axis]
        rows += [pairs[:, 0], pairs[:, 1]]
        cols += [pairs[:, 1], pairs[:, 0]]
        shifts += [np.tile(shift, (len(pairs), 1)), np.tile(-shift, (len(pairs), 1))]
    offset = np.zeros((n_pieces, 3), dtype=np.int64)
    if rows:
        rows, cols, shifts = np.concatenate(rows), np.concatenate(cols), np.concatenate(shifts)
        edge = {(int(a), int(b)): s for a, b, s in zip(rows, cols, shifts)}
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_pieces, n_pieces)).tocsr()
        n_comp, comp = connected_components(graph, directed=False)
        sizes = np.bincount(piece.ravel(), minlength=n_pieces)
        done = np.zeros(n_pieces, dtype=bool)
        for root in np.argsort(-sizes, kind='stable'):        # largest piece of each body stays
            if done[root]:
                continue
            nodes, pred = breadth_first_order(graph, root, directed=False)
            for v in nodes[1:]:
                offset[v] = offset[pred[v]] + edge[(int(pred[v]), int(v))]
            done[nodes] = True
    ijk = np.indices(labels.shape).reshape(3, -1).T
    return (ijk + offset[piece.ravel()]).astype(float)


def _kmeans(points: np.ndarray, k: int, rng, n_init: int = 10) -> np.ndarray:
    """Best of n_init k-means++ runs (smallest inertia), like sklearn's KMeans(n_init=10)."""
    best, best_inertia = None, np.inf
    for _ in range(n_init):
        centres, label = kmeans2(points, k, minit='++', seed=rng)
        inertia = float(np.sum((points - centres[label]) ** 2))
        if inertia < best_inertia:
            best, best_inertia = label, inertia
    return best


def _neighbour_pairs(field: np.ndarray, constraint: np.ndarray, periodic, candidates: np.ndarray):
    """(id, neighbour id) for every voxel of an id in `candidates` and each of its 26 neighbours
    with another positive id and the same constraint value (grain or packet)."""
    out = []
    mine = np.isin(field, candidates)
    if not mine.any():
        return np.zeros((0, 2), dtype=np.int64)
    idx = np.argwhere(mine)
    shape = np.array(field.shape)
    for d in _NEIGHBOURS:
        nb = idx + d
        ok = np.ones(len(idx), dtype=bool)
        for a in range(3):
            if periodic[a]:
                nb[:, a] %= shape[a]
            else:
                ok &= (nb[:, a] >= 0) & (nb[:, a] < shape[a])
        src, dst = idx[ok], nb[ok]
        a_id = field[tuple(src.T)]
        b_id = field[tuple(dst.T)]
        keep = (b_id > 0) & (b_id != a_id) & (constraint[tuple(src.T)] == constraint[tuple(dst.T)])
        out.append(np.column_stack([a_id[keep], b_id[keep]]))
    return np.concatenate(out)


def _merge_small(field: np.ndarray, constraint: np.ndarray, periodic, min_voxels: int,
                 iterations: int, small_targets: bool):
    """Merge ids with fewer than min_voxels voxels into their most frequent neighbour id with the
    same constraint value. Returns the field and the number of merges."""
    field = field.copy()
    merged = 0
    for _ in range(iterations):
        ids, counts = np.unique(field[field > 0], return_counts=True)
        small = ids[counts < min_voxels]
        if not len(small):
            break
        pairs = _neighbour_pairs(field, constraint, periodic, small)
        if not small_targets:
            pairs = pairs[~np.isin(pairs[:, 1], small)]
        if not len(pairs):
            break
        uniq, votes = np.unique(pairs, axis=0, return_counts=True)
        # per small id the neighbour with most votes (ties: smallest neighbour id)
        order = np.lexsort((uniq[:, 1], -votes, uniq[:, 0]))
        uniq = uniq[order]
        first = np.r_[True, uniq[1:, 0] != uniq[:-1, 0]]
        mapping = np.arange(int(field.max()) + 1)
        mapping[uniq[first, 0]] = uniq[first, 1]
        positive = field > 0
        field[positive] = mapping[field[positive]]
        merged += int(first.sum())
    return field, merged


def _renumber(ids: np.ndarray) -> np.ndarray:
    """Positive ids to 1 ... n in ascending order, -1 stays."""
    out = np.full(ids.shape, -1, dtype=np.int64)
    positive = ids > 0
    _, inv = np.unique(ids[positive], return_inverse=True)
    out[positive] = inv + 1
    return out


def _thickness_pool(config: SubstructureConfig) -> np.ndarray:
    import pandas as pd
    raw = pd.read_csv(config.block_file)[THICKNESS_COLUMN].dropna().to_numpy(dtype=float)
    raw = raw[raw >= config.min_physical_thickness]
    if not len(raw):
        raise ValueError(f'{config.block_file}: no block thickness >= {config.min_physical_thickness}')
    lo, hi = np.percentile(raw, [config.lower_percentile, config.upper_percentile])
    pool = raw[(raw >= lo) & (raw <= hi)]
    if not len(pool):
        raise ValueError(f'{config.block_file}: no block thickness left after percentile clipping')
    return pool


def _variable_slices(d: np.ndarray, pool: np.ndarray, rng):
    """Slices along d with thicknesses drawn from pool (legacy _variable_blocks)."""
    start, end_all = d.min(), d.max()
    local = np.full(len(d), -1, dtype=int)
    sampled = []
    k = 0
    while start <= end_all:
        t = float(rng.choice(pool))
        end = start + t
        local[((d >= start) if k == 0 else (d > start)) & (d <= end)] = k
        sampled.append(t)
        start, k = end, k + 1
    local[local < 0] = k - 1
    return local, np.asarray(sampled)


def _element_regions(result, voxel_region, region_grain, region_of_grain) -> np.ndarray:
    tet = result.tet_mesh
    shape = np.array(voxel_region.shape)
    size = np.asarray(result.box_size_m)
    centroid = tet.points[tet.cells[:, :4]].mean(axis=1)
    ijk = np.clip(np.floor(centroid / size * shape).astype(int), 0, shape - 1)
    grain = tet.grain_ids
    region = voxel_region[tuple(ijk.T)].astype(np.int64)
    bad = region_grain[region - 1] != grain
    for d in _NEIGHBOURS:
        if not bad.any():
            break
        nb = (ijk[bad] + d) % shape
        cand = voxel_region[tuple(nb.T)]
        ok = region_grain[cand - 1] == grain[bad]
        idx = np.flatnonzero(bad)[ok]
        region[idx] = cand[ok]
        bad[idx] = False
    if bad.any():                     # far from any voxel of its grain: first region of the grain
        first = np.zeros(len(region_of_grain), dtype=np.int64)
        for r in range(len(region_grain), 0, -1):
            first[region_grain[r - 1]] = r
        region[bad] = first[grain[bad]]
        log.warning('%d elements got the first region of their grain (no voxel of the grain nearby)', int(bad.sum()))
    return region.astype(np.int32)
