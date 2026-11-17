"""Grain boundary surfaces for the 3D view, as compact binary buffers.

Staircase source: the voxel faces between different labels plus the box faces, as in
dragen.mesh.surface.staircase_mesh, but merged: per axis and plane, faces with the same grain pair
and orientation are first joined into runs along one in-plane axis, then runs with the same extent
in consecutive rows into rectangles (a vectorised row-run variant of greedy meshing). The merged
surface covers exactly the voxel faces (same area); rectangles may meet in T-junctions, which is
fine for display. A downsample step takes every step-th voxel per axis first.

Smoothed source: the triangles of the grain boundary model a meshing job stored in mesh.npz
(surf_points, surf_triangles, surf_regions).

Every triangle carries (front, back): the grain seen when looking at its front face (the grain
behind the surface, i.e. the normal points out of it) and the region on the other side, -1 for the
box. Payload layout (little endian):

    b'DRGS', uint32 header length, header JSON (padded to 4 bytes),
    float32 points (V, 3), uint32 triangles (T, 3), int32 regions (T, 2), uint8 edge mask (T,)

edge mask bit k hides the edge opposite vertex k (the diagonal of a rectangle).
"""
import json
import struct
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

BOX = -1


@dataclass
class Surface:
    points: np.ndarray
    """(V, 3) float32 coordinates in micrometres."""
    triangles: np.ndarray
    """(T, 3) uint32."""
    regions: np.ndarray
    """(T, 2) int32 (front grain, back region; -1 = box)."""
    edge_mask: np.ndarray
    """(T,) uint8."""
    meta: dict = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        header = dict(self.meta, n_points=int(len(self.points)), n_triangles=int(len(self.triangles)))
        text = json.dumps(header).encode('utf-8')
        text += b' ' * (-len(text) % 4)
        mask = np.ascontiguousarray(self.edge_mask, dtype=np.uint8).tobytes()
        mask += b'\0' * (-len(mask) % 4)
        return b''.join([b'DRGS', struct.pack('<I', len(text)), text,
                         np.ascontiguousarray(self.points, dtype='<f4').tobytes(),
                         np.ascontiguousarray(self.triangles, dtype='<u4').tobytes(),
                         np.ascontiguousarray(self.regions, dtype='<i4').tobytes(), mask])


def face_rectangles(labels: np.ndarray):
    """Merged voxel faces of a 3D label field.

    Returns per axis a (rectangles, keys) pair: rectangles (R, 5) int64 = plane, b0, b1, c0, c1
    on the axis order (a, b, c) with b = (a + 1) % 3, c = (a + 2) % 3 (b1, c1 exclusive), and keys
    (R, 2) = (front grain, back region) with front > 0. The face normal points from front to back:
    +a when the front grain is on the lower side of the plane, -a otherwise.
    """
    labels = np.asarray(labels).astype(np.int64)
    out = []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        lab = labels.transpose((a, b, c))
        n, nb, nc = lab.shape
        cap = np.full((1, nb, nc), BOX, dtype=np.int64)
        ext = np.concatenate([cap, lab, cap])
        left, right = ext[:-1], ext[1:]                     # (n + 1, nb, nc)
        face = left != right
        # front = the grain on the lower side unless that side is the box (lower box face)
        flip = left < 0
        front = np.where(flip, right, left)
        back = np.where(flip, left, right)
        span = int(labels.max()) + 2
        key = np.where(face, (front * span + (back + 1)) * 2 + flip, -1)
        out.append(_merge_plane_runs(key, span))
    return out


def _merge_plane_runs(key: np.ndarray, span: int):
    """Rectangles of equal keys (>= 0) in every plane of key (P, B, C)."""
    p_n, b_n, c_n = key.shape
    flat = key.reshape(-1, c_n)
    # runs along c
    start = np.ones_like(flat, dtype=bool)
    start[:, 1:] = flat[:, 1:] != flat[:, :-1]
    start &= flat >= 0
    rows, c0 = np.nonzero(start)
    if not len(rows):
        return np.zeros((0, 5), dtype=np.int64), np.zeros((0, 2), dtype=np.int64)
    k = flat[rows, c0]
    # run end: the next position in the row where the key changes, or the row end
    change = np.zeros_like(flat, dtype=bool)
    change[:, 1:] = flat[:, 1:] != flat[:, :-1]
    changes = np.flatnonzero(change)                         # global positions row * C + c, sorted
    begin = rows * c_n + c0
    nxt = np.searchsorted(changes, begin, side='right')
    end = np.where(nxt < len(changes), changes[np.minimum(nxt, len(changes) - 1)], np.iinfo(np.int64).max)
    c1 = np.minimum(end, rows * c_n + c_n) - rows * c_n
    plane, row = np.divmod(rows, b_n)
    # merge runs of consecutive rows with the same plane, extent and key
    order = np.lexsort((row, k, c1, c0, plane))
    plane, row, c0, c1, k = plane[order], row[order], c0[order], c1[order], k[order]
    same = np.zeros(len(order), dtype=bool)
    same[1:] = ((plane[1:] == plane[:-1]) & (c0[1:] == c0[:-1]) & (c1[1:] == c1[:-1]) & (k[1:] == k[:-1])
                & (row[1:] == row[:-1] + 1))
    rect = np.cumsum(~same) - 1
    first = np.flatnonzero(~same)
    last = np.concatenate([first[1:] - 1, [len(order) - 1]])
    rects = np.column_stack([plane[first], row[first], row[last] + 1, c0[first], c1[first]])
    code = k[first]
    flip = code % 2
    pair = code // 2
    front, back = pair // span, pair % span - 1
    keys = np.column_stack([front, back, flip])
    return rects, keys


def staircase_surface(labels: np.ndarray, spacing: Sequence[float], step: int = 1) -> Surface:
    labels = np.asarray(labels)
    full_shape = labels.shape
    if step > 1:
        labels = labels[::step, ::step, ::step]
    spacing = np.asarray(spacing, dtype=float) * step
    faces = int(sum(np.count_nonzero(np.diff(np.concatenate([np.full_like(np.take(labels, [0], a), BOX), labels,
                                                              np.full_like(np.take(labels, [0], a), BOX)], axis=a),
                                             axis=a) != 0) for a in range(3)))
    corners, regions, masks = [], [], []
    area = 0
    for a, (rects, keys) in enumerate(face_rectangles(labels)):
        if not len(rects):
            continue
        b, c = (a + 1) % 3, (a + 2) % 3
        p, b0, b1, c0, c1 = rects.T
        area += int(((b1 - b0) * (c1 - c0)).sum())
        quad = np.zeros((len(rects), 4, 3), dtype=np.int64)
        quad[:, :, a] = p[:, None]
        quad[:, 0, b], quad[:, 0, c] = b0, c0
        quad[:, 1, b], quad[:, 1, c] = b1, c0
        quad[:, 2, b], quad[:, 2, c] = b1, c1
        quad[:, 3, b], quad[:, 3, c] = b0, c1
        flip = keys[:, 2].astype(bool)
        quad[flip] = quad[flip][:, [0, 3, 2, 1]]
        corners.append(quad)
        regions.append(keys[:, :2])
    if corners:
        quad = np.concatenate(corners)
        pair = np.concatenate(regions)
    else:
        quad = np.zeros((0, 4, 3), dtype=np.int64)
        pair = np.zeros((0, 2), dtype=np.int64)
    # shared lattice points
    dims = np.array(labels.shape) + 1
    flat = np.ravel_multi_index(quad.reshape(-1, 3).T, dims) if len(quad) else np.zeros(0, dtype=np.int64)
    used, inverse = np.unique(flat, return_inverse=True)
    points = (np.stack(np.unravel_index(used, dims), axis=1) * spacing).astype(np.float32)
    quad_idx = inverse.reshape(-1, 4)
    triangles = np.concatenate([quad_idx[:, [0, 1, 2]], quad_idx[:, [0, 2, 3]]]).astype(np.uint32)
    regions_t = np.concatenate([pair, pair]).astype(np.int32)
    # hide the diagonal 0-2: opposite vertex 1 in (0, 1, 2), opposite vertex 2 in (0, 2, 3)
    mask = np.concatenate([np.full(len(quad), 0b010), np.full(len(quad), 0b100)]).astype(np.uint8)
    meta = {'source': 'staircase', 'step': step, 'shape': list(labels.shape), 'full_shape': list(full_shape),
            'spacing': spacing.tolist(), 'size': (np.array(labels.shape) * spacing).tolist(),
            'voxel_faces': faces, 'rectangles': int(len(quad)), 'face_area_voxels': area,
            'merged_triangles': int(len(triangles)), 'unmerged_triangles': 2 * faces}
    return Surface(points, triangles, regions_t, mask, meta)


def smoothed_surface(mesh_npz, size: Sequence[float]) -> Optional[Surface]:
    """The grain boundary triangles a meshing job stored, or None if the file has none."""
    data = np.load(mesh_npz)
    if 'surf_points' not in data.files:
        return None
    regions = data['surf_regions'].astype(np.int64)
    regions = np.where(regions < 0, BOX, regions).astype(np.int32)
    tris = data['surf_triangles'].astype(np.uint32)
    meta = {'source': 'smoothed', 'step': 1, 'size': list(map(float, size)),
            'merged_triangles': int(len(tris)), 'unmerged_triangles': int(len(tris))}
    return Surface(data['surf_points'].astype(np.float32), tris, regions,
                   np.zeros(len(tris), dtype=np.uint8), meta)


def auto_step(labels_shape: Sequence[int], budget_voxels: int = 2 ** 21) -> int:
    """Smallest step that keeps the downsampled field below budget_voxels (128^3 at step 1)."""
    step, n = 1, int(np.prod(labels_shape))
    while n / step ** 3 > budget_voxels:
        step += 1
    return step
