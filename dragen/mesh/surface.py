"""One conforming triangle mesh for all grain boundaries of an RVE.

Every triangle separates two regions: `grains[f, 0]` and `grains[f, 1]`. Its normal (right-hand
rule over the vertex order) points out of `grains[f, 0]`. Regions are grain IDs (> 0, 0 for
unassigned space), one of the six box faces, coded negative by `box_face_code`, or GAP. Neighbouring
grains share the triangles between them, so the surface of every grain is watertight by
construction as long as the mesh stays conforming; nothing has to be stitched.

The mesh lives in the box [0, size]. A grain that crosses a periodic boundary is cut there and
gets box-face triangles on both sides; the periodic partner check is in `dragen.mesh.quality`.

`staircase_mesh` builds this mesh directly from the voxel faces. It is the exact reference for
the quality metrics (its grain volumes equal the voxel volumes) and the starting point for the
smoothing in M2/M3.
"""
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


GAP = -7
"""Region on the far side of a face that belongs to one element only and does not lie on the box:
a hole between grains in a non-conforming volume mesh (see volume.grain_boundary_mesh)."""


def box_face_code(axis: int, side: int) -> int:
    """Region code of a box face: -1 / -2 for the lower / upper x face, -3 / -4 for y, -5 / -6 for z."""
    return -(2 * axis + side + 1)


def is_box_face(region) -> np.ndarray:
    return np.asarray(region) < 0


@dataclass
class InterfaceMesh:
    points: np.ndarray
    """(P, 3) coordinates."""
    triangles: np.ndarray
    """(F, 3) point indices."""
    grains: np.ndarray
    """(F, 2) regions on both sides; the normal points out of grains[:, 0]."""
    corner: Optional[np.ndarray] = None
    """(P, 3) voxel corner index of every point, for meshes built on the voxel lattice."""

    def grain_ids(self) -> np.ndarray:
        g = np.unique(self.grains)
        return g[~is_box_face(g)]

    def sided(self):
        """Every triangle once per adjacent grain, oriented outward for that grain.

        Returns (grain, triangles, source): grain (S,), triangles (S, 3) and the index of the
        interface triangle each row came from. Box faces get no rows of their own.
        """
        keep0 = ~is_box_face(self.grains[:, 0])
        keep1 = ~is_box_face(self.grains[:, 1])
        idx = np.arange(len(self.triangles))
        grain = np.concatenate([self.grains[keep0, 0], self.grains[keep1, 1]])
        tris = np.concatenate([self.triangles[keep0], self.triangles[keep1][:, ::-1]])
        source = np.concatenate([idx[keep0], idx[keep1]])
        return grain, tris, source


def staircase_mesh(labels: np.ndarray, spacing: Sequence[float]) -> InterfaceMesh:
    """Interface mesh made of the voxel faces between different labels, two triangles per face.

    labels: 3D array of region IDs >= 0, axes x, y, z. Faces on the box boundary separate the
    voxel from the box face. Points are the voxel corners in use, at corner index * spacing.
    """
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError('staircase_mesh needs a 3D label field')
    if labels.min() < 0:
        raise ValueError('labels must be >= 0; negative codes are reserved for box faces')
    corner_shape = tuple(n + 1 for n in labels.shape)
    quads, sides = [], []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        cap = list(labels.shape)
        cap[a] = 1
        ext = np.concatenate([np.full(cap, box_face_code(a, 0), dtype=np.int64),
                              labels.astype(np.int64),
                              np.full(cap, box_face_code(a, 1), dtype=np.int64)], axis=a)
        n = labels.shape[a]
        left = np.take(ext, np.arange(n + 1), axis=a)
        right = np.take(ext, np.arange(1, n + 2), axis=a)
        face = left != right
        # corner lattice position of the face's lowest corner; axis a holds the plane index
        p = np.argwhere(face)
        pair = np.stack([left[face], right[face]], axis=1)
        eb, ec = np.eye(3, dtype=np.int64)[b], np.eye(3, dtype=np.int64)[c]
        # normal eb x ec = +ea, i.e. out of the lower voxel
        corners = np.stack([p, p + eb, p + eb + ec, p + ec], axis=1)
        flip = pair[:, 0] < 0  # lower box face: the grain is on the right, turn the face around
        # keep corner 0 first, so the split diagonal p -> p + eb + ec is the same on opposite box
        # faces and the triangles mirror each other across a periodic boundary
        corners[flip] = corners[flip][:, [0, 3, 2, 1]]
        pair[flip] = pair[flip][:, ::-1]
        quads.append(np.ravel_multi_index(corners.reshape(-1, 3).T, corner_shape).reshape(-1, 4))
        sides.append(pair)
    quads = np.concatenate(quads)
    sides = np.concatenate(sides)
    used, inverse = np.unique(quads, return_inverse=True)
    quads = inverse.reshape(-1, 4)
    corner = np.stack(np.unravel_index(used, corner_shape), axis=1)
    triangles = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    grains = np.concatenate([sides, sides])
    return InterfaceMesh(points=corner * np.asarray(spacing, dtype=float), triangles=triangles,
                         grains=grains, corner=corner)


def corner_grain_counts(labels: np.ndarray, periodic: Sequence[bool]) -> np.ndarray:
    """Number of distinct labels among the voxels around every voxel corner.

    Returns an array of shape labels.shape + 1 per axis. On a periodic axis a corner on the box
    boundary sees the voxels on both sides; on an open axis only those inside the box. This is
    the node classification of the meshing concept: 2 = boundary face, 3 = triple line,
    >= 4 = quadruple point (Section 9.2 of the process notes).

    Only grains are counted here. The topology (dragen.mesh.topology) counts the box faces as
    virtual grains as well and does not look across periodic boundaries.
    """
    labels = np.asarray(labels).astype(np.int64)
    padded = labels
    for axis, per in enumerate(periodic):
        first = np.take(padded, [0], axis=axis)
        last = np.take(padded, [-1], axis=axis)
        if per:
            padded = np.concatenate([last, padded, first], axis=axis)
        else:
            padded = np.concatenate([np.full_like(last, -1), padded, np.full_like(first, -1)], axis=axis)
    n = labels.shape
    around = np.stack([padded[i:i + n[0] + 1, j:j + n[1] + 1, k:k + n[2] + 1]
                       for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    around = np.sort(around, axis=0)
    new = np.concatenate([around[:1] >= 0, (np.diff(around, axis=0) != 0) & (around[1:] >= 0)])
    return new.sum(axis=0)
