"""The finite element mesh every solver export writes: the tetrahedral (2D: triangular) mesh of an
RVEResult or a voxel mesh (one 8-node hexahedron per voxel; 2D: one 4-node quadrilateral per
pixel), with what the solver files need on top of the elements: nodes and element faces (2D:
edges) on the box faces, and the periodic node classes.

Node order of the cells is the one of VTK, meshio, Abaqus and Exodus II (they agree for these
types):
    hexahedron  bottom face (z low) 0-1-2-3 counter-clockwise seen from +z, then 4-7 above them
    tetra       0-3, positive volume
    tetra10     corners 0-3, midside nodes 4 (0-1), 5 (1-2), 6 (0-2), 7 (0-3), 8 (1-3), 9 (2-3)
    quad        0-1-2-3 counter-clockwise
    triangle    0-2 counter-clockwise
    triangle6   corners 0-2, midside nodes 3 (0-1), 4 (1-2), 5 (2-0)
"""
from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

MeshKind = Literal['auto', 'voxel', 'tet']

BOX_FACES: Tuple[Tuple[str, int, int], ...] = (
    ('left', 0, 0), ('right', 0, 1), ('bottom', 1, 0), ('top', 1, 1), ('back', 2, 0), ('front', 2, 1))
"""(name, axis, side) of the box faces; side 0 is the face at 0, side 1 the face at the box size.
The names are those of MOOSE's GeneratedMesh and of the legacy MOOSE export. 2D: the first four."""


def box_faces(ndim: int) -> Tuple[Tuple[str, int, int], ...]:
    return BOX_FACES[:2 * ndim]

# Local faces (corner nodes, 0-based, pointing out of the element) in the side numbering of
# Exodus II; Abaqus numbers the faces of C3D8 and C3D4/C3D10 differently (see abaqus.py).
EXODUS_SIDES = {
    'hexahedron': ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (0, 4, 7, 3), (0, 3, 2, 1), (4, 5, 6, 7)),
    'tetra': ((0, 1, 3), (1, 2, 3), (0, 3, 2), (0, 2, 1)),
}
EXODUS_SIDES['tetra10'] = EXODUS_SIDES['tetra']
EXODUS_SIDES['quad'] = ((0, 1), (1, 2), (2, 3), (3, 0))
EXODUS_SIDES['triangle'] = ((0, 1), (1, 2), (2, 0))
EXODUS_SIDES['triangle6'] = EXODUS_SIDES['triangle']
CORNERS = {'tetra10': 4, 'triangle6': 3}

_HEX_CORNERS = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
_QUAD_CORNERS = np.array([(0, 0), (1, 0), (1, 1), (0, 1)])


@dataclass
class SolverMesh:
    """points (P, d) metres (d = 2 or 3); cells (E, k) int64; grain_ids (E,) int32 (1 ... G);
    periodic_pairs per periodic axis (n, 2) node indices (node on the face at 0, partner on the
    opposite face); box_size_m (x, y[, z])."""

    points: np.ndarray
    cells: np.ndarray
    cell_type: str
    grain_ids: np.ndarray
    periodic_pairs: Dict[int, np.ndarray]
    box_size_m: Tuple[float, ...]

    @property
    def ndim(self) -> int:
        return len(self.box_size_m)

    @property
    def corners(self) -> np.ndarray:
        """Corner nodes of every cell (midside nodes dropped)."""
        return self.cells[:, :CORNERS.get(self.cell_type, self.cells.shape[1])]

    def on_face(self, axis: int, side: int, tol: float = 1e-9) -> np.ndarray:
        """Boolean per node: lies on the box face (axis, side); tol relative to the box size."""
        target = 0.0 if side == 0 else self.box_size_m[axis]
        return np.abs(self.points[:, axis] - target) <= tol * max(self.box_size_m)

    def face_nodes(self) -> Dict[str, np.ndarray]:
        """Sorted node indices on each box face (BOX_FACES names), corner and midside nodes."""
        return {name: np.flatnonzero(self.on_face(axis, side)) for name, axis, side in box_faces(self.ndim)}

    def face_sides(self, local_faces) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Per box face the element faces lying in it: (element index, local face index, 0-based
        in `local_faces`). A face lies in the box face when all its corner nodes do."""
        out = {}
        for name, axis, side in box_faces(self.ndim):
            on = self.on_face(axis, side)
            elems, faces = [], []
            for f, nodes in enumerate(local_faces):
                hit = np.flatnonzero(on[self.cells[:, list(nodes)]].all(axis=1))
                elems.append(hit)
                faces.append(np.full(len(hit), f))
            elems, faces = np.concatenate(elems), np.concatenate(faces)
            order = np.lexsort((faces, elems))
            out[name] = (elems[order], faces[order])
        return out

    def periodic_classes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nodes that are the same point of the periodic structure, as (slave, master, offset):
        every class has one master, the node with the smallest coordinates; every other node of
        the class is a slave with offset (k_x, k_y, k_z) in {0, 1}: slave = master + sum_a k_a L_a.
        One row per slave, so a corner of a fully periodic box gives 7 rows, not 12 pairs."""
        if not self.periodic_pairs:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, np.zeros((0, 3), dtype=np.int64)
        pairs = np.concatenate(list(self.periodic_pairs.values()))
        n = len(self.points)
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, comp = connected_components(graph, directed=False)
        nodes = np.unique(pairs)
        comp = comp[nodes]
        # master: smallest coordinate sum in its class (a class is a product set of translates,
        # so this node is lowest along every axis)
        key = self.points[nodes].sum(axis=1)
        order = np.lexsort((key, comp))
        nodes, comp = nodes[order], comp[order]
        first = np.r_[True, comp[1:] != comp[:-1]]
        master = nodes[first][np.cumsum(first) - 1]
        slave_rows = ~first
        slave, master = nodes[slave_rows], master[slave_rows]
        offset = np.rint((self.points[slave] - self.points[master]) / np.asarray(self.box_size_m)).astype(np.int64)
        return slave, master, offset


def voxel_mesh(result) -> SolverMesh:
    """One 8-node hexahedron per voxel (2D: 4-node quadrilateral per pixel), in the order of
    result.grain_ids.ravel() (C order, x slowest); node (i, j, k) of the (nx+1, ny+1, nz+1) grid
    has index (i*(ny+1) + j)*(nz+1) + k. Periodic pairs as for the tetrahedral mesh: every node of
    the face at 0 with its partner."""
    labels = np.asarray(result.grain_ids)
    ndim = labels.ndim
    shape = np.array(labels.shape)
    nodes_shape = tuple(shape + 1)
    spacing = np.asarray(result.box_size_m, dtype=float) / shape
    ijk = np.indices(nodes_shape).reshape(ndim, -1).T
    points = ijk * spacing
    points[:, :] = np.where(ijk == shape, np.asarray(result.box_size_m), points)   # exact upper faces
    base = np.indices(tuple(shape)).reshape(ndim, -1).T
    offsets = _HEX_CORNERS if ndim == 3 else _QUAD_CORNERS
    corner_ijk = base[:, None, :] + offsets[None, :, :]
    cells = np.ravel_multi_index(tuple(corner_ijk.transpose(2, 0, 1)), nodes_shape).astype(np.int64)
    index = np.arange(np.prod(nodes_shape), dtype=np.int64).reshape(nodes_shape)
    pairs = {}
    for axis in range(ndim):
        if result.periodic[axis]:
            lo = np.take(index, 0, axis=axis).ravel()
            hi = np.take(index, shape[axis], axis=axis).ravel()
            pairs[axis] = np.column_stack([lo, hi])
    return SolverMesh(points=points, cells=cells, cell_type='hexahedron' if ndim == 3 else 'quad',
                      grain_ids=labels.ravel().astype(np.int32), periodic_pairs=pairs,
                      box_size_m=tuple(float(s) for s in result.box_size_m))


def solver_mesh(result, mesh: MeshKind = 'auto') -> SolverMesh:
    """'tet': the tetrahedral mesh (error if there is none); 'voxel': the voxel mesh; 'auto':
    the tetrahedral mesh if the RVE has one, else the voxel mesh."""
    if mesh not in ('auto', 'voxel', 'tet'):
        raise ValueError(f"mesh must be 'auto', 'voxel' or 'tet', got {mesh!r}")
    if mesh == 'tet' or (mesh == 'auto' and result.tet_mesh is not None):
        tet = result.tet_mesh
        if tet is None:
            raise ValueError("the RVE has no tetrahedral mesh; generate it with RVESpec(mesh='tet')")
        return SolverMesh(points=tet.points, cells=tet.cells, cell_type=tet.cell_type,
                          grain_ids=tet.grain_ids, periodic_pairs=tet.periodic_pairs,
                          box_size_m=tuple(float(s) for s in result.box_size_m))
    return voxel_mesh(result)
