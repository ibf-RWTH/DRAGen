"""Volume meshes (one grain ID per element) and the element geometry needed to judge them.

Node order follows Abaqus and VTK:
  tet4   corners 0-3
  tet10  corners 0-3, midside nodes 4 (0-1), 5 (1-2), 6 (0-2), 7 (0-3), 8 (1-3), 9 (2-3);
         gmsh numbers 8 and 9 the other way round
  hex8   0-3 bottom face counter-clockwise seen from above, 4-7 the top face above them
"""
from dataclasses import dataclass

import numpy as np

from dragen.mesh.surface import GAP, InterfaceMesh, box_face_code

CORNERS = {'tet4': 4, 'tet10': 4, 'hex8': 8}
NODES = {'tet4': 4, 'tet10': 10, 'hex8': 8}

# element faces by local corner index, ordered so that the normal points out of a positively
# oriented element
FACES = {
    'tet': np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
    'hex8': np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]),
}
TET10_EDGES = np.array([[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]])
# the three neighbours of every hex corner, right-handed
HEX_CORNER_EDGES = np.array([[1, 3, 4], [2, 0, 5], [3, 1, 6], [0, 2, 7],
                             [7, 5, 0], [4, 6, 1], [5, 7, 2], [6, 4, 3]])
_HEX_SIGNS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                       [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)


@dataclass
class VolumeMesh:
    points: np.ndarray
    """(P, 3) node coordinates."""
    cells: np.ndarray
    """(E, k) node indices, order as in the module docstring."""
    cell_type: str
    """'tet4', 'tet10' or 'hex8'."""
    grain: np.ndarray
    """(E,) grain ID of every element."""

    def __post_init__(self):
        if self.cell_type not in NODES:
            raise ValueError(f'unknown cell type {self.cell_type!r}')
        if self.cells.shape[1] != NODES[self.cell_type]:
            raise ValueError(f'{self.cell_type} needs {NODES[self.cell_type]} nodes per element')

    @property
    def corners(self) -> np.ndarray:
        return self.cells[:, :CORNERS[self.cell_type]]

    @property
    def faces(self) -> np.ndarray:
        return FACES['hex8'] if self.cell_type == 'hex8' else FACES['tet']


_HEX_CORNER_OFFSETS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])


def voxel_hex_mesh(labels: np.ndarray, spacing) -> VolumeMesh:
    """One hex8 element per voxel, element grain = voxel label (the unsmoothed legacy C3D8 mesh)."""
    labels = np.asarray(labels)
    corner_shape = tuple(n + 1 for n in labels.shape)
    voxel = np.argwhere(np.ones(labels.shape, dtype=bool))       # C order, matches labels.ravel()
    cells = np.ravel_multi_index((voxel[:, None, :] + _HEX_CORNER_OFFSETS[None]).reshape(-1, 3).T,
                                 corner_shape).reshape(-1, 8)
    used, inverse = np.unique(cells, return_inverse=True)
    points = np.stack(np.unravel_index(used, corner_shape), axis=1) * np.asarray(spacing, dtype=float)
    return VolumeMesh(points, inverse.reshape(-1, 8), 'hex8', labels.ravel().copy())


def _kuhn_tets() -> np.ndarray:
    """Six positively oriented tetrahedra along the diagonal 0-6 of a hex; neighbouring hexes split
    their shared face along the same diagonal, so the result is conforming."""
    index = {tuple(c): i for i, c in enumerate(_HEX_CORNER_OFFSETS)}
    tets = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        c = np.zeros(3, dtype=int)
        path = [index[tuple(c)]]
        for axis in perm:
            c[axis] = 1
            path.append(index[tuple(c)])
        x = _HEX_CORNER_OFFSETS[path].astype(float)
        if np.linalg.det(x[1:] - x[0]) < 0:
            path[1], path[2] = path[2], path[1]
        tets.append(path)
    return np.array(tets)


def hex_to_tets(mesh: VolumeMesh) -> VolumeMesh:
    """Split every hex8 into six tet4 (Kuhn split), conforming across elements."""
    cells = mesh.cells[:, _kuhn_tets()].reshape(-1, 4)
    return VolumeMesh(mesh.points, cells, 'tet4', np.repeat(mesh.grain, 6))


def tet4_to_tet10(mesh: VolumeMesh) -> VolumeMesh:
    """Straight-sided tet10: one shared midside node at the middle of every edge."""
    edges = mesh.cells[:, TET10_EDGES]                              # (E, 6, 2)
    key = np.sort(edges.reshape(-1, 2), axis=1)
    unique, inverse = np.unique(key, axis=0, return_inverse=True)
    mids = mesh.points[unique].mean(axis=1)
    points = np.vstack([mesh.points, mids])
    cells = np.hstack([mesh.cells, len(mesh.points) + inverse.reshape(-1, 6)])
    return VolumeMesh(points, cells, 'tet10', mesh.grain)


def grain_boundary_mesh(mesh: VolumeMesh, box_lower, box_upper, tol: float = 1e-9) -> InterfaceMesh:
    """The element faces between different grains and on the box, as an InterfaceMesh.

    Uses the corner nodes only (tet10 faces become flat triangles), quads are split into two
    triangles. A face used by one element that does not lie on the box gets the region GAP.
    tol is relative to the box size.
    """
    local = mesh.faces
    nodes = mesh.corners[:, local].reshape(-1, local.shape[1])
    owner = np.repeat(np.arange(len(mesh.cells)), len(local))
    _, face_id, count = np.unique(np.sort(nodes, axis=1), axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(face_id.ravel(), kind='stable')
    starts = np.searchsorted(face_id.ravel()[order], np.arange(len(count)))
    first = order[starts]                      # one row per face, in face ID order
    partner = np.full(len(count), -1, dtype=np.int64)
    shared = count >= 2
    partner[shared] = owner[order[starts[shared] + 1]]

    grain = mesh.grain[owner[first]]
    other = np.where(shared, mesh.grain[np.maximum(partner, 0)], 0)
    region = np.where(shared, other, GAP)
    lower, upper = np.asarray(box_lower, float), np.asarray(box_upper, float)
    eps = tol * np.max(upper - lower)
    x = mesh.points[nodes[first]]
    for axis in range(3):
        for side, bound in enumerate((lower[axis], upper[axis])):
            on = np.all(np.abs(x[:, :, axis] - bound) <= eps, axis=1)
            region[on & ~shared] = box_face_code(axis, side)
    keep = ~shared | (grain != other)
    faces = nodes[first[keep]]
    pair = np.column_stack([grain[keep], region[keep]])
    if faces.shape[1] == 4:
        # split along the diagonal through the corner with the smallest coordinate sum, so that
        # mirrored quads on opposite box faces are split the same way
        start = np.argmin(mesh.points[faces].sum(axis=2), axis=1)
        faces = np.take_along_axis(faces, (start[:, None] + np.arange(4)) % 4, axis=1)
        faces = np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]])
        pair = np.concatenate([pair, pair])
    used, inverse = np.unique(faces, return_inverse=True)
    return InterfaceMesh(mesh.points[used], inverse.reshape(faces.shape), pair)


def _shape_derivatives(cell_type: str, xi: np.ndarray) -> np.ndarray:
    """dN/dxi at natural coordinates xi (Q, 3); returns (Q, nodes, 3)."""
    q = len(xi)
    if cell_type == 'tet4':
        d = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        return np.broadcast_to(d, (q, 4, 3)).copy()
    if cell_type == 'tet10':
        lam = np.column_stack([1 - xi.sum(axis=1), xi])
        dlam = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        out = np.empty((q, 10, 3))
        out[:, :4] = (4 * lam - 1)[:, :, None] * dlam[None]
        for m, (a, b) in enumerate(TET10_EDGES):
            out[:, 4 + m] = 4 * (lam[:, a, None] * dlam[b] + lam[:, b, None] * dlam[a])
        return out
    if cell_type == 'hex8':
        f = 1 + xi[:, None, :] * _HEX_SIGNS[None]  # (Q, 8, 3)
        out = np.empty((q, 8, 3))
        for j in range(3):
            others = [k for k in range(3) if k != j]
            out[:, :, j] = _HEX_SIGNS[None, :, j] * f[:, :, others[0]] * f[:, :, others[1]] / 8
        return out
    raise ValueError(cell_type)


def _sample_points(cell_type: str) -> np.ndarray:
    """Natural coordinates where the Jacobian is checked: corners, plus midsides and centroid for tet10."""
    if cell_type == 'hex8':
        return _HEX_SIGNS.copy()
    corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    if cell_type == 'tet4':
        return corners
    mids = (corners[TET10_EDGES[:, 0]] + corners[TET10_EDGES[:, 1]]) / 2
    return np.vstack([corners, mids, [[0.25, 0.25, 0.25]]])


def _quadrature(cell_type: str):
    """Points and weights that integrate the Jacobian determinant exactly."""
    if cell_type == 'hex8':
        g = 1 / np.sqrt(3)
        return _HEX_SIGNS * g, np.ones(8)
    # degree-3 rule on the unit tetrahedron (volume 1/6)
    xi = np.array([[0.25, 0.25, 0.25], [1 / 6, 1 / 6, 1 / 6], [0.5, 1 / 6, 1 / 6],
                   [1 / 6, 0.5, 1 / 6], [1 / 6, 1 / 6, 0.5]])
    w = np.array([-4 / 5, 9 / 20, 9 / 20, 9 / 20, 9 / 20]) / 6
    return xi, w


def jacobian_determinants(mesh: VolumeMesh, xi: np.ndarray = None) -> np.ndarray:
    """det J of every element at natural coordinates xi (default: the check points); (E, Q)."""
    if xi is None:
        xi = _sample_points(mesh.cell_type)
    dn = _shape_derivatives(mesh.cell_type, xi)              # (Q, n, 3)
    x = mesh.points[mesh.cells]                               # (E, n, 3)
    jac = np.einsum('eni,qnj->eqij', x, dn)                   # (E, Q, 3, 3)
    return np.linalg.det(jac)


def element_volumes(mesh: VolumeMesh) -> np.ndarray:
    xi, w = _quadrature(mesh.cell_type)
    return jacobian_determinants(mesh, xi) @ w


def scaled_jacobian(mesh: VolumeMesh) -> np.ndarray:
    """Corner-based scaled Jacobian, 1 for a regular tetrahedron or a cube, <= 0 when inverted.

    Uses only the corner nodes, so for tet10 it judges the straight-sided parent tetrahedron.
    """
    x = mesh.points[mesh.corners]
    if mesh.cell_type == 'hex8':
        e = x[:, HEX_CORNER_EDGES] - x[:, :, None, :]          # (E, 8, 3, 3)
        det = np.linalg.det(e)
        norm = np.prod(np.linalg.norm(e, axis=3), axis=2)
        return (det / norm).min(axis=1)
    e = x[:, 1:] - x[:, :1]
    det = np.linalg.det(e)
    lengths = []
    for k in range(4):
        others = [j for j in range(4) if j != k]
        lengths.append(np.prod(np.linalg.norm(x[:, others] - x[:, [k]], axis=2), axis=1))
    return np.sqrt(2) * det / np.max(lengths, axis=0)


def tet_dihedral_angles(mesh: VolumeMesh) -> np.ndarray:
    """(E, 6) dihedral angles in degrees of the corner tetrahedron, edges as TET10_EDGES."""
    x = mesh.points[mesh.corners]
    normals = []
    for k in range(4):
        i, j, l = [m for m in range(4) if m != k]
        n = np.cross(x[:, j] - x[:, i], x[:, l] - x[:, i])
        away = np.einsum('ij,ij->i', n, x[:, i] - x[:, k])
        normals.append(n * np.sign(away)[:, None])
    normals = np.stack(normals, axis=1)
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    out = []
    for a, b in TET10_EDGES:
        k, l = [m for m in range(4) if m not in (a, b)]   # the faces opposite k and l share edge a-b
        cos = np.clip(np.einsum('ij,ij->i', normals[:, k], normals[:, l]), -1, 1)
        out.append(180 - np.degrees(np.arccos(cos)))
    return np.stack(out, axis=1)


def tet_radius_ratio(mesh: VolumeMesh) -> np.ndarray:
    """3 * inradius / circumradius of the corner tetrahedron; 1 for a regular one, 0 when flat."""
    x = mesh.points[mesh.corners]
    a, b, c = (x[:, 1] - x[:, 0]), (x[:, 2] - x[:, 0]), (x[:, 3] - x[:, 0])
    det = np.abs(np.einsum('ij,ij->i', a, np.cross(b, c)))
    area = sum(np.linalg.norm(np.cross(x[:, j] - x[:, i], x[:, l] - x[:, i]), axis=1) / 2
               for i, j, l in FACES['tet'])
    r_in = (det / 2) / area                                   # 3 V / A with V = det / 6
    num = (np.einsum('ij,ij->i', a, a)[:, None] * np.cross(b, c)
           + np.einsum('ij,ij->i', b, b)[:, None] * np.cross(c, a)
           + np.einsum('ij,ij->i', c, c)[:, None] * np.cross(a, b))
    with np.errstate(divide='ignore', invalid='ignore'):
        r_circ = np.linalg.norm(num, axis=1) / (2 * det)
        return np.nan_to_num(3 * r_in / r_circ)
