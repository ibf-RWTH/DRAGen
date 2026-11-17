"""Volume mesh of the smoothed grain boundary network with gmsh (phase M4).

The surface model of M2 with the smoothed points of M3 goes into gmsh as discrete entities with
their mesh: anchors as points, lines as curves (two-node segments along the vertex chain),
surfaces as discrete surfaces with their triangles. gmsh does not remesh discrete entities, so the
grain boundary triangles stay exactly as they come from the smoothing (author's decision, process
notes section 2) and only the volumes are filled with tetrahedra.

Volumes: gmsh fills a volume only where its surface loop bounds one connected region. A grain cut
by the box on a periodic axis consists of several pieces in the box, and a grain around another
grain has a cavity. Every grain is therefore split into bodies: the shell surfaces are grouped
into closed components (connected through shared curves); a component that encloses positive
volume with the grain's outward orientation is a body, one with negative volume a cavity, which
goes to the body that contains it. Each body becomes one gmsh volume (geo kernel, surface loops of
the discrete surfaces, as in gmsh tutorial t13); `body_grain` maps it back to the grain.

Quality: on the unchanged smoothed triangles almost all slivers are tetrahedra whose four nodes
lie on the boundary and that bridge two nearly coplanar boundary triangles; the Netgen optimizer
splits them with interior nodes. On the adaptively remeshed surface (M4a) HXT alone gives a third
fewer elements in a fifth of the meshing time, but leaves single slivers down to 0.03 degrees and
the coarsening feedback rarely helps; the Netgen optimizer after HXT lifts the smallest angle above
1.2 degrees in five of the six worst RVEs and makes the feedback work again (default HXT + Netgen
since 30.10.2026, author's decision, process notes 7.12). HXT runs on one thread: with several
threads its result changes from run to run.

Element size: a Distance field to all grain boundary and box surfaces, size = h0 + growth *
distance, capped at max_size; h0 is the smallest voxel edge. The tetrahedra thus get coarser
towards the grain interiors as far as the grain size allows.

Second order: gmsh adds one midside node per edge. Discrete surfaces have no parametrisation, so
the midside nodes lie on the straight edges (tet10 with straight sides); neighbouring elements and
mirrored box faces share them.

Node numbering of the result: node i < len(points) is surface vertex i of the model, the other
nodes (volume vertices, midside nodes) follow. Elements in VolumeMesh order (tet10 nodes 8 and 9
swapped against gmsh).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from dragen.mesh import quality
from dragen.mesh.model import SurfaceModel
from dragen.mesh.quality import _group, unique_pairs
from dragen.mesh.volume import FACES, TET10_EDGES, VolumeMesh, element_volumes

GMSH_TET10_TO_VTK = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]
TET10_FACES = np.array([[0, 2, 1, 6, 5, 4], [0, 1, 3, 4, 8, 7], [0, 3, 2, 7, 9, 6], [1, 2, 3, 5, 9, 8]])
"""Element faces with their midside nodes, corners as volume.FACES['tet']."""


@dataclass
class TetMeshConfig:
    growth: float = 0.5
    """Increase of the element size per unit distance from the nearest surface."""
    max_size: Optional[float] = None
    """Largest element size in voxel edges (None: only limited by the distance to the surfaces)."""
    order: int = 2
    optimize: str = 'Netgen'
    """Extra gmsh optimizer runs after meshing, comma separated ('' none, 'Netgen', 'Relocate3D',
    'Gmsh', ...). On the unchanged smoothed surface Netgen splits the slivers spanning two boundary
    triangles (smallest dihedral angle at 100^3: 0.04 -> 0.58 degrees, p1 8.8 -> 13.3) for about
    three times the meshing time; after HXT on the remeshed surface it removes the worst slivers
    for about half more elements (process notes 7.12)."""
    algorithm: int = 10
    """gmsh Mesh.Algorithm3D: 1 Delaunay, 4 Frontal, 10 HXT."""
    threads: int = 1
    """gmsh threads; HXT is reproducible only on one thread."""
    verbosity: int = 2


@dataclass
class Body:
    grain: int
    shells: List[List[int]]
    """Surface indices of every closed shell; the first is the outer one, the others cavities."""


@dataclass
class TetMeshResult:
    mesh: VolumeMesh
    body: np.ndarray
    """(E,) body index of every element (into `bodies`)."""
    bodies: List[Body]
    stats: dict = field(default_factory=dict)


def grain_bodies(model: SurfaceModel, points: np.ndarray) -> List[Body]:
    """Splits every grain shell into closed bodies, cavities assigned to the body around them."""
    bodies = []
    for grain, shell in model.shells.items():
        components = _shell_components(model, shell)
        outer, cavities = [], []
        for comp in components:
            tris = _outward_triangles(model, shell[comp])
            (outer if _enclosed_volume(points, tris) > 0 else cavities).append((comp, tris))
        grain_bodies_ = [Body(int(grain), [list(shell[comp, 0])]) for comp, _ in outer]
        for comp, tris in cavities:
            probe = points[tris[0, 0]]
            inside = [k for k, (_, t) in enumerate(outer) if abs(_winding_number(points, t, probe)) > 0.5]
            if len(inside) != 1:
                raise ValueError(f'grain {grain}: a cavity lies in {len(inside)} bodies')
            grain_bodies_[inside[0]].shells.append(list(shell[comp, 0]))
        bodies += grain_bodies_
    return bodies


def tet_mesh(model: SurfaceModel, points: np.ndarray, spacing: Sequence[float],
             config: TetMeshConfig = TetMeshConfig()) -> TetMeshResult:
    """Tetrahedral mesh of all grains in one gmsh run on the given surface points."""
    import gmsh
    import time
    stats = {'seconds': {}}
    t = time.perf_counter()
    bodies = grain_bodies(model, points)
    stats['seconds']['bodies'] = time.perf_counter() - t
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber('General.Terminal', 1 if config.verbosity else 0)
        gmsh.option.setNumber('General.Verbosity', config.verbosity)
        gmsh.option.setNumber('Mesh.Renumber', 0)          # node tag = surface vertex + 1
        gmsh.option.setNumber('Mesh.Algorithm3D', config.algorithm)
        gmsh.option.setNumber('General.NumThreads', config.threads)
        gmsh.model.add('rve')
        t = time.perf_counter()
        _add_surfaces(gmsh, model, points)
        for i, body in enumerate(bodies):
            loops = [gmsh.model.geo.addSurfaceLoop([s + 1 for s in shell]) for shell in body.shells]
            gmsh.model.geo.addVolume(loops, i + 1)
        gmsh.model.geo.synchronize()
        _size_field(gmsh, model, spacing, config)
        stats['seconds']['entities'] = time.perf_counter() - t
        t = time.perf_counter()
        gmsh.model.mesh.generate(3)
        _optimize(gmsh, config)
        stats['seconds']['generate'] = time.perf_counter() - t
        t = time.perf_counter()
        if config.order > 1:
            gmsh.model.mesh.setOrder(config.order)
        stats['seconds']['order'] = time.perf_counter() - t
        mesh, body = _extract(gmsh, bodies, len(points), config.order)
    finally:
        gmsh.finalize()
    stats['bodies'] = len(bodies)
    stats['cavities'] = sum(len(b.shells) - 1 for b in bodies)
    return TetMeshResult(mesh=mesh, body=body, bodies=bodies, stats=stats)


def _optimize(gmsh, config: TetMeshConfig) -> None:
    for method in config.optimize.split(','):
        if method:
            gmsh.model.mesh.optimize(method)


def periodic_node_pairs(points: np.ndarray, box_upper: Sequence[float], axes: Sequence[int],
                        nodes: Optional[np.ndarray] = None, tol: float = 1e-9) -> Dict[int, np.ndarray]:
    """Per axis the (n, 2) pairs (node on the lower face, node at the same place on the upper
    face) among `nodes` (default: all); tol is relative to the box size. Raises if a node on one
    face has no partner on the other."""
    upper = np.asarray(box_upper, float)
    eps = tol * upper.max()
    nodes = np.arange(len(points)) if nodes is None else np.asarray(nodes)
    out = {}
    for axis in axes:
        lo = nodes[np.abs(points[nodes, axis]) <= eps]
        hi = nodes[np.abs(points[nodes, axis] - upper[axis]) <= eps]
        shifted = points[hi].copy()
        shifted[:, axis] = 0.0
        dist, idx = cKDTree(shifted).query(points[lo], distance_upper_bound=2 * eps + 1e-300) \
            if len(hi) else (np.full(len(lo), np.inf), np.zeros(len(lo), int))
        if len(lo) != len(hi) or not np.all(np.isfinite(dist)):
            raise ValueError(f'axis {axis}: {len(lo)} nodes on the lower face, {len(hi)} on the upper, '
                             f'{np.count_nonzero(~np.isfinite(dist))} without partner')
        out[int(axis)] = np.column_stack([lo, hi[idx]])
    return out


def enclosed_volumes(model: SurfaceModel, points: np.ndarray) -> Dict[int, float]:
    """Volume enclosed by every grain shell (divergence theorem)."""
    t = points[model.oriented]
    tri_vol = np.einsum('ij,ij->i', t[:, 0], np.cross(t[:, 1], t[:, 2])) / 6
    patch_vol = np.bincount(model.topology.patch, weights=tri_vol, minlength=len(model.surfaces))
    return {g: float(np.sum(patch_vol[shell[:, 0]] * shell[:, 1])) for g, shell in model.shells.items()}


def check(model: SurfaceModel, surface_points: np.ndarray, mesh: VolumeMesh, shape: Sequence[int],
          spacing: Sequence[float], periodic: Sequence[bool], reference_volume: Dict[int, float]) -> dict:
    """quality.volume_report plus the M4 rules: the surface vertices did not move, the grain
    boundary and box faces of the tet mesh are exactly the model's triangles, and every node on a
    periodic box face (corner or midside) has its partner. The counts *_missing, *_extra and
    periodic_unmatched_* must be 0."""
    upper = np.array(shape) * np.asarray(spacing, dtype=float)
    axes = [a for a in range(3) if periodic[a]]
    rep = quality.volume_report(mesh, (0, 0, 0), upper, reference_volume=reference_volume,
                                periodic_axes=axes)
    ids = np.unique(mesh.grain)
    per_grain = np.bincount(np.searchsorted(ids, mesh.grain), weights=element_volumes(mesh), minlength=len(ids))
    rep['volume_vs_surface'] = quality._volume_deviation(ids, per_grain, enclosed_volumes(model, surface_points))
    rep['surface_nodes_moved'] = float(np.abs(mesh.points[:len(surface_points)] - surface_points).max())
    # element faces between grains or on the box against the model's triangles
    faces = np.sort(mesh.corners[:, FACES['tet']].reshape(-1, 3), axis=1)
    owner = np.repeat(mesh.grain, 4)
    face_id = _group(*faces.T)
    order = np.argsort(face_id, kind='stable')
    count = np.bincount(face_id)
    starts = np.searchsorted(face_id[order], np.arange(len(count)))
    first = order[starts]
    second = np.where(count > 1, order[np.minimum(starts + 1, len(order) - 1)], first)
    boundary = faces[first[(count == 1) | (owner[first] != owner[second])]]
    triangles = np.sort(model.mesh.triangles, axis=1)
    both = np.concatenate([boundary, triangles])
    source = np.concatenate([np.zeros(len(boundary), int), np.ones(len(triangles), int)])
    ids = _group(*both.T)
    in_tets = np.bincount(ids[source == 0], minlength=ids.max() + 1) > 0
    in_model = np.bincount(ids[source == 1], minlength=ids.max() + 1) > 0
    rep['interface_faces_missing'] = int(np.count_nonzero(in_model & ~in_tets))
    rep['interface_faces_extra'] = int(np.count_nonzero(in_tets & ~in_model))
    # periodic partners of all nodes on the box faces, and mirrored element faces (with midside nodes)
    local = TET10_FACES if mesh.cell_type == 'tet10' else FACES['tet']
    facets = mesh.cells[:, local].reshape(-1, local.shape[1])[count[face_id] == 1]
    per = quality.periodic_mismatch(mesh.points, facets, (0, 0, 0), upper, axes)
    rep['periodic_unmatched_nodes'] = int(sum(v['unmatched_nodes'] for v in per.values()))
    rep['periodic_unmatched_facets'] = int(sum(v['unmatched_facets'] for v in per.values()))
    rep['periodic_max_distance'] = float(max([v['max_distance'] for v in per.values()], default=0.0))
    h0 = float(np.min(spacing))
    ends = mesh.corners[:, TET10_EDGES]
    longest = np.linalg.norm(mesh.points[ends[..., 0]] - mesh.points[ends[..., 1]], axis=2).max(axis=1) / h0
    rep['longest_edge_voxels'] = {'median': float(np.median(longest)), 'p99': float(np.percentile(longest, 99)),
                                  'max': float(longest.max())}
    rep['elements_per_voxel'] = len(mesh.cells) / float(np.prod(shape))
    return rep


# --------------------------------------------------------------------------------------------- #

def _shell_components(model: SurfaceModel, shell: np.ndarray) -> List[np.ndarray]:
    """Rows of the shell grouped into components connected through shared curves."""
    rows, cols = [], []
    for i, s in enumerate(shell[:, 0]):
        for loop in model.surfaces[s].loops:
            rows += [i] * len(loop)
            cols += list(loop[:, 0])
    n = len(shell)
    if rows:
        incidence = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(model.curves)))
        _, comp = connected_components(incidence @ incidence.T, directed=False)
    else:
        comp = np.arange(n)
    return [np.flatnonzero(comp == c) for c in np.unique(comp)]


def _outward_triangles(model: SurfaceModel, rows: np.ndarray) -> np.ndarray:
    return np.concatenate([model.oriented[model.surfaces[s].triangles][:, ::int(sign)] for s, sign in rows])


def _enclosed_volume(points, tris) -> float:
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    return float(np.einsum('ij,ij->', a, np.cross(b, c)) / 6)


def _winding_number(points, tris, q) -> float:
    """Generalised winding number of a closed triangle set around q (solid angles / 4 pi)."""
    a, b, c = points[tris[:, 0]] - q, points[tris[:, 1]] - q, points[tris[:, 2]] - q
    la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
    num = np.einsum('ij,ij->i', a, np.cross(b, c))
    den = la * lb * lc + np.einsum('ij,ij->i', a, b) * lc + np.einsum('ij,ij->i', b, c) * la \
        + np.einsum('ij,ij->i', c, a) * lb
    return float(np.sum(2 * np.arctan2(num, den)) / (4 * np.pi))


def _add_surfaces(gmsh, model: SurfaceModel, points: np.ndarray) -> None:
    """Anchors, lines and surfaces as discrete entities with their nodes and elements; node tag =
    vertex index + 1, entity tag = index + 1.

    Written to a MSH 4.1 file and merged in one go: adding them one entity at a time through the
    API costs time per call that grows with the mesh (427 s at 100^3)."""
    import os
    import tempfile
    fd, path = tempfile.mkstemp(suffix='.msh')
    os.close(fd)
    try:
        with open(path, 'w') as f:
            _write_msh(f, model, points)
        gmsh.merge(path)
    finally:
        os.remove(path)


def _write_msh(f, model: SurfaceModel, points: np.ndarray) -> None:
    """MSH 4.1 (ASCII) with the discrete points, curves and surfaces of the model. Coordinates with
    17 significant digits, so they read back bit for bit."""
    topo = model.topology
    anchor_tag = np.zeros(len(points), dtype=np.int64)
    anchor_tag[topo.anchors] = np.arange(1, len(topo.anchors) + 1)
    lines, surfaces = topo.lines, model.surfaces
    surface_tris = [model.oriented[s.triangles] for s in surfaces]

    def bbox(v):
        x = points[v]
        # repr of a Python float: shortest exact form; numpy 2 would print np.float64(...)
        return ' '.join(repr(float(c)) for c in np.concatenate([x.min(axis=0), x.max(axis=0)]))

    f.write('$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Entities\n')
    f.write(f'{len(topo.anchors)} {len(lines)} {len(surfaces)} 0\n')
    for i, v in enumerate(topo.anchors):
        f.write(f'{i + 1} {" ".join(repr(float(c)) for c in points[v])} 0\n')
    for c, line in enumerate(lines):
        v = line.vertices
        f.write(f'{c + 1} {bbox(v)} 0 2 {anchor_tag[v[0]]} {-anchor_tag[v[-1]]}\n')
    for p, surface in enumerate(surfaces):
        boundary = [int(sign) * (int(c) + 1) for loop in surface.loops for c, sign in loop]
        f.write(f'{p + 1} {bbox(surface_tris[p].ravel())} 0 {len(boundary)} {" ".join(map(str, boundary))}\n')
    f.write('$EndEntities\n')

    # nodes: every vertex once, in the entity of lowest dimension that contains it
    owner_dim = np.full(len(points), 2, dtype=np.int64)
    owner_tag = np.zeros(len(points), dtype=np.int64)
    for p, tris in enumerate(surface_tris):
        owner_tag[tris.ravel()] = p + 1
    for c, line in enumerate(lines):
        owner_dim[line.vertices] = 1
        owner_tag[line.vertices] = c + 1
    owner_dim[topo.anchors] = 0
    owner_tag[topo.anchors] = np.arange(1, len(topo.anchors) + 1)
    order = np.lexsort((np.arange(len(points)), owner_tag, owner_dim))
    blocks = unique_pairs(owner_dim[order], owner_tag[order], return_counts=True)
    f.write('$Nodes\n')
    f.write(f'{len(blocks[0])} {len(points)} 1 {len(points)}\n')
    start = 0
    for (dim, tag), n in zip(*blocks):
        rows = order[start:start + n]
        start += n
        f.write(f'{dim} {tag} 0 {n}\n')
        f.write('\n'.join(map(str, rows + 1)) + '\n')
        np.savetxt(f, points[rows], fmt='%.17g')
    f.write('$EndNodes\n')

    n_segments = sum(len(line.vertices) - 1 for line in lines)
    n_tris = sum(len(t) for t in surface_tris)
    f.write('$Elements\n')
    f.write(f'{len(lines) + len(surfaces)} {n_segments + n_tris} 1 {n_segments + n_tris}\n')
    tag = 1
    for c, line in enumerate(lines):
        v = line.vertices
        rows = np.column_stack([tag + np.arange(len(v) - 1), v[:-1] + 1, v[1:] + 1])
        f.write(f'1 {c + 1} 1 {len(rows)}\n')
        np.savetxt(f, rows, fmt='%d')
        tag += len(rows)
    for p, tris in enumerate(surface_tris):
        rows = np.column_stack([tag + np.arange(len(tris)), tris + 1])
        f.write(f'2 {p + 1} 2 {len(rows)}\n')
        np.savetxt(f, rows, fmt='%d')
        tag += len(rows)
    f.write('$EndElements\n')


def _size_field(gmsh, model: SurfaceModel, spacing, config: TetMeshConfig) -> None:
    h0 = float(np.min(spacing))
    field_ = gmsh.model.mesh.field
    field_.add('Distance', 1)
    field_.setNumbers(1, 'SurfacesList', list(range(1, len(model.surfaces) + 1)))
    field_.add('MathEval', 2)
    expr = f'{h0} + {config.growth} * F1'
    if config.max_size is not None:
        expr = f'Min({expr}, {config.max_size * h0})'
    field_.setString(2, 'F', expr)
    field_.setAsBackgroundMesh(2)
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)


def _extract(gmsh, bodies: List[Body], n_surface: int, order: int) -> Tuple[VolumeMesh, np.ndarray]:
    tags, coords, _ = gmsh.model.mesh.getNodes()
    tags = tags.astype(np.int64) - 1
    xyz = coords.reshape(-1, 3)
    n_nodes = 10 if order > 1 else 4
    cells, body = [], []
    for i in range(len(bodies)):
        _, _, nodes = gmsh.model.mesh.getElements(3, i + 1)
        if not len(nodes) or not len(nodes[0]):
            raise ValueError(f'gmsh left body {i} (grain {bodies[i].grain}) empty')
        c = nodes[0].astype(np.int64).reshape(-1, n_nodes) - 1
        cells.append(c[:, GMSH_TET10_TO_VTK] if order > 1 else c)
        body.append(np.full(len(c), i))
    cells, body = np.concatenate(cells), np.concatenate(body)
    # keep surface vertex i as node i, number the other used nodes after them
    used = np.unique(cells)
    extra = used[used >= n_surface]
    new = np.full(max(int(tags.max()), int(used.max())) + 1, -1, dtype=np.int64)
    new[:n_surface] = np.arange(n_surface)
    new[extra] = n_surface + np.arange(len(extra))
    points = np.zeros((n_surface + len(extra), 3))
    keep = new[tags] >= 0
    points[new[tags[keep]]] = xyz[keep]
    grain = np.array([b.grain for b in bodies])[body]
    return VolumeMesh(points, new[cells], 'tet10' if order > 1 else 'tet4', grain), body
