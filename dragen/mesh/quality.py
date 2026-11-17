"""Quality metrics for grain boundary meshes and volume meshes (phase M0).

Every later meshing step (M1-M4) is judged with these functions, and so is the legacy mesher.
They take plain arrays, know nothing about how the mesh was made and are checked against the
staircase mesh, whose answers are known exactly (tests/unit/test_mesh_quality.py).

Surface checks, per grain, on the triangles of an InterfaceMesh oriented outward for that grain:
  open_edges          edges used by an odd number of the grain's triangles (hole in the surface)
  nonmanifold_edges   edges used by more than two of them (e.g. two voxels touching at an edge)
  misoriented_edges   closed edges whose two triangles run the same way along them
  nonmanifold_vertices vertices around which the grain's triangles form more than one fan
                       (e.g. two voxels touching at a corner)
  components / euler  connected pieces of the surface (connected through shared edges) and the
                       Euler characteristic V - E + F of each; 2 for a sphere, 0 for a torus
  volume              enclosed volume (divergence theorem); equals the voxel volume for the
                       staircase mesh

Interface checks: duplicate triangles, degenerate triangles, triangles with the same grain on
both sides, triangles facing a GAP (non-conforming neighbours), smallest angle and radius ratio,
pairwise self-intersections. For smoothed meshes (M3): kink angle between neighbouring triangles
of a patch, triangles folded against their staircase position.

Volume checks: inverted elements (det J <= 0 at a check point), scaled Jacobian, Jacobian ratio,
dihedral angles and radius ratio of tetrahedra, faces used once that do not lie on the box
(gaps between grains = non-conforming mesh), faces used more than twice, tet10 midside nodes that
differ between neighbours, element volume per grain, connected element bodies per grain.

Periodicity: every node on a lower box face has a partner at the same position on the upper face,
and the boundary facets on both faces match through these partners.
"""
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from dragen.mesh.surface import GAP, InterfaceMesh, is_box_face
from dragen.mesh.volume import (TET10_EDGES, VolumeMesh, element_volumes, jacobian_determinants,
                                scaled_jacobian, tet_dihedral_angles, tet_radius_ratio)


def _group(*cols) -> np.ndarray:
    """Group ID of every row of the given integer columns (rows with equal values share one)."""
    order = np.lexsort(cols[::-1])
    keys = np.stack([np.asarray(c)[order] for c in cols])
    start = np.concatenate([[True], np.any(keys[:, 1:] != keys[:, :-1], axis=0)])
    ids = np.empty(len(order), dtype=np.int64)
    ids[order] = np.cumsum(start) - 1
    return ids


def unique_pairs(a, b, return_inverse: bool = False, return_counts: bool = False):
    """np.unique over the rows (a, b) of two integer columns.

    Encodes each row in one int64 key and sorts once; several times faster than
    np.unique(..., axis=0), which sorts rows as structured records. Returns the unique rows
    (n, 2), then the inverse and the counts if asked for.
    """
    a = np.asarray(a, dtype=np.int64).ravel()
    b = np.asarray(b, dtype=np.int64).ravel()
    if not len(a):
        rows = np.zeros((0, 2), dtype=np.int64)
        extra = [np.zeros(0, dtype=np.int64)] * (return_inverse + return_counts)
        return (rows, *extra) if extra else rows
    a0, b0 = a.min(), b.min()
    span = int(b.max() - b0) + 1
    if (int(a.max() - a0) + 1) * span >= 2 ** 62:
        raise OverflowError('pair values too large to encode')
    out = np.unique((a - a0) * span + (b - b0), return_inverse=return_inverse, return_counts=return_counts)
    keys = out[0] if isinstance(out, tuple) else out
    rows = np.column_stack([keys // span + a0, keys % span + b0])
    if isinstance(out, tuple):
        return (rows, *[o.ravel() for o in out[1:]])
    return rows


def _components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    graph = sparse.coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def _chain_pairs(group: np.ndarray):
    """Positions of consecutive rows within each group (after sorting by group)."""
    order = np.argsort(group, kind='stable')
    same = group[order[1:]] == group[order[:-1]]
    return order[:-1][same], order[1:][same]


# --------------------------------------------------------------------------------------------- #
# surfaces

def grain_surfaces(mesh: InterfaceMesh) -> pd.DataFrame:
    """One row per grain with the surface checks from the module docstring."""
    grain, tris, _ = mesh.sided()
    s = len(tris)
    t = np.repeat(np.arange(s), 3)
    k = np.tile(np.arange(3), s)
    u = tris.ravel()
    v = tris[:, [1, 2, 0]].ravel()
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    g = grain[t]
    edge = _group(g, lo, hi)
    n_edges = edge.max() + 1 if len(edge) else 0
    use = np.bincount(edge, minlength=n_edges)
    turn = np.bincount(edge, weights=np.where(u < v, 1, -1), minlength=n_edges)
    edge_grain = np.zeros(n_edges, dtype=np.int64)
    edge_grain[edge] = g

    # triangles connected through shared edges; corners connected around a vertex
    a, b = _chain_pairs(edge)
    component = _components(s, t[a], t[b])
    corner_lo = 3 * t + np.where(u < v, k, (k + 1) % 3)
    corner_hi = 3 * t + np.where(u < v, (k + 1) % 3, k)
    fan = _components(3 * s, np.concatenate([corner_lo[a], corner_hi[a]]),
                      np.concatenate([corner_lo[b], corner_hi[b]]))
    corner_vertex = tris.ravel()
    corner_grain = np.repeat(grain, 3)
    gv = _group(corner_grain, corner_vertex)
    fans_per_gv = np.bincount(unique_pairs(gv, fan)[:, 0])
    gv_grain = np.zeros(len(fans_per_gv), dtype=np.int64)
    gv_grain[gv] = corner_grain

    # Euler characteristic per connected piece
    n_comp = component.max() + 1 if s else 0
    faces = np.bincount(component, minlength=n_comp)
    edges = np.bincount(unique_pairs(component[t], edge)[:, 0], minlength=n_comp)
    verts = np.bincount(unique_pairs(np.repeat(component, 3), corner_vertex)[:, 0], minlength=n_comp)
    euler = verts - edges + faces
    comp_grain = np.zeros(n_comp, dtype=np.int64)
    comp_grain[component] = grain

    p = mesh.points[tris]
    volume = np.einsum('ij,ij->i', p[:, 0], np.cross(p[:, 1], p[:, 2])) / 6

    ids = np.unique(grain)
    pos = {gid: i for i, gid in enumerate(ids)}

    def per_grain(owner, values):
        out = np.zeros(len(ids), dtype=np.asarray(values).dtype)
        np.add.at(out, np.searchsorted(ids, owner), values)
        return out

    euler_lists = [[] for _ in ids]
    for c in np.argsort(euler, kind='stable'):
        euler_lists[pos[comp_grain[c]]].append(int(euler[c]))
    open_edges = per_grain(edge_grain, (use % 2 == 1).astype(np.int64))
    nonmanifold_edges = per_grain(edge_grain, (use > 2).astype(np.int64))
    misoriented = per_grain(edge_grain, ((use % 2 == 0) & (turn != 0)).astype(np.int64))
    nonmanifold_vertices = per_grain(gv_grain, (fans_per_gv > 1).astype(np.int64))
    return pd.DataFrame({
        'grain': ids,
        'triangles': np.bincount(np.searchsorted(ids, grain), minlength=len(ids)),
        'open_edges': open_edges,
        'nonmanifold_edges': nonmanifold_edges,
        'misoriented_edges': misoriented,
        'nonmanifold_vertices': nonmanifold_vertices,
        'components': per_grain(comp_grain, np.ones(n_comp, dtype=np.int64)),
        'euler': [tuple(e) for e in euler_lists],
        'closed_manifold': (open_edges + nonmanifold_edges + misoriented + nonmanifold_vertices) == 0,
        'volume': per_grain(grain, volume),
    })


def interface_patches(mesh: InterfaceMesh) -> Dict[tuple, int]:
    """Connected pieces of the boundary between every pair of grains (box faces left out).

    Triangles of the same pair are connected when they share an edge.
    """
    inner = ~is_box_face(mesh.grains).any(axis=1)
    tris = mesh.triangles[inner]
    pair = np.sort(mesh.grains[inner], axis=1)
    if not len(tris):
        return {}
    f = len(tris)
    t = np.repeat(np.arange(f), 3)
    u, v = tris.ravel(), tris[:, [1, 2, 0]].ravel()
    edge = _group(pair[t, 0], pair[t, 1], np.minimum(u, v), np.maximum(u, v))
    a, b = _chain_pairs(edge)
    piece = _components(f, t[a], t[b])
    keys, counts = np.unique(np.unique(np.column_stack([pair, piece]), axis=0)[:, :2], axis=0,
                             return_counts=True)
    return {(int(g1), int(g2)): int(n) for (g1, g2), n in zip(keys, counts)}


def surface_kinks(points: np.ndarray, triangles: np.ndarray, patch: np.ndarray) -> np.ndarray:
    """Angle in degrees between the normals of every two triangles that share an edge inside the
    same patch (0 = flat; the staircase has 0 and 90). Triangles must be consistently oriented
    within a patch."""
    t = points[triangles]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
    tri = np.repeat(np.arange(len(triangles)), 3)
    u, v = triangles.ravel(), triangles[:, [1, 2, 0]].ravel()
    key = unique_pairs(np.minimum(u, v), np.maximum(u, v), return_inverse=True)[1]
    order = np.argsort(key, kind='stable')
    same = key[order[1:]] == key[order[:-1]]
    i, j = tri[order[:-1]][same], tri[order[1:]][same]
    keep = patch[i] == patch[j]
    cos = np.einsum('ij,ij->i', n[i[keep]], n[j[keep]])
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def folded_triangles(reference: np.ndarray, points: np.ndarray, triangles: np.ndarray) -> int:
    """Triangles whose normal turned by 90 degrees or more against the reference positions."""
    def normals(x):
        t = x[triangles]
        return np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    return int(np.count_nonzero(np.einsum('ij,ij->i', normals(points), normals(reference)) <= 0))


def triangle_quality(points: np.ndarray, triangles: np.ndarray) -> Dict[str, np.ndarray]:
    """Smallest angle (degrees), radius ratio 2 r_in / r_circ (1 = equilateral) and area per triangle."""
    p = points[triangles]
    e = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 1], p[:, 0] - p[:, 2]], axis=1)
    length = np.linalg.norm(e, axis=2)
    area = np.linalg.norm(np.cross(e[:, 0], -e[:, 2]), axis=1) / 2
    angles = []
    for i in range(3):
        a, b = -e[:, i - 1], e[:, i]          # the two edges leaving vertex i
        cos = np.einsum('ij,ij->i', a, b) / np.maximum(length[:, i - 1] * length[:, i], 1e-300)
        angles.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    with np.errstate(divide='ignore', invalid='ignore'):
        r_in = 2 * area / length.sum(axis=1)
        r_circ = np.prod(length, axis=1) / (4 * area)
        ratio = np.nan_to_num(2 * r_in / r_circ)
    return {'min_angle': np.min(angles, axis=0), 'radius_ratio': ratio, 'area': area}


def self_intersections(points: np.ndarray, triangles: np.ndarray, tol: float = 1e-9) -> int:
    """Number of triangle pairs without a common vertex whose interiors cross.

    Broad phase: pairs whose centroids are closer than the sum of their bounding radii.
    Narrow phase: an edge of one triangle passes through the interior of the other
    (Moeller-Trumbore). Coplanar overlaps are not detected; exact duplicates are counted by
    `interface_report` instead.
    """
    return len(self_intersection_pairs(points, triangles, tol=tol))


def self_intersection_pairs(points: np.ndarray, triangles: np.ndarray, candidates: np.ndarray = None,
                            tol: float = 1e-9, shared_vertex: bool = False, touching: bool = False) -> np.ndarray:
    """The crossing pairs (k, 2) of `self_intersections`; with `candidates` (triangle indices)
    only pairs that involve at least one candidate are tested. With shared_vertex, pairs that share
    one vertex are tested as well (only pairs sharing an edge are skipped); a hit at the shared
    vertex itself does not count. With touching, a corner of one triangle that lies on the other
    one (inside or on its boundary) but is none of its corners counts as well: a T-junction, which
    a volume mesher rejects (gmsh HXT: "a vertex and a facet intersect"), and so does an edge that
    crosses an edge of the other triangle in a single point inside both (gmsh HXT: "Two segments
    intersect", O51)."""
    p = points[triangles]
    lo, hi = p.min(axis=1), p.max(axis=1)
    pairs = _box_pairs(lo, hi, candidates)
    if len(pairs) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    i, j = pairs[:, 0], pairs[:, 1]
    # eq[k, u, v]: corner u of triangle i equals corner v of triangle j
    eq = triangles[i][:, :, None] == triangles[j][:, None, :]
    common = eq.any(axis=2).sum(axis=1)
    keep = common < (2 if shared_vertex else 1)
    i, j, eq = i[keep], j[keep], eq[keep]
    # per triangle in a pair: edge vectors, normal and their lengths (gathered per pair below)
    used = np.zeros(len(triangles), dtype=bool)
    used[i] = True
    used[j] = True
    used = np.flatnonzero(used)
    slot = np.zeros(len(triangles), dtype=np.int64)
    slot[used] = np.arange(len(used))
    pu = p[used]
    p0 = pu[:, 0]
    e1, e2 = pu[:, 1] - p0, pu[:, 2] - p0
    normal = np.cross(e1, e2)
    normal_len = np.linalg.norm(normal, axis=1)
    e1_len, e2_len = np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1)
    p, i, j = pu, slot[i], slot[j]
    # separating planes: if all corners of one triangle lie clearly on one side of the other's
    # plane, the two cannot cross (corners close to the plane keep the pair); a shared corner is
    # left out, since touching there does not count
    for swap in (False, True):
        a, b = (j, i) if swap else (i, j)
        rel = p[b] - p[a, :1]
        d = np.einsum('ikj,ij->ik', rel, normal[a])
        eps = (tol * normal_len[a] * np.abs(rel).max(axis=(1, 2)))[:, None]
        shared = eq.any(axis=2) if swap else eq.any(axis=1)      # per corner of b
        apart = np.all((d > eps) | shared, axis=1) | np.all((d < -eps) | shared, axis=1)
        i, j, eq = i[~apart], j[~apart], eq[~apart]
    hit = np.zeros(len(i), dtype=bool)
    for swap, (a, b) in enumerate(((i, j), (j, i))):
        # corners of a close to the plane of b: only those segments can lie in it
        rel = p[a] - p[b, :1]
        d = np.abs(np.einsum('ikj,ij->ik', rel, normal[b]))
        near = d <= 1e-9 * (normal_len[b] * np.maximum(np.abs(rel).max(axis=(1, 2)), 1e-300))[:, None]
        if touching:
            own = eq.any(axis=1) if swap else eq.any(axis=2)          # corner of a is one of b's
            rows = np.flatnonzero((near & ~own).any(axis=1))
            if len(rows):
                ra, rb = a[rows], b[rows]
                hit[rows] |= _corners_on_triangle(p[ra], p[rb], normal[rb], (near & ~own)[rows], tol)
        tb = (p[b, 0], e1[b], e2[b], e1_len[b], e2_len[b])
        for k in range(3):
            hit |= _segment_hits_triangle(p[a, k], p[a, (k + 1) % 3], None, tol, tb, closed=touching)
        flat = near & near[:, [1, 2, 0]]                        # edge k: corners k and k + 1
        rows = np.flatnonzero(flat.any(axis=1))
        if len(rows):
            ra, rb = a[rows], b[rows]
            hit[rows] |= _coplanar_edges_hit_triangle(p[ra], p[rb], normal[rb], normal_len[rb], flat[rows], tol)
    i, j = used[i[hit]], used[j[hit]]
    order = np.lexsort((j, i))
    return np.column_stack([i[order], j[order]])


def _box_pairs(lo: np.ndarray, hi: np.ndarray, candidates=None) -> np.ndarray:
    """Pairs (i < j) of boxes [lo, hi] that overlap, at least one of them among the candidates.

    Broad phase on a uniform grid with cells twice the median box size; a box is entered in every
    cell it touches, pairs are formed within the cells and then tested for overlap."""
    n = len(lo)
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64)
    cell = max(2.0 * float(np.median((hi - lo).max(axis=1))), 1e-12)
    origin = lo.min(axis=0)
    a = np.floor((lo - origin) / cell).astype(np.int64)
    b = np.floor((hi - origin) / cell).astype(np.int64)
    dims = b.max(axis=0) + 1
    span = b - a

    def cells(sel):
        """(box, cell key) for every cell touched by the selected boxes."""
        ext = span[sel] + 1
        count = ext.prod(axis=1)
        rows = np.repeat(sel, count)
        k = np.arange(len(rows)) - np.repeat(np.cumsum(count) - count, count)
        ey, ez = np.repeat(ext[:, 1], count), np.repeat(ext[:, 2], count)
        c = a[rows] + np.column_stack([k // (ey * ez), (k // ez) % ey, k % ez])
        return rows, c[:, 0] + dims[0] * (c[:, 1] + dims[1] * c[:, 2])

    flag = None
    if candidates is not None:
        flag = np.zeros(n, dtype=bool)
        flag[np.asarray(candidates, dtype=np.int64)] = True
        if not flag.any():
            return np.zeros((0, 2), dtype=np.int64)
    if flag is not None and int(np.prod(dims)) <= 50_000_000:
        # only the cells that hold a candidate matter, marked on a dense grid (no sorting); a box
        # can reach a marked cell only if its lowest cell lies at most its span below one, so the
        # marks are first spread downwards by the largest span to preselect the boxes
        mark = np.zeros(tuple(dims[::-1]), dtype=bool)
        mark.ravel()[cells(np.flatnonzero(flag))[1]] = True
        reach = mark.copy()
        for axis, s in zip((2, 1, 0), span.max(axis=0)):
            r = reach.copy()
            for shift in range(1, int(s) + 1):
                sl = [slice(None)] * 3
                sl[axis] = slice(0, -shift)
                src = [slice(None)] * 3
                src[axis] = slice(shift, None)
                r[tuple(sl)] |= reach[tuple(src)]
            reach = r
        a_key = a[:, 0] + dims[0] * (a[:, 1] + dims[1] * a[:, 2])
        rows, keys = cells(np.flatnonzero(reach.ravel()[a_key]))
        keep = mark.ravel()[keys]
        rows, keys = rows[keep], keys[keep]
    else:
        rows, keys = cells(np.arange(n))
        if flag is not None:
            keep = np.isin(keys, np.unique(keys[flag[rows]]))
            rows, keys = rows[keep], keys[keep]
    # within each cell the entries that are no candidates come first; a candidate entry is paired
    # with those and with all entries after it, so every pair in a cell is formed once
    cand = np.ones(len(rows), dtype=bool) if flag is None else flag[rows]
    order = np.argsort(keys * 2 + cand)
    rows, keys, cand = rows[order], keys[order], cand[order]
    starts = np.flatnonzero(np.concatenate([[True], keys[1:] != keys[:-1]]))
    counts = np.diff(np.concatenate([starts, [len(keys)]]))
    others = np.add.reduceat(~cand, starts) if len(starts) else np.zeros(0, dtype=np.int64)
    src = np.flatnonzero(cand)
    cell_of = np.repeat(np.arange(len(starts)), counts)[src]
    n_before = others[cell_of]
    n_after = starts[cell_of] + counts[cell_of] - 1 - src
    total = n_before + n_after
    first = np.repeat(src, total)
    offset = np.arange(int(total.sum())) - np.repeat(np.cumsum(total) - total, total)
    before = np.repeat(n_before, total)
    second = np.where(offset < before, np.repeat(starts[cell_of], total) + offset,
                      first + 1 + offset - before)
    i, j = rows[first], rows[second]
    # each pair is kept once: in its first common cell (the larger of the two lowest cells, which
    # both boxes touch if they overlap)
    keep = np.ones(len(i), dtype=bool)
    key = keys[first]
    for d, k in enumerate((key % dims[0], (key // dims[0]) % dims[1], key // (dims[0] * dims[1]))):
        ad = a[:, d]
        keep &= np.maximum(ad[i], ad[j]) == k
    i, j = i[keep], j[keep]
    overlap = np.ones(len(i), dtype=bool)
    for d in range(3):
        lod, hid = lo[:, d], hi[:, d]
        overlap &= (lod[i] <= hid[j]) & (lod[j] <= hid[i])
    i, j = i[overlap], j[overlap]
    return np.column_stack([np.minimum(i, j), np.maximum(i, j)])


def _corners_on_triangle(pa, pb, n, sel, tol):
    """Whether a selected corner of triangle a (pa, (m, 3, 3)), lying in the plane of triangle b
    (pb, normal n), lies inside b or on its boundary, in the projection that drops the dominant
    normal component."""
    drop = np.argmax(np.abs(n), axis=1)
    keep = np.array([[1, 2], [0, 2], [0, 1]])[drop][:, None, :]
    qa = np.take_along_axis(pa, keep, axis=2)
    qt = np.take_along_axis(pb, keep, axis=2)
    t = [qt[:, m] for m in range(3)]

    def orient(p, q, r):
        return (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])

    sign = np.sign(orient(t[0], t[1], t[2]))
    size = np.maximum(np.maximum(np.abs(qa).max(axis=(1, 2)), np.abs(qt).max(axis=(1, 2))), 1e-300)
    e = tol * size ** 2
    out = np.zeros(len(pa), dtype=bool)
    for k in range(3):
        o = np.stack([orient(t[m], t[(m + 1) % 3], qa[:, k]) for m in range(3)], axis=1) * sign[:, None]
        out |= sel[:, k] & (sign != 0) & np.all(o >= -e[:, None], axis=1)
    return out


def _coplanar_edges_hit_triangle(pa, pb, n, nn, flat, tol):
    """Whether an edge of triangle a (corners pa, (m, 3, 3)) lying in the plane of triangle b
    (corners pb, normal n, |n| nn) crosses one of b's edges or ends strictly inside it (the 3D test
    cannot see this: the edge is parallel to the plane); flat (m, 3) selects the edges k (corners
    k, k + 1) to test. Touching at vertices or along edges does not count."""
    t0 = pb[:, 0]
    rel = pa - t0[:, None, :]
    dist = np.abs(np.einsum('ikj,ij->ik', rel, n))                 # corner distances to the plane
    scale = np.abs(rel).max(axis=2)
    # project onto the coordinate plane that drops the dominant normal component
    drop = np.argmax(np.abs(n), axis=1)
    keep = np.array([[1, 2], [0, 2], [0, 1]])[drop][:, None, :]
    qa = np.take_along_axis(pa, keep, axis=2)                       # (m, 3, 2)
    qt = np.take_along_axis(pb, keep, axis=2)
    size_a = np.abs(qa).max(axis=2)
    size_t = np.abs(qt).max(axis=(1, 2))
    t = [qt[:, m] for m in range(3)]

    def orient(p, q, r):
        return (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])

    sign = np.sign(orient(t[0], t[1], t[2]))[:, None]
    out = np.zeros(len(pa), dtype=bool)
    for k in range(3):
        k1 = (k + 1) % 3
        eps = 1e-9 * nn * np.maximum(np.maximum(scale[:, k], scale[:, k1]), 1e-300)
        idx = np.flatnonzero(flat[:, k] & (dist[:, k] <= eps) & (dist[:, k1] <= eps) & (nn > 0))
        if not len(idx):
            continue
        a, b = qa[idx, k], qa[idx, k1]
        tk = [v[idx] for v in t]
        size = np.maximum(np.maximum(np.maximum(size_a[idx, k], size_a[idx, k1]), size_t[idx]), 1e-300)
        e = tol * size ** 2
        hit = np.zeros(len(idx), dtype=bool)
        for m in range(3):
            c, d = tk[m], tk[(m + 1) % 3]
            o1, o2 = orient(a, b, c), orient(a, b, d)
            o3, o4 = orient(c, d, a), orient(c, d, b)
            hit |= (((o1 > e) & (o2 < -e)) | ((o1 < -e) & (o2 > e))) & (((o3 > e) & (o4 < -e)) | ((o3 < -e) & (o4 > e)))
        sg = sign[idx]
        for pnt in (a, b):
            o = np.stack([orient(tk[m], tk[(m + 1) % 3], pnt) for m in range(3)], axis=1) * sg
            hit |= np.all(o > e[:, None], axis=1)
        out[idx] |= hit
    return out


def _segment_hits_triangle(s0, s1, tri, tol, parts=None, closed=False):
    """parts: (corner 0, edge 1, edge 2, |edge 1|, |edge 2|) of the triangles, instead of tri.
    With closed, a hit on the triangle's boundary counts as well (the segment's own ends still do
    not): an edge that crosses an edge of the triangle in a single point."""
    if parts is None:
        e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        parts = (tri[:, 0], e1, e2, np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1))
    t0, e1, e2, e1_len, e2_len = parts
    d = s1 - s0
    h = np.cross(d, e2)
    det = np.einsum('ij,ij->i', e1, h)
    ok = np.abs(det) > tol * np.linalg.norm(d, axis=1) * e1_len * e2_len
    det = np.where(ok, det, 1.0)
    s = s0 - t0
    uu = np.einsum('ij,ij->i', s, h) / det
    q = np.cross(s, e1)
    vv = np.einsum('ij,ij->i', d, q) / det
    tt = np.einsum('ij,ij->i', e2, q) / det
    lim = -tol if closed else tol
    return ok & (uu > lim) & (vv > lim) & (uu + vv < 1 - lim) & (tt > tol) & (tt < 1 - tol)


def interface_report(mesh: InterfaceMesh, reference_volume: Optional[Dict[int, float]] = None,
                     intersections: bool = False) -> dict:
    """Summary of all surface checks; reference_volume maps grain ID -> expected volume."""
    grains = grain_surfaces(mesh)
    quality = triangle_quality(mesh.points, mesh.triangles)
    key = np.sort(mesh.triangles, axis=1)
    _, dup_counts = np.unique(key, axis=0, return_counts=True)
    scale = np.median(quality['area']) if len(quality['area']) else 1.0
    out = {
        'n_points': int(len(mesh.points)),
        'n_triangles': int(len(mesh.triangles)),
        'n_grains': int(len(grains)),
        'duplicate_triangles': int((dup_counts - 1).sum()),
        'degenerate_triangles': int(np.count_nonzero(quality['area'] < 1e-10 * scale)),
        'same_region_triangles': int(np.count_nonzero(mesh.grains[:, 0] == mesh.grains[:, 1])),
        'gap_triangles': int(np.count_nonzero(mesh.grains == GAP)),
        'grains_not_closed_manifold': int(np.count_nonzero(~grains['closed_manifold'])),
        'open_edges': int(grains['open_edges'].sum()),
        'nonmanifold_edges': int(grains['nonmanifold_edges'].sum()),
        'misoriented_edges': int(grains['misoriented_edges'].sum()),
        'nonmanifold_vertices': int(grains['nonmanifold_vertices'].sum()),
        'grains_with_several_surfaces': int(np.count_nonzero(grains['components'] > 1)),
        'min_angle_deg': _quantiles(quality['min_angle']),
        'radius_ratio': _quantiles(quality['radius_ratio']),
    }
    if reference_volume is not None:
        out['volume_deviation'] = _volume_deviation(grains['grain'].to_numpy(), grains['volume'].to_numpy(),
                                                    reference_volume)
    if intersections:
        out['self_intersections'] = self_intersections(mesh.points, mesh.triangles)
    return out


# --------------------------------------------------------------------------------------------- #
# volume meshes

def volume_report(mesh: VolumeMesh, box_lower: Sequence[float], box_upper: Sequence[float],
                  reference_volume: Optional[Dict[int, float]] = None, tol: float = 1e-9,
                  periodic_axes: Sequence[int] = ()) -> dict:
    """Summary of the volume checks from the module docstring. tol is relative to the box size.

    On periodic_axes, elements of the same grain on opposite box faces whose faces mirror each
    other belong to the same body.
    """
    det = jacobian_determinants(mesh)
    volume = element_volumes(mesh)
    sj = scaled_jacobian(mesh)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.nan_to_num(det.min(axis=1) / det.max(axis=1))
    faces, owner, use = _element_faces(mesh)
    once = use[faces[1]] == 1
    on_box = _on_box(mesh.points, faces[0], box_lower, box_upper, tol)
    out = {
        'cell_type': mesh.cell_type,
        'n_points': int(len(mesh.points)),
        'n_elements': int(len(mesh.cells)),
        'unused_points': int(len(mesh.points) - len(np.unique(mesh.cells))),
        'inverted_elements': int(np.count_nonzero(det.min(axis=1) <= 0)),
        'scaled_jacobian': _quantiles(sj),
        'jacobian_ratio': _quantiles(ratio),
        'open_inner_faces': int(np.count_nonzero(once & ~on_box)),
        'overused_faces': int(np.count_nonzero(np.bincount(faces[1]) > 2)),
        'grains_with_several_bodies': int(np.count_nonzero(
            _bodies_per_grain(mesh, faces, owner, box_lower, box_upper, periodic_axes, tol) > 1)),
    }
    if mesh.cell_type in ('tet4', 'tet10'):
        dihedral = tet_dihedral_angles(mesh)
        out['dihedral_min_deg'] = _quantiles(dihedral.min(axis=1))
        out['dihedral_max_deg'] = _quantiles(dihedral.max(axis=1), upper=True)
        out['radius_ratio'] = _quantiles(tet_radius_ratio(mesh))
    if mesh.cell_type == 'tet10':
        out['midside_mismatch'] = _midside_mismatch(mesh)
    if reference_volume is not None:
        ids = np.unique(mesh.grain)
        per_grain = np.bincount(np.searchsorted(ids, mesh.grain), weights=volume, minlength=len(ids))
        out['volume_deviation'] = _volume_deviation(ids, per_grain, reference_volume)
    return out


def _element_faces(mesh: VolumeMesh):
    """Faces as rows of sorted corner nodes, a face ID per (element, face) and use counts."""
    local = mesh.faces
    nodes = mesh.corners[:, local]                       # (E, f, k)
    rows = np.sort(nodes.reshape(-1, local.shape[1]), axis=1)
    face_id = _group(*rows.T)
    owner = np.repeat(np.arange(len(mesh.cells)), len(local))
    return (rows, face_id), owner, np.bincount(face_id)


def _on_box(points, rows, lower, upper, tol):
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    eps = tol * np.max(upper - lower)
    x = points[rows]                                     # (n, k, 3)
    flat = np.zeros(len(rows), dtype=bool)
    for bound in (lower, upper):
        flat |= np.any(np.all(np.abs(x - bound) <= eps, axis=1), axis=1)
    return flat


def _bodies_per_grain(mesh, faces, owner, lower, upper, periodic_axes, tol):
    rows, face_id = faces
    a, b = _chain_pairs(face_id)
    a, b = owner[a], owner[b]
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    eps = tol * np.max(upper - lower)
    once = np.bincount(face_id)[face_id] == 1
    centre = mesh.points[rows].mean(axis=1)
    for axis in periodic_axes:
        lo = np.flatnonzero(once & (np.abs(centre[:, axis] - lower[axis]) <= eps))
        hi = np.flatnonzero(once & (np.abs(centre[:, axis] - upper[axis]) <= eps))
        if not len(lo) or not len(hi):
            continue
        shifted = centre[lo].copy()
        shifted[:, axis] = upper[axis]
        dist, idx = cKDTree(centre[hi]).query(shifted, distance_upper_bound=2 * eps + 1e-300)
        found = np.isfinite(dist)
        a = np.concatenate([a, owner[lo[found]]])
        b = np.concatenate([b, owner[hi[idx[found]]]])
    same = mesh.grain[a] == mesh.grain[b]
    body = _components(len(mesh.cells), a[same], b[same])
    pairs = unique_pairs(mesh.grain, body)
    return np.unique(pairs[:, 0], return_counts=True)[1]


def _midside_mismatch(mesh):
    """Element edges that share both corners with another element edge but not the midside node."""
    corners = mesh.cells[:, TET10_EDGES]                 # (E, 6, 2)
    mid = mesh.cells[:, 4:].ravel()
    lo = corners.min(axis=2).ravel()
    hi = corners.max(axis=2).ravel()
    edge = _group(lo, hi)
    variants = unique_pairs(edge, mid)[:, 0]
    return int(np.count_nonzero(np.bincount(variants) > 1))


# --------------------------------------------------------------------------------------------- #
# periodicity and anchors

def periodic_mismatch(points: np.ndarray, facets: np.ndarray, box_lower: Sequence[float],
                      box_upper: Sequence[float], axes: Sequence[int], tol: float = 1e-9) -> dict:
    """Partner check between opposite box faces on the given axes.

    facets: boundary facets (triangles or element faces, any number of nodes); facets that do
    not lie on one of the two faces are ignored. tol is relative to the box size.
    Returns per axis the node counts on both faces, nodes without a partner, the largest
    distance to the partner, and facets without a mirrored facet.
    """
    lower, upper = np.asarray(box_lower, float), np.asarray(box_upper, float)
    eps = tol * np.max(upper - lower)
    facets = np.asarray(facets)
    out = {}
    for axis in axes:
        on_lo = np.flatnonzero(np.abs(points[:, axis] - lower[axis]) <= eps)
        on_hi = np.flatnonzero(np.abs(points[:, axis] - upper[axis]) <= eps)
        shifted = points[on_hi].copy()
        shifted[:, axis] = lower[axis]
        partner = np.full(len(points), -1, dtype=np.int64)
        dist = np.zeros(0)
        if len(on_lo) and len(on_hi):
            dist, idx = cKDTree(shifted).query(points[on_lo], distance_upper_bound=2 * eps + 1e-300)
            found = np.isfinite(dist)
            partner[on_lo[found]] = on_hi[idx[found]]
            dist = dist[found]
        matched_hi = np.zeros(len(points), dtype=bool)
        matched_hi[partner[partner >= 0]] = True
        lo_facets = facets[np.all(np.isin(facets, on_lo), axis=1)]
        hi_facets = facets[np.all(np.isin(facets, on_hi), axis=1)]
        mapped = partner[lo_facets]
        good = np.all(mapped >= 0, axis=1)
        lo_keys = {tuple(r) for r in np.sort(mapped[good], axis=1)}
        hi_keys = {tuple(r) for r in np.sort(hi_facets, axis=1)}
        out[int(axis)] = {
            'nodes_lower': int(len(on_lo)),
            'nodes_upper': int(len(on_hi)),
            'unmatched_nodes': int(np.count_nonzero(partner[on_lo] < 0)
                                   + np.count_nonzero(~matched_hi[on_hi])),
            'max_distance': float(dist.max()) if len(dist) else 0.0,
            'unmatched_facets': int(np.count_nonzero(~good) + len(lo_keys ^ hi_keys)),
        }
    return out


def anchor_displacement(before: np.ndarray, after: np.ndarray, nodes: np.ndarray) -> dict:
    """How far the given nodes (e.g. quadruple points) moved; 0 when anchors are kept fixed."""
    d = np.linalg.norm(np.asarray(after)[nodes] - np.asarray(before)[nodes], axis=1)
    return {'n': int(len(d)), 'max': float(d.max()) if len(d) else 0.0,
            'mean': float(d.mean()) if len(d) else 0.0, 'moved': int(np.count_nonzero(d > 0))}


# --------------------------------------------------------------------------------------------- #

def _quantiles(values: np.ndarray, upper: bool = False) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {}
    if upper:
        return {'max': float(values.max()), 'p99': float(np.percentile(values, 99)),
                'median': float(np.median(values))}
    return {'min': float(values.min()), 'p1': float(np.percentile(values, 1)),
            'median': float(np.median(values))}


def _volume_deviation(ids: np.ndarray, actual: np.ndarray, reference: Dict[int, float]) -> dict:
    ref = np.array([reference.get(int(g), np.nan) for g in ids])
    keep = np.isfinite(ref) & (ref > 0)
    rel = (actual[keep] - ref[keep]) / ref[keep]
    if not len(rel):
        return {}
    return {'max_abs': float(np.abs(rel).max()), 'p90_abs': float(np.percentile(np.abs(rel), 90)),
            'median': float(np.median(rel)), 'total': float(actual[keep].sum() / ref[keep].sum() - 1)}
