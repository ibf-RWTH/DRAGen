"""Topology of the grain boundary network (phase M1): anchors, junction lines and patches.

Works on the staircase mesh of a cleaned label field (dragen.mesh.cleanup). The six box faces are
virtual grains (author's decision, process notes section 2), also on periodic axes, so

  vertex class = distinct regions around a vertex, real grains plus the box faces it lies on:
                 2 inside a boundary patch or a box face, 3 on a junction line or box edge,
                 >= 4 at a junction point or box corner.

Junction lines are chains of mesh edges around which at least three regions meet (triple lines,
grain boundaries meeting a box face, box edges; four or more grains along one voxel edge give a
line of higher order). Anchors are the vertices that later stay fixed: vertices with more regions
than the lines through them (quadruple points, box corners, a grain boundary meeting a box edge),
vertices where the line graph branches or ends, and vertices where two lines with different region
sets meet. Along a line of higher order every vertex has four or more regions; these vertices are
regular line points, not anchors. A closed line without any anchor (a ring) gets one artificial
anchor.
Lines run from anchor to anchor. Patches are the connected pieces of the boundary between one
pair of regions, cut along all lines; they may have holes or no boundary at all.

Periodic axes: the mesh is cut at the box, so the pattern of lines on the lower face comes from
the first voxel layer and the one on the upper face from the last. Both faces must end up with the
same triangulation and move together, so every line lying in a periodic box face is imprinted on
the opposite face as well (`imprinted` lines separate nothing on that side) and anchors are
closed under the periodic images. `periodic_class` names the image class of every vertex.

Check: for the box (a ball) cut into anchors, lines, patches and grains,
    V - E + sum(chi(patch) - chi(boundary of patch)) - sum(chi(grain)) = 1
with V anchors, E lines, chi(patch) the Euler characteristic of the patch surface, chi(boundary)
that of the lines around it (0 for separate loops) and chi(grain) half the Euler characteristic of
the grain's boundary. `euler_check` evaluates it; it holds only if every line ends in anchors and
no anchor lies inside a patch. A patch whose boundary touches itself at an anchor (a "pinched"
patch: the boundary between two grains meets itself at a point where four grains meet) is legal
but needs care when meshing; `pinched_patches` counts them.
"""
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from dragen.mesh.quality import _chain_pairs, _components, _group, grain_surfaces, unique_pairs
from dragen.mesh.surface import InterfaceMesh, staircase_mesh

IMPRINT = -100
"""Pseudo-region added to the region set of imprinted edges, so they never join real lines."""


@dataclass
class Line:
    vertices: np.ndarray
    """Mesh vertices from anchor to anchor; a closed line starts and ends with the same anchor."""
    regions: Tuple[int, ...]
    """Regions around the line (grains and box faces), sorted; IMPRINT for imprinted lines."""

    @property
    def closed(self) -> bool:
        return len(self.vertices) > 2 and self.vertices[0] == self.vertices[-1]

    @property
    def imprinted(self) -> bool:
        return IMPRINT in self.regions

    @property
    def order(self) -> int:
        """Number of real regions around the line (3 = triple line)."""
        return sum(r != IMPRINT for r in self.regions)


@dataclass
class Topology:
    mesh: InterfaceMesh
    shape: Tuple[int, int, int]
    periodic: Tuple[bool, bool, bool]
    vertex_regions: np.ndarray
    """(P,) number of distinct regions around every vertex."""
    edges: np.ndarray
    """(K, 2) all mesh edges, lower vertex first."""
    edge_line: np.ndarray
    """(K,) index of the line an edge belongs to, -1 for edges inside a patch."""
    anchors: np.ndarray
    ring_anchors: np.ndarray
    """Anchors added only to split a ring (subset of anchors)."""
    lines: List[Line]
    patch: np.ndarray
    """(F,) patch of every triangle."""
    patch_regions: np.ndarray
    """(n_patches, 2) region pair of every patch, sorted."""
    periodic_class: np.ndarray
    """(P,) smallest vertex index among the periodic images of every vertex."""

    def patch_euler(self) -> np.ndarray:
        tris = self.mesh.triangles
        t = np.repeat(np.arange(len(tris)), 3)
        p = self.patch
        n = len(self.patch_regions)
        faces = np.bincount(p, minlength=n)
        verts = np.bincount(unique_pairs(p[t], tris.ravel())[:, 0], minlength=n)
        edge_of = _edge_index(self.edges, tris.ravel(), tris[:, [1, 2, 0]].ravel(), len(self.mesh.points))
        edges = np.bincount(unique_pairs(p[t], edge_of)[:, 0], minlength=n)
        return verts - edges + faces

    def _patch_boundary(self):
        """(patch, edge) rows of line edges around every patch and their (patch, vertex) rows."""
        tris = self.mesh.triangles
        t = np.repeat(np.arange(len(tris)), 3)
        edge_of = _edge_index(self.edges, tris.ravel(), tris[:, [1, 2, 0]].ravel(), len(self.mesh.points))
        on = self.edge_line[edge_of] >= 0
        pe = unique_pairs(self.patch[t[on]], edge_of[on])
        pv = np.column_stack([np.repeat(pe[:, 0], 2), self.edges[pe[:, 1]].ravel()])
        return pe, pv

    def patch_boundary_euler(self) -> np.ndarray:
        pe, pv = self._patch_boundary()
        n = len(self.patch_regions)
        vertices = np.bincount(unique_pairs(pv[:, 0], pv[:, 1])[:, 0], minlength=n)
        return vertices - np.bincount(pe[:, 0], minlength=n)

    def pinched_patches(self) -> int:
        """Patches with a boundary vertex used by more than two of their boundary edges."""
        _, pv = self._patch_boundary()
        rows, uses = unique_pairs(pv[:, 0], pv[:, 1], return_counts=True)
        return int(len(np.unique(rows[uses > 2, 0])))

    def isolated_anchors(self) -> int:
        on_line = np.zeros(len(self.mesh.points), dtype=bool)
        on_line[self.edges[self.edge_line >= 0].ravel()] = True
        return int(np.count_nonzero(~on_line[self.anchors]))

    def euler_check(self) -> int:
        grains = grain_surfaces(self.mesh)
        chi_grains = sum(sum(e) for e in grains['euler']) // 2
        open_patches = self.patch_euler() - self.patch_boundary_euler()
        return int(len(self.anchors) - len(self.lines) + open_patches.sum() - chi_grains)

    def summary(self) -> dict:
        real = [ln for ln in self.lines if not ln.imprinted]
        orders = np.array([ln.order for ln in real], dtype=int)
        return {
            'anchors': int(len(self.anchors)),
            'ring_anchors': int(len(self.ring_anchors)),
            'lines': int(len(self.lines)),
            'imprinted_lines': int(len(self.lines) - len(real)),
            'rings': int(sum(ln.closed for ln in self.lines)),
            'triple_lines': int(np.count_nonzero(orders == 3)),
            'higher_order_lines': int(np.count_nonzero(orders > 3)),
            'patches': int(len(self.patch_regions)),
            'patches_without_boundary': int(len(self.patch_regions) - len(np.unique(
                self.patch[_triangles_on_lines(self)]))),
            'pinched_patches': self.pinched_patches(),
            'isolated_anchors': self.isolated_anchors(),
            'euler_check': self.euler_check(),
        }


def build_topology(labels: np.ndarray, spacing: Sequence[float], periodic: Sequence[bool]) -> Topology:
    """Topology of a label field without edge or point self-contacts (see ManifoldCleaner)."""
    mesh = staircase_mesh(labels, spacing)
    return topology_of(mesh, labels.shape, periodic)


def topology_of(mesh: InterfaceMesh, shape: Sequence[int], periodic: Sequence[bool]) -> Topology:
    if mesh.corner is None:
        raise ValueError('topology needs a mesh on the voxel lattice (InterfaceMesh.corner)')
    shape = tuple(int(n) for n in shape)
    periodic = tuple(bool(p) for p in periodic)
    tris, n_points = mesh.triangles, len(mesh.points)

    # mesh edges and the regions around them
    t = np.repeat(np.arange(len(tris)), 3)
    u, v = tris.ravel(), tris[:, [1, 2, 0]].ravel()
    edges, edge_of = unique_pairs(np.minimum(u, v), np.maximum(u, v), return_inverse=True)
    edge_reg = unique_pairs(np.concatenate([edge_of, edge_of]),
                            np.concatenate([mesh.grains[t, 0], mesh.grains[t, 1]]))
    n_reg = np.bincount(edge_reg[:, 0], minlength=len(edges))
    feature = n_reg >= 3
    vert_reg = unique_pairs(np.repeat(tris.ravel(), 2), np.repeat(mesh.grains, 3, axis=0).ravel())
    vertex_regions = np.bincount(vert_reg[:, 0], minlength=n_points)

    # periodic images: imprint lines in periodic box faces on the opposite face
    lattice = np.full(int(np.prod(np.array(shape) + 1)), -1, dtype=np.int64)
    flat = np.ravel_multi_index(mesh.corner.T, tuple(n + 1 for n in shape))
    lattice[flat] = np.arange(n_points)
    imprinted = np.zeros(len(edges), dtype=bool)
    for axis in range(3):
        if not periodic[axis]:
            continue
        ends = mesh.corner[edges]                                     # (K, 2, 3)
        for side, target in ((0, shape[axis]), (shape[axis], 0)):
            in_face = feature & np.all(ends[:, :, axis] == side, axis=1)
            mirrored = ends[in_face].copy()
            mirrored[:, :, axis] = target
            ids = lattice[np.ravel_multi_index(mirrored.reshape(-1, 3).T, tuple(n + 1 for n in shape))]
            if np.any(ids < 0):
                raise ValueError('periodic image of a line is not part of the mesh')
            ids = np.sort(ids.reshape(-1, 2), axis=1)
            k = _edge_index(edges, ids[:, 0], ids[:, 1], n_points)
            imprinted[k[~feature[k]]] = True
            feature[k] = True
    canonical = mesh.corner.copy()
    for axis in range(3):
        if periodic[axis]:
            canonical[:, axis] %= shape[axis]
    cls = _group(*canonical.T)
    periodic_class = np.full(cls.max() + 1, n_points, dtype=np.int64)
    np.minimum.at(periodic_class, cls, np.arange(n_points))
    periodic_class = periodic_class[cls]

    # signature of every feature edge: its region set, imprinted edges marked
    sig_rows = edge_reg[feature[edge_reg[:, 0]]]
    width = int(n_reg[feature].max()) + 1 if feature.any() else 1
    table = np.full((len(edges), width), np.iinfo(np.int64).max, dtype=np.int64)
    rank = np.arange(len(sig_rows)) - np.searchsorted(sig_rows[:, 0], sig_rows[:, 0])
    table[sig_rows[:, 0], rank] = sig_rows[:, 1]
    table[imprinted, -1] = IMPRINT                                 # last column is always free
    table = np.sort(table, axis=1)
    feat_ids = np.flatnonzero(feature)
    signature = np.full(len(edges), -1, dtype=np.int64)
    if len(feat_ids):
        signature[feat_ids] = _group(*table[feat_ids].T)

    # anchors: a vertex with four or more regions is a regular point of a line of the same order
    # (four grains along one voxel edge), not an anchor, unless the line branches, ends or changes
    fe = edges[feat_ids]
    degree = np.bincount(fe.ravel(), minlength=n_points)
    anchor = (degree > 0) & (degree != 2)
    line_order = np.zeros(n_points, dtype=np.int64)
    if len(feat_ids):
        ends = np.concatenate([fe[:, 0], fe[:, 1]])
        sigs = np.concatenate([signature[feat_ids]] * 2)
        pairs = unique_pairs(ends, sigs)
        anchor |= np.bincount(pairs[:, 0], minlength=n_points) > 1
        real = (table < np.iinfo(np.int64).max) & (table != IMPRINT)
        np.maximum.at(line_order, ends, np.concatenate([real[feat_ids].sum(axis=1)] * 2))
    anchor |= (vertex_regions >= 4) & (vertex_regions > line_order)
    anchor = _close_over_images(anchor, periodic_class)

    lines, edge_line, ring_anchor = _trace_lines(edges, feat_ids, anchor, periodic_class, table, n_points)
    anchor |= ring_anchor

    # patches: triangles connected through edges that are not on a line
    inner = edge_line[edge_of] < 0
    a, b = _chain_pairs(np.where(inner, edge_of, -1 - np.arange(len(edge_of))))
    patch = _components(len(tris), t[a], t[b])
    patch = np.unique(patch, return_inverse=True)[1].ravel()
    patch_regions = np.zeros((patch.max() + 1, 2), dtype=np.int64)
    patch_regions[patch] = np.sort(mesh.grains, axis=1)

    return Topology(mesh=mesh, shape=shape, periodic=periodic, vertex_regions=vertex_regions,
                    edges=edges, edge_line=edge_line, anchors=np.flatnonzero(anchor),
                    ring_anchors=np.flatnonzero(ring_anchor), lines=lines, patch=patch,
                    patch_regions=patch_regions, periodic_class=periodic_class)


# --------------------------------------------------------------------------------------------- #

def _edge_index(edges: np.ndarray, lo: np.ndarray, hi: np.ndarray, n_points: int) -> np.ndarray:
    """Row of (min(lo, hi), max(lo, hi)) in the sorted edge array."""
    a, b = np.minimum(lo, hi).astype(np.int64), np.maximum(lo, hi).astype(np.int64)
    keys = edges[:, 0].astype(np.int64) * n_points + edges[:, 1]
    idx = np.searchsorted(keys, a * n_points + b)
    if np.any(idx >= len(keys)) or np.any(keys[np.minimum(idx, len(keys) - 1)] != a * n_points + b):
        raise ValueError('edge is not part of the mesh')
    return idx


def _close_over_images(flag: np.ndarray, periodic_class: np.ndarray) -> np.ndarray:
    hit = np.zeros(len(flag), dtype=bool)
    hit[periodic_class[flag]] = True
    return hit[periodic_class]


def _trace_lines(edges, feat_ids, anchor, periodic_class, table, n_points):
    """Walk the line graph from anchor to anchor; rings get an artificial anchor first."""
    edge_line = np.full(len(edges), -1, dtype=np.int64)
    if not len(feat_ids):
        return [], edge_line, np.zeros(n_points, dtype=bool)
    fe = edges[feat_ids]
    order = np.argsort(np.concatenate([fe[:, 0], fe[:, 1]]), kind='stable')
    nbr_vertex = np.concatenate([fe[:, 1], fe[:, 0]])[order]
    nbr_edge = np.concatenate([feat_ids, feat_ids])[order]
    start = np.searchsorted(np.concatenate([fe[:, 0], fe[:, 1]])[order], np.arange(n_points + 1))
    used = np.zeros(len(edges), dtype=bool)
    lines: List[Line] = []

    def walk(first, edge, nxt):
        chain, cur, e = [first], nxt, edge
        while True:
            used[e] = True
            edge_line[e] = len(lines)
            chain.append(cur)
            if anchor[cur]:
                break
            s, f = start[cur], start[cur + 1]
            follow = [(w, k) for w, k in zip(nbr_vertex[s:f], nbr_edge[s:f]) if not used[k]]
            if not follow:
                break
            cur, e = follow[0]
        regions = tuple(int(r) for r in table[edge] if r != np.iinfo(np.int64).max)
        lines.append(Line(vertices=np.array(chain), regions=regions))

    for vtx in np.flatnonzero(anchor):
        for w, k in zip(nbr_vertex[start[vtx]:start[vtx + 1]], nbr_edge[start[vtx]:start[vtx + 1]]):
            if not used[k]:
                walk(vtx, k, w)

    ring_anchor = np.zeros(n_points, dtype=bool)
    rest = feat_ids[~used[feat_ids]]
    if len(rest):
        comp = _components(n_points, edges[rest, 0], edges[rest, 1])
        on_ring = np.unique(edges[rest].ravel())
        chosen = {}
        for vtx in on_ring:
            c = comp[vtx]
            if c not in chosen or periodic_class[vtx] < periodic_class[chosen[c]]:
                chosen[c] = vtx
        for vtx in chosen.values():
            ring_anchor[vtx] = True
        anchor |= ring_anchor
        for vtx in sorted(chosen.values()):
            s, f = start[vtx], start[vtx + 1]
            for w, k in zip(nbr_vertex[s:f], nbr_edge[s:f]):
                if not used[k]:
                    walk(vtx, k, w)
                    break
    return lines, edge_line, ring_anchor


def _triangles_on_lines(topo: Topology) -> np.ndarray:
    tris = topo.mesh.triangles
    edge_of = _edge_index(topo.edges, tris.ravel(), tris[:, [1, 2, 0]].ravel(), len(topo.mesh.points))
    return np.flatnonzero(np.any((topo.edge_line[edge_of] >= 0).reshape(-1, 3), axis=1))
