"""Discrete surface model of the grain boundary network (phase M2).

Turns the topology of M1 into the entities a volume mesher and the smoothing need:

  points    the anchors
  curves    the lines (anchor to anchor), one per Topology.lines entry, same index
  surfaces  the patches, one per Topology patch, same index. A surface is oriented so that its
            normal points out of `regions[0]`, the larger region ID (always a grain; the other
            side is a grain or a box face, so box surfaces point out of the box). Its boundary is a
            list of loops; a loop is a cyclic list of curve uses (curve, +1 along the curve's vertex
            order, -1 against it) with the surface on the left. A surface without loops is closed
            (a grain inside another grain).
  shells    per grain the surfaces around it, with +1 where the surface normal points out of the
            grain and -1 where it points in.

Loops are found by walking the boundary half-edges of a surface and turning around each vertex
through the surface's own triangles, so a surface whose boundary touches itself at an anchor
(pinched patch, process notes 7.9) still gets clean loops.

Periodicity: entities lying in a periodic box face are grouped with their images on the
opposite face(s). One member of each group is the master (the one nearest the lower faces), the
others are slaves with an integer voxel translation (slave = master + translation * spacing),
which is what gmsh's setPeriodic expects.

Vertex constraints for the smoothing (author's rules, process notes section 2): anchors are fixed;
line vertices move along their line; the other vertices move in their surface; on top of that a
vertex on a box face keeps the coordinate normal to that face, so box-edge vertices only move along
the edge and box corners not at all. Periodic images move together (`periodic_class`, `shift`).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from dragen.mesh.quality import _components, unique_pairs
from dragen.mesh.surface import is_box_face
from dragen.mesh.topology import Topology, _edge_index

FIXED, ON_LINE, ON_SURFACE = 2, 1, 0


@dataclass
class Surface:
    regions: Tuple[int, int]
    """(outer, other): the normal points out of regions[0]."""
    triangles: np.ndarray
    """Indices into the mesh triangles."""
    loops: List[np.ndarray] = field(default_factory=list)
    """Each loop an (n, 2) array of (curve, sign)."""


@dataclass
class PeriodicLink:
    slave: int
    master: int
    translation: Tuple[int, int, int]
    """In voxels: slave = master + translation * spacing."""


@dataclass
class SurfaceModel:
    topology: Topology
    surfaces: List[Surface]
    shells: Dict[int, np.ndarray]
    """grain -> (k, 2) array of (surface, sign)."""
    oriented: np.ndarray
    """(F, 3) mesh triangles, each oriented like the surface it belongs to."""
    vertex_kind: np.ndarray
    """(P,) FIXED, ON_LINE or ON_SURFACE."""
    vertex_entity: np.ndarray
    """(P,) anchor index for FIXED, curve for ON_LINE, surface for ON_SURFACE."""
    box_lock: np.ndarray
    """(P, 3) True where a coordinate must not change (vertex on a box face)."""
    shift: np.ndarray
    """(P, 3) voxel offset of every vertex from its periodic class representative's position."""
    periodic_points: List[PeriodicLink]
    periodic_curves: List[PeriodicLink]
    periodic_surfaces: List[PeriodicLink]

    @property
    def mesh(self):
        return self.topology.mesh

    @property
    def curves(self):
        return self.topology.lines

    @property
    def points(self) -> np.ndarray:
        return self.topology.anchors

    # ------------------------------------------------------------------------------------- #

    def check(self) -> dict:
        """Consistency of the model; every count except the totals should be 0."""
        topo = self.topology
        uses = np.zeros(len(self.curves), dtype=np.int64)
        slit_curves = 0
        for surface in self.surfaces:
            seen = {}
            for loop in surface.loops:
                for c, sign in loop:
                    uses[c] += 1
                    seen[c] = seen.get(c, 0) + sign
            slit_curves += sum(1 for c, net in seen.items() if net == 0)
        unbalanced = 0
        for grain, shell in self.shells.items():
            net = {}
            for s, sign in shell:
                for loop in self.surfaces[s].loops:
                    for c, csign in loop:
                        net[c] = net.get(c, 0) + sign * csign
            unbalanced += sum(1 for v in net.values() if v != 0)
        n_loops = np.array([len(s.loops) for s in self.surfaces])
        chi = topo.patch_euler()
        pinched = set(_pinched(topo))
        genus = [(2 - n - x) / 2 for n, x, k in zip(n_loops, chi, range(len(chi))) if k not in pinched]
        return {
            'surfaces': len(self.surfaces),
            'curves': len(self.curves),
            'points': int(len(self.points)),
            'loops': int(n_loops.sum()),
            'surfaces_with_holes': int(np.count_nonzero(n_loops > 1)),
            'closed_surfaces': int(np.count_nonzero(n_loops == 0)),
            'surfaces_with_genus': int(sum(g > 0 for g in genus)),
            'periodic_points': len(self.periodic_points),
            'periodic_curves': len(self.periodic_curves),
            'periodic_surfaces': len(self.periodic_surfaces),
            # all of the following must be 0
            'unused_curves': int(np.count_nonzero(uses == 0)),
            'curves_used_once': int(np.count_nonzero(uses == 1)),
            'slit_curves': int(slit_curves),
            'unbalanced_shell_curves': int(unbalanced),
            'non_integer_genus': int(sum(g != int(g) or g < 0 for g in genus)),
            'misoriented_triangles': int(_misoriented(self)),
            'unmatched_periodic_surfaces': int(_unmatched_periodic_surfaces(self)),
            'inexact_periodic_links': int(_inexact_links(self)),
        }

    def to_pyvista(self):
        """MultiBlock with 'surfaces' (cell data surface, outer, other), 'curves' (cell data curve,
        order, imprinted) and 'points' (the anchors), for a look in ParaView."""
        import pyvista as pv
        pts = self.mesh.points
        tri = np.column_stack([np.full(len(self.oriented), 3), self.oriented]).ravel()
        surf = pv.PolyData(pts, faces=tri)
        surf.cell_data['surface'] = self.topology.patch
        regions = np.array([s.regions for s in self.surfaces])
        surf.cell_data['outer'] = regions[self.topology.patch, 0]
        surf.cell_data['other'] = regions[self.topology.patch, 1]
        cells, curve, order, imprinted = [], [], [], []
        for i, line in enumerate(self.curves):
            cells.append(np.concatenate([[len(line.vertices)], line.vertices]))
            curve.append(i)
            order.append(line.order)
            imprinted.append(line.imprinted)
        lines = pv.PolyData(pts, lines=np.concatenate(cells) if cells else None)
        if cells:
            lines.cell_data['curve'] = np.array(curve)
            lines.cell_data['order'] = np.array(order)
            lines.cell_data['imprinted'] = np.array(imprinted)
        return pv.MultiBlock({'surfaces': surf, 'curves': lines, 'points': pv.PolyData(pts[self.points])})


def build_model(topo: Topology) -> SurfaceModel:
    mesh = topo.mesh
    n_points = len(mesh.points)
    tris = mesh.triangles
    n_patch = len(topo.patch_regions)

    # orientation: normal out of the larger region of the patch
    outer = topo.patch_regions[:, 1]
    other = topo.patch_regions[:, 0]
    flip = mesh.grains[:, 0] != outer[topo.patch]
    oriented = np.where(flip[:, None], tris[:, ::-1], tris)

    # half-edges of the oriented triangles
    f = len(oriented)
    he_tri = np.repeat(np.arange(f), 3)
    he_from = oriented.ravel()
    he_to = oriented[:, [1, 2, 0]].ravel()
    he_patch = topo.patch[he_tri]
    he_edge = _edge_index(topo.edges, he_from, he_to, n_points)
    he_line = topo.edge_line[he_edge]
    on_line = he_line >= 0
    nxt = (np.arange(3 * f) // 3) * 3 + (np.arange(3 * f) % 3 + 1) % 3     # next half-edge in triangle
    # twin within the same patch: same patch and edge, other direction (edge index and a direction
    # bit keep the keys small enough for int64 even on very large meshes)
    direction = (he_from > he_to).astype(np.int64)
    key_rows, key_inv = unique_pairs(he_patch, 2 * he_edge + direction, return_inverse=True)
    twin_key = _lookup(key_rows, he_patch, 2 * he_edge + 1 - direction)
    lookup_he = np.full(len(key_rows), -1, dtype=np.int64)
    lookup_he[key_inv] = np.arange(3 * f)
    twin = np.where(twin_key >= 0, lookup_he[np.maximum(twin_key, 0)], -1)

    # successor of every boundary half-edge: turn around its end vertex through the patch
    boundary = np.flatnonzero(on_line)
    succ = nxt[boundary]
    pending = ~on_line[succ]
    for _ in range(64):
        if not pending.any():
            break
        t = twin[succ[pending]]
        if np.any(t < 0):
            raise ValueError('open surface: a patch edge is neither on a line nor shared')
        succ[pending] = nxt[t]
        pending = ~on_line[succ]
    else:
        raise ValueError(f'{np.count_nonzero(pending)} boundary half-edges find no successor')
    succ_of = np.full(3 * f, -1, dtype=np.int64)
    succ_of[boundary] = succ

    # direction of each line edge along its curve
    forward_from = np.full(len(topo.edges), -1, dtype=np.int64)
    if topo.lines:
        tails = np.concatenate([ln.vertices[:-1] for ln in topo.lines])
        heads = np.concatenate([ln.vertices[1:] for ln in topo.lines])
        forward_from[_edge_index(topo.edges, tails, heads, n_points)] = tails
    he_sign = np.where(he_from == forward_from[he_edge], 1, -1)

    surfaces = [Surface(regions=(int(outer[p]), int(other[p])), triangles=np.zeros(0, dtype=np.int64))
                for p in range(n_patch)]
    order = np.argsort(topo.patch, kind='stable')
    starts = np.searchsorted(topo.patch[order], np.arange(n_patch + 1))
    for p in range(n_patch):
        surfaces[p].triangles = order[starts[p]:starts[p + 1]]

    # loops: cycles of succ, cut into curve uses at anchors
    visited = np.zeros(3 * f, dtype=bool)
    anchor = np.zeros(n_points, dtype=bool)
    anchor[topo.anchors] = True
    for h0 in boundary:
        if visited[h0]:
            continue
        cycle = [h0]
        visited[h0] = True
        h = succ_of[h0]
        while h != h0:
            visited[h] = True
            cycle.append(h)
            h = succ_of[h]
        cycle = np.array(cycle)
        # rotate so the loop starts at a curve start (an anchor)
        starts_at_anchor = np.flatnonzero(anchor[he_from[cycle]])
        cycle = np.roll(cycle, -starts_at_anchor[0])
        new_use = anchor[he_from[cycle]]
        use_id = np.cumsum(new_use) - 1
        first = cycle[new_use]
        loop = np.column_stack([he_line[first], he_sign[first]])
        lengths = np.bincount(use_id)
        expected = np.array([len(topo.lines[c].vertices) - 1 for c in loop[:, 0]])
        if np.any(lengths != expected):
            raise ValueError('a loop does not run along whole curves')
        surfaces[he_patch[h0]].loops.append(loop)

    # shells
    shells: Dict[int, List[Tuple[int, int]]] = {}
    for p, s in enumerate(surfaces):
        shells.setdefault(s.regions[0], []).append((p, 1))
        if not is_box_face(s.regions[1]):
            shells.setdefault(s.regions[1], []).append((p, -1))
    shells = {g: np.array(v, dtype=np.int64) for g, v in sorted(shells.items())}

    # vertex constraints
    kind = np.full(n_points, ON_SURFACE, dtype=np.int64)
    entity = np.full(n_points, -1, dtype=np.int64)
    entity[oriented.ravel()] = np.repeat(topo.patch, 3)
    for i, line in enumerate(topo.lines):
        kind[line.vertices] = ON_LINE
        entity[line.vertices] = i
    kind[topo.anchors] = FIXED
    entity[topo.anchors] = np.arange(len(topo.anchors))
    shape = np.array(topo.shape)
    box_lock = (mesh.corner == 0) | (mesh.corner == shape)
    canonical = mesh.corner.copy()
    for a in range(3):
        if topo.periodic[a]:
            canonical[:, a] %= shape[a]
    shift = mesh.corner - canonical

    model = SurfaceModel(topology=topo, surfaces=surfaces, shells=shells, oriented=oriented,
                         vertex_kind=kind, vertex_entity=entity, box_lock=box_lock, shift=shift,
                         periodic_points=[], periodic_curves=[], periodic_surfaces=[])
    _periodic_links(model)
    return model


# --------------------------------------------------------------------------------------------- #

def _lookup(rows: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row index of (a, b) in the sorted unique rows, -1 if absent."""
    span = int(max(rows[:, 1].max(), b.max())) + 1
    keys = rows[:, 0] * span + rows[:, 1]
    query = np.asarray(a, dtype=np.int64) * span + b
    idx = np.searchsorted(keys, query)
    idx = np.minimum(idx, len(keys) - 1)
    return np.where(keys[idx] == query, idx, -1)


def _in_periodic_face(topo: Topology, vertices: np.ndarray) -> bool:
    c = topo.mesh.corner[vertices]
    for a in range(3):
        if topo.periodic[a] and (np.all(c[:, a] == 0) or np.all(c[:, a] == topo.shape[a])):
            return True
    return False


def _group_links(entities, model) -> List[PeriodicLink]:
    """entities: (id, vertices). Groups images by their periodic classes; the master is the image
    nearest the lower box faces, the translation the shift of the bounding box."""
    cls = model.topology.periodic_class
    corner = model.topology.mesh.corner
    families: Dict[tuple, list] = {}
    for eid, verts in entities:
        families.setdefault(tuple(np.unique(cls[verts])), []).append((eid, corner[verts].min(axis=0)))
    links = []
    for members in families.values():
        if len(members) < 2:
            continue
        m_id, m_low = min(members, key=lambda m: (int(m[1].sum()), tuple(m[1])))
        for eid, low in members:
            if eid != m_id:
                links.append(PeriodicLink(slave=int(eid), master=int(m_id),
                                          translation=tuple(int(t) for t in low - m_low)))
    return links


def _periodic_links(model: SurfaceModel) -> None:
    topo = model.topology
    if not any(topo.periodic):
        return
    on_face = model.box_lock & np.array(topo.periodic)
    face_vertex = on_face.any(axis=1)
    model.periodic_points = _group_links(
        [(i, np.array([v])) for i, v in enumerate(topo.anchors) if face_vertex[v]], model)
    model.periodic_curves = _group_links(
        [(i, ln.vertices) for i, ln in enumerate(topo.lines) if _in_periodic_face(topo, ln.vertices)], model)
    surfaces = []
    for p, s in enumerate(model.surfaces):
        verts = np.unique(model.oriented[s.triangles])
        if _in_periodic_face(topo, verts):
            surfaces.append((p, verts))
    model.periodic_surfaces = _group_links(surfaces, model)


def _pinched(topo: Topology) -> List[int]:
    _, pv = topo._patch_boundary()
    rows, uses = unique_pairs(pv[:, 0], pv[:, 1], return_counts=True)
    return list(np.unique(rows[uses > 2, 0]))


def _misoriented(model: SurfaceModel) -> int:
    """Triangle edges inside a surface that are not traversed in opposite directions."""
    tris = model.oriented
    topo = model.topology
    patch = np.repeat(topo.patch, 3)
    u, v = tris.ravel(), tris[:, [1, 2, 0]].ravel()
    edge = _edge_index(topo.edges, u, v, len(model.mesh.points))
    _, counts = unique_pairs(patch, 2 * edge + (u > v), return_counts=True)
    return int(np.count_nonzero(counts > 1))


def _unmatched_periodic_surfaces(model: SurfaceModel) -> int:
    topo = model.topology
    if not any(topo.periodic):
        return 0
    linked = {l.slave for l in model.periodic_surfaces} | {l.master for l in model.periodic_surfaces}
    count = 0
    for p, s in enumerate(model.surfaces):
        verts = np.unique(model.oriented[s.triangles])
        if _in_periodic_face(topo, verts) and p not in linked:
            count += 1
    return count


def _inexact_links(model: SurfaceModel) -> int:
    """Links whose slave is not exactly the translated master (vertex by vertex, via classes)."""
    topo = model.topology
    corner = topo.mesh.corner
    shape = np.array(topo.shape)

    def verts_of(kind, i):
        if kind == 'point':
            return np.array([topo.anchors[i]])
        if kind == 'curve':
            return topo.lines[i].vertices
        return np.unique(model.oriented[model.surfaces[i].triangles])

    bad = 0
    for kind, links in (('point', model.periodic_points), ('curve', model.periodic_curves),
                        ('surface', model.periodic_surfaces)):
        for link in links:
            sv, mv = verts_of(kind, link.slave), verts_of(kind, link.master)
            moved = {tuple(c) for c in corner[mv] + np.array(link.translation)}
            if moved != {tuple(c) for c in corner[sv]} or np.any(np.abs(link.translation) % shape != 0):
                bad += 1
    return bad
