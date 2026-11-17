"""Triangle mesh of a 2D RVE with smoothed grain boundaries (2D counterpart of mesh_rve, E2).

Steps:
    cleanup    a grain touching itself only diagonally (2x2 checkerboard a b / c a) gets the
               pixel b, so every grain is a face-connected body without pinch points
    curves     the pixel edges between different grains form polylines between anchors: nodes
               where three or more regions meet (the box sides count as regions), box corners and
               the ends of curves on the box; a closed curve without anchor gets one fixed node
    smoothing  interior curves: Taubin smoothing (anchors fixed), resampling to the boundary mesh
               size, area correction (normal offset per curve, least squares over all grains)
               and a crossing check; a curve that crosses another one falls back to fewer
               iterations, at worst to its staircase
    box sides  keep the pixel corners, unsmoothed, identical on opposite sides: periodic node
               pairs are exact (midside nodes included)
    gmsh       one plane surface per grain piece (outer loop and holes), every boundary segment a
               transfinite line with two nodes, so the mesh uses exactly these nodes; size grows
               from h at the boundaries to h_max inside (Distance/Threshold field)

Lengths in the units of `spacing` (micrometres in DRAGen).
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

BOX = {(0, 0): -1, (0, 1): -2, (1, 0): -3, (1, 1): -4}
"""Pseudo regions of the box sides: (axis, side) -> -1 x = 0, -2 x = L, -3 y = 0, -4 y = L."""


@dataclass(frozen=True)
class Mesh2DConfig:
    """order: 1 (3-node) or 2 (6-node) triangles. iterations: Taubin steps per curve (lambda 0.5,
    mu -0.53). h: boundary segment length, h_max: largest element size inside a grain, dist_max:
    distance over which the size grows from h to h_max, all in pixel edges. area_tolerance:
    relative grain area deviation the correction aims at."""

    order: int = 2
    iterations: int = 20
    h: float = 1.0
    h_max: float = 6.0
    dist_max: float = 6.0
    area_tolerance: float = 0.005
    cleanup: bool = True
    verbosity: int = 0


@dataclass
class Mesh2DResult:
    points: np.ndarray
    cells: np.ndarray
    grain: np.ndarray
    labels: np.ndarray
    curves: List[np.ndarray]
    stats: Dict = field(default_factory=dict)

    @property
    def cell_type(self) -> str:
        return 'triangle6' if self.cells.shape[1] == 6 else 'triangle'


# --------------------------------------------------------------------------- cleanup

def cleanup_pinches(labels: np.ndarray, periodic: Sequence[bool]) -> Tuple[np.ndarray, int]:
    """Remove diagonal-only self contacts: in a 2x2 block a b / c a (a != b, a != c) pixel b
    becomes a. Repeated until none is left. Returns the labels and the number of changed pixels."""
    labels = labels.copy()
    changed = 0
    for _ in range(100):
        p = _pad(labels, periodic)
        a, b = p[:-1, :-1], p[1:, :-1]
        c, d = p[:-1, 1:], p[1:, 1:]
        # diagonal a-d same, off-diagonal different from it; and the mirrored diagonal b-c
        pinch1 = (a == d) & (a != b) & (a != c) & (a > 0)
        pinch2 = (b == c) & (b != a) & (b != d) & (b > 0)
        if not (pinch1.any() or pinch2.any()):
            break
        n0, n1 = labels.shape
        for (i, j) in np.argwhere(pinch1):            # set pixel (i+1, j) (= b) to a
            labels[(i + 1) % n0 if periodic[0] else min(i + 1, n0 - 1), j % n1] = labels[i % n0, j % n1]
            changed += 1
            break
        else:
            for (i, j) in np.argwhere(pinch2):        # set pixel (i, j) (= a) to b
                labels[i % n0, j % n1] = labels[(i + 1) % n0 if periodic[0] else min(i + 1, n0 - 1), j % n1]
                changed += 1
                break
    return labels, changed


def _pad(labels, periodic):
    """Labels with one more row/column at the upper end: the wrapped one on periodic axes, 0
    (no grain) otherwise; the 2x2 windows of the result are the pixel corners inside the box."""
    out = labels
    for axis in range(2):
        extra = np.take(out, [0], axis=axis) if periodic[axis] else np.zeros_like(np.take(out, [0], axis=axis))
        out = np.concatenate([out, extra], axis=axis)
    return out


# --------------------------------------------------------------------------- curves

def boundary_segments(labels: np.ndarray):
    """Pixel edges between different regions, as (a, b, left, right) with node ids of the
    (nx+1, ny+1) corner grid (id = i*(ny+1) + j) and the region on each side; box sides are
    the pseudo regions of BOX. Every edge is oriented with a grain on its left; edges between
    two grains appear once."""
    nx, ny = labels.shape
    nid = lambda i, j: i * (ny + 1) + j                                   # noqa: E731
    segs = []
    # vertical edges x = i between pixels (i-1, j) and (i, j); direction +y has the left side at x < i
    for i in range(nx + 1):
        left = labels[i - 1] if i > 0 else np.full(ny, BOX[(0, 0)])
        right = labels[i] if i < nx else np.full(ny, BOX[(0, 1)])
        j = np.flatnonzero(left != right)
        segs.append(np.column_stack([nid(i, j), nid(i, j + 1), left[j], right[j]]))
    # horizontal edges y = j between pixels (i, j-1) and (i, j); direction +x has the left side at y > j
    for j in range(ny + 1):
        below = labels[:, j - 1] if j > 0 else np.full(nx, BOX[(1, 0)])
        above = labels[:, j] if j < ny else np.full(nx, BOX[(1, 1)])
        i = np.flatnonzero(below != above)
        segs.append(np.column_stack([nid(i, j), nid(i + 1, j), above[i], below[i]]))
    segs = np.concatenate(segs).astype(np.int64)
    flip = segs[:, 2] < 0                                                 # box on the left: reverse
    segs[flip] = segs[flip][:, [1, 0, 3, 2]]
    return segs


def trace_curves(segs: np.ndarray, n_nodes: int):
    """Chains of segments between anchors (nodes with a degree other than 2 or where the pair of
    regions changes). Returns a list of (node chain, left region, right region)."""
    deg = np.bincount(segs[:, :2].ravel(), minlength=n_nodes)
    # a node of degree 2 whose two segments separate different region pairs is an anchor too
    pair = np.sort(segs[:, 2:], axis=1)
    anchor = deg != 2
    by_node: Dict[int, List[int]] = {}
    for s, (a, b) in enumerate(segs[:, :2]):
        by_node.setdefault(int(a), []).append(s)
        by_node.setdefault(int(b), []).append(s)
    for node, ss in by_node.items():
        if len(ss) == 2 and not np.array_equal(pair[ss[0]], pair[ss[1]]):
            anchor[node] = True
    used = np.zeros(len(segs), dtype=bool)
    curves = []

    def walk(start_seg, start_node):
        chain, s, node = [start_node], start_seg, start_node
        while True:
            used[s] = True
            a, b = segs[s, :2]
            nxt = int(b if a == node else a)
            chain.append(nxt)
            if anchor[nxt] or nxt == start_node:
                return chain
            s = next(t for t in by_node[nxt] if not used[t])
            node = nxt

    for node in np.flatnonzero(anchor):
        for s in by_node.get(int(node), []):
            if not used[s]:
                chain = walk(s, int(node))
                curves.append(_oriented(chain, segs, s))
    for s in range(len(segs)):                         # closed curves without anchor
        if not used[s]:
            node = int(segs[s, 0])
            anchor[node] = True
            chain = walk(s, node)
            curves.append(_oriented(chain, segs, s))
    return curves, anchor


def _oriented(chain, segs, first_seg):
    """(chain, left, right) with the chain running in the direction of its first segment."""
    a, b, left, right = segs[first_seg]
    if chain[0] != a:
        chain = chain[::-1]
    return np.asarray(chain, dtype=np.int64), int(left), int(right)


# --------------------------------------------------------------------------- smoothing

def _taubin(p: np.ndarray, iterations: int, closed: bool) -> np.ndarray:
    if closed:                                            # chain repeats its first node at the end
        q = _taubin_open_or_ring(p[:-1].copy(), iterations, True)
        return np.vstack([q, q[:1]])
    return _taubin_open_or_ring(p.copy(), iterations, False)


def _taubin_open_or_ring(p: np.ndarray, iterations: int, closed: bool) -> np.ndarray:
    for _ in range(iterations):
        for f in (0.5, -0.53):
            if closed:
                lap = 0.5 * (np.roll(p, 1, 0) + np.roll(p, -1, 0)) - p
                lap[0] = 0.0                                  # the fixed node
            else:
                lap = np.zeros_like(p)
                lap[1:-1] = 0.5 * (p[:-2] + p[2:]) - p[1:-1]
            p += f * lap
    return p


def _resample(p: np.ndarray, h: float, closed: bool) -> np.ndarray:
    """Equal arc-length spacing, about h; end points kept."""
    q = np.vstack([p, p[:1]]) if closed and not np.allclose(p[0], p[-1]) else p
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    n = max(3 if closed else 1, int(round(s[-1] / h)))
    t = np.linspace(0.0, s[-1], n + 1)
    out = np.column_stack([np.interp(t, s, q[:, k]) for k in range(2)])
    out[0], out[-1] = q[0], q[-1]
    return out


def _segments_cross(curves_pts: List[np.ndarray]) -> np.ndarray:
    """Per curve: does one of its segments cross a segment of any curve (sharing no node)?"""
    starts, ends, owner = [], [], []
    for c, p in enumerate(curves_pts):
        starts.append(p[:-1])
        ends.append(p[1:])
        owner.append(np.full(len(p) - 1, c))
    a, b, owner = np.concatenate(starts), np.concatenate(ends), np.concatenate(owner)
    mid = 0.5 * (a + b)
    reach = np.linalg.norm(b - a, axis=1).max()
    pairs = cKDTree(mid).query_pairs(reach + 1e-12, output_type='ndarray')
    bad = np.zeros(len(curves_pts), dtype=bool)
    if not len(pairs):
        return bad
    i, j = pairs[:, 0], pairs[:, 1]
    p, r = a[i], b[i] - a[i]
    q, s = a[j], b[j] - a[j]
    shared = (np.isclose(a[i], a[j]).all(1) | np.isclose(a[i], b[j]).all(1) |
              np.isclose(b[i], a[j]).all(1) | np.isclose(b[i], b[j]).all(1))
    cross = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
    qp = q - p
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / cross
        u = (qp[:, 0] * r[:, 1] - qp[:, 1] * r[:, 0]) / cross
    eps = 1e-9
    hit = (~shared) & (np.abs(cross) > 1e-14) & (t > -eps) & (t < 1 + eps) & (u > -eps) & (u < 1 + eps)
    bad[owner[i[hit]]] = True
    bad[owner[j[hit]]] = True
    return bad


def _polygon_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _region_areas(curves_pts, curves, n_grains) -> np.ndarray:
    """Area per grain from the boundary curves (Green: sum over its oriented curves)."""
    area = np.zeros(n_grains + 1)
    for p, (_, left, right) in zip(curves_pts, curves):
        x, y = p[:, 0], p[:, 1]
        w = 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
        if left > 0:
            area[left] += w
        if right > 0:
            area[right] -= w
    return area


def _normals(p: np.ndarray) -> np.ndarray:
    """Left unit normals at the interior nodes (zero at the end points)."""
    n = np.zeros_like(p)
    t = p[2:] - p[:-2]
    n[1:-1] = np.column_stack([-t[:, 1], t[:, 0]])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)


def _correct_areas(curves_pts, curves, movable, target, iterations=4):
    """Shift every movable curve by delta_c along its left normal so that the grain areas match
    target (least norm solution of the linearised system)."""
    n_grains = len(target) - 1
    for _ in range(iterations):
        area = _region_areas(curves_pts, curves, n_grains)
        diff = target[1:] - area[1:]
        rows, cols, vals = [], [], []
        for c in np.flatnonzero(movable):
            p = curves_pts[c]
            # moving the curve to its left (into the left region) by delta shrinks the left
            # region by about delta * length and grows the right one
            length = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
            _, left, right = curves[c]
            if left > 0:
                rows.append(left - 1); cols.append(c); vals.append(-length)
            if right > 0:
                rows.append(right - 1); cols.append(c); vals.append(length)
        if not rows:
            return
        A = np.zeros((n_grains, len(curves_pts)))
        A[rows, cols] = vals
        delta = np.linalg.lstsq(A, diff, rcond=None)[0]
        for c in np.flatnonzero(movable):
            if delta[c]:
                # moving interior nodes changes the area by about delta * length * (n - 1) / n
                curves_pts[c] = curves_pts[c] + delta[c] * _normals(curves_pts[c])


# --------------------------------------------------------------------------- main

def mesh_rve_2d(labels: np.ndarray, spacing: Sequence[float], periodic: Sequence[bool],
                config: Mesh2DConfig = Mesh2DConfig()) -> Mesh2DResult:
    import gmsh

    start = time.perf_counter()
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError('mesh_rve_2d needs a 2D label field')
    if config.order not in (1, 2):
        raise ValueError(f'order must be 1 or 2, got {config.order}')
    spacing = np.asarray(spacing, dtype=float)
    stats = {}
    changed = 0
    if config.cleanup:
        labels, changed = cleanup_pinches(labels, periodic)
    stats['cleanup_changed_pixels'] = changed
    nx, ny = labels.shape
    n_grains = int(labels.max())
    target = np.r_[0.0, np.bincount(labels.ravel(), minlength=n_grains + 1)[1:] * float(np.prod(spacing))]
    corner = np.indices((nx + 1, ny + 1)).reshape(2, -1).T * spacing
    corner[:, 0] = np.where(np.indices((nx + 1, ny + 1))[0].ravel() == nx, nx * spacing[0], corner[:, 0])
    corner[:, 1] = np.where(np.indices((nx + 1, ny + 1))[1].ravel() == ny, ny * spacing[1], corner[:, 1])

    segs = boundary_segments(labels)
    curves, anchor = trace_curves(segs, corner.shape[0])
    h = config.h * float(spacing.min())
    on_box = np.array([min(l, r) < 0 for _, l, r in curves])
    closed = np.array([c[0] == c[-1] for c, _, _ in curves])
    raw = [corner[c] for c, _, _ in curves]

    def build(iterations):
        pts = []
        for k, p in enumerate(raw):
            if on_box[k]:
                pts.append(p.copy())                          # box sides: pixel corners
                continue
            q = _taubin(p, iterations[k], closed[k]) if iterations[k] else p
            pts.append(_resample(q, h, closed[k]) if iterations[k] else q.copy())
        return pts

    iterations = np.where(on_box, 0, config.iterations)
    pts = build(iterations)
    movable = ~on_box & (iterations > 0)
    _correct_areas(pts, curves, movable, target)
    fallbacks = 0
    for _ in range(12):
        bad = _segments_cross(pts) & ~on_box
        if not bad.any():
            break
        fallbacks += int(bad.sum())
        iterations = np.where(bad, iterations // 2, iterations)
        pts = build(iterations)
        movable = ~on_box & (iterations > 0)
        _correct_areas(pts, curves, movable, target)
    else:
        raise RuntimeError('grain boundary curves still cross after reducing the smoothing')
    area = _region_areas(pts, curves, n_grains)
    rel = np.abs(area[1:] - target[1:]) / target[1:]
    used_nodes = np.zeros(len(anchor), dtype=bool)
    used_nodes[segs[:, :2].ravel()] = True
    stats.update(curves=len(curves), anchors=int((anchor & used_nodes).sum()), smoothing_fallbacks=fallbacks,
                 area_deviation_max=float(rel.max()) if len(rel) else 0.0,
                 area_deviation_mean=float(rel.mean()) if len(rel) else 0.0)

    # ---------------------------------------------------------------- gmsh geometry
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber('General.Terminal', 1 if config.verbosity else 0)
        gmsh.option.setNumber('General.Verbosity', config.verbosity)
        gmsh.model.add('rve2d')
        geo = gmsh.model.geo
        point_tag: Dict[Tuple[int, int], int] = {}       # anchors by corner id, others new
        tag = [0]

        def add_point(xy, key=None):
            if key is not None and key in point_tag:
                return point_tag[key]
            tag[0] += 1
            geo.addPoint(float(xy[0]), float(xy[1]), 0.0, h, tag[0])
            if key is not None:
                point_tag[key] = tag[0]
            return tag[0]

        # every curve: list of gmsh line tags; segment k of a curve runs from node k to k+1
        line_tags: List[List[int]] = []
        line_count = 0
        for (chain, _, _), p, box in zip(curves, pts, on_box):
            ends = (('c', int(chain[0])), ('c', int(chain[-1])))
            ids = []
            for k, xy in enumerate(p):
                if k == 0:
                    ids.append(add_point(xy, ends[0]))
                elif k == len(p) - 1:
                    ids.append(add_point(xy, ends[1]))
                elif box:
                    ids.append(add_point(xy, ('c', int(chain[k]))))
                else:
                    ids.append(add_point(xy))
            tags = []
            for a, b in zip(ids[:-1], ids[1:]):
                line_count += 1
                geo.addLine(a, b, line_count)
                tags.append(line_count)
            line_tags.append(tags)

        # loops per grain: oriented line tags with the grain on the left
        surfaces_of: Dict[int, List[int]] = {}
        loop_id = surf_id = 0
        for g in range(1, n_grains + 1):
            loops = _grain_loops(g, curves, pts, line_tags)
            outers = [l for l in loops if l[1] > 0]
            holes = [l for l in loops if l[1] < 0]
            assign = {i: [] for i in range(len(outers))}
            for hole in holes:
                probe = hole[2][0]
                inside = [i for i, o in enumerate(outers) if _point_in_polygon(probe, o[2])]
                if not inside:
                    raise RuntimeError(f'grain {g}: hole loop outside every outer loop')
                best = min(inside, key=lambda i: outers[i][1])
                assign[best].append(hole)
            for i, outer in enumerate(outers):
                loop_id += 1
                geo.addCurveLoop(outer[0], loop_id)
                ring = [loop_id]
                for hole in assign[i]:
                    loop_id += 1
                    geo.addCurveLoop(hole[0], loop_id)
                    ring.append(loop_id)
                surf_id += 1
                geo.addPlaneSurface(ring, surf_id)
                surfaces_of.setdefault(g, []).append(surf_id)
        geo.synchronize()
        for t in range(1, line_count + 1):
            gmsh.model.mesh.setTransfiniteCurve(t, 2)
        for g, surfs in surfaces_of.items():
            gmsh.model.addPhysicalGroup(2, surfs, g)
        interior = [t for tags, box in zip(line_tags, on_box) if not box for t in tags]
        f_dist = gmsh.model.mesh.field.add('Distance')
        gmsh.model.mesh.field.setNumbers(f_dist, 'CurvesList', interior or [1])
        f_thr = gmsh.model.mesh.field.add('Threshold')
        gmsh.model.mesh.field.setNumber(f_thr, 'InField', f_dist)
        gmsh.model.mesh.field.setNumber(f_thr, 'SizeMin', h)
        gmsh.model.mesh.field.setNumber(f_thr, 'SizeMax', config.h_max * float(spacing.min()))
        gmsh.model.mesh.field.setNumber(f_thr, 'DistMin', 0.0)
        gmsh.model.mesh.field.setNumber(f_thr, 'DistMax', config.dist_max * float(spacing.min()))
        gmsh.model.mesh.field.setAsBackgroundMesh(f_thr)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
        gmsh.option.setNumber('Mesh.Algorithm', 6)
        gmsh.model.mesh.generate(2)
        if config.order == 2:
            gmsh.option.setNumber('Mesh.HighOrderOptimize', 0)
            gmsh.option.setNumber('Mesh.SecondOrderLinear', 1)
            gmsh.model.mesh.setOrder(2)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        index = np.full(int(node_tags.max()) + 1, -1, dtype=np.int64)
        index[node_tags.astype(np.int64)] = np.arange(len(node_tags))
        points = coords.reshape(-1, 3)[:, :2].copy()
        etype = 9 if config.order == 2 else 2
        cells, grain = [], []
        for g, surfs in surfaces_of.items():
            for s in surfs:
                types, _, nodes = gmsh.model.mesh.getElements(2, s)
                for t, n in zip(types, nodes):
                    if t == etype:
                        c = index[n.astype(np.int64)].reshape(-1, 6 if etype == 9 else 3)
                        cells.append(c)
                        grain.append(np.full(len(c), g, dtype=np.int32))
    finally:
        gmsh.finalize()
    cells = np.concatenate(cells)
    grain = np.concatenate(grain)
    # unused nodes (gmsh keeps geometry points) and orientation
    used = np.unique(cells)
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    points, cells = points[used], remap[cells]
    area2 = _signed_area2(points, cells)
    flip = area2 < 0
    if flip.any():
        cells[flip] = cells[flip][:, [0, 2, 1, 5, 4, 3]] if cells.shape[1] == 6 else cells[flip][:, [0, 2, 1]]
    stats.update(quality(points, cells, grain, target, spacing))
    stats.update(elements=int(len(cells)), nodes=int(len(points)),
                 seconds=round(time.perf_counter() - start, 3))
    log.info('2D mesh: %d triangles, %d nodes, min angle %.1f deg', len(cells), len(points), stats['min_angle_deg'])
    return Mesh2DResult(points=points, cells=cells, grain=grain, labels=labels,
                        curves=pts, stats=stats)


def _grain_loops(g, curves, pts, line_tags):
    """Closed loops of grain g: (signed line tags, signed area, polygon points). Segments with g
    on the left keep their direction, with g on the right are reversed; at a node with several
    continuations the one turning most to the left... see _next_segment."""
    segs = []                                # (start xy, end xy, signed tag)
    for (chain, left, right), p, tags in zip(curves, pts, line_tags):
        if left == g:
            segs += [(p[k], p[k + 1], tags[k]) for k in range(len(tags))]
        elif right == g:
            segs += [(p[k + 1], p[k], -tags[k]) for k in range(len(tags))]
    if not segs:
        return []
    key = lambda xy: (round(float(xy[0]), 9), round(float(xy[1]), 9))     # noqa: E731
    out_of: Dict[tuple, List[int]] = {}
    for i, (a, _, _) in enumerate(segs):
        out_of.setdefault(key(a), []).append(i)
    used = np.zeros(len(segs), dtype=bool)
    loops = []
    for first in range(len(segs)):
        if used[first]:
            continue
        ring, poly, i = [], [], first
        while not used[i]:
            used[i] = True
            a, b, t = segs[i]
            ring.append(int(t))
            poly.append(a)
            cands = [j for j in out_of.get(key(b), []) if not used[j]]
            if not cands:
                break
            # face on the left: the next segment is the first one clockwise from the way back
            back = np.arctan2(a[1] - b[1], a[0] - b[0])
            ang = [((back - np.arctan2(segs[j][1][1] - b[1], segs[j][1][0] - b[0])) % (2 * np.pi)) or 2 * np.pi
                   for j in cands]
            i = cands[int(np.argmin(ang))]
        poly = np.asarray(poly)
        loops.append((ring, _polygon_area(poly), poly))
    return loops


def _point_in_polygon(pt, poly) -> bool:
    x, y = pt
    inside = False
    xs, ys = poly[:, 0], poly[:, 1]
    j = len(poly) - 1
    for i in range(len(poly)):
        if (ys[i] > y) != (ys[j] > y) and x < (xs[j] - xs[i]) * (y - ys[i]) / (ys[j] - ys[i]) + xs[i]:
            inside = not inside
        j = i
    return inside


def _signed_area2(points, cells):
    a, b, c = points[cells[:, 0]], points[cells[:, 1]], points[cells[:, 2]]
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])


def quality(points, cells, grain, target_area, spacing) -> dict:
    """Smallest and largest interior angle, inverted triangles, grain areas against the pixels."""
    a, b, c = points[cells[:, 0]], points[cells[:, 1]], points[cells[:, 2]]

    def angle(p, q, r):
        u, v = q - p, r - p
        cos = np.einsum('ij,ij->i', u, v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))
    ang = np.column_stack([angle(a, b, c), angle(b, c, a), angle(c, a, b)])
    area = 0.5 * _signed_area2(points, cells)
    per_grain = np.bincount(grain, weights=area, minlength=len(target_area))
    mask = target_area > 0
    dev = np.abs(per_grain[mask] - target_area[mask]) / target_area[mask]
    return {'min_angle_deg': float(ang.min()), 'max_angle_deg': float(ang.max()),
            'angles_below_15': int(np.count_nonzero(ang.min(axis=1) < 15)),
            'inverted': int(np.count_nonzero(area <= 0)),
            'mesh_area_deviation_max': float(dev.max()) if len(dev) else 0.0,
            'total_area_error': float(abs(area.sum() - target_area.sum()) / target_area.sum())}
