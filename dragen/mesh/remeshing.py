"""Adaptive remeshing of the smoothed grain boundary network (phase M4a).

The smoothed mesh of M3 still has the voxel resolution everywhere and many poor triangles; tets
built on it contain slivers (process notes 7.12). This module rebuilds the triangulation with an
edge length that follows a size field: fine where a grain is thin, where a surface or a line is
strongly curved and on small grains, coarse on wide flat boundaries (author's decision, process
notes section 2).

The remeshing works on the one conforming mesh of all grain boundaries, so neighbouring grains keep
sharing their triangles. Per iteration (Botsch and Kobbelt, "A remeshing approach to multiresolution
modeling", 2004, with a size field):

  split     edges longer than 4/3 h get a midpoint
  collapse  edges shorter than 4/5 h are collapsed: a surface vertex into any neighbour of its
            surface, a line vertex only into its neighbour along the line; anchors never move
  flip      edges inside a surface are flipped where that raises the smallest angle
  relax     vertices move towards the 1/h-weighted centre of their neighbours, tangentially, and
            are projected back onto the smoothed surface of M3 (line vertices onto their line)

Each operation is carried out for an independent set of edges at once. An operation is refused
when it would change the topology (link condition), fold a triangle against its previous normal or
produce an edge longer than 4/3 h. Periodic images are treated as one: an operation on a vertex in a
periodic box face is done on all its images or on none, and the images keep exactly the same
coordinates in the face. Coordinates normal to a box face never change.

The size field comes from the smoothed mesh: h = min(d / 4 of the smaller adjacent grain,
thickness_fraction * local grain thickness, sqrt(8 chord_error / curvature)), clipped to
[h_min, h_max] and graded so that it grows by at most `gradation` per unit length. Lengths are in
voxel edges (smallest spacing).

Afterwards the grain volumes are corrected as in M3 (one offset per inner surface).
"""
from dataclasses import dataclass, field, replace
from typing import Dict

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree

from dragen.mesh.model import FIXED, ON_LINE, ON_SURFACE, Surface, SurfaceModel
from dragen.mesh.quality import self_intersection_pairs, unique_pairs
from dragen.mesh.surface import InterfaceMesh, is_box_face
from dragen.mesh.topology import Line, Topology


@dataclass
class RemeshConfig:
    h_min: float = 1.0
    """Smallest edge length in voxel edges (author's decision)."""
    h_max_fraction: float = 0.25
    """Largest edge length as a fraction of the equivalent diameter of the smaller adjacent grain."""
    thickness_fraction: float = 0.5
    chord_error: float = 0.1
    """Allowed distance between a straight edge and the curved surface, in voxel edges."""
    gradation: float = 0.4
    iterations: int = 16
    relax_steps: int = 2
    stop_fraction: float = 0.005
    """Stop early once an iteration splits and collapses fewer edges than this fraction of the
    triangles."""
    collapse_batches: int = 4
    """Collapse batches per iteration; one batch only collapses edges far enough apart."""
    min_cos: float = 0.2
    """A triangle changed by an operation must keep at least this cosine with its old normal."""
    min_angle: float = 10.0
    """A collapse or a move must not leave a triangle with a smallest angle (degrees) below this
    unless the triangle was worse before and does not get worse."""
    min_wedge: float = 10.0
    """Smallest angle (degrees) between neighbouring surfaces around a line edge, as in M3; an
    operation that closes a wedge further is refused."""
    normal_smoothing: int = 5
    """Laplacian passes over the vertex normals before the curvature is taken (voxel noise)."""
    volume_iterations: int = 3
    volume_tolerance: float = 0.01
    seed: int = 0


@dataclass
class RemeshResult:
    model: SurfaceModel
    points: np.ndarray
    stats: dict = field(default_factory=dict)


class Remesher:
    def __init__(self, model: SurfaceModel, points: np.ndarray, labels: np.ndarray, spacing,
                 target_volume: Dict[int, float], config: RemeshConfig = RemeshConfig()):
        self.model = model
        self.config = config
        self.spacing = np.asarray(spacing, dtype=float)
        self.h0 = float(self.spacing.min())
        self.box = np.array(model.topology.shape) * self.spacing
        self.target = target_volume
        self.rng = np.random.default_rng(config.seed)
        topo = model.topology
        # reference: the smoothed mesh of M3
        self.ref_x = np.asarray(points, dtype=float)
        self.ref_tris = model.oriented
        self.ref_patch = topo.patch
        self.ref_tree = cKDTree(self.ref_x[self.ref_tris].mean(axis=1))
        self.ref_normal = _normal(self.ref_x[self.ref_tris])
        self.ref_normal /= np.maximum(np.linalg.norm(self.ref_normal, axis=1, keepdims=True), 1e-300)
        seg, seg_curve = [], []
        for c, line in enumerate(topo.lines):
            v = line.vertices
            seg.append(np.column_stack([v[:-1], v[1:]]))
            seg_curve.append(np.full(len(v) - 1, c))
        self.ref_seg = np.concatenate(seg) if seg else np.zeros((0, 2), np.int64)
        self.ref_seg_curve = np.concatenate(seg_curve) if seg_curve else np.zeros(0, np.int64)
        self.seg_tree = cKDTree(self.ref_x[self.ref_seg].mean(axis=1)) if len(self.ref_seg) else None
        # arc length along every reference line: segment start parameters, per curve its length,
        # a global parameter (curve base + s) that increases along the concatenated segments
        n_curves = len(topo.lines)
        self.seg_len = np.linalg.norm(self.ref_x[self.ref_seg[:, 1]] - self.ref_x[self.ref_seg[:, 0]], axis=1)
        self.curve_len = np.bincount(self.ref_seg_curve, weights=self.seg_len, minlength=n_curves)
        self.curve_base = np.concatenate([[0.0], np.cumsum(self.curve_len)[:-1]]) + np.arange(n_curves)
        cum = np.cumsum(self.seg_len) - self.seg_len
        first_seg = np.searchsorted(self.ref_seg_curve, np.arange(n_curves))
        self.seg_first = first_seg
        self.seg_last = np.searchsorted(self.ref_seg_curve, np.arange(n_curves), side='right') - 1
        self.seg_s0 = cum - cum[first_seg[self.ref_seg_curve]]
        self.seg_g0 = self.curve_base[self.ref_seg_curve] + self.seg_s0
        self.curve_closed = np.array([ln.closed for ln in topo.lines], dtype=bool)
        self.regions = np.array([s.regions for s in model.surfaces])
        self.ref_h = self._size_field(labels)
        self.ref_vertex_tree = cKDTree(self.ref_x)

        # working mesh, starts as the reference
        self.x = self.ref_x.copy()
        self.tris = self.ref_tris.copy()
        self.patch = self.ref_patch.copy()
        self.tnorm = self.ref_normal.copy()
        """(F, 3) unit normal of the smoothed surface under every triangle, inherited through the
        operations; the fold guard compares against it."""
        self.kind = model.vertex_kind.copy()
        self.entity = model.vertex_entity.copy()
        self.lock = model.box_lock.copy()
        self.shift = model.shift.copy()
        self.cls = np.unique(topo.periodic_class, return_inverse=True)[1].ravel()
        self.cedge = np.sort(self.ref_seg, axis=1)
        self.ccurve = self.ref_seg_curve.copy()
        self.anchor = np.zeros(len(self.x), dtype=bool)
        self.anchor[topo.anchors] = True
        self.curve_start = np.array([ln.vertices[0] for ln in topo.lines], dtype=np.int64)
        self.s = np.full(len(self.x), np.nan)
        for c, line in enumerate(topo.lines):
            v = line.vertices
            seg = slice(self.seg_first[c], self.seg_last[c] + 1)
            self.s[v[1:-1]] = (self.seg_s0[seg] + self.seg_len[seg])[:-1]
        self.stats = {'refused': {'split': 0, 'collapse': 0, 'flip': 0}}

    # ------------------------------------------------------------------------------------- #
    # size field on the reference mesh

    def _size_field(self, labels: np.ndarray) -> np.ndarray:
        cfg, h0 = self.config, self.h0
        x, tris = self.ref_x, self.ref_tris
        n = len(x)
        # grain size: d / 4 of the smaller adjacent grain
        counts = np.bincount(labels.ravel())
        d = (6 * counts * np.prod(self.spacing) / np.pi) ** (1 / 3)
        reg = self.regions[self.ref_patch]
        d_tri = np.where(is_box_face(reg[:, 1]), d[reg[:, 0]],
                         np.minimum(d[reg[:, 0]], d[np.maximum(reg[:, 1], 0)]))
        h_max = np.full(n, np.inf)
        np.minimum.at(h_max, tris.ravel(), np.repeat(cfg.h_max_fraction * d_tri, 3))
        h_max = np.maximum(h_max, cfg.h_min * h0)
        h = h_max.copy()
        # thickness: distance to the nearest grain boundary voxel along the inward normals
        normals = self._vertex_normals(x, tris)
        h = np.minimum(h, cfg.thickness_fraction * self._thickness(labels, x, tris, normals, h_max))
        # curvature of the surfaces (smoothed normals) and of the lines
        kappa = self._surface_curvature(x, tris, normals)
        kappa = np.maximum(kappa, self._line_curvature(x))
        with np.errstate(divide='ignore'):
            h = np.minimum(h, np.sqrt(8 * cfg.chord_error * h0 / np.maximum(kappa, 1e-12)))
        h = np.clip(h, cfg.h_min * h0, h_max)
        # gradation: h(v) <= h(u) + g |uv|
        e = unique_pairs(np.minimum(tris.ravel(), tris[:, [1, 2, 0]].ravel()),
                         np.maximum(tris.ravel(), tris[:, [1, 2, 0]].ravel()))
        length = np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1)
        for _ in range(1000):
            new = h.copy()
            np.minimum.at(new, e[:, 0], h[e[:, 1]] + cfg.gradation * length)
            np.minimum.at(new, e[:, 1], h[e[:, 0]] + cfg.gradation * length)
            if np.array_equal(new, h):
                break
            h = new
        return h

    def _thickness(self, labels, x, tris, normals, h_max) -> np.ndarray:
        """Per vertex the smallest local thickness of the grains on both sides: 2 * the largest
        distance to a boundary voxel seen from the vertex into the grain, plus one voxel. The grain
        is probed along its inward normal at the vertex and along six directions tilted by 45
        degrees, so that a vertex next to a triple line still sees the bulk of its grain."""
        shape = np.array(labels.shape)
        periodic = self.model.topology.periodic
        boundary = np.zeros(labels.shape, dtype=bool)
        for a in range(3):
            for s in (1, -1):
                other = np.roll(labels, s, axis=a)
                if not periodic[a]:
                    idx = [slice(None)] * 3
                    idx[a] = 0 if s == 1 else -1
                    other[tuple(idx)] = labels[tuple(idx)]
                boundary |= other != labels
        reach = int(np.ceil(np.max(h_max) / self.spacing.min())) + 2
        pad = [(reach, reach) if periodic[a] else (0, 0) for a in range(3)]
        dist = ndimage.distance_transform_edt(~np.pad(boundary, pad, mode='wrap'), sampling=self.spacing)
        dist = dist[tuple(slice(p[0], p[0] + n) for p, n in zip(pad, labels.shape))]
        thick = np.full(len(x), np.inf)
        reg = self.regions[self.ref_patch]
        t = x[tris]
        fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        steps = np.arange(0.5, np.max(h_max) / self.h0 + 1, 0.5) * self.h0
        for side, sign in ((0, -1.0), (1, 1.0)):
            valid = ~is_box_face(reg[:, side])
            vert = tris[valid].ravel()
            grain = np.repeat(reg[valid, side], 3)
            vg, inv = unique_pairs(vert, grain, return_inverse=True)
            v, g = vg[:, 0], vg[:, 1]
            inward = np.zeros((len(vg), 3))
            np.add.at(inward, inv, sign * np.repeat(fn[valid], 3, axis=0))
            inward /= np.maximum(np.linalg.norm(inward, axis=1, keepdims=True), 1e-300)
            # two tangents and the probe directions
            helper = np.where(np.abs(inward[:, [0]]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
            t1 = np.cross(inward, helper)
            t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
            t2 = np.cross(inward, t1)
            directions = [inward] + [(inward + np.cos(phi) * t1 + np.sin(phi) * t2) / np.sqrt(2)
                                     for phi in np.arange(6) * np.pi / 3]
            best = np.zeros(len(v))
            for d in directions:
                inside = np.ones(len(v), dtype=bool)
                for step in steps:
                    p = x[v] + step * d
                    idx = np.floor(p / self.spacing).astype(np.int64)
                    for a in range(3):
                        idx[:, a] = idx[:, a] % shape[a] if periodic[a] else np.clip(idx[:, a], 0, shape[a] - 1)
                    inside &= labels[idx[:, 0], idx[:, 1], idx[:, 2]] == g
                    best = np.where(inside, np.maximum(best, dist[idx[:, 0], idx[:, 1], idx[:, 2]]), best)
            np.minimum.at(thick, v, 2 * best + self.h0)
        return thick

    def _surface_curvature(self, x, tris, normals) -> np.ndarray:
        """Largest |n_a - n_b| / |x_a - x_b| over the edges inside a surface, with the vertex normals
        averaged within each surface first so the voxel steps do not count as curvature."""
        patch = np.repeat(self.ref_patch, 3)
        u, v = tris.ravel(), tris[:, [1, 2, 0]].ravel()
        # normals per (vertex, patch): average over the patch's triangles only
        vp, inv = unique_pairs(u, patch, return_inverse=True)
        t = x[tris]
        fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        nrm = np.zeros((len(vp), 3))
        np.add.at(nrm, inv, np.repeat(fn, 3, axis=0))
        # neighbours within the patch
        key_u = inv
        key_v = _lookup_pairs(vp, v, patch)
        adj = sparse.csr_matrix((np.ones(len(key_u)), (key_u, key_v)), shape=(len(vp), len(vp)))
        adj = ((adj + adj.T) > 0).astype(float)
        deg = np.asarray(adj.sum(axis=1)).ravel() + 1
        for _ in range(self.config.normal_smoothing):
            nrm = (adj @ nrm + nrm) / deg[:, None]
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-300)
        a, b = key_u, key_v
        k = np.linalg.norm(nrm[a] - nrm[b], axis=1) / np.maximum(np.linalg.norm(x[u] - x[v], axis=1), 1e-300)
        kappa = np.zeros(len(x))
        np.maximum.at(kappa, u, k)
        return kappa

    def _line_curvature(self, x) -> np.ndarray:
        """Turning angle per unit length along every line, averaged over neighbouring vertices."""
        kappa = np.zeros(len(x))
        for line in self.model.topology.lines:
            v = line.vertices
            if len(v) < 3:
                continue
            p = x[v]
            t = np.diff(p, axis=0)
            length = np.linalg.norm(t, axis=1)
            t /= np.maximum(length[:, None], 1e-300)
            # tangents averaged over a few segments to remove the voxel steps
            for _ in range(self.config.normal_smoothing):
                t[1:-1] = (t[:-2] + 2 * t[1:-1] + t[2:]) / 4
                t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-300)
            turn = np.arccos(np.clip(np.einsum('ij,ij->i', t[:-1], t[1:]), -1, 1))
            k = turn / np.maximum((length[:-1] + length[1:]) / 2, 1e-300)
            np.maximum.at(kappa, v[1:-1], k)
        return kappa

    @staticmethod
    def _vertex_normals(x, tris) -> np.ndarray:
        t = x[tris]
        fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        n = np.zeros_like(x)
        np.add.at(n, tris.ravel(), np.repeat(fn, 3, axis=0))
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)

    # ------------------------------------------------------------------------------------- #

    def run(self) -> RemeshResult:
        cfg = self.config
        self.stats['triangles_before'] = int(len(self.tris))
        self.stats['passes'] = []
        for _ in range(cfg.iterations):
            rec = {'split': self._guarded(self._split), 'collapse': 0}
            for _ in range(cfg.collapse_batches):
                n = self._guarded(self._collapse)
                self._compact()
                rec['collapse'] += n
                if not n:
                    break
            rec['flip'] = self._guarded(self._flip)
            self._relax_round(cfg.relax_steps)
            rec['triangles'] = int(len(self.tris))
            self.stats['passes'].append(rec)
            if rec['split'] + rec['collapse'] < cfg.stop_fraction * len(self.tris):
                break
        self.stats['final_flips'] = self._guarded(self._flip)
        self._relax_round(1)
        self.stats['volume_before_correction'] = self._volume_error()
        x0 = self.x.copy()
        for _ in range(cfg.volume_iterations):
            self._correct_volume()
        self._undo_crossings(x0)
        self.stats['volume_final'] = self._volume_error()
        self.stats['triangles_after'] = int(len(self.tris))
        model = self._build_model()
        return RemeshResult(model=model, points=self.x.copy(), stats=self.stats)

    def _relax_round(self, steps):
        x0 = self.x.copy()
        for _ in range(steps):
            self._relax()
        self._undo_crossings(x0)

    def _undo_crossings(self, x0):
        """Vertices of triangles that cross after a series of moves go back to x0."""
        moved = np.any(self.x != x0, axis=1)
        candidates = np.flatnonzero(moved[self.tris].any(axis=1))
        for _ in range(16):
            back = self._crossing_vertices(self.x, candidates) & moved
            if not back.any():
                return
            back = self._close_images(back)
            self.x[back] = x0[back]
            moved &= ~back
            candidates = np.flatnonzero(back[self.tris].any(axis=1))
            # the reverted triangles and everything they may now cross
        self.stats['unresolved_crossings'] = int(len(self_intersection_pairs(self.x, self.tris, shared_vertex=True, touching=True)))

    def refine_near(self, centres: np.ndarray, iterations: int = 4) -> RemeshResult:
        """Halves the size field (down to h_min) within 2 h of the given points, e.g. the centres
        of poor tetrahedra, and continues remeshing for a few iterations."""
        self._scale_size_near(centres, 0.5, floor=self.config.h_min * self.h0)
        self.config = replace(self.config, iterations=iterations)
        return self.run()

    def coarsen_near(self, centres: np.ndarray, iterations: int = 4) -> RemeshResult:
        """Doubles the size field within 2 h of the given points, beyond h_max, and continues
        remeshing: vertices there are collapsed, so a feature finer than the mesh (a narrow gap
        or wedge that forces flat tetrahedra) is simplified away. The guards still hold, the
        grain geometry changes locally by a fraction of h (author's proposal, 29.10.2026)."""
        self._scale_size_near(centres, 2.0)
        self.config = replace(self.config, iterations=iterations)
        return self.run()

    def _scale_size_near(self, centres, factor, floor=0.0):
        """Scales the size field once at every reference vertex within 2 h of any centre (also
        where several centres are close: no factor**k for k clustered poor elements, O52)."""
        if not len(centres):
            return
        near = self.ref_vertex_tree.query_ball_point(centres, 2 * self.ref_h.max())
        sel = np.zeros(len(self.ref_h), dtype=bool)
        for c, idx in zip(centres, near):
            idx = np.asarray(idx, dtype=np.int64)
            sel[idx[np.linalg.norm(self.ref_x[idx] - c, axis=1) < 2 * self.ref_h[idx]]] = True
        self.ref_h[sel] = np.maximum(self.ref_h[sel] * factor, floor)

    _STATE = ('x', 's', 'curve_start', 'tris', 'patch', 'tnorm', 'kind', 'entity', 'lock', 'shift', 'cls', 'anchor', 'cedge', 'ccurve')

    def _guarded(self, op) -> int:
        """Runs a batch operation. Where it closes a wedge at a line edge below min_wedge or makes
        triangles cross, the operations touching the vertices involved are excluded and the batch
        is run again with the remaining operations of the previous try only, so it shrinks."""
        n_old = len(self.x)
        exclude = np.zeros(n_old, dtype=bool)
        self._only = None
        self._batch_seed = int(self.rng.integers(2 ** 62))
        for _ in range(16):
            saved = {k: getattr(self, k).copy() for k in self._STATE}
            self._parents = np.zeros((0, 2), dtype=np.int64)
            self._accepted = np.zeros(0, dtype=np.int64)
            self._changed = np.zeros(0, dtype=np.int64)
            n = op(exclude)
            self._only = self._accepted
            # wedges change only at edges of changed triangles; narrow wedges that were there
            # before do not count
            changed = np.zeros(len(self.tris), dtype=bool)
            changed[self._changed] = True
            bad = self._wedges_around(self.x, self.tris, changed)
            bad[:n_old] &= ~self._wedges_around(saved['x'], saved['tris'], bad[:n_old][saved['tris']].any(axis=1))
            bad |= self._crossing_vertices(self.x, self._changed)
            if not bad.any():
                return n
            ids = np.flatnonzero(bad)
            exclude[ids[ids < n_old]] = True
            exclude[self._parents[ids[ids >= n_old] - n_old].ravel()] = True
            for k, v in saved.items():
                setattr(self, k, v)
        self._only = None
        self.stats['refused']['wedge_batches'] = self.stats['refused'].get('wedge_batches', 0) + 1
        return 0

    def _in_box_plane(self, a, b) -> np.ndarray:
        """True where the edge (a, b) lies in a box face plane (both ends locked there)."""
        return np.any(self.lock[a] & self.lock[b] & (self.x[a] == self.x[b]), axis=1)

    def _crossing_vertices(self, x, candidates) -> np.ndarray:
        """Vertices of triangles that cross another one, among pairs involving the candidates."""
        out = np.zeros(len(x), dtype=bool)
        if len(candidates):
            pairs = self_intersection_pairs(x, self.tris, candidates=candidates, shared_vertex=True, touching=True)
            out[self.tris[pairs.ravel()].ravel()] = True
        return out

    def _wedges_around(self, x, tris, tmask) -> np.ndarray:
        """`_wedge_vertices` of the edges of the triangles in tmask, exactly: every triangle around
        such an edge shares a vertex with a triangle in tmask."""
        if not tmask.any():
            return np.zeros(len(x), dtype=bool)
        near = np.zeros(len(x), dtype=bool)
        near[tris[tmask].ravel()] = True
        return self._wedge_vertices(x, tris[near[tris].any(axis=1)])

    def _wedge_vertices(self, x, tris) -> np.ndarray:
        """Vertices of the triangles that bound a wedge narrower than min_wedge around an edge
        with three or more triangles (a line edge)."""
        if not len(tris):
            return np.zeros(len(x), dtype=bool)
        u, v, w =tris.ravel(), tris[:, [1, 2, 0]].ravel(), tris[:, [2, 0, 1]].ravel()
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        _, inv, cnt = unique_pairs(lo, hi, return_inverse=True, return_counts=True)
        sel = np.flatnonzero(cnt[inv] >= 3)
        out = np.zeros(len(x), dtype=bool)
        if not len(sel):
            return out
        a, b, c, k = lo[sel], hi[sel], w[sel], inv[sel]
        t = x[b] - x[a]
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-300)
        p = x[c] - x[a]
        p -= _dot(p, t)[:, None] * t
        order = np.lexsort((np.arange(len(k)), k))
        first = order[np.concatenate([[True], k[order[1:]] != k[order[:-1]]])]
        r = np.zeros((int(k.max()) + 1, 3))
        r[k[first]] = p[first]
        r = r[k] / np.maximum(np.linalg.norm(r[k], axis=1, keepdims=True), 1e-300)
        ang = np.degrees(np.arctan2(_dot(p, np.cross(t, r)), _dot(p, r))) % 360.0
        o = np.lexsort((ang, k))
        ks, angs, src = k[o], ang[o], sel[o] // 3
        same = ks[1:] == ks[:-1]
        narrow = same & (np.diff(angs) < self.config.min_wedge)
        starts = np.flatnonzero(np.concatenate([[True], ~same]))
        ends = np.concatenate([starts[1:] - 1, [len(ks) - 1]])
        cyclic = 360.0 - (angs[ends] - angs[starts]) < self.config.min_wedge
        tri_bad = np.concatenate([src[:-1][narrow], src[1:][narrow], src[starts[cyclic]], src[ends[cyclic]]])
        out[tris[tri_bad].ravel()] = True
        return out

    # ------------------------------------------------------------------------------------- #
    # mesh tables

    def _edges(self):
        """Unique edges (E, 2), edge of every half-edge (3F,), triangles per edge, curve per edge."""
        u, v = self.tris.ravel(), self.tris[:, [1, 2, 0]].ravel()
        edges, he_edge = unique_pairs(np.minimum(u, v), np.maximum(u, v), return_inverse=True)
        n_tri = np.bincount(he_edge, minlength=len(edges))
        curve = np.full(len(edges), -1, dtype=np.int64)
        if len(self.cedge):
            k = _find_rows(edges, self.cedge)
            if np.any(k < 0):
                raise RuntimeError('a line edge is missing from the mesh')
            curve[k] = self.ccurve
        return edges, he_edge, n_tri, curve

    def _h(self, x=None) -> np.ndarray:
        if x is not None:
            return self.ref_h[self.ref_vertex_tree.query(x)[1]]
        # nearest reference vertex of the current vertices, looked up again only where they moved
        x = self.x
        old_x, old_idx = getattr(self, '_nearest', (np.zeros((0, 3)), np.zeros(0, dtype=np.int64)))
        m = min(len(old_x), len(x))
        stale = np.ones(len(x), dtype=bool)
        stale[:m] = np.any(old_x[:m] != x[:m], axis=1)
        idx = np.empty(len(x), dtype=np.int64)
        idx[:m] = old_idx[:m]
        if stale.any():
            idx[stale] = self.ref_vertex_tree.query(x[stale])[1]
        self._nearest = (x.copy(), idx)
        return self.ref_h[idx]

    def _adjacency(self, edges, n):
        a = sparse.csr_matrix((np.ones(2 * len(edges), dtype=np.int32),
                               (np.concatenate([edges[:, 0], edges[:, 1]]),
                                np.concatenate([edges[:, 1], edges[:, 0]]))), shape=(n, n))
        a.data[:] = 1
        return a

    def _vertex_triangles(self):
        n, f = len(self.x), len(self.tris)
        return sparse.csr_matrix((np.ones(3 * f, dtype=np.int8),
                                  (self.tris.ravel(), np.repeat(np.arange(f), 3))), shape=(n, f))

    def _group_key(self, a, b):
        """Same key for all periodic images of the edge (a, b)."""
        ca, cb = self.cls[a], self.cls[b]
        return np.minimum(ca, cb) * (int(self.cls.max()) + 1) + np.maximum(ca, cb)

    def _priority(self, key):
        """Pseudo-random priority, equal for equal keys, distinct for different keys, and in the
        same order for the same keys throughout one guarded batch."""
        mixed = (key.astype(np.uint64) * np.uint64(11400714819323198485) + np.uint64(self._batch_seed)) >> np.uint64(1)
        order = np.lexsort((key, mixed))
        rank = np.empty(len(key), dtype=np.int64)
        new_group = np.concatenate([[True], np.diff(key[order]) != 0]) if len(key) else np.zeros(0, bool)
        rank[order] = np.cumsum(new_group)
        return rank

    def _accept_groups(self, key, ok, a, b):
        """True where every candidate with the same key is ok and the group holds all periodic
        images of the edge (a, b): as many as the smaller periodic class of its two vertices."""
        uniq, inv = np.unique(key, return_inverse=True)
        inv = inv.ravel()
        bad = np.bincount(inv, weights=~ok, minlength=len(uniq)) > 0
        size = np.bincount(self.cls)
        complete = np.bincount(inv, minlength=len(uniq))[inv] == np.minimum(size[self.cls[a]], size[self.cls[b]])
        ok = ok & ~bad[inv] & complete
        if getattr(self, '_only', None) is not None:
            ok &= np.isin(key, self._only)
        self._accepted = np.unique(key[ok])
        return ok

    # ------------------------------------------------------------------------------------- #
    # operations

    def _split(self, exclude) -> int:
        edges, he_edge, n_tri, curve = self._edges()
        h = self._h()
        length = np.linalg.norm(self.x[edges[:, 0]] - self.x[edges[:, 1]], axis=1)
        cand = np.flatnonzero(length > 4 / 3 * (h[edges[:, 0]] + h[edges[:, 1]]) / 2)
        if not len(cand):
            return 0
        key = self._group_key(edges[cand, 0], edges[cand, 1])
        prio = np.zeros(len(edges), dtype=np.int64)
        prio[cand] = self._priority(key)
        # independent: at most one split edge per triangle
        ok = ~exclude[edges[cand, 0]] & ~exclude[edges[cand, 1]]
        index = np.full(len(edges), -1, dtype=np.int64)
        index[cand] = np.arange(len(cand))
        he_cand = index[he_edge]
        has = he_cand >= 0
        ok = _independent_set(ok, prio[cand], he_cand[has], np.repeat(np.arange(len(self.tris)), 3)[has],
                              len(self.tris))
        ok = self._accept_groups(key, ok, edges[cand, 0], edges[cand, 1])
        self.stats['refused']['split'] += int(np.count_nonzero(~ok))
        sel = cand[ok]
        if not len(sel):
            return 0
        a, b = edges[sel, 0], edges[sel, 1]
        m = len(self.x) + np.arange(len(sel))
        self._parents = np.column_stack([a, b])
        # new vertices
        mid = (self.x[a] + self.x[b]) / 2
        lock = self.lock[a] & self.lock[b]
        on_curve = curve[sel] >= 0
        s_mid = np.full(len(sel), np.nan)
        if on_curve.any():
            ca, cb, cc = a[on_curve], b[on_curve], curve[sel][on_curve]
            sa = self._param(ca, cc, self.s[cb])
            sb = self._param(cb, cc, sa)
            sa = self._param(ca, cc, sb)                               # both anchors of a closed curve
            both = self.anchor[ca] & self.anchor[cb]
            sa = np.where(both, 0.0, sa)
            sb = np.where(both, self.curve_len[cc], sb)
            s_mid[on_curve] = (sa + sb) / 2
            mid[on_curve] = self._curve_point(cc, s_mid[on_curve])
        mid = np.where(lock, self.x[a], mid)                       # exactly in the box face
        kind = np.where(on_curve, ON_LINE, ON_SURFACE)
        tri_of = np.zeros(len(edges), dtype=np.int64)
        tri_of[he_edge] = np.repeat(np.arange(len(self.tris)), 3)
        entity = np.where(on_curve, curve[sel], self.patch[tri_of[sel]])
        group = np.unique(key[ok], return_inverse=True)[1].ravel()
        cls = int(self.cls.max()) + 1 + group
        shift = np.where(lock, self.shift[a], 0)
        self.x = np.vstack([self.x, mid])
        self.s = np.concatenate([self.s, s_mid])
        self.kind = np.concatenate([self.kind, kind])
        self.entity = np.concatenate([self.entity, entity])
        self.lock = np.vstack([self.lock, lock])
        self.shift = np.vstack([self.shift, shift])
        self.cls = np.concatenate([self.cls, cls])
        self.anchor = np.concatenate([self.anchor, np.zeros(len(sel), dtype=bool)])
        # triangles: (v0, v1, v2) with the split edge v_i -> v_i+1 becomes (v_i, m, v_i+2), (m, v_i+1, v_i+2)
        new_of_edge = np.full(len(edges), -1, dtype=np.int64)
        new_of_edge[sel] = m
        he_new = new_of_edge[he_edge].reshape(-1, 3)
        t_idx, local = np.nonzero(he_new >= 0)
        tri = self.tris[t_idx]
        vi, vj, vk = (tri[np.arange(len(t_idx)), (local + s) % 3] for s in range(3))
        mm = he_new[t_idx, local]
        self.tris[t_idx] = np.column_stack([vi, mm, vk])
        self._changed = np.concatenate([t_idx, len(self.tris) + np.arange(len(t_idx))])
        self.tris = np.vstack([self.tris, np.column_stack([mm, vj, vk])])
        self.patch = np.concatenate([self.patch, self.patch[t_idx]])
        self.tnorm = np.vstack([self.tnorm, self.tnorm[t_idx]])
        # line edges
        if on_curve.any():
            sc = sel[on_curve]
            rows = _find_rows(self.cedge, edges[sc])
            keep = np.ones(len(self.cedge), dtype=bool)
            keep[rows] = False
            mc = new_of_edge[sc]
            add = np.vstack([np.column_stack([edges[sc, 0], mc]), np.column_stack([mc, edges[sc, 1]])])
            self.cedge = np.vstack([self.cedge[keep], np.sort(add, axis=1)])
            self.ccurve = np.concatenate([self.ccurve[keep], curve[sc], curve[sc]])
        return int(len(sel))

    def _collapse(self, exclude) -> int:
        cfg = self.config
        edges, he_edge, n_tri, curve = self._edges()
        h = self._h()
        x = self.x
        length = np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], axis=1)
        h_edge = (h[edges[:, 0]] + h[edges[:, 1]]) / 2
        short = np.flatnonzero(length < 4 / 5 * h_edge)
        # surface vertices with only three neighbours go regardless of length: they make poor
        # triangles and block the collapse of the edge they sit next to
        degree = np.bincount(edges.ravel(), minlength=len(x))
        val3 = (degree == 3) & (self.kind == ON_SURFACE)
        v3 = np.flatnonzero(val3[edges[:, 0]] | val3[edges[:, 1]])
        short = np.union1d(short, v3)
        if not len(short):
            return 0
        # both directions, then the allowed ones: a disappears into b
        a = np.concatenate([edges[short, 0], edges[short, 1]])
        b = np.concatenate([edges[short, 1], edges[short, 0]])
        e = np.concatenate([short, short])
        keep = (length[e] < 4 / 5 * h_edge[e]) | val3[a]
        a, b, e = a[keep], b[keep], e[keep]
        kind_a = self.kind[a]
        allowed = (kind_a != FIXED) & ~self.anchor[a]
        allowed &= np.where(kind_a == ON_LINE, curve[e] == self.entity[a], curve[e] < 0)
        allowed &= np.all(~self.lock[a] | self.lock[b], axis=1)
        # a closed line keeps at least three edges
        edges_per_curve = np.bincount(self.ccurve, minlength=len(self.model.curves))
        closed = np.append(self.curve_closed, False)
        allowed &= ~((kind_a == ON_LINE) & closed[np.where(kind_a == ON_LINE, self.entity[a], -1)]
                     & (edges_per_curve[np.where(kind_a == ON_LINE, self.entity[a], 0)] <= 3))
        # prefer moving surface vertices onto lines; otherwise the direction from the lower periodic
        # class, so that all images of an edge collapse the same way
        a, b, e = a[allowed], b[allowed], e[allowed]
        if not len(a):
            return 0
        rank = (~val3[a]) * 4 + np.where(self.kind[a] == ON_SURFACE, 0, 1) * 2 + (self.cls[a] > self.cls[b])
        order = np.lexsort((rank, e))
        first = np.concatenate([[True], e[order[1:]] != e[order[:-1]]])
        pick = order[first]
        a, b, e = a[pick], b[pick], e[pick]
        if self._only is not None:
            # a repeated batch: only the collapses accepted in the previous try are evaluated
            pick = np.flatnonzero(np.isin(self._group_key(a, b), self._only))
            a, b, e = a[pick], b[pick], e[pick]
            if not len(a):
                return 0

        # link condition: common neighbours of a and b = opposite vertices of the edge's triangles
        adj = self._adjacency(edges, len(x))
        common = np.asarray(adj[a].multiply(adj[b]).sum(axis=1)).ravel()
        ok = common == n_tri[e]
        # a line vertex: its other line edge (a, c) becomes (b, c), which must not exist yet (two
        # lines between the same anchors would otherwise merge into one edge)
        line_a = np.flatnonzero(self.kind[a] == ON_LINE)
        if len(line_a):
            rows_a = np.concatenate([self.cedge[:, 0], self.cedge[:, 1]])
            rows_c = np.concatenate([self.cedge[:, 1], self.cedge[:, 0]])
            order_ = np.argsort(rows_a, kind='stable')
            lo_ = np.searchsorted(rows_a[order_], a[line_a], side='left')
            hi_ = np.searchsorted(rows_a[order_], a[line_a], side='right')
            for k in range(2):                                    # a line vertex has two line edges
                pos = np.minimum(lo_ + k, len(order_) - 1)
                has = lo_ + k < hi_
                c = rows_c[order_[pos]]
                other = has & (c != b[line_a])
                exists = _find_rows(edges, np.sort(np.column_stack([b[line_a], c]), axis=1)) >= 0
                ok[line_a[other & exists]] = False
        # exclusions first: a candidate touching an excluded vertex is not evaluated further
        vt = self._vertex_triangles()
        rows = vt[a]
        all_c = np.repeat(np.arange(len(a)), np.diff(rows.indptr))
        pv = np.repeat(all_c, 3)
        vv = self.tris[rows.indices].ravel()
        ok &= ~(np.bincount(pv, weights=exclude[vv], minlength=len(a)) > 0)
        # triangles around a that survive, with a replaced by b; checked cheapest first, and a
        # candidate is dropped with all its triangles as soon as one of them fails
        cand_idx, t = all_c, rows.indices
        tri = self.tris[t]
        sel = ok[cand_idx] & ~np.any(tri == b[cand_idx, None], axis=1)
        cand_idx, t, tri = cand_idx[sel], t[sel], tri[sel]
        new = np.where(tri == a[cand_idx, None], b[cand_idx, None], tri)

        def keep(fail):
            nonlocal cand_idx, t, tri, new
            ok[cand_idx[fail]] = False
            k = ok[cand_idx]
            cand_idx, t, tri, new = cand_idx[k], t[k], tri[k], new[k]

        # tiny edges (below h / 4) may leave edges up to 2 h behind
        tiny = (np.linalg.norm(x[a] - x[b], axis=1) < h_edge[e] / 4) | val3[a]
        p_new = x[new]
        hv = h[new]
        el = np.linalg.norm(p_new - p_new[:, [1, 2, 0]], axis=2)
        factor = np.where(tiny[cand_idx], 2.0, 4 / 3)[:, None]
        keep(~np.all(el <= factor * (hv + hv[:, [1, 2, 0]]) / 2 + 1e-12, axis=1))
        p_old, p_new = x[tri], x[new]
        n_old = np.cross(p_old[:, 1] - p_old[:, 0], p_old[:, 2] - p_old[:, 0])
        n_new = np.cross(p_new[:, 1] - p_new[:, 0], p_new[:, 2] - p_new[:, 0])
        good = _dot(n_new, n_old) > cfg.min_cos * np.linalg.norm(n_new, axis=1) * np.linalg.norm(n_old, axis=1)
        good[good] = ~self._against_reference(p_new[good], self.tnorm[t[good]])
        # smallest angle: at least min_angle, or not smaller than before
        g = np.flatnonzero(good)
        ang_new = _min_angle(p_new[g])
        low = ang_new < cfg.min_angle
        good[g[low]] = ang_new[low] >= _min_angle(p_old[g[low]])
        keep(~good)
        # a triangle of a grain boundary surface must not get an edge inside a box face plane
        # that is not a line edge (it would cut through the box face triangles)
        inner_surface = ~is_box_face(self.regions[self.patch[t], 1])
        if inner_surface.any():
            good = np.ones(len(t), dtype=bool)
            bb = b[cand_idx]
            for k in range(3):
                w = new[:, k]
                rows_ = np.flatnonzero(good & inner_surface & (w != bb))
                plane = rows_[self._in_box_plane(bb[rows_], w[rows_])]
                if len(plane):
                    line = (_find_rows(self.cedge, np.sort(np.column_stack([bb[plane], w[plane]]), axis=1)) >= 0)                         | (_find_rows(self.cedge, np.sort(np.column_stack([a[cand_idx[plane]], w[plane]]), axis=1)) >= 0)
                    good[plane[~line]] = False
            keep(~good)
        # a new triangle must not exist already (a closed surface of four vertices would fold
        # onto itself; the link condition alone does not catch it)
        around_b = np.unique(vt[np.unique(b[cand_idx])].indices)
        keep(_rows_in(np.sort(new, axis=1), np.sort(self.tris[around_b], axis=1)))
        # independent set: the vertices of the triangles around a
        key = self._group_key(a, b)
        prio = self._priority(key)
        ok = _independent_set(ok, prio, pv, vv, len(x))
        ok = self._accept_groups(key, ok, a, b)
        self.stats['refused']['collapse'] += int(np.count_nonzero(~ok))
        a, b = a[ok], b[ok]
        if not len(a):
            return 0
        to = np.arange(len(x), dtype=np.int64)
        to[a] = b
        touched = np.any(np.isin(self.tris, a), axis=1)
        self.tris = to[self.tris]
        alive = (self.tris[:, 0] != self.tris[:, 1]) & (self.tris[:, 1] != self.tris[:, 2]) \
            & (self.tris[:, 0] != self.tris[:, 2])
        self.tris, self.patch, self.tnorm = self.tris[alive], self.patch[alive], self.tnorm[alive]
        self._changed = np.flatnonzero(touched[alive])
        ce = to[self.cedge]
        keep = ce[:, 0] != ce[:, 1]
        self.cedge, self.ccurve = np.sort(ce[keep], axis=1), self.ccurve[keep]
        return int(len(a))

    def _flip(self, exclude) -> int:
        cfg = self.config
        edges, he_edge, n_tri, curve = self._edges()
        inner = np.flatnonzero((n_tri == 2) & (curve < 0))
        if not len(inner):
            return 0
        f = len(self.tris)
        he_tri = np.repeat(np.arange(f), 3)
        he_local = np.tile(np.arange(3), f)
        order = np.argsort(he_edge, kind='stable')
        starts = np.searchsorted(he_edge[order], np.arange(len(edges)))
        h1, h2 = order[starts[inner]], order[starts[inner] + 1]
        t1, t2 = he_tri[h1], he_tri[h2]
        tri1, tri2 = self.tris[t1], self.tris[t2]
        l1 = he_local[h1]
        a = tri1[np.arange(len(inner)), l1]
        b = tri1[np.arange(len(inner)), (l1 + 1) % 3]
        c = tri1[np.arange(len(inner)), (l1 + 2) % 3]
        d = tri2[np.arange(len(inner)), (he_local[h2] + 2) % 3]
        if self._only is not None:
            # a repeated batch: only the flips accepted in the previous try are evaluated
            sel = np.isin(self._group_key(a, b), self._only)
            inner, t1, t2, tri1, tri2, a, b, c, d = (v[sel] for v in (inner, t1, t2, tri1, tri2, a, b, c, d))
            if not len(inner):
                return 0
        x = self.x
        old = np.stack([x[tri1], x[tri2]], axis=1)                 # (k, 2, 3, 3)
        new1, new2 = np.column_stack([a, d, c]), np.column_stack([d, b, c])
        new = np.stack([x[new1], x[new2]], axis=1)
        ang_old = np.minimum(_min_angle(old[:, 0]), _min_angle(old[:, 1]))
        ang_new = np.minimum(_min_angle(new[:, 0]), _min_angle(new[:, 1]))
        ok = (ang_new > ang_old + 1e-3) & ~(exclude[a] | exclude[b] | exclude[c] | exclude[d])
        n_old = _normal(old[:, 0]) + _normal(old[:, 1])
        for k in range(2):
            nn = _normal(new[:, k])
            ok &= _dot(nn, n_old) > cfg.min_cos * np.linalg.norm(nn, axis=1) * np.linalg.norm(n_old, axis=1)
        ok &= _find_rows(edges, np.sort(np.column_stack([c, d]), axis=1)) < 0
        ok &= ~(self._in_box_plane(c, d) & ~is_box_face(self.regions[self.patch[t1], 1]))
        n_ref = self.tnorm[t1] + self.tnorm[t2]
        n_ref /= np.maximum(np.linalg.norm(n_ref, axis=1, keepdims=True), 1e-300)
        for k in range(2):
            ok[ok] = ~self._against_reference(new[ok, k], n_ref[ok])
        key = self._group_key(a, b)
        prio = self._priority(key)
        # resources: both triangles, and the new edge (two flips in different quads must not
        # create the same edge)
        cd = np.sort(np.column_stack([c, d]), axis=1)
        cd_key = unique_pairs(cd[:, 0], cd[:, 1], return_inverse=True)[1]
        idx = np.arange(len(inner))
        ok = _independent_set(ok, prio, np.concatenate([idx, idx, idx]),
                              np.concatenate([t1, t2, f + cd_key]), f + int(cd_key.max()) + 1)
        ok = self._accept_groups(key, ok, a, b)
        self.stats['refused']['flip'] += int(np.count_nonzero(~ok))
        self._changed = np.concatenate([t1[ok], t2[ok]])
        self.tris[t1[ok]] = new1[ok]
        self.tris[t2[ok]] = new2[ok]
        self.tnorm[t1[ok]] = n_ref[ok]
        self.tnorm[t2[ok]] = n_ref[ok]
        return int(np.count_nonzero(ok))

    def _relax(self) -> None:
        cfg = self.config
        edges, he_edge, n_tri, curve = self._edges()
        x, n = self.x, len(self.x)
        h = self._h()
        w = 1.0 / h
        normals = self._vertex_normals(x, self.tris)
        # surface vertices: all neighbours; line vertices: neighbours along their line
        on_line_edge = curve >= 0
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        line = np.concatenate([on_line_edge, on_line_edge])
        use = np.where(self.kind[rows] == ON_LINE, line, self.kind[rows] == ON_SURFACE)
        rows, cols = rows[use], cols[use]
        wsum = np.bincount(rows, weights=w[cols], minlength=n)
        target = np.stack([np.bincount(rows, weights=w[cols] * x[cols, k], minlength=n) for k in range(3)], axis=1)
        moving = (wsum > 0) & (self.kind != FIXED)
        target[moving] /= wsum[moving, None]
        step = np.zeros_like(x)
        step[moving] = 0.5 * (target[moving] - x[moving])
        surf = moving & (self.kind == ON_SURFACE)
        step[surf] -= _dot(step[surf], normals[surf])[:, None] * normals[surf]
        new = x + step
        new[surf] = self._project_surface(new[surf], self.entity[surf])
        # line vertices move along their reference line in arc length, towards the 1/h-weighted
        # mean parameter of their two line neighbours; the order along the line is kept
        lin = np.flatnonzero(moving & (self.kind == ON_LINE))
        s_new = self.s.copy()
        if len(lin):
            le = edges[curve >= 0]
            lc = curve[curve >= 0]
            v = np.concatenate([le[:, 0], le[:, 1]])
            u = np.concatenate([le[:, 1], le[:, 0]])
            c = np.concatenate([lc, lc])
            sel = (self.kind[v] == ON_LINE) & moving[v]
            v, u, c = v[sel], u[sel], c[sel]
            su = self._param(u, c, self.s[v])
            ws = np.bincount(v, weights=w[u], minlength=n)
            ts = np.bincount(v, weights=w[u] * su, minlength=n)
            s_target = ts[lin] / ws[lin]
            s_new[lin] = self.s[lin] + 0.5 * (s_target - self.s[lin])
            new[lin] = self._curve_point(self.entity[lin], s_new[lin])
        new = np.where(self.lock, x, new)
        new = self._sync_images(new)
        new = self._guard_move(x, new)
        # images took their master's position: their parameter follows from it
        moved_line = (self.kind == ON_LINE) & np.any(new != x, axis=1)
        idx = np.flatnonzero(moved_line)
        if len(idx):
            s_new[idx] = self._project_curve_param(new[idx], self.entity[idx])
        back = (self.kind == ON_LINE) & ~moved_line
        s_new[back] = self.s[back]
        self.x, self.s = new, s_new

    def _guard_move(self, x, new):
        """Puts back the vertices of triangles that fold, of wedges that close and vertices that
        would leave the box or touch a box face plane they are not locked to."""
        outside = lambda y: np.any(~self.lock & ((y <= 0) | (y >= self.box)), axis=1)
        for _ in range(8):
            back = np.zeros(len(x), dtype=bool)
            back[self.tris[self._folded(x, new)].ravel()] = True
            # only edges of triangles with a moved vertex change their wedges; narrow wedges that
            # were there before do not count
            closing = self._wedges_around(new, self.tris, np.any(new != x, axis=1)[self.tris].any(axis=1))
            back |= closing & ~self._wedges_around(x, self.tris, closing[self.tris].any(axis=1))
            back |= outside(new) & ~outside(x)
            back &= np.any(new != x, axis=1)
            if not back.any():
                break
            back = self._close_images(back)
            new[back] = x[back]
        return new

    def _folded(self, x_old, x_new) -> np.ndarray:
        """Triangles that fold against their normal before the move or against the smoothed
        surface, or nearly vanish."""
        t_old, t_new = x_old[self.tris], x_new[self.tris]
        n_old, n_new = _normal(t_old), _normal(t_new)
        area_old, area_new = np.linalg.norm(n_old, axis=1), np.linalg.norm(n_new, axis=1)
        bad = (_dot(n_new, n_old) <= self.config.min_cos * area_old * area_new) | (area_new < 0.05 * area_old)
        moved = np.any(np.any(t_old != t_new, axis=2), axis=1)
        idx = np.flatnonzero(moved & ~bad)
        ang_new, ang_old = _min_angle(t_new[idx]), _min_angle(t_old[idx])
        bad[idx] = (ang_new < self.config.min_angle) & (ang_new < ang_old)
        idx = idx[~bad[idx]]
        bad[idx] = self._against_reference(x_new[self.tris[idx]], self.tnorm[idx])
        return bad

    def _against_reference(self, tri, ref) -> np.ndarray:
        """True where a triangle's normal has less than min_cos with its reference normal."""
        if not len(tri):
            return np.zeros(0, dtype=bool)
        n = _normal(tri)
        return _dot(n, ref) <= self.config.min_cos * np.linalg.norm(n, axis=1)

    # ------------------------------------------------------------------------------------- #
    # projection and periodic images

    def _project_surface(self, p, patch):
        return self._closest_ref(p, patch)[0]

    def _closest_ref(self, p, patch, k=8):
        """Closest point on the smoothed surface `patch` and the triangle it lies on."""
        out = p.copy()
        tri_of = np.zeros(len(p), dtype=np.int64)
        best = np.full(len(p), np.inf)
        todo = np.arange(len(p))
        for kk in (k, 64, 512, len(self.ref_tris)):
            if not len(todo):
                break
            _, idx = self.ref_tree.query(p[todo], k=min(kk, len(self.ref_tris)))
            idx = idx.reshape(len(todo), -1)
            for j in range(idx.shape[1]):
                t = idx[:, j]
                same = self.ref_patch[t] == patch[todo]
                q = _closest_on_triangles(p[todo], self.ref_x[self.ref_tris[t]])
                d = np.linalg.norm(q - p[todo], axis=1)
                better = same & (d < best[todo])
                out[todo[better]] = q[better]
                tri_of[todo[better]] = t[better]
                best[todo[better]] = d[better]
            todo = todo[~np.isfinite(best[todo])]
        return out, tri_of

    def _curve_point(self, curve, s):
        """Point at arc length s on the reference line `curve`."""
        s = np.clip(s, 0.0, self.curve_len[curve])
        g = self.curve_base[curve] + s
        idx = np.clip(np.searchsorted(self.seg_g0, g, side='right') - 1, self.seg_first[curve], self.seg_last[curve])
        t = np.clip((g - self.seg_g0[idx]) / np.maximum(self.seg_len[idx], 1e-300), 0.0, 1.0)
        a, b = self.ref_x[self.ref_seg[idx, 0]], self.ref_x[self.ref_seg[idx, 1]]
        return a + t[:, None] * (b - a)

    def _param(self, u, curve, near):
        """Arc length of vertex u on `curve`; for an anchor 0 or the curve length, whichever end
        it is (on a closed curve the one nearer to `near`)."""
        out = self.s[u].copy()
        anc = self.anchor[u]
        length = self.curve_len[curve]
        at_start = u == self.curve_start[curve]
        closed = self.curve_closed[curve]
        out[anc] = np.where(closed[anc], np.where(np.nan_to_num(near[anc], nan=0.0) < length[anc] / 2, 0.0, length[anc]),
                            np.where(at_start[anc], 0.0, length[anc]))
        return out

    def _project_curve_param(self, p, curve):
        """Arc length of the closest point of p on its reference line."""
        out = np.zeros(len(p))
        best = np.full(len(p), np.inf)
        todo = np.arange(len(p))
        for kk in (8, 64, len(self.ref_seg)):
            if not len(todo):
                break
            _, idx = self.seg_tree.query(p[todo], k=min(kk, len(self.ref_seg)))
            idx = idx.reshape(len(todo), -1)
            for j in range(idx.shape[1]):
                sg = idx[:, j]
                same = self.ref_seg_curve[sg] == curve[todo]
                a, b = self.ref_x[self.ref_seg[sg, 0]], self.ref_x[self.ref_seg[sg, 1]]
                ab = b - a
                tpar = np.clip(_dot(p[todo] - a, ab) / np.maximum(_dot(ab, ab), 1e-300), 0, 1)
                d = np.linalg.norm(a + tpar[:, None] * ab - p[todo], axis=1)
                better = same & (d < best[todo])
                out[todo[better]] = self.seg_s0[sg[better]] + tpar[better] * self.seg_len[sg[better]]
                best[todo[better]] = d[better]
            todo = todo[~np.isfinite(best[todo])]
        return out

    def _representatives(self):
        """Per periodic class the image with the smallest shift (the master)."""
        n_cls = int(self.cls.max()) + 1
        score = self.shift.sum(axis=1) * (len(self.x) + 1) + np.arange(len(self.x))
        best = np.full(n_cls, np.iinfo(np.int64).max)
        np.minimum.at(best, self.cls, score)
        return best[self.cls] % (len(self.x) + 1)

    def _sync_images(self, x):
        """Images take the in-face coordinates of their master; locked coordinates stay."""
        rep = self._representatives()
        return np.where(self.lock, x, x[rep])

    def _close_images(self, flag):
        n_cls = int(self.cls.max()) + 1
        c = np.zeros(n_cls, dtype=bool)
        c[self.cls[flag]] = True
        return c[self.cls]

    def _compact(self):
        used = np.zeros(len(self.x), dtype=bool)
        used[self.tris.ravel()] = True
        used |= self.anchor
        new = np.cumsum(used, dtype=np.int64) - 1
        self.tris = new[self.tris]
        self.cedge = new[self.cedge]
        self.curve_start = new[self.curve_start]
        for name in ('x', 's', 'kind', 'entity', 'lock', 'shift', 'cls', 'anchor'):
            setattr(self, name, getattr(self, name)[used])
        self.cls = np.unique(self.cls, return_inverse=True)[1].ravel()

    # ------------------------------------------------------------------------------------- #
    # volumes

    def _grain_volumes(self, x=None) -> Dict[int, float]:
        x = self.x if x is None else x
        t = x[self.tris]
        tri_vol = np.einsum('ij,ij->i', t[:, 0], np.cross(t[:, 1], t[:, 2])) / 6
        patch_vol = np.bincount(self.patch, weights=tri_vol, minlength=len(self.model.surfaces))
        return {g: float(np.sum(patch_vol[s[:, 0]] * s[:, 1])) for g, s in self.model.shells.items()}

    def _allowed(self, volume):
        return max(self.config.volume_tolerance * volume, float(np.prod(self.spacing)))

    def _volume_error(self) -> dict:
        vol = self._grain_volumes()
        rel = np.array([(vol[g] - self.target[g]) / self.target[g] for g in self.target])
        off = sum(abs(vol[g] - self.target[g]) > self._allowed(self.target[g]) for g in self.target)
        return {'max_abs': float(np.abs(rel).max()), 'p90_abs': float(np.percentile(np.abs(rel), 90)),
                'median': float(np.median(rel)), 'grains_off_tolerance': int(off)}

    def _correct_volume(self):
        x, tris = self.x, self.tris
        t = x[tris]
        an = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]) / 2
        vn = np.zeros_like(x)
        np.add.at(vn, tris.ravel(), np.repeat(an, 3, axis=0) / 3)
        on_surface = self.kind == ON_SURFACE
        surf_of = np.where(on_surface, self.entity, 0)
        moving_surface = ~is_box_face(self.regions[:, 1])
        movable = on_surface & moving_surface[surf_of]
        dvol = np.bincount(surf_of[movable], weights=np.linalg.norm(vn[movable], axis=1),
                           minlength=len(self.model.surfaces))
        grains = list(self.model.shells)
        rows, cols, vals = [], [], []
        for i, g in enumerate(grains):
            shell = self.model.shells[g]
            rows.append(np.full(len(shell), i))
            cols.append(shell[:, 0])
            vals.append(shell[:, 1] * dvol[shell[:, 0]])
        a = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(len(grains), len(self.model.surfaces)))
        vol = self._grain_volumes()
        residual = np.array([self.target[g] - vol[g] for g in grains])
        offset = lsqr(a, residual, atol=1e-12, btol=1e-12)[0]
        norm = np.linalg.norm(vn, axis=1)
        step = np.zeros_like(x)
        movable &= norm > 0
        step[movable] = vn[movable] / norm[movable, None] * offset[surf_of[movable]][:, None]
        new = self._sync_images(np.where(self.lock, x, x + step))
        self.x = self._guard_move(x, new)

    # ------------------------------------------------------------------------------------- #

    def _build_model(self) -> SurfaceModel:
        """A SurfaceModel with the same points, curves, surfaces, loops and shells (same indices)
        on the new triangles."""
        old = self.model
        topo = old.topology
        edges, he_edge, n_tri, curve = self._edges()
        anchors = np.flatnonzero(self.anchor)
        # anchors keep their order: the old anchor indices were sorted and compaction keeps order
        lines = []
        u, v = self.tris.ravel(), self.tris[:, [1, 2, 0]].ravel()
        he_patch = np.repeat(self.patch, 3)
        he_curve = curve[he_edge]
        # direction from a surface that uses the curve: its half-edges run along the curve where
        # the loop sign is +1
        use_patch, use_sign = _first_uses(old)
        on = np.flatnonzero((he_curve >= 0) & (he_patch == use_patch[np.maximum(he_curve, 0)]))
        forward = use_sign[he_curve[on]] > 0
        frm_all = np.where(forward, u[on], v[on])
        to_all = np.where(forward, v[on], u[on])
        order = np.argsort(he_curve[on], kind='stable')
        bounds = np.searchsorted(he_curve[on][order], np.arange(len(topo.lines) + 1))
        for c, line in enumerate(topo.lines):
            sl = order[bounds[c]:bounds[c + 1]]
            nxt = dict(zip(frm_all[sl].tolist(), to_all[sl].tolist()))
            start = int(anchors[np.searchsorted(topo.anchors, line.vertices[0])])
            chain = [start]
            while True:
                chain.append(nxt[chain[-1]])
                if chain[-1] == start or self.anchor[chain[-1]]:
                    break
            lines.append(Line(vertices=np.array(chain, dtype=np.int64), regions=line.regions))
        grains = self.regions[self.patch]
        mesh = InterfaceMesh(points=self.x.copy(), triangles=self.tris.copy(), grains=grains)
        rep = self._representatives()
        new_topo = Topology(mesh=mesh, shape=topo.shape, periodic=topo.periodic,
                            vertex_regions=np.zeros(len(self.x), dtype=np.int64), edges=edges,
                            edge_line=curve, anchors=anchors,
                            ring_anchors=anchors[np.isin(topo.anchors, topo.ring_anchors)],
                            lines=lines, patch=self.patch.copy(), patch_regions=topo.patch_regions,
                            periodic_class=rep)
        order = np.argsort(self.patch, kind='stable')
        starts = np.searchsorted(self.patch[order], np.arange(len(old.surfaces) + 1))
        surfaces = [Surface(regions=s.regions, triangles=order[starts[p]:starts[p + 1]], loops=s.loops)
                    for p, s in enumerate(old.surfaces)]
        entity = self.entity.copy()
        entity[anchors] = np.arange(len(anchors))
        return SurfaceModel(topology=new_topo, surfaces=surfaces, shells=old.shells, oriented=self.tris.copy(),
                            vertex_kind=self.kind.copy(), vertex_entity=entity, box_lock=self.lock.copy(),
                            shift=self.shift.copy(), periodic_points=old.periodic_points,
                            periodic_curves=old.periodic_curves, periodic_surfaces=old.periodic_surfaces)


def remesh(model: SurfaceModel, points: np.ndarray, labels: np.ndarray, spacing,
           target_volume: Dict[int, float], config: RemeshConfig = RemeshConfig()) -> RemeshResult:
    return Remesher(model, points, labels, spacing, target_volume, config).run()


def check(original: SurfaceModel, result: RemeshResult, spacing) -> dict:
    """Topology and geometry checks of a remeshed model; the counts under 'errors' must be 0."""
    model, x = result.model, result.points
    topo, tris = model.topology, model.oriented
    old = original.topology
    chi_old = _patch_euler(old.mesh.triangles, old.patch, len(original.surfaces))
    chi_new = _patch_euler(tris, topo.patch, len(model.surfaces))
    angle = np.degrees(_angles(x[tris])).min(axis=1)
    h0 = float(np.min(spacing))
    ends = topo.edges
    length = np.linalg.norm(x[ends[:, 0]] - x[ends[:, 1]], axis=1) / h0
    box = np.array(topo.shape) * np.asarray(spacing, float)
    from dragen.mesh import quality
    box_tris = tris[is_box_face(topo.mesh.grains).any(axis=1)]
    per = quality.periodic_mismatch(x, box_tris, (0, 0, 0), box, [a for a in range(3) if topo.periodic[a]])
    anchors_moved = float(np.abs(x[topo.anchors] - original.mesh.points[old.anchors]).max()) \
        if len(old.anchors) else 0.0
    return {
        'triangles': int(len(tris)), 'vertices': int(len(x)),
        'min_angle_deg': {'min': float(angle.min()), 'p1': float(np.percentile(angle, 1)),
                          'median': float(np.median(angle))},
        'edge_voxels': {'min': float(length.min()), 'median': float(np.median(length)),
                        'max': float(length.max())},
        'errors': {
            'patch_euler_changed': int(np.count_nonzero(chi_old != chi_new)),
            'lost_patches': int(np.count_nonzero(np.bincount(topo.patch, minlength=len(model.surfaces)) == 0)),
            'self_intersections': int(len(self_intersection_pairs(x, tris, shared_vertex=True, touching=True))),
            'anchors_moved': int(anchors_moved > 0),
            'periodic_unmatched_nodes': int(sum(v['unmatched_nodes'] for v in per.values())),
            'periodic_unmatched_facets': int(sum(v['unmatched_facets'] for v in per.values())),
            'misoriented_edges': int(_misoriented(tris, topo.patch)),
            'edges_in_box_plane': int(_edges_in_box_plane(model, x)),
            'vertices_outside_box': int(np.count_nonzero(np.any(~model.box_lock & ((x <= 0) | (x >= box)), axis=1))),
        },
    }


# --------------------------------------------------------------------------------------------- #

def _independent_set(ok, prio, cand_of, res_of, n_res, rounds=16):
    """A maximal set of the ok candidates whose resources are disjoint, built in rounds: in every
    round a candidate wins if it has the highest priority on all its resources among the candidates
    still available; the resources of the winners are then blocked.

    cand_of, res_of: one row per (candidate, resource) pair."""
    selected = np.zeros(len(ok), dtype=bool)
    if not len(cand_of):
        return selected
    # rows of ok candidates sorted by resource, then priority: the last available row of a resource
    # holds its best (the order among equal priorities does not matter: they win or lose together)
    rows = np.flatnonzero(ok[cand_of])
    cand_s, res_s = cand_of[rows], res_of[rows]
    rank = np.unique(prio[cand_s], return_inverse=True)[1].ravel()
    order = np.argsort(res_s.astype(np.int64) * (int(rank.max(initial=0)) + 1) + rank)
    cand_s, res_s = cand_s[order], res_s[order]
    blocked = np.zeros(n_res, dtype=bool)
    alive = np.ones(len(ok), dtype=bool)
    best = np.zeros(n_res, dtype=np.int64)
    for _ in range(rounds):
        # drop the rows of candidates that are selected or touch a blocked resource
        dead = np.zeros(len(ok), dtype=bool)
        dead[cand_s[blocked[res_s]]] = True
        alive &= ~dead & ~selected
        keep = alive[cand_s]
        cand_s, res_s = cand_s[keep], res_s[keep]
        if not len(cand_s):
            break
        last = np.concatenate([res_s[1:] != res_s[:-1], [True]])
        best[res_s[last]] = prio[cand_s[last]]
        lose = np.zeros(len(ok), dtype=bool)
        lose[cand_s[best[res_s] != prio[cand_s]]] = True
        present = np.zeros(len(ok), dtype=bool)
        present[cand_s] = True
        win = present & ~lose
        if not win.any():
            break
        selected |= win
        blocked[res_s[win[cand_s]]] = True
    return selected


def _edges_in_box_plane(model: SurfaceModel, x) -> int:
    """Edges of grain boundary (not box face) triangles that lie in a box face plane but are not
    line edges: they cut through the triangles of the box face."""
    tris, patch = model.oriented, model.topology.patch
    regions = np.array([s.regions for s in model.surfaces])
    inner = ~is_box_face(regions[patch, 1])
    a = tris[inner].ravel()
    b = tris[inner][:, [1, 2, 0]].ravel()
    lock = model.box_lock
    plane = np.any(lock[a] & lock[b] & (x[a] == x[b]), axis=1)
    if not plane.any():
        return 0
    seg = np.concatenate([np.column_stack([ln.vertices[:-1], ln.vertices[1:]]) for ln in model.curves])
    edge = np.sort(np.column_stack([a[plane], b[plane]]), axis=1)
    on_line = _find_rows(unique_pairs(np.min(seg, axis=1), np.max(seg, axis=1)), edge) >= 0
    return int(len(unique_pairs(edge[~on_line, 0], edge[~on_line, 1])))


def _first_uses(model: SurfaceModel):
    """Per curve the first surface that uses it and the sign of that use."""
    patch = np.full(len(model.curves), -1, dtype=np.int64)
    sign = np.zeros(len(model.curves), dtype=np.int64)
    for p in range(len(model.surfaces) - 1, -1, -1):
        for loop in model.surfaces[p].loops[::-1]:
            patch[loop[::-1, 0]] = p
            sign[loop[::-1, 0]] = loop[::-1, 1]
    if np.any(patch < 0):
        raise ValueError('a curve is not used by any surface')
    return patch, sign


def _patch_euler(tris, patch, n_patch):
    tris = np.asarray(tris, dtype=np.int64)
    u, v = tris.ravel(), tris[:, [1, 2, 0]].ravel()
    p = np.repeat(patch, 3)
    verts = np.bincount(unique_pairs(p, u)[:, 0], minlength=n_patch)
    n = int(max(u.max(), v.max())) + 1
    edges = np.bincount(unique_pairs(p, np.minimum(u, v) * n + np.maximum(u, v))[:, 0], minlength=n_patch)
    return verts - edges + np.bincount(patch, minlength=n_patch)


def _misoriented(tris, patch):
    tris = np.asarray(tris, dtype=np.int64)
    u, v = tris.ravel(), tris[:, [1, 2, 0]].ravel()
    n = int(tris.max()) + 1
    _, counts = unique_pairs(np.repeat(patch, 3), u * n + v, return_counts=True)
    return np.count_nonzero(counts > 1)


def _find_rows(table: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Index of each row (sorted pair) in the sorted unique pair table, -1 if absent."""
    if not len(table):
        return np.full(len(rows), -1, dtype=np.int64)
    table, rows = np.asarray(table, dtype=np.int64), np.asarray(rows, dtype=np.int64)
    span = int(max(table.max(), rows.max())) + 1
    keys = table[:, 0] * span + table[:, 1]
    order = np.argsort(keys)
    q = rows[:, 0] * span + rows[:, 1]
    pos = np.minimum(np.searchsorted(keys[order], q), len(keys) - 1)
    return np.where(keys[order][pos] == q, order[pos], -1)


def _rows_in(rows: np.ndarray, table: np.ndarray) -> np.ndarray:
    """True for every row of the (n, 3) int array that also occurs in table."""
    if not len(rows) or not len(table):
        return np.zeros(len(rows), dtype=bool)
    void = np.dtype((np.void, 3 * 8))
    t = np.sort(np.ascontiguousarray(table, dtype=np.int64).view(void).ravel())
    r = np.ascontiguousarray(rows, dtype=np.int64).view(void).ravel()
    pos = np.minimum(np.searchsorted(t, r), len(t) - 1)
    return t[pos] == r


def _lookup_pairs(rows: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _find_rows(rows, np.column_stack([a, b]))


def _dot(a, b):
    return np.einsum('ij,ij->i', a, b)


def _normal(t):
    return np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])


def _angles(t):
    """(n, 3) interior angles in radians of triangles t (n, 3, 3)."""
    out = []
    for i in range(3):
        u = t[:, (i + 1) % 3] - t[:, i]
        v = t[:, (i + 2) % 3] - t[:, i]
        c = _dot(u, v) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-300)
        out.append(np.arccos(np.clip(c, -1, 1)))
    return np.stack(out, axis=1)


def _min_angle(t):
    """Smallest interior angle in degrees (the largest cosine, one arccos per triangle)."""
    cos = []
    for i in range(3):
        u = t[:, (i + 1) % 3] - t[:, i]
        v = t[:, (i + 2) % 3] - t[:, i]
        cos.append(_dot(u, v) / np.maximum(np.sqrt(_dot(u, u) * _dot(v, v)), 1e-300))
    return np.degrees(np.arccos(np.clip(np.max(cos, axis=0), -1, 1)))


def _closest_on_triangles(p, tri):
    """Closest point to p (n, 3) on the triangles tri (n, 3, 3)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    n = np.cross(b - a, c - a)
    nn = np.maximum(_dot(n, n), 1e-300)
    q = p - (_dot(p - a, n) / nn)[:, None] * n
    # barycentric coordinates of the projection
    v0, v1, v2 = b - a, c - a, q - a
    d00, d01, d11 = _dot(v0, v0), _dot(v0, v1), _dot(v1, v1)
    d20, d21 = _dot(v2, v0), _dot(v2, v1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-300)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    inside = (v >= 0) & (w >= 0) & (v + w <= 1)
    best = q.copy()
    dist = np.where(inside, 0.0, np.inf)
    for s0, s1 in ((a, b), (b, c), (c, a)):
        e = s1 - s0
        t = np.clip(_dot(p - s0, e) / np.maximum(_dot(e, e), 1e-300), 0, 1)
        r = s0 + t[:, None] * e
        d = np.linalg.norm(r - p, axis=1)
        take = ~inside & (d < dist)
        best[take] = r[take]
        dist = np.where(take, d, dist)
    return best
