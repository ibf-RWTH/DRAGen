"""Hierarchical smoothing of the grain boundary network (phase M3).

Works on the surface model of M2 and only moves points; the triangles stay as they are.

  1. anchors stay fixed
  2. lines are smoothed along their vertex chain, end points fixed
  3. surfaces are smoothed with their boundary (lines and anchors) fixed
  4. volume correction: every inner surface is offset along its vertex normals so that every
     grain gets back the volume of its voxels
  5. untangling: where surfaces still cross, a triangle folded or a grain wedge closed, the
     vertices involved and their neighbours are blended back towards their voxel corners (the
     staircase has none of these problems)
  6. fallback: a grain still more than `volume_tolerance` off its voxel volume (e.g. a grain of
     one voxel, which cannot be rounded without losing volume) is put back on the staircase: all
     vertices of its surfaces except anchors return to their voxel corners. Neighbours that drift
     out of tolerance by this are treated the same way in the next round.

Smoothing is Taubin's lambda/mu scheme (a shrinking Laplacian step followed by an inflating one),
which removes the voxel steps without shrinking the shape as plain Laplacian smoothing does.
Throughout, the rules of the author hold (process notes section 2): coordinates normal to a box
face never change (`SurfaceModel.box_lock`), periodic images move together, and no vertex moves
further than `max_displacement` voxel edges from its voxel corner.

The volume correction solves, in the least-squares sense with minimum norm, for one offset per
surface: the volume change of a grain is the sum over its surfaces of (offset x area moved),
positive where the surface normal points out of the grain. Box surfaces do not move. Vertices on
lines and anchors do not move, so the correction is spread over the surface interiors.
"""
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from dragen.mesh.model import FIXED, ON_LINE, ON_SURFACE, SurfaceModel
from dragen.mesh.quality import self_intersection_pairs, unique_pairs
from dragen.mesh.surface import is_box_face


@dataclass
class SmoothingConfig:
    line_iterations: int = 10
    surface_iterations: int = 15
    lam: float = 0.5
    mu: float = -0.53
    max_displacement: float = 1.0
    """Largest distance of a vertex from its voxel corner, in voxel edges (smallest spacing)."""
    volume_iterations: int = 3
    min_cos: float = 0.2
    """Guard: a triangle's normal must keep at least this cosine with its staircase normal ..."""
    min_area: float = 0.1
    """... and its area at least this fraction of its staircase area; otherwise the step of its
    vertices is halved (up to `guard_halvings` times, then dropped)."""
    guard_halvings: int = 4
    volume_tolerance: float = 0.01
    """Allowed volume error per grain: this fraction of its volume or one voxel volume,
    whichever is larger (author's decision, process notes section 2)."""
    fallback_rounds: int = 1000
    stop_tolerance: float = 1e-3
    """A smoothing stage ends early once no vertex moves more than this (voxel edges) in an
    iteration; steps below a thousandth of it are dropped and not checked."""
    untangle_rounds: int = 6
    """Halvings of the move towards the voxel corner per crossing; the last round resets fully."""
    min_wedge: float = 10.0
    """Guard: the interior angle of every grain at every edge of its shell must stay within
    [min_wedge, 360 - min_wedge] degrees, so surfaces of a grain never pass through each other."""


@dataclass
class SmoothingResult:
    points: np.ndarray
    grain_volume_target: Dict[int, float]
    stats: dict = field(default_factory=dict)


class Smoother:
    """Applies the four steps to a SurfaceModel built on the voxel lattice."""

    def __init__(self, model: SurfaceModel, spacing, config: SmoothingConfig = SmoothingConfig()):
        self.model = model
        self.config = config
        self.spacing = np.asarray(spacing, dtype=float)
        topo = model.topology
        self.origin = model.mesh.points.copy()
        self.free = ~model.box_lock
        # periodic classes: dense index for averaging moves of all images
        self.cls = np.unique(topo.periodic_class, return_inverse=True)[1].ravel()
        self.n_cls = int(self.cls.max()) + 1 if len(self.cls) else 0
        self.limit = config.max_displacement * float(self.spacing.min())
        self.n0 = self._normals(self.origin)
        self.area0 = np.sqrt(_dot(self.n0, self.n0))
        self.guarded = 0
        self._wedges()
        self._line_neighbours()
        self._surface_neighbours()

    # ------------------------------------------------------------------------------------- #

    def run(self) -> SmoothingResult:
        x = self.origin.copy()
        target = self._voxel_volumes()
        stats = {'volume_before': self._volume_error(x, target)}
        stats['iterations'] = {}
        for name, n, mover, nbr in (('lines', self.config.line_iterations, self.line_mover, self.line_nbr),
                                    ('surfaces', self.config.surface_iterations, self.surf_mover, self.surf_nbr)):
            done = 0
            for done in range(1, n + 1):
                before = x
                x = self._taubin(x, mover, nbr)
                moved = np.sqrt(_dot(x - before, x - before)).max() if len(x) else 0.0
                if moved < self.config.stop_tolerance * float(self.spacing.min()):
                    break
            stats['iterations'][name] = done
        stats['volume_after_smoothing'] = self._volume_error(x, target)
        for _ in range(self.config.volume_iterations):
            x = self._correct_volume(x, target)
        stats['volume_after_correction'] = self._volume_error(x, target)
        x, stats['untangling'], touched = self._untangle(x)
        if touched.any():
            # restore the volumes without moving the untangled vertices again, then check again
            for _ in range(self.config.volume_iterations):
                x = self._correct_volume(x, target, frozen=touched)
            x, stats['untangling_after_correction'], again = self._untangle(x)
        x, fallback = self._revert_off_volume(x, target)
        stats['reverted_grains'], stats['fallback_rounds'] = fallback['grains'], fallback['rounds']
        stats['volume_final'] = self._volume_error(x, target)
        d = np.linalg.norm(x - self.origin, axis=1) / float(self.spacing.min())
        stats['displacement_voxels'] = {'max': float(d.max()) if len(d) else 0.0,
                                        'median_moved': float(np.median(d[d > 0])) if np.any(d > 0) else 0.0,
                                        'at_limit': int(np.count_nonzero(d >= self.config.max_displacement - 1e-9))}
        kind = self.model.vertex_kind
        stats['vertices'] = {'fixed': int(np.count_nonzero(kind == FIXED)),
                             'on_line': int(np.count_nonzero(kind == ON_LINE)),
                             'on_surface': int(np.count_nonzero(kind == ON_SURFACE))}
        stats['guarded_vertex_steps'] = int(self.guarded)
        return SmoothingResult(points=x, grain_volume_target=target, stats=stats)

    # ------------------------------------------------------------------------------------- #

    def _line_neighbours(self):
        rows, cols = [], []
        for line in self.model.curves:
            v = line.vertices
            if len(v) < 3:
                continue
            inner = v[1:-1]
            rows += [inner, inner]
            cols += [v[:-2], v[2:]]
        n = len(self.origin)
        self.line_nbr = _umbrella(np.concatenate(rows) if rows else np.zeros(0, int),
                                  np.concatenate(cols) if cols else np.zeros(0, int), n)
        self.line_mover = self.model.vertex_kind == ON_LINE

    def _surface_neighbours(self):
        tris = self.model.oriented
        u = tris.ravel()
        v = tris[:, [1, 2, 0]].ravel()
        rows = np.concatenate([u, v])
        cols = np.concatenate([v, u])
        pairs = unique_pairs(rows, cols)          # each edge appears once per adjacent triangle
        self.surf_nbr = _umbrella(pairs[:, 0], pairs[:, 1], len(self.origin))
        self.surf_mover = self.model.vertex_kind == ON_SURFACE

    def _taubin(self, x, mover, nbr):
        for factor in (self.config.lam, self.config.mu):
            step = (nbr @ x - x) * factor
            step[~mover] = 0.0
            x = self._apply(x, step)
        return x

    def _wedges(self):
        """For every edge of every grain shell: the edge (a -> b) and the third vertices c1, c2
        of the grain's two triangles there; the grain lies on the inner side of both."""
        model = self.model
        tris = model.oriented
        regions = np.array([s.regions for s in model.surfaces])
        patch = model.topology.patch
        inner = ~is_box_face(regions[patch, 1])
        side_tri = np.concatenate([tris, tris[inner][:, ::-1]])           # outward for side_grain
        side_grain = np.concatenate([regions[patch, 0], regions[patch[inner], 1]])
        a = side_tri.ravel()
        b = side_tri[:, [1, 2, 0]].ravel()
        c = side_tri[:, [2, 0, 1]].ravel()
        g = np.repeat(side_grain, 3)
        n = len(self.origin)
        key = unique_pairs(g, np.minimum(a, b) * n + np.maximum(a, b), return_inverse=True)[1]
        order = np.argsort(key, kind='stable')
        same = key[order[1:]] == key[order[:-1]]
        first, second = order[:-1][same], order[1:][same]
        self.w = np.column_stack([a[first], b[first], c[first], c[second]])
        # a flat edge inside one surface needs no wedge check: the fold guard keeps both normals
        # within acos(min_cos) of their common staircase normal, so the wedge stays open
        side_src = np.concatenate([np.arange(len(tris)), np.flatnonzero(inner)])
        same_patch = patch[side_src[first // 3]] == patch[side_src[second // 3]]
        flat = np.abs(self._wedge_angles(self.origin, slice(None)) - 180.0) < 1.0
        self.w = self.w[~(same_patch & flat)]
        self.w_bad0 = self._bad_wedges(self.origin, slice(None))
        # incidence, to re-check only what a changed vertex touches
        nw, nt = len(self.w), len(tris)
        self.wedges_of = sparse.csr_matrix((np.ones(4 * nw, dtype=np.float32),
                                            (np.repeat(np.arange(nw), 4), self.w.ravel())), shape=(nw, n))
        self.tris_of = sparse.csr_matrix((np.ones(3 * nt, dtype=np.float32),
                                          (np.repeat(np.arange(nt), 3), tris.ravel())), shape=(nt, n))

    def _bad_wedges(self, x, idx):
        """Wedges (interior angle of the grain at a shell edge) outside [min_wedge, 360 - min_wedge]."""
        phi = self._wedge_angles(x, idx)
        return (phi < self.config.min_wedge) | (phi > 360.0 - self.config.min_wedge)

    def _wedge_angles(self, x, idx):
        w = self.w[idx]
        a = x[w[:, 0]]
        e = x[w[:, 1]] - a
        d1, d2 = x[w[:, 2]] - a, x[w[:, 3]] - a
        ee = _dot(e, e)
        p1 = d1 - (_dot(d1, e) / np.maximum(ee, 1e-300))[:, None] * e
        p2 = d2 - (_dot(d2, e) / np.maximum(ee, 1e-300))[:, None] * e
        inward = -_cross(e, d1)                        # into the grain, perpendicular to e
        # angle from p1 towards the grain interior to p2: |inward| = |e| |p1|, so
        # atan2(p2.inward / (|e||p1|), p2.p1 / |p1|) = atan2(p2.inward, p2.p1 |e|)
        return np.degrees(np.arctan2(_dot(p2, inward), _dot(p2, p1) * np.sqrt(ee))) % 360.0

    def _normals(self, x, idx=slice(None)):
        t = x[self.model.oriented[idx]]
        return _cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])

    def _bad_triangles(self, x, idx, area_old, was_bad):
        """Triangles that fold against their staircase normal or collapse, and got worse."""
        n = self._normals(x, idx)
        area = np.sqrt(_dot(n, n))
        bad = (_dot(n, self.n0[idx]) < self.config.min_cos * area * self.area0[idx])             | (area < self.config.min_area * self.area0[idx])
        return bad & (~was_bad[idx] | (area < area_old[idx]))

    def _apply(self, x, step):
        """Couple periodic images, respect box locks, clamp to the displacement limit and halve
        the step of vertices whose triangles would fold or collapse or whose grain wedges would
        close; after the first try only what the halved vertices touch is checked again."""
        if self.n_cls < len(x):
            mean = np.zeros((self.n_cls, 3))
            np.add.at(mean, self.cls, step)
            count = np.bincount(self.cls, minlength=self.n_cls)[:, None]
            step = mean[self.cls] / count[self.cls]
        step = step * self.free
        moving = _dot(step, step) > (1e-3 * self.config.stop_tolerance * float(self.spacing.min())) ** 2
        if not moving.any():
            return x
        step[~moving] = 0.0
        scale = np.ones(len(x))
        tris = self.model.oriented
        nt, nw = len(tris), len(self.w)
        mv = moving.astype(np.float32)
        t_mov = np.flatnonzero(self.tris_of @ mv)
        w_mov = np.flatnonzero(self.wedges_of @ mv)
        area_old = np.zeros(nt)
        was_bad = np.zeros(nt, dtype=bool)
        n_old = self._normals(x, t_mov)
        area_old[t_mov] = np.sqrt(_dot(n_old, n_old))
        was_bad[t_mov] = (_dot(n_old, self.n0[t_mov]) < self.config.min_cos * area_old[t_mov] * self.area0[t_mov]) \
            | (area_old[t_mov] < self.config.min_area * self.area0[t_mov])
        new = x.copy()
        new[moving] = self._clamp(x[moving] + step[moving], moving)
        bad_t = np.zeros(nt, dtype=bool)
        bad_w = np.zeros(nw, dtype=bool)
        bad_t[t_mov] = self._bad_triangles(new, t_mov, area_old, was_bad)
        bad_w[w_mov] = self._bad_wedges(new, w_mov) & ~self.w_bad0[w_mov]
        attempt = 0
        while bad_t.any() or bad_w.any():
            hit = np.zeros(len(x), dtype=bool)
            hit[tris[bad_t].ravel()] = True
            hit[self.w[bad_w].ravel()] = True
            hit &= moving & (scale > 0)
            if self.n_cls < len(x):                       # all periodic images alike
                cls_hit = np.zeros(self.n_cls, dtype=bool)
                cls_hit[self.cls[hit]] = True
                hit = cls_hit[self.cls]
            if not hit.any():
                break
            self.guarded += int(np.count_nonzero(hit))
            scale[hit] = 0.0 if attempt >= self.config.guard_halvings - 1 else scale[hit] * 0.5
            attempt += 1
            new[hit] = self._clamp(x[hit] + step[hit] * scale[hit, None], hit)
            hv = hit.astype(np.float32)
            t_idx = np.flatnonzero(self.tris_of @ hv)
            w_idx = np.flatnonzero(self.wedges_of @ hv)
            bad_t[t_idx] = self._bad_triangles(new, t_idx, area_old, was_bad)
            bad_w[w_idx] = self._bad_wedges(new, w_idx) & ~self.w_bad0[w_idx]
        return new

    def _clamp(self, new, rows=slice(None)):
        origin = self.origin[rows]
        offset = new - origin
        dist = np.sqrt(_dot(offset, offset))
        over = dist > self.limit
        new = new.copy()
        new[over] = origin[over] + offset[over] * (self.limit / dist[over])[:, None]
        return new

    def _problems(self, x, candidates=None):
        """Crossing triangle pairs, triangles folded against the staircase and closed grain
        wedges; returns their counts and the vertices involved."""
        tris = self.model.mesh.triangles
        pairs = self_intersection_pairs(x, tris, candidates=candidates)
        folded = _dot(self._normals(x), self.n0) <= 0
        wedge = self._bad_wedges(x, slice(None)) & ~self.w_bad0
        verts = np.concatenate([tris[pairs].ravel(), self.model.oriented[folded].ravel(), self.w[wedge].ravel()])
        return (len(pairs), int(folded.sum()), int(wedge.sum())), verts

    def _untangle(self, x):
        """Blend the vertices of every problem (see _problems) and their neighbours back towards
        their voxel corners until none is left; the staircase itself has none."""
        tris = self.model.mesh.triangles
        smoothed = x.copy()
        t = np.ones(len(x))
        counts, verts = self._problems(x)
        stats = {'crossings_before': counts[0], 'folded_before': counts[1], 'wedges_before': counts[2],
                 'rounds': 0, 'vertices': 0}
        touched = np.zeros(len(x), dtype=bool)
        while len(verts):
            hit = np.zeros(len(x), dtype=bool)
            hit[verts] = True
            hit |= (self.surf_nbr @ hit.astype(float)) > 0               # and their neighbours
            hit &= (self.model.vertex_kind != FIXED) & (t > 0)
            if self.n_cls < len(x):
                cls_hit = np.zeros(self.n_cls, dtype=bool)
                cls_hit[self.cls[hit]] = True
                hit = cls_hit[self.cls] & (t > 0)
            if not hit.any():
                break
            last = stats['rounds'] >= self.config.untangle_rounds - 1
            t[hit] = 0.0 if last else t[hit] * 0.5
            touched |= hit
            x = self.origin + t[:, None] * (smoothed - self.origin)
            near = np.flatnonzero(np.any(touched[tris], axis=1))
            counts, verts = self._problems(x, candidates=near)
            stats['rounds'] += 1
        stats['vertices'] = int(np.count_nonzero(touched))
        stats['crossings_after'], stats['folded_after'], stats['wedges_after'] = counts
        return x, stats, touched

    def _allowed(self, volume: float) -> float:
        return max(self.config.volume_tolerance * volume, float(np.prod(self.spacing)))

    def _revert_off_volume(self, x, target):
        """Put every grain that is off its volume back on the staircase (all its surface and line
        vertices, with their periodic images), untangle, and repeat while grains are off."""
        model = self.model
        reverted = set()
        rounds = 0
        while rounds < self.config.fallback_rounds:
            vol = self._grain_volumes(x)
            off = [g for g in target if abs(vol[g] - target[g]) > self._allowed(target[g])]
            if not off:
                break
            surfaces = np.unique(np.concatenate([model.shells[g][:, 0] for g in off]))
            back = np.zeros(len(x), dtype=bool)
            back[model.oriented[np.isin(model.topology.patch, surfaces)].ravel()] = True
            back &= model.vertex_kind != FIXED
            if self.n_cls < len(x):
                cls_back = np.zeros(self.n_cls, dtype=bool)
                cls_back[self.cls[back]] = True
                back = cls_back[self.cls]
            x = x.copy()
            x[back] = self.origin[back]
            x, _, _ = self._untangle(x)
            reverted.update(off)
            rounds += 1
        return x, {'grains': len(reverted), 'rounds': rounds}

    # ------------------------------------------------------------------------------------- #

    def _voxel_volumes(self) -> Dict[int, float]:
        return {g: v for g, v in self._grain_volumes(self.origin).items()}

    def _grain_volumes(self, x) -> Dict[int, float]:
        t = x[self.model.oriented]
        tri_vol = np.einsum('ij,ij->i', t[:, 0], np.cross(t[:, 1], t[:, 2])) / 6
        patch_vol = np.bincount(self.model.topology.patch, weights=tri_vol,
                                minlength=len(self.model.surfaces))
        return {g: float(np.sum(patch_vol[shell[:, 0]] * shell[:, 1]))
                for g, shell in self.model.shells.items()}

    def _volume_error(self, x, target) -> dict:
        vol = self._grain_volumes(x)
        rel = np.array([(vol[g] - target[g]) / target[g] for g in target])
        off = sum(abs(vol[g] - target[g]) > self._allowed(target[g]) for g in target)
        return {'max_abs': float(np.abs(rel).max()), 'p90_abs': float(np.percentile(np.abs(rel), 90)),
                'median': float(np.median(rel)), 'grains_off_tolerance': int(off)}

    def _correct_volume(self, x, target, frozen=None):
        model = self.model
        tris = model.oriented
        t = x[tris]
        area_normal = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]) / 2       # (F, 3), |.| = area
        vn = np.zeros_like(x)
        np.add.at(vn, tris.ravel(), np.repeat(area_normal, 3, axis=0) / 3)     # area-weighted normals
        on_surface = model.vertex_kind == ON_SURFACE
        surf_of = np.where(on_surface, model.vertex_entity, 0)
        moving_surface = np.array([not is_box_face(s.regions[1]) for s in model.surfaces])
        movable = on_surface & moving_surface[surf_of]
        if frozen is not None:
            movable &= ~frozen
        # volume change per unit offset of each surface: sum of |vertex normal| (vertex area)
        dvol = np.bincount(surf_of[movable], weights=np.linalg.norm(vn[movable], axis=1),
                           minlength=len(model.surfaces))
        rows, cols, vals = [], [], []
        grains = list(model.shells)
        for i, g in enumerate(grains):
            shell = model.shells[g]
            rows.append(np.full(len(shell), i))
            cols.append(shell[:, 0])
            vals.append(shell[:, 1] * dvol[shell[:, 0]])
        a = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(len(grains), len(model.surfaces)))
        vol = self._grain_volumes(x)
        residual = np.array([target[g] - vol[g] for g in grains])
        offset = lsqr(a, residual, atol=1e-12, btol=1e-12)[0]
        unit = np.zeros_like(x)
        norm = np.linalg.norm(vn, axis=1)
        unit[movable] = vn[movable] / norm[movable][:, None]
        step = np.zeros_like(x)
        step[movable] = unit[movable] * offset[surf_of[movable]][:, None]
        return self._apply(x, step)


def _dot(a, b):
    return np.einsum('ij,ij->i', a, b)


def _cross(a, b):
    return np.column_stack([a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
                            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]])


def _umbrella(rows, cols, n):
    """Row-normalised adjacency: (A @ x)[i] = mean of the neighbours of i."""
    if not len(rows):
        return sparse.csr_matrix((n, n))
    a = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    a.data[:] = 1.0
    deg = np.asarray(a.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    return sparse.diags(1.0 / deg) @ a


def smooth(model: SurfaceModel, spacing, config: SmoothingConfig = SmoothingConfig()) -> SmoothingResult:
    return Smoother(model, spacing, config).run()
