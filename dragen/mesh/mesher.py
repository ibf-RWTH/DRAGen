"""From a voxel label field to a C3D10 mesh with smoothed, adaptively meshed grain boundaries.

    clean-up (M1) -> topology (M1) -> surface model (M2) -> smoothing (M3)
    -> adaptive remeshing of the grain boundaries (M4a) -> gmsh tetrahedra (M4)

Feedback: tetrahedra with a smallest dihedral angle below `min_dihedral` are almost always forced
by the surface (a grain locally thinner than its boundary triangles, a narrow wedge at a line, or
a slightly concave edge bridged by one flat element; process notes 7.12). Around them the surface
is remeshed further and the volume meshed again, one step of `feedback` per round: 'coarsen'
doubles the size field there so vertices are collapsed and the fine feature is simplified away
(author's proposal), 'refine' halves it down to h_min. What remains is reported.

Resolution: where h_max (a quarter of the grain diameter) is not above h_min for most grains, the
voxels are too coarse for the grain size; the mesh is made anyway and a warning is given.
"""
import time
from dataclasses import dataclass, field
import logging
from typing import Optional, Sequence, Tuple

import numpy as np

from dragen.mesh.cleanup import ManifoldCleaner
from dragen.mesh.model import SurfaceModel, build_model
from dragen.mesh.remeshing import RemeshConfig, Remesher
from dragen.mesh.smoothing import SmoothingConfig, SmoothingResult, smooth
from dragen.mesh.tetmesh import TetMeshConfig, TetMeshResult, tet_mesh
from dragen.mesh.topology import build_topology
from dragen.mesh.volume import tet_dihedral_angles


@dataclass
class MeshConfig:
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    remeshing: Optional[RemeshConfig] = field(default_factory=RemeshConfig)
    """None: keep the smoothed triangles as they are (M4 without M4a)."""
    tetmesh: TetMeshConfig = field(default_factory=TetMeshConfig)
    min_dihedral: float = 5.0
    """Acceptance of the author: smallest dihedral angle of every tetrahedron, degrees."""
    feedback: Tuple[str, ...] = ('coarsen', 'coarsen', 'coarsen')
    """One step per feedback round: 'coarsen' or 'refine' around the poor tetrahedra. The round
    with the fewest poor tetrahedra is returned."""
    feedback_iterations: int = 4
    coarse_warning: float = 0.5
    """Warn when more than this share of the grains has d / 4 <= h_min (voxels too coarse)."""


@dataclass
class MeshResult:
    labels: np.ndarray
    """The cleaned label field the mesh was built from."""
    staircase: SurfaceModel
    smoothed: SmoothingResult
    surface: SurfaceModel
    """The final grain boundary model (remeshed, or the staircase model with smoothed points)."""
    points: np.ndarray
    """Points of `surface`."""
    tets: TetMeshResult
    stats: dict = field(default_factory=dict)


def mesh_rve(labels: np.ndarray, spacing: Sequence[float], periodic: Sequence[bool],
             config: MeshConfig = MeshConfig()) -> MeshResult:
    stats = {'seconds': {}}
    seconds = stats['seconds']

    t = time.perf_counter()
    cleaned, cleanup = ManifoldCleaner(periodic).run(labels)
    seconds['cleanup'] = time.perf_counter() - t
    stats['cleanup'] = {'changed_voxels': cleanup.changed_voxels, 'critical_after': cleanup.critical_after}

    stats['warnings'] = []
    if config.remeshing is not None:
        counts = np.bincount(cleaned.ravel())[1:]
        d = (6 * counts[counts > 0] * np.prod(spacing) / np.pi) ** (1 / 3)
        coarse = float(np.mean(config.remeshing.h_max_fraction * d <= config.remeshing.h_min * min(spacing)))
        stats['coarse_grain_share'] = coarse
        if coarse > config.coarse_warning:
            msg = (f'{coarse:.0%} of the grains are smaller than 4 voxels across; the resolution is too coarse '
                   f'for the grain size and the adaptive surface cannot be coarser than the voxels there')
            stats['warnings'].append(msg)
            logging.getLogger(__name__).warning(msg)

    t = time.perf_counter()
    staircase = build_model(build_topology(cleaned, spacing, periodic))
    seconds['model'] = time.perf_counter() - t

    t = time.perf_counter()
    smoothed = smooth(staircase, spacing, config.smoothing)
    seconds['smoothing'] = time.perf_counter() - t

    if config.remeshing is None:
        surface, points, remesher = staircase, smoothed.points, None
    else:
        t = time.perf_counter()
        remesher = Remesher(staircase, smoothed.points, cleaned, spacing, smoothed.grain_volume_target,
                            config.remeshing)
        remeshed = remesher.run()
        seconds['remeshing'] = time.perf_counter() - t
        surface, points = remeshed.model, remeshed.points

    stats['feedback'] = []
    seconds['tetmesh'] = 0.0
    best = None
    for round_ in range(len(config.feedback) + 1):
        t = time.perf_counter()
        try:
            tets = tet_mesh(surface, points, spacing, config.tetmesh)
        except Exception as error:
            # a feedback round gmsh cannot mesh (e.g. edges the remeshing made intersect) falls back
            # to the best earlier round
            if best is None:
                raise
            stats['feedback'].append({'triangles': int(len(surface.oriented)), 'error': repr(error)})
            logging.getLogger(__name__).warning('feedback round %d failed (%r); best earlier round kept',
                                                round_, error)
            break
        finally:
            seconds['tetmesh'] += time.perf_counter() - t
        dihedral = tet_dihedral_angles(tets.mesh).min(axis=1)
        bad = np.flatnonzero(dihedral < config.min_dihedral)
        stats['feedback'].append({'triangles': int(len(surface.oriented)), 'elements': int(len(dihedral)),
                                  'below_min_dihedral': int(len(bad)), 'min_dihedral': float(dihedral.min())})
        # keep the best round: fewest poor tetrahedra, then the largest smallest angle
        score = (len(bad), -float(dihedral.min()))
        if best is None or score < best[0]:
            best = (score, round_, surface, points, tets)
        if not len(bad) or remesher is None or round_ == len(config.feedback):
            break
        t = time.perf_counter()
        centres = tets.mesh.points[tets.mesh.corners[bad]].mean(axis=1)
        step = remesher.coarsen_near if config.feedback[round_] == 'coarsen' else remesher.refine_near
        stats['feedback'][-1]['next'] = config.feedback[round_]
        remeshed = step(centres, config.feedback_iterations)
        seconds['remeshing'] += time.perf_counter() - t
        surface, points = remeshed.model, remeshed.points
    _, stats['feedback_best_round'], surface, points, tets = best
    if remesher is not None:
        stats['remeshing'] = remesher.stats
    return MeshResult(labels=cleaned, staircase=staircase, smoothed=smoothed, surface=surface, points=points,
                      tets=tets, stats=stats)
