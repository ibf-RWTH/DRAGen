"""Library interface: one call from an RVE specification to voxel labels, orientations and,
optionally, a tetrahedral mesh. No GUI, no change of the working directory, no global state, no
output to stdout (messages go to `logging`).

Conventions of this interface (the rest of DRAGen works in micrometres and Bunge Euler angles):

Lengths
    Metres. The grain sources (GAN .pkl, CSV) stay in micrometres, as everywhere in DRAGen.
Axes
    Index (i, j, k) of `grain_ids` is (x, y, z); x = rolling direction (RD), y = transverse
    direction (TD), z = normal direction (ND). Grain shapes are tilted by their angle alpha about
    ND. DRAGen does not reorient the input: the grain data must be given in this frame (GAN and
    CSV sources trained/measured in RD x ND sections are).
Grain IDs
    1 ... G, as in DRAGen and the solver files. Row g - 1 of `orientations` and `phase_ids`
    belongs to grain g; `RVEResult.orientation_of(ids)` does the lookup.
Orientations
    Unit quaternions (w, x, y, z) with w >= 0, ACTIVE rotation from the crystal into the sample
    frame: v_sample = q * v_crystal * q^-1. The columns of its rotation matrix are the crystal
    axes [100], [010], [001] expressed in RD, TD, ND (Hamilton product, as scipy). Equivalent to
    Bunge Euler angles (phi1, PHI, phi2), which describe the passive rotation sample -> crystal.
    The four numbers are the same as DAMASK's quaternion of that orientation: DAMASK reads them
    as passive sample -> crystal with its sign convention P = -1, which is the same rotation.
    Example: Goss, Euler (0, 45, 0) deg -> q = (cos 22.5, sin 22.5, 0, 0), [100] || RD, [011] || ND.
Reproducibility
    The same spec (and the same source files) gives the same RVE. The tetrahedral mesh is
    reproducible because gmsh runs on one thread.
"""
import dataclasses
import hashlib
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from dragen.config.phases import Phase
from dragen.config.rve import BandConfig, BoxConfig, InclusionConfig, PhaseConfig, RveConfig

log = logging.getLogger(__name__)

MATRIX_PHASES = ('ferrite', 'martensite', 'pearlite', 'bainite', 'austenite')
"""Phase names a PhaseSpec accepts (DRAGen's fixed phases; bands and inclusions are features)."""

_EULER = ['phi1', 'PHI', 'phi2']


# ---------------------------------------------------------------------- specification

@dataclass(frozen=True)
class PhaseSpec:
    """One phase of the RVE.

    name: one of MATRIX_PHASES.
    fraction: volume fraction; the fractions of all phases add up to 1.
    source: grain data in micrometres. A trained WGAN (.pkl, extra 'ml') or a CSV with one row per
        measured grain: a (semi-axis), optional b, c, alpha (tilt about ND, degrees) and
        phi1, PHI, phi2 (Bunge, degrees). Grain size, shape and, if present, orientation are drawn
        together, so their correlations are kept.
    orientations: optional CSV with columns phi1, PHI, phi2 (Bunge, degrees), one row per measured
        grain. If given, every grain of this phase gets an orientation drawn from it, replacing
        those of the source. Without it the source's orientations are used, or random ones if it
        has none.
    """

    name: str
    fraction: float
    source: Path
    orientations: Optional[Path] = None

    def __post_init__(self):
        name = str(self.name).lower()
        if name not in MATRIX_PHASES:
            raise ValueError(f'unknown phase {self.name!r}, expected one of {MATRIX_PHASES}')
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'source', Path(self.source))
        if self.orientations is not None:
            object.__setattr__(self, 'orientations', Path(self.orientations))


@dataclass(frozen=True)
class BandSpec:
    """Bands of one phase (e.g. martensite bands in DP steel), parallel to a box face.

    count: number of bands.
    thickness_m: (min, max) thickness in metres; each band draws its thickness uniformly.
    source: grain data of the band grains (GAN .pkl or CSV, micrometres, as PhaseSpec.source).
    phase: phase of the band, one of MATRIX_PHASES. Band grains take the orientation file of the
        PhaseSpec of this phase, if there is one.
    normal_axis: 0, 1 or 2 (x, y, z); the bands lie in the plane normal to it. Default: the last
        axis (z, in 2D y).
    filling: volume fraction of each band slab taken by band grains; the rest is matrix.

    The grains of a band are merged into one grain after the tessellation (one orientation per
    band). The phase fractions of the PhaseSpecs refer to the matrix, i.e. the box without the
    band grains.
    """

    count: int
    thickness_m: Tuple[float, float]
    source: Path
    phase: str = 'martensite'
    normal_axis: Optional[int] = None
    filling: float = 0.5

    def __post_init__(self):
        phase = str(self.phase).lower()
        if phase not in MATRIX_PHASES:
            raise ValueError(f'unknown band phase {self.phase!r}, expected one of {MATRIX_PHASES}')
        object.__setattr__(self, 'phase', phase)
        object.__setattr__(self, 'source', Path(self.source))
        object.__setattr__(self, 'thickness_m', tuple(float(t) for t in self.thickness_m))

    def to_config(self) -> BandConfig:
        return BandConfig(count=int(self.count), phase=self.phase, source=str(self.source),
                          thickness=tuple(t * 1e6 for t in self.thickness_m),
                          normal_axis=-1 if self.normal_axis is None else int(self.normal_axis),
                          filling=float(self.filling))


@dataclass(frozen=True)
class InclusionSpec:
    """Non-growing inclusions or pores, placed inside grains after the tessellation (phase
    'inclusions'). fraction: volume fraction of the box; source: their size and shape data
    (GAN .pkl or CSV, micrometres). Inclusions that fit into no grain are left out (warning)."""

    source: Path
    fraction: float

    def __post_init__(self):
        object.__setattr__(self, 'source', Path(self.source))

    def to_config(self) -> InclusionConfig:
        return InclusionConfig(source=str(self.source), fraction=float(self.fraction))


@dataclass(frozen=True)
class RVESpec:
    """What generate_rve builds.

    box_size_m: edge length of a cube, or (x, y, z) edge lengths, in metres; (x, y) for a 2D RVE
        (pixels, triangles; grain_ids (nx, ny)).
    resolution: voxels along x; voxels are cubes, so y and z get size / voxel size (rounded).
        The tetrahedral mesh is built on these voxels.
    phases: the phases with their sources and fractions.
    periodic: all axes, or per axis (x, y, z). A non-periodic axis is a free surface.
    mesh: 'voxel' (labels only) or 'tet' (also a tetrahedral mesh with smoothed grain boundaries;
        2D: triangles).
    tet_order: 1 (4-node) or 2 (10-node) tetrahedra (2D: 3- or 6-node triangles).
    seed: random seed; the same seed and input give the same RVE.
    bands: optional BandSpec.
    inclusions: optional InclusionSpec.
    """

    box_size_m: Union[float, Tuple[float, float, float]]
    resolution: int
    phases: Sequence[PhaseSpec]
    periodic: Union[bool, Tuple[bool, bool, bool]] = True
    mesh: Literal['voxel', 'tet'] = 'voxel'
    tet_order: Literal[1, 2] = 2
    seed: int = 0
    bands: Optional[BandSpec] = None
    inclusions: Optional[InclusionSpec] = None

    def __post_init__(self):
        size = self.box_size_m
        size = (float(size),) * 3 if np.isscalar(size) else tuple(float(s) for s in size)
        if len(size) not in (2, 3):
            raise ValueError(f'box_size_m needs 1, 2 (2D) or 3 entries, got {len(size)}')
        object.__setattr__(self, 'box_size_m', size)
        periodic = self.periodic
        periodic = (bool(periodic),) * len(size) if isinstance(periodic, (bool, np.bool_)) else tuple(bool(p) for p in periodic)
        if len(periodic) != len(size):
            raise ValueError(f'periodic needs {len(size)} entries, got {len(periodic)}')
        object.__setattr__(self, 'periodic', periodic)
        object.__setattr__(self, 'phases', tuple(self.phases))
        if int(self.resolution) < 2:
            raise ValueError(f'resolution must be at least 2 voxels, got {self.resolution}')
        if self.mesh not in ('voxel', 'tet'):
            raise ValueError(f"mesh must be 'voxel' or 'tet', got {self.mesh!r}")
        if self.tet_order not in (1, 2):
            raise ValueError(f'tet_order must be 1 or 2, got {self.tet_order}')
        names = [p.name for p in self.phases]
        if len(set(names)) != len(names):
            raise ValueError(f'phases listed twice: {names}')
        self.to_config()            # all remaining checks (fractions, box) live in RveConfig

    def to_config(self) -> RveConfig:
        """The DRAGen configuration (micrometres) this spec stands for."""
        size_um = tuple(s * 1e6 for s in self.box_size_m)
        box = BoxConfig(size=size_um, resolution=int(self.resolution) / size_um[0], periodic=self.periodic)
        phases = tuple(PhaseConfig(phase=p.name, source=str(p.source), fraction=p.fraction) for p in self.phases)
        extra = {}
        if self.bands is not None:
            band = self.bands.to_config()
            if band.normal_axis < 0:
                from dataclasses import replace
                band = replace(band, normal_axis=len(self.box_size_m) - 1)
            extra['bands'] = band
        if self.inclusions is not None:
            extra['inclusions'] = self.inclusions.to_config()
        return RveConfig(box=box, phases=phases, seed=int(self.seed), **extra)

    def to_dict(self) -> dict:
        def plain(value):
            if isinstance(value, Path):
                return value.as_posix()
            if isinstance(value, (list, tuple)):
                return [plain(v) for v in value]
            if isinstance(value, dict):
                return {k: plain(v) for k, v in value.items()}
            return value
        return plain(dataclasses.asdict(self))


# ---------------------------------------------------------------------- result

@dataclass
class TetMesh:
    """Tetrahedral mesh with smoothed grain boundaries.

    points: (P, 3) float64, metres, box from 0 to box_size_m. 2D: (P, 2) and triangles
        ('triangle', 'triangle6': corners 0-2 counter-clockwise, midside 3 (0-1), 4 (1-2), 5 (2-0)).
    cells: (E, 4) or (E, 10) int64, node order of VTK/Abaqus/meshio ('tetra', 'tetra10'):
        corners 0-3, midside nodes 4 (0-1), 5 (1-2), 6 (0-2), 7 (0-3), 8 (1-3), 9 (2-3).
    grain_ids: (E,) int32, grain of every element (same IDs as RVEResult.grain_ids).
    periodic_pairs: per periodic axis (0 = x, 1 = y, 2 = z) an (n, 2) int64 array of node
        indices: node on the face at 0, node at the same place on the opposite face.
    """

    points: np.ndarray
    cells: np.ndarray
    grain_ids: np.ndarray
    periodic_pairs: Dict[int, np.ndarray]

    @property
    def cell_type(self) -> str:
        return {10: 'tetra10', 4: 'tetra', 6: 'triangle6', 3: 'triangle'}[self.cells.shape[1]]


@dataclass
class RVEResult:
    """A generated RVE; conventions in the module docstring.

    grain_ids: (nx, ny, nz) int32, grain ID 1 ... G per voxel.
    orientations: (G, 4) float64 unit quaternions (w, x, y, z), crystal -> sample, active.
    euler_bunge_deg: (G, 3) float64, the same orientations as Bunge Euler angles in degrees.
    phase_ids: (G,) int32 DRAGen phase ID per grain (1 ferrite, 2 martensite, 3 pearlite,
        4 bainite, 5 austenite, 6 inclusions); `phase_names` maps them to names.
    band_ids: (G,) int32 band index 0 ... of a grain, -1 outside bands. Band k is grain k + 1;
        band grains without contact to their band stay grains of their own with its index.
    radii_m: (G, 3) semi-axes a, b, c of the sampled ellipsoid in metres (for a band the sphere
        of its summed grain volume), alpha_deg: (G,) its tilt about ND. The grain's voxels are
        the result of the tessellation, not this ellipsoid.
    parent_grain_ids, packet_ids: (G,) only after dragen.substructure.apply_substructure: the
        regions are then blocks and grains without substructure; parent grain and packet of each
        region (packet -1 for grains without substructure).
    box_size_m: (x, y, z) edge lengths in metres.
    tet_mesh: only for spec.mesh == 'tet'.
    metadata: dragen version, git commit, seed, SHA-256 of the spec and the input files, run
        times, warnings, phase fractions.
    """

    grain_ids: np.ndarray
    orientations: np.ndarray
    euler_bunge_deg: np.ndarray
    phase_ids: np.ndarray
    phase_names: Dict[int, str]
    box_size_m: Tuple[float, float, float]
    periodic: Tuple[bool, bool, bool]
    band_ids: Optional[np.ndarray] = None
    radii_m: Optional[np.ndarray] = None
    alpha_deg: Optional[np.ndarray] = None
    tet_mesh: Optional[TetMesh] = None
    parent_grain_ids: Optional[np.ndarray] = None
    packet_ids: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.band_ids is None:
            self.band_ids = np.full(len(self.orientations), -1, dtype=np.int32)

    @property
    def n_grains(self) -> int:
        return len(self.orientations)

    @property
    def voxel_size_m(self) -> Tuple[float, float, float]:
        return tuple(s / n for s, n in zip(self.box_size_m, self.grain_ids.shape))

    def orientation_of(self, grain_ids) -> np.ndarray:
        """Quaternions of the given grain IDs (any shape), e.g. per voxel or per element."""
        return self.orientations[np.asarray(grain_ids) - 1]

    def save(self, out_dir: Path) -> None:
        """rve.npz (arrays, metres) and metadata.json into out_dir."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays = dict(grain_ids=self.grain_ids, orientations=self.orientations,
                      euler_bunge_deg=self.euler_bunge_deg, phase_ids=self.phase_ids,
                      band_ids=self.band_ids,
                      **({} if self.radii_m is None else dict(radii_m=self.radii_m, alpha_deg=self.alpha_deg)),
                      **({} if self.parent_grain_ids is None else dict(parent_grain_ids=self.parent_grain_ids,
                                                                       packet_ids=self.packet_ids)),
                      box_size_m=np.array(self.box_size_m), periodic=np.array(self.periodic))
        if self.tet_mesh is not None:
            arrays.update(tet_points=self.tet_mesh.points, tet_cells=self.tet_mesh.cells,
                          tet_grain_ids=self.tet_mesh.grain_ids,
                          **{f'tet_periodic_pairs_{a}': p for a, p in self.tet_mesh.periodic_pairs.items()})
        np.savez_compressed(out_dir / 'rve.npz', **arrays)
        meta = dict(self.metadata, phase_names={str(k): v for k, v in self.phase_names.items()})
        (out_dir / 'metadata.json').write_text(json.dumps(meta, indent=2, default=_json_default), encoding='utf-8')

    @classmethod
    def load(cls, out_dir: Path) -> 'RVEResult':
        """Inverse of save."""
        out_dir = Path(out_dir)
        data = np.load(out_dir / 'rve.npz')
        meta = json.loads((out_dir / 'metadata.json').read_text(encoding='utf-8'))
        names = {int(k): v for k, v in meta.pop('phase_names').items()}
        tet = None
        if 'tet_cells' in data:
            pairs = {int(k.rsplit('_', 1)[1]): data[k] for k in data.files if k.startswith('tet_periodic_pairs_')}
            tet = TetMesh(points=data['tet_points'], cells=data['tet_cells'], grain_ids=data['tet_grain_ids'],
                          periodic_pairs=pairs)
        return cls(grain_ids=data['grain_ids'], orientations=data['orientations'],
                   euler_bunge_deg=data['euler_bunge_deg'], phase_ids=data['phase_ids'], phase_names=names,
                   box_size_m=tuple(float(s) for s in data['box_size_m']),
                   periodic=tuple(bool(p) for p in data['periodic']),
                   band_ids=data['band_ids'] if 'band_ids' in data else None,
                   radii_m=data['radii_m'] if 'radii_m' in data else None,
                   alpha_deg=data['alpha_deg'] if 'alpha_deg' in data else None,
                   parent_grain_ids=data['parent_grain_ids'] if 'parent_grain_ids' in data else None,
                   packet_ids=data['packet_ids'] if 'packet_ids' in data else None,
                   tet_mesh=tet, metadata=meta)


# ---------------------------------------------------------------------- generation

def euler_to_quaternion(euler_bunge_deg: np.ndarray) -> np.ndarray:
    """Bunge Euler angles (degrees) -> unit quaternions (w, x, y, z), active crystal -> sample,
    w >= 0. scipy's intrinsic 'ZXZ' matrix Rz(phi1) Rx(PHI) Rz(phi2) is the transpose of the
    Bunge matrix g (sample -> crystal), i.e. exactly the rotation crystal -> sample."""
    xyzw = Rotation.from_euler('ZXZ', np.asarray(euler_bunge_deg, float).reshape(-1, 3), degrees=True) \
        .as_quat(canonical=True)
    return np.ascontiguousarray(xyzw[:, [3, 0, 1, 2]])


def generate_rve(spec: RVESpec, out_dir: Optional[Path] = None) -> RVEResult:
    """Generate the RVE described by `spec`. With out_dir, also save it there (RVEResult.save)."""
    from dragen.pipeline import generate

    start = time.perf_counter()
    config = spec.to_config()
    log.info('generating RVE: %s voxels, %d phases, seed %d', config.box.shape, len(spec.phases), spec.seed)
    with _capture_warnings() as warnings:
        result = generate(config)
        labels = result.labels.astype(np.int32, copy=False)
        seconds = dict(result.seconds)
        euler = np.array(result.grains.euler, dtype=np.float64)
        _orientations_from_files(spec, euler, np.asarray(result.grains.phase, dtype=np.int32))
        tet, metadata_mesh = None, {}
        if spec.mesh == 'tet':
            t = time.perf_counter()
            tet, metadata_mesh = _tet_mesh(labels, config.box, spec.tet_order)
            seconds['tet_mesh'] = time.perf_counter() - t
    seconds['total'] = time.perf_counter() - start

    out = result_from_generation(config, labels, result.grains, tet=tet, euler=euler, metadata={
        'seed': spec.seed, 'spec': spec.to_dict(), 'spec_sha256': _spec_hash(spec),
        'seconds': seconds, 'warnings': warnings, **metadata_mesh})
    log.info('RVE with %d grains in %.1f s', out.n_grains, seconds['total'])
    if out_dir is not None:
        out.save(Path(out_dir))
    return out


def result_from_generation(config: RveConfig, labels: np.ndarray, grains, tet: Optional[TetMesh] = None,
                           euler: Optional[np.ndarray] = None, metadata: Optional[dict] = None) -> RVEResult:
    """RVEResult from the output of dragen.pipeline.generate (micrometres, GrainTable), e.g. for
    runs made with an RveConfig (GUI, CLI). euler replaces the grains' angles if given."""
    labels = np.asarray(labels).astype(np.int32, copy=False)
    euler = np.array(grains.euler if euler is None else euler, dtype=np.float64)
    phase_ids = np.asarray(grains.phase, dtype=np.int32)
    counts = np.bincount(labels.ravel(), minlength=len(grains) + 1)[1:]
    fractions = {Phase(p).name.lower(): float(counts[phase_ids == p].sum() / labels.size)
                 for p in np.unique(phase_ids)}
    radii = np.asarray(grains.radii, dtype=np.float64) * 1e-6
    if radii.shape[1] == 2:
        radii = np.column_stack([radii, radii[:, :1]])
    return RVEResult(
        grain_ids=labels,
        orientations=euler_to_quaternion(euler),
        euler_bunge_deg=euler,
        phase_ids=phase_ids,
        phase_names={int(p): Phase(p).name.lower() for p in Phase},
        box_size_m=tuple(float(s) * 1e-6 for s in config.box.size),
        periodic=tuple(bool(p) for p in config.box.periodic),
        band_ids=np.asarray(grains.band, dtype=np.int32),
        radii_m=radii,
        alpha_deg=np.asarray(grains.alpha, dtype=np.float64),
        tet_mesh=tet,
        metadata={'dragen_version': _version(), 'git_commit': _git_commit(), 'seed': config.seed,
                  'n_grains': int(len(grains)), 'shape': list(labels.shape),
                  'phase_fractions': fractions, **(metadata or {})},
    )


def tet_mesh_from_arrays(points_um: np.ndarray, cells: np.ndarray, grain: np.ndarray, box: BoxConfig) -> TetMesh:
    """TetMesh (metres, periodic node pairs) from a mesh in micrometres (dragen.mesh output)."""
    from dragen.mesh.tetmesh import periodic_node_pairs
    axes = [a for a, p in enumerate(box.periodic) if p]
    pairs = periodic_node_pairs(points_um, box.size, axes) if axes else {}
    return TetMesh(points=np.asarray(points_um, dtype=np.float64) * 1e-6,
                   cells=np.asarray(cells, dtype=np.int64),
                   grain_ids=np.asarray(grain, dtype=np.int32),
                   periodic_pairs={a: np.asarray(p, dtype=np.int64) for a, p in pairs.items()})


def _orientations_from_files(spec: RVESpec, euler: np.ndarray, phase_ids: np.ndarray) -> None:
    """Replace, per phase with an orientation file, the Euler angles of its grains by rows drawn
    from that file (with replacement). Own random stream, so the geometry does not depend on it."""
    rng = np.random.default_rng([int(spec.seed), 1])
    for p in spec.phases:
        if p.orientations is None:
            continue
        data = pd.read_csv(p.orientations)
        missing = [c for c in _EULER if c not in data]
        if missing:
            raise ValueError(f'{p.orientations}: columns {missing} missing (need phi1, PHI, phi2 in degrees)')
        table = data[_EULER].dropna().to_numpy(dtype=float)
        if not len(table):
            raise ValueError(f'{p.orientations}: no orientations')
        mine = np.flatnonzero(phase_ids == int(Phase.parse(p.name)))
        euler[mine] = table[rng.integers(len(table), size=len(mine))]


def _tet_mesh(labels: np.ndarray, box: BoxConfig, order: int):
    if labels.ndim == 2:
        return _tri_mesh(labels, box, order)
    from dragen.mesh.mesher import MeshConfig, mesh_rve
    from dragen.mesh.tetmesh import TetMeshConfig

    result = mesh_rve(labels, box.spacing, box.periodic,
                      MeshConfig(tetmesh=TetMeshConfig(order=order, verbosity=0)))
    mesh = result.tets.mesh
    tet = tet_mesh_from_arrays(mesh.points, mesh.cells, mesh.grain, box)
    feedback = result.stats.get('feedback', [])
    best = feedback[result.stats.get('feedback_best_round', 0)] if feedback else {}
    meta = {'tet_mesh': {'elements': int(len(mesh.cells)), 'nodes': int(len(mesh.points)),
                         'min_dihedral_deg': best.get('min_dihedral'),
                         'below_min_dihedral': best.get('below_min_dihedral'),
                         'cleanup_changed_voxels': result.stats.get('cleanup', {}).get('changed_voxels')}}
    return tet, meta


def _tri_mesh(labels: np.ndarray, box: BoxConfig, order: int):
    from dragen.mesh.mesh2d import Mesh2DConfig, mesh_rve_2d

    result = mesh_rve_2d(labels, box.spacing, box.periodic, Mesh2DConfig(order=order))
    tet = tet_mesh_from_arrays(result.points, result.cells, result.grain, box)
    return tet, {'tri_mesh': {k: v for k, v in result.stats.items() if not isinstance(v, (list, dict))}}


class _capture_warnings:
    """Collects WARNING records of the dragen loggers during a run, for the metadata."""

    def __enter__(self) -> List[str]:
        self.records: List[str] = []
        records = self.records

        class Handler(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        self.handler = Handler(level=logging.WARNING)
        self.logger = logging.getLogger('dragen')
        self.logger.addHandler(self.handler)
        return self.records

    def __exit__(self, *exc):
        self.logger.removeHandler(self.handler)
        return False


def _file_hash(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _spec_hash(spec: RVESpec) -> str:
    """SHA-256 over the spec and the content of every input file (so a changed source changes it)."""
    files = {}
    for p in spec.phases:
        files[p.source.as_posix()] = _file_hash(p.source)
        if p.orientations is not None:
            files[p.orientations.as_posix()] = _file_hash(p.orientations)
    for feature in (spec.bands, spec.inclusions):
        if feature is not None:
            files[feature.source.as_posix()] = _file_hash(feature.source)
    payload = json.dumps({'spec': spec.to_dict(), 'files': files}, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _version() -> Optional[str]:
    try:
        from importlib.metadata import version
        return version('dragen')
    except Exception:
        return None


def _git_commit() -> Optional[str]:
    """Commit of the DRAGen checkout, with '-dirty' for uncommitted changes; None outside git."""
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True,
                                text=True, timeout=5, check=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return commit + ('-dirty' if dirty else '')
    except Exception:
        return None


def _json_default(value):
    """numpy scalars and arrays in metadata (e.g. from a configuration) as plain JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f'{type(value).__name__} is not JSON serializable')
