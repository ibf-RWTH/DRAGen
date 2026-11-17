"""Settings of the martensite substructure (packets, blocks, orientations)."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from dragen.config.phases import Phase

ORIENTATION_MODES = ('KS', 'experimental')
THICKNESS_COLUMN = 'block_thickness'


@dataclass(frozen=True)
class SubstructureConfig:
    """One substructure run. Lengths in micrometres, sizes in voxels.

    block_thickness: mean block thickness ('user' mode, as t_mu in the legacy code), or
    block_file: CSV with a column 'block_thickness' (EBSD); each block draws its thickness from
        the values in [lower_percentile, upper_percentile] above min_physical_thickness.
    phases: phases that get packets and blocks (names or IDs; legacy: martensite, pearlite,
        bainite).
    min_packet_voxels: a grain is split by k-means into min(4, voxels // min_packet_voxels + 1)
        packets.
    min_block_voxels: packets with at most this many voxels stay one block.
    min_voxels_per_packet / min_voxels_per_block: smaller packets / blocks are merged into the
        neighbour (26-neighbourhood, same grain / packet) with the most contacts.
    orientation_mode: 'KS' (24 Kurdjumov-Sachs variants, packet = habit plane class, block = pair
        of variants) or 'experimental' (measured parent -> child transformations from
        parent_orientation_file and child_orientation_file, both CSV with grain_id, phi1, PHI,
        phi2; the parent file may omit grain_id, rows are then grains 1, 2, ...).
    seed: random seed of k-means, block planes and variants.
    """

    block_thickness: Optional[float] = None
    block_file: Optional[str] = None
    lower_percentile: float = 5.0
    upper_percentile: float = 95.0
    min_physical_thickness: float = 0.1
    phases: Tuple = ('martensite', 'pearlite', 'bainite')
    min_packet_voxels: int = 100
    min_block_voxels: int = 10
    min_voxels_per_packet: int = 5
    min_voxels_per_block: int = 5
    max_merge_iterations: int = 50
    orientation_mode: str = 'KS'
    parent_orientation_file: Optional[str] = None
    child_orientation_file: Optional[str] = None
    seed: int = 42
    phase_ids: Tuple[int, ...] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'phase_ids', tuple(int(Phase.parse(p)) for p in self.phases))
        object.__setattr__(self, 'phases', tuple(Phase(p).name.lower() for p in self.phase_ids))
        if (self.block_thickness is None) == (self.block_file is None):
            raise ValueError('give either block_thickness (micrometres) or block_file (EBSD csv)')
        if self.block_thickness is not None and self.block_thickness <= 0:
            raise ValueError(f'block_thickness must be > 0, got {self.block_thickness}')
        if self.block_file is not None:
            _check_csv(self.block_file, 'block_file', [THICKNESS_COLUMN])
        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise ValueError('percentiles need 0 <= lower < upper <= 100')
        for name in ('min_packet_voxels', 'min_block_voxels', 'min_voxels_per_packet',
                     'min_voxels_per_block', 'max_merge_iterations'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be >= 1')
        if self.orientation_mode not in ORIENTATION_MODES:
            raise ValueError(f'orientation_mode must be one of {ORIENTATION_MODES}, got {self.orientation_mode!r}')
        if self.orientation_mode == 'experimental':
            if not self.parent_orientation_file or not self.child_orientation_file:
                raise ValueError("orientation_mode 'experimental' needs parent_orientation_file and "
                                 'child_orientation_file')
            _check_csv(self.parent_orientation_file, 'parent_orientation_file', ['phi1', 'PHI', 'phi2'])
            _check_csv(self.child_orientation_file, 'child_orientation_file', ['grain_id', 'phi1', 'PHI', 'phi2'])

    @property
    def mode(self) -> str:
        """'user' (one mean thickness) or 'file' (measured distribution)."""
        return 'user' if self.block_thickness is not None else 'file'

    def to_dict(self) -> dict:
        out = asdict(self)
        out.pop('phase_ids')
        out['phases'] = list(out['phases'])
        return out

    @classmethod
    def from_dict(cls, data: dict) -> 'SubstructureConfig':
        data = dict(data)
        if 'phases' in data:
            data['phases'] = tuple(data['phases'])
        return cls(**data)


def _check_csv(path, label, columns):
    if not Path(path).is_file():
        raise FileNotFoundError(f'{label} not found: {path}')
    if not str(path).lower().endswith('.csv'):
        raise ValueError(f'{label} must be a csv, got {path}')
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in columns if c not in header]
    if missing:
        raise ValueError(f'{label} ({path}) misses column(s) {missing}')
