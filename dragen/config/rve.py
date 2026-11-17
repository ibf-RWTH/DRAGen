"""Configuration dataclasses. Lengths are in micrometres, angles in degrees."""
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import yaml

from dragen.config.phases import Phase


@dataclass(frozen=True)
class BoxConfig:
    """Size, resolution and periodicity of the RVE box.

    size: edge length per axis; two entries give a 2D RVE, three a 3D RVE.
    resolution: voxels per micrometre.
    periodic: per axis; defaults to fully periodic. A non-periodic axis models a free surface,
        e.g. the thickness direction of a sheet.
    """

    size: Tuple[float, ...]
    resolution: float
    periodic: Optional[Tuple[bool, ...]] = None

    def __post_init__(self):
        size = tuple(float(s) for s in self.size)
        if len(size) not in (2, 3):
            raise ValueError(f'box size needs 2 or 3 entries, got {len(size)}')
        if any(s <= 0 for s in size):
            raise ValueError(f'box size must be positive, got {size}')
        if self.resolution <= 0:
            raise ValueError(f'resolution must be positive, got {self.resolution}')
        periodic = (True,) * len(size) if self.periodic is None else tuple(bool(p) for p in self.periodic)
        if len(periodic) != len(size):
            raise ValueError(f'periodic needs {len(size)} entries, got {len(periodic)}')
        object.__setattr__(self, 'size', size)
        object.__setattr__(self, 'periodic', periodic)
        if min(self.shape) < 2:
            raise ValueError(f'resolution {self.resolution} gives fewer than 2 voxels on an axis: {self.shape}')

    @property
    def ndim(self) -> int:
        return len(self.size)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Voxels per axis: size * resolution, rounded to the nearest integer."""
        return tuple(max(1, int(round(s * self.resolution))) for s in self.size)

    @property
    def spacing(self) -> Tuple[float, ...]:
        """Exact voxel edge length per axis, so that shape * spacing == size."""
        return tuple(s / n for s, n in zip(self.size, self.shape))

    @property
    def volume(self) -> float:
        return math.prod(self.size)


@dataclass(frozen=True)
class PhaseConfig:
    """One phase: where its grains come from and which volume fraction it takes.

    source: .csv with columns a[, b, c, alpha, phi1, PHI, phi2] or a trained WGAN .pkl.
    """

    phase: Phase
    source: str
    fraction: float

    def __post_init__(self):
        object.__setattr__(self, 'phase', Phase.parse(self.phase))
        if not 0 < self.fraction <= 1:
            raise ValueError(f'{self.phase.name}: fraction must be in (0, 1], got {self.fraction}')


@dataclass(frozen=True)
class RsaConfig:
    """Discrete random sequential addition.

    shrink_volume_fraction: each ellipsoid is placed with this fraction of its volume (legacy 0.3).
    max_overlap: accepted fraction of a seed lying on already occupied voxels (legacy 0.05).
    max_attempts_per_grain: failed placements before the run is aborted.
    """

    shrink_volume_fraction: float = 0.3
    max_overlap: float = 0.05
    max_attempts_per_grain: int = 10000

    def __post_init__(self):
        if not 0 < self.shrink_volume_fraction <= 1:
            raise ValueError(f'shrink_volume_fraction must be in (0, 1], got {self.shrink_volume_fraction}')
        if not 0 <= self.max_overlap < 1:
            raise ValueError(f'max_overlap must be in [0, 1), got {self.max_overlap}')

    @property
    def shrink_factor(self) -> float:
        """Factor applied to every radius."""
        return self.shrink_volume_fraction ** (1 / 3)


@dataclass(frozen=True)
class TessellationConfig:
    """Discrete tessellation (ellipsoid growth with volume control).

    connected_growth: a grain may only take voxels face-adjacent to its own body, so no grain
        ends up split into several bodies.
    kernel_margin: how far a grain can grow, as a multiple of its unshrunk radii. Voxels no
        kernel reaches are filled from adjacent grains at the end. At 2.0 that fill-up was empty
        in all benchmark cases; 1.25 left about 4 % of the volume to it.
    growth_exponent: 0 is the growth law of the legacy generator (every grain grows by the same
        factor per epoch), 1 moves every grain front by about one voxel per epoch. Values in
        between speed up small grains, which the legacy law lets larger neighbours enclose.
        See dragen.pipeline.growth_rates. The default 0.3 is calibrated on three cases with five
        seeds each (benchmarks/calibrate_growth.py): 14 of 15 runs met every acceptance criterion,
        against 6 of 15 with 0.
    """

    connected_growth: bool = True
    kernel_margin: float = 2.0
    growth_exponent: float = 0.3

    def __post_init__(self):
        if self.kernel_margin < 1:
            raise ValueError(f'kernel_margin must be >= 1, got {self.kernel_margin}')
        if not 0 <= self.growth_exponent <= 1:
            raise ValueError(f'growth_exponent must be in [0, 1], got {self.growth_exponent}')


@dataclass(frozen=True)
class BandConfig:
    """Bands of a second phase (e.g. martensite bands in DP steel).

    count: number of bands; 0 switches the feature off.
    thickness: (min, max) band thickness, sampled uniformly per band.
    normal_axis: axis normal to the band plane (2 = bands in the x-y plane).
    filling: volume fraction of each band taken by band grains.
    """

    count: int = 0
    phase: Phase = Phase.MARTENSITE
    source: Optional[str] = None
    thickness: Tuple[float, float] = (0.0, 0.0)
    normal_axis: int = 2
    filling: float = 0.5

    def __post_init__(self):
        object.__setattr__(self, 'phase', Phase.parse(self.phase))
        object.__setattr__(self, 'thickness', tuple(float(t) for t in self.thickness))
        if self.count < 0:
            raise ValueError(f'band count must be >= 0, got {self.count}')
        if self.count:
            if self.source is None:
                raise ValueError('bands need a grain source')
            lo, hi = self.thickness
            if not 0 < lo <= hi:
                raise ValueError(f'band thickness needs 0 < min <= max, got {self.thickness}')
            if not 0 < self.filling < 1:
                raise ValueError(f'band filling must be in (0, 1), got {self.filling}')


@dataclass(frozen=True)
class InclusionConfig:
    """Non-growing inclusions or pores, placed inside grains after tessellation."""

    source: Optional[str] = None
    fraction: float = 0.0

    def __post_init__(self):
        if self.fraction < 0 or self.fraction >= 1:
            raise ValueError(f'inclusion fraction must be in [0, 1), got {self.fraction}')
        if self.fraction > 0 and self.source is None:
            raise ValueError('inclusions need a grain source')


@dataclass(frozen=True)
class RveConfig:
    """Everything one generation run needs."""

    box: BoxConfig
    phases: Tuple[PhaseConfig, ...]
    rsa: RsaConfig = field(default_factory=RsaConfig)
    tessellation: TessellationConfig = field(default_factory=TessellationConfig)
    bands: BandConfig = field(default_factory=BandConfig)
    inclusions: InclusionConfig = field(default_factory=InclusionConfig)
    seed: Optional[int] = None
    n_rves: int = 1

    def __post_init__(self):
        phases = tuple(self.phases)
        if not phases:
            raise ValueError('at least one phase is required')
        ids = [p.phase for p in phases]
        if len(set(ids)) != len(ids):
            raise ValueError(f'phases listed twice: {ids}')
        total = sum(p.fraction for p in phases)
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f'phase fractions must add up to 1, got {total}')
        if self.bands.count and self.bands.normal_axis >= self.box.ndim:
            raise ValueError(f'band normal_axis {self.bands.normal_axis} does not exist in a {self.box.ndim}D box')
        if self.n_rves < 1:
            raise ValueError(f'n_rves must be >= 1, got {self.n_rves}')
        object.__setattr__(self, 'phases', phases)

    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        def plain(value):
            if isinstance(value, Phase):
                return value.name.lower()
            if isinstance(value, dict):
                return {k: plain(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(v) for v in value]
            return value
        return plain(asdict(self))

    @classmethod
    def from_dict(cls, data: dict) -> 'RveConfig':
        data = dict(data)
        return cls(
            box=BoxConfig(**data.pop('box')),
            phases=tuple(PhaseConfig(**p) for p in data.pop('phases')),
            rsa=RsaConfig(**data.pop('rsa', {})),
            tessellation=TessellationConfig(**data.pop('tessellation', {})),
            bands=BandConfig(**data.pop('bands', {})),
            inclusions=InclusionConfig(**data.pop('inclusions', {})),
            **data,
        )

    def save(self, path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding='utf-8')

    @classmethod
    def load(cls, path) -> 'RveConfig':
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding='utf-8')))
