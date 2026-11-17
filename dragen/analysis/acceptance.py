"""Acceptance criteria every generated RVE has to meet.

The limits are calibrated on the legacy generator (benchmarks/results/legacy) so the rework must
be at least as faithful to its input, and they add hard requirements the legacy code misses:
no grain may be split into several bodies, because the boundary meshing builds one closed
surface per grain.

The lower volume bound (p10) applies to the main body of each grain. The legacy code reaches a
p10 of 0.68-0.89 over all voxels of a grain, but only 0.62-0.78 when the pieces it scattered
behind neighbouring grains are not counted (benchmarks/results/legacy, lognormal and gan_dual
cases). Without those pieces small grains enclosed by larger ones stay below target.

The upper bound (p90) was 1.20 (legacy: 1.11-1.19) and is 1.25 since the growth exponent 0.3
became the default: speeding up small grains lets some of them overshoot. Over the calibration
runs p90 reached at most 1.204 at exponent 0.3, while the worst p10 rose from 0.40 to 0.685.
"""
from dataclasses import dataclass
from typing import Dict, List

from dragen.analysis.metrics import RveReport


@dataclass(frozen=True)
class Limits:
    min_fill_ratio: float = 1.0
    max_empty_grains: int = 0
    max_fragmented_grains: int = 0
    volume_ratio_median: tuple = (0.95, 1.05)
    min_main_body_ratio_p10: float = 0.60
    max_volume_ratio_p90: float = 1.25
    max_volume_ks: float = 0.12
    max_phase_fraction_error: float = 0.03


DEFAULT_LIMITS = Limits()


def violations(rep: RveReport, target_phase_fractions: Dict[int, float],
               limits: Limits = DEFAULT_LIMITS) -> List[str]:
    """Return one readable line per violated criterion; an empty list means accepted."""
    out = []
    if rep.fill_ratio < limits.min_fill_ratio:
        out.append(f'fill_ratio {rep.fill_ratio:.4f} < {limits.min_fill_ratio}')
    if rep.empty_grains > limits.max_empty_grains:
        out.append(f'{rep.empty_grains} grains have no voxel')
    if rep.fragmented_grains > limits.max_fragmented_grains:
        out.append(f'{rep.fragmented_grains} of {rep.n_grains} grains are split into several bodies')
    lo, hi = limits.volume_ratio_median
    if not lo <= rep.volume_ratio_median <= hi:
        out.append(f'median volume ratio {rep.volume_ratio_median:.3f} outside [{lo}, {hi}]')
    if rep.main_body_ratio_p10 < limits.min_main_body_ratio_p10:
        out.append(f'p10 main-body volume ratio {rep.main_body_ratio_p10:.3f} < {limits.min_main_body_ratio_p10}')
    if rep.volume_ratio_p90 > limits.max_volume_ratio_p90:
        out.append(f'p90 volume ratio {rep.volume_ratio_p90:.3f} > {limits.max_volume_ratio_p90}')
    if rep.volume_ks > limits.max_volume_ks:
        out.append(f'volume KS distance {rep.volume_ks:.3f} > {limits.max_volume_ks}')
    for phase, target in target_phase_fractions.items():
        actual = rep.phase_fractions.get(int(phase), 0.0)
        if abs(actual - target) > limits.max_phase_fraction_error:
            out.append(f'phase {phase}: fraction {actual:.3f}, target {target:.3f}')
    return out


@dataclass(frozen=True)
class FeatureLimits:
    max_band_filling_error: float = 0.10
    """absolute deviation of the filling inside each band slab from the configured filling"""
    max_band_outside_fraction: float = 0.20
    """share of a band's grain volume that grew out of its slab"""
    max_inclusion_fraction_error: float = 0.25
    """relative deviation of the volume fraction of the placed inclusions from the volume they were
    sampled with (an inclusion that fits into no grain is left out on purpose and not counted)"""


def feature_violations(bands: List[dict], inclusions: dict, band_filling: float, inclusion_fraction: float,
                       limits: FeatureLimits = FeatureLimits()) -> List[str]:
    """Criteria for bands (dragen.analysis.features.band_report) and inclusions (inclusion_report).

    Inclusions only go into grains that can hold them (Henrich et al. 2023: inside the grown
    grains); inclusions that fit nowhere are not a violation. Checked are that every placed
    inclusion is complete (`placed_volume_fraction` against `expected_placed_fraction`) and that
    none touches a grain boundary.
    """
    out = []
    for b in bands:
        if abs(b['filling'] - band_filling) > limits.max_band_filling_error:
            out.append(f"band {b['band']}: filling {b['filling']:.3f}, target {band_filling:.3f}")
        if b['outside_fraction'] > limits.max_band_outside_fraction:
            out.append(f"band {b['band']}: {b['outside_fraction']:.1%} of its grain volume lies outside the band")
        if b.get('pieces', 1) != 1:
            out.append(f"band {b['band']}: the band grain consists of {b['pieces']} pieces")
    if inclusion_fraction > 0 and inclusions['placed']:
        if inclusions['touching_boundary']:
            out.append(f"{inclusions['touching_boundary']} inclusions touch a grain boundary or another inclusion")
        expected = inclusions['expected_placed_fraction']
        error = abs(inclusions['volume_fraction'] - expected) / expected
        if error > limits.max_inclusion_fraction_error:
            out.append(f"placed inclusions fill {inclusions['volume_fraction']:.4f} of the box, "
                       f"their ellipsoids {expected:.4f}")
    return out
