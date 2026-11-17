"""Generation pipeline: grain sampling -> RSA (bands first) -> tessellation -> repair -> band
merge -> inclusions.

Solver export follows in a later phase (E1).

Volume bookkeeping: band grains hold `filling` times the volume of their band slab; the matrix
phases share the rest of the box according to their fractions. Matrix grains are not seeded
inside a band. After the tessellation all grains of a band are merged into one grain (author's
decision 29.10.2026): the band keeps the rough, interlocked boundary its grains grew, but it is a
single grain with one orientation for the later stages. Inclusions are cut out of the grains after tessellation, their volume fraction
refers to the whole box. An inclusion only fits into a grain that can hold it together with its
shell; inclusions that fit nowhere are left out and logged.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from dragen.config.phases import Phase
from dragen.config.rve import RveConfig
from dragen.core.domain import VoxelDomain
from dragen.core.grains import GrainTable
from dragen.core.kernel import build_kernel
from dragen.sources import open_source
from dragen.stages.bands import Band, place_bands
from dragen.stages.inclusions import InclusionPlacer, InclusionStats
from dragen.stages.repair import merge_detached_pieces
from dragen.stages.rsa import RsaPlacer
from dragen.stages.tessellation import TessellationStats, Tessellator


@dataclass
class RveResult:
    labels: np.ndarray
    """Grain ID per voxel; grain i of `grains` has ID i + 1. Band k is grain k + 1 (its grains
    merged), then band grains detached from their band, the matrix grains, the inclusions (phase
    INCLUSIONS)."""
    grains: GrainTable
    domain: VoxelDomain
    target_voxels: np.ndarray
    """Discrete target volume per grain (voxels of the unshrunk ellipsoid)."""
    bands: List[Band] = field(default_factory=list)
    band_grains: List[dict] = field(default_factory=list)
    """Per band the number of grains merged into it and of band grains left detached."""
    seconds: Dict[str, float] = field(default_factory=dict)
    tessellation: TessellationStats = None
    inclusions: InclusionStats = None
    rsa_attempts: int = 0
    repaired_voxels: int = 0


def _by_volume(grains: GrainTable) -> GrainTable:
    """Largest first, as in the legacy RSA."""
    return grains.take(np.argsort(-grains.volume, kind='stable'))


def _source_limits(domain: VoxelDomain) -> dict:
    return dict(min_radius=min(domain.spacing), max_radius=min(domain.size) / 2)


def sample_band_grains(config: RveConfig, domain: VoxelDomain, bands: List[Band],
                       rng: np.random.Generator) -> List[GrainTable]:
    if not bands:
        return []
    source = open_source(config.bands.source, config.bands.phase, ndim=domain.ndim)
    tables = []
    for k, band in enumerate(bands):
        slab_volume = np.count_nonzero(band.mask(domain)) * domain.voxel_volume
        table = source.sample(config.bands.filling * slab_volume, rng, **_source_limits(domain),
                              max_extent=(band.axis, band.thickness))
        tables.append(_by_volume(table).with_band(k))
    return tables


def sample_grains(config: RveConfig, domain: VoxelDomain, rng: np.random.Generator,
                  band_volume: float = 0.0) -> GrainTable:
    matrix_volume = config.box.volume - band_volume
    tables = []
    for phase in config.phases:
        source = open_source(phase.source, phase.phase, ndim=domain.ndim)
        tables.append(source.sample(matrix_volume * phase.fraction, rng, **_source_limits(domain)))
    return _by_volume(GrainTable.concat(tables))


def growth_rates(grains: GrainTable, shrink: float, bin_size: float, margin: float, exponent: float):
    """Relative growth rate per grain and axis, and per grain the epoch at which its kernel
    reaches margin * radii.

    rate[g, i] = bin_size / (R_i ** (1 - exponent) * r_g ** exponent)

    R_i is the largest shrunk radius along axis i over all grains, r_g the largest shrunk radius
    of grain g. exponent = 0 is the legacy law (Henrich et al. 2020): every grain grows by the
    same factor per epoch, so small grains advance their front by a fraction of a voxel per epoch.
    exponent = 1 moves the front of every grain by about one voxel per epoch along its longest
    axis, keeping its shape.
    """
    r0 = grains.radii * shrink
    axis_max = r0.max(axis=0)
    grain_max = r0.max(axis=1, keepdims=True)
    rate = bin_size / (axis_max ** (1 - exponent) * grain_max ** exponent)
    max_epoch = np.ceil(((margin / shrink) - 1) / rate.min(axis=1)).astype(int)
    return rate, max_epoch


def generate(config: RveConfig) -> RveResult:
    rng = np.random.default_rng(config.seed)
    domain = VoxelDomain.from_box(config.box)
    seconds = {}

    t = time.perf_counter()
    bands = place_bands(config.bands, domain, rng) if config.bands.count else []
    band_tables = sample_band_grains(config, domain, bands, rng)
    band_volume = config.bands.filling * sum(np.count_nonzero(b.mask(domain)) for b in bands) * domain.voxel_volume
    matrix = sample_grains(config, domain, rng, band_volume=band_volume)
    grains = GrainTable.concat(band_tables + [matrix])
    seconds['sampling'] = time.perf_counter() - t

    t = time.perf_counter()
    shrink = config.rsa.shrink_factor
    rate, max_epoch = growth_rates(grains, shrink, bin_size=min(domain.spacing),
                                   margin=config.tessellation.kernel_margin,
                                   exponent=config.tessellation.growth_exponent)
    kernels = [build_kernel(r, a, domain.spacing, shrink, g, k)
               for r, a, g, k in zip(grains.radii, grains.alpha, rate, max_epoch)]
    seconds['kernels'] = time.perf_counter() - t

    labels = domain.new_labels()
    t = time.perf_counter()
    placer = RsaPlacer(domain, config.rsa, rng)
    centers, start = [], 0
    for band, table in zip(bands, band_tables):
        centers.append(placer.place(labels, kernels[start:start + len(table)], first_id=start + 1,
                                    region=band.mask(domain)))
        start += len(table)
    # matrix grains may grow into a band but not be seeded there (at most 10 % of a seed inside)
    outside_bands = None
    if bands:
        outside_bands = ~np.any([b.mask(domain) for b in bands], axis=0)
    centers.append(placer.place(labels, kernels[start:], first_id=start + 1, region=outside_bands))
    centers = np.concatenate(centers)
    seconds['rsa'] = time.perf_counter() - t

    t = time.perf_counter()
    tessellator = Tessellator(domain, config.tessellation, rng)
    tessellator.run(labels, kernels, centers, no_fill_up=grains.band >= 0)
    seconds['tessellation'] = time.perf_counter() - t

    t = time.perf_counter()
    repaired = merge_detached_pieces(labels, domain)
    seconds['repair'] = time.perf_counter() - t

    grains = grains.with_centers(centers)
    targets = [k.target_voxels for k in kernels]
    band_grains = []
    if bands:
        labels, grains, targets, band_grains = merge_bands(labels, grains, targets, bands, domain)
    inclusion_stats = None
    if config.inclusions.fraction > 0:
        t = time.perf_counter()
        source = open_source(config.inclusions.source, Phase.INCLUSIONS, ndim=domain.ndim)
        inclusions = _by_volume(source.sample(config.inclusions.fraction * config.box.volume, rng,
                                              **_source_limits(domain)))
        inclusion_kernels = [build_kernel(r, a, domain.spacing, 1.0, np.zeros(domain.ndim), 0)
                             for r, a in zip(inclusions.radii, inclusions.alpha)]
        inclusion_placer = InclusionPlacer(domain, rng)
        placed, inclusion_centers = inclusion_placer.place(labels, inclusion_kernels, first_id=len(grains) + 1)
        grains = GrainTable.concat([grains, inclusions.take(placed).with_centers(inclusion_centers)])
        targets += [inclusion_kernels[i].target_voxels for i in placed]
        inclusion_stats = inclusion_placer.stats
        if inclusion_stats.not_placed:
            logging.getLogger(__name__).warning(
                '%d of %d inclusions fit into no grain and were left out',
                inclusion_stats.not_placed, inclusion_stats.not_placed + inclusion_stats.placed)
        seconds['inclusions'] = time.perf_counter() - t

    return RveResult(labels=labels, grains=grains, domain=domain, target_voxels=np.array(targets),
                     bands=bands, band_grains=band_grains, seconds=seconds, tessellation=tessellator.stats,
                     inclusions=inclusion_stats, rsa_attempts=placer.attempts, repaired_voxels=repaired)


def merge_bands(labels: np.ndarray, grains: GrainTable, target_voxels: List[int], bands: List[Band],
                domain: VoxelDomain):
    """Merges the grains of band k into grain k + 1. Only the grains of the largest connected piece
    of the band (across periodic boundaries) are merged; a band grain without contact to it stays
    a grain of its own (band phase and index, own orientation), so phase fractions do not change
    and every grain stays one body. Order of the IDs: merged bands, detached band grains, matrix
    grains.

    The row of a band in the grain table describes the merged grain: radii of the sphere with the
    summed volume of its grains, the orientation of its first grain, its phase, the centre of the
    slab. Its target is the sum of its grains' targets. Returns labels, grains, targets and per
    band the number of merged and of detached grains.
    """
    from dragen.analysis.metrics import component_labels
    n_bands = len(bands)
    band = grains.band
    n = len(grains)
    flat = labels.ravel()
    # the piece of the band union every band grain lies in (a grain is one body after the repair)
    owner = np.where(flat > 0, np.concatenate([[-1], band])[np.clip(flat, 0, n)], -1)
    band_labels = np.where(owner >= 0, owner + 1, 0).reshape(labels.shape)
    piece = component_labels(band_labels, domain.periodic)
    merged = np.zeros(n, dtype=bool)
    for k in range(n_bands):
        mine = np.flatnonzero(owner == k)
        if not len(mine):
            continue
        pieces, size = np.unique(piece[mine], return_counts=True)
        main = pieces[np.argmax(size)]
        grains_in_main = np.unique(flat[mine[piece[mine] == main]]) - 1
        merged[grains_in_main] = True
    detached = np.flatnonzero((band >= 0) & ~merged)
    matrix = np.flatnonzero(band < 0)

    new_id = np.zeros(n + 1, dtype=labels.dtype)
    new_id[1:][merged] = band[merged] + 1
    new_id[1:][detached] = n_bands + 1 + np.arange(len(detached))
    new_id[1:][matrix] = n_bands + len(detached) + 1 + np.arange(len(matrix))
    labels = new_id[labels]

    volume = np.bincount(band[merged], weights=grains.volume[merged], minlength=n_bands)
    first = np.array([np.flatnonzero(merged & (band == k))[0] for k in range(n_bands)])
    if domain.ndim == 3:
        r = (3 * volume / (4 * np.pi)) ** (1 / 3)
    else:
        r = np.sqrt(volume / np.pi)
    centre = np.tile((np.asarray(domain.shape) // 2).astype(float), (n_bands, 1))
    for k, b in enumerate(bands):
        centre[k, b.axis] = np.floor(b.center / domain.spacing[b.axis])
    band_rows = GrainTable(radii=np.repeat(r[:, None], domain.ndim, axis=1), alpha=np.zeros(n_bands),
                           euler=grains.euler[first], phase=grains.phase[first], band=np.arange(n_bands),
                           centers=centre.astype(grains.centers.dtype))
    grains_out = GrainTable.concat([band_rows, grains.take(detached), grains.take(matrix)])
    targets = np.asarray(target_voxels)
    band_targets = np.bincount(band[merged], weights=targets[merged], minlength=n_bands)
    targets = list(np.round(band_targets).astype(int)) + list(targets[detached]) + list(targets[matrix])
    counts = {'merged': np.bincount(band[merged], minlength=n_bands),
              'detached': np.bincount(band[detached], minlength=n_bands)}
    stats = [{'merged': int(counts['merged'][k]), 'detached': int(counts['detached'][k])} for k in range(n_bands)]
    return labels, grains_out, targets, stats


def expected_phase_fractions(config: RveConfig, bands: List[Band], domain: VoxelDomain) -> Dict[int, float]:
    """Volume fraction each phase should end up with, given the band and inclusion settings."""
    band_fraction = config.bands.filling * sum(np.count_nonzero(b.mask(domain)) for b in bands) / domain.n_voxels
    keep = 1.0 - config.inclusions.fraction
    out: Dict[int, float] = {}
    for p in config.phases:
        out[int(p.phase)] = out.get(int(p.phase), 0.0) + (1 - band_fraction) * p.fraction * keep
    if bands:
        out[int(config.bands.phase)] = out.get(int(config.bands.phase), 0.0) + band_fraction * keep
    if config.inclusions.fraction > 0:
        out[int(Phase.INCLUSIONS)] = config.inclusions.fraction
    return out
