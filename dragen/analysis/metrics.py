"""Metrics on a voxel label field, independent of the generator that produced it.

A label field is an integer array with one grain ID per voxel. IDs > 0 are grains, 0 is
unassigned space. Negative IDs are legacy markers (bands, inclusions) and count as assigned but
belong to no grain. Every function takes plain numpy arrays so the legacy generator and the
rework can be measured with the same code.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import ks_2samp


def fill_ratio(labels: np.ndarray) -> float:
    """Fraction of voxels that carry any label (grain, band or inclusion)."""
    return float(np.count_nonzero(labels) / labels.size)


def grain_voxel_counts(labels: np.ndarray, n_grains: Optional[int] = None) -> np.ndarray:
    """Voxel count per grain ID; index i holds grain i, index 0 is unused."""
    positive = labels[labels > 0].ravel()
    minlength = (n_grains + 1) if n_grains is not None else 0
    return np.bincount(positive, minlength=minlength)


def phase_fractions(labels: np.ndarray, phase_of_grain: np.ndarray) -> dict:
    """Volume fraction per phase ID over all grain voxels.

    phase_of_grain[i] is the phase of grain i; index 0 is ignored.
    """
    counts = grain_voxel_counts(labels, n_grains=len(phase_of_grain) - 1)
    phases = np.asarray(phase_of_grain)[1:len(counts)]
    total = counts[1:].sum()
    if total == 0:
        return {}
    return {int(p): float(counts[1:][phases == p].sum() / total) for p in np.unique(phases)}


def component_labels(labels: np.ndarray, periodic: Sequence[bool]) -> np.ndarray:
    """Flat array assigning every voxel the ID of its face-connected same-label component.

    Neighbours across the box boundary are connected on axes flagged periodic. Voxels with
    label <= 0 get their own singleton components and should be ignored by the caller.
    """
    flat = labels.ravel()
    index = np.arange(flat.size).reshape(labels.shape)
    rows, cols = [], []
    for axis in range(labels.ndim):
        neighbour = np.roll(index, -1, axis=axis)
        same = (labels == labels.take(neighbour.ravel()).reshape(labels.shape)) & (labels > 0)
        if not periodic[axis]:
            # drop the wrap-around pairs on the last slice of a non-periodic axis
            last = [slice(None)] * labels.ndim
            last[axis] = -1
            same[tuple(last)] = False
        rows.append(index[same])
        cols.append(neighbour[same])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    graph = sparse.coo_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)),
                              shape=(flat.size, flat.size))
    _, component = connected_components(graph, directed=False)
    return component


def components_per_grain(labels: np.ndarray, periodic: Sequence[bool]) -> np.ndarray:
    """Number of face-connected voxel components per grain ID (index 0 unused).

    A grain that came out of tessellation as one body has exactly one component.
    """
    return _component_sizes(labels, periodic)[0]


def main_body_voxel_counts(labels: np.ndarray, periodic: Sequence[bool]) -> np.ndarray:
    """Voxel count of the largest connected body of each grain ID (index 0 unused)."""
    return _component_sizes(labels, periodic)[1]


def _component_sizes(labels: np.ndarray, periodic: Sequence[bool]):
    flat = labels.ravel()
    component = component_labels(labels, periodic)
    grain_voxels = flat > 0
    pairs, sizes = np.unique(np.stack([flat[grain_voxels], component[grain_voxels]]), axis=1,
                             return_counts=True)
    n = int(flat.max()) + 1
    largest = np.zeros(n, dtype=np.int64)
    np.maximum.at(largest, pairs[0], sizes)
    return np.bincount(pairs[0], minlength=n), largest


def ks_distance(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (0 = identical distributions)."""
    return float(ks_2samp(np.asarray(sample_a), np.asarray(sample_b)).statistic)


@dataclass
class RveReport:
    n_voxels: int
    n_grains: int
    fill_ratio: float
    empty_grains: int
    fragmented_grains: int
    volume_ratio_median: float
    volume_ratio_p10: float
    volume_ratio_p90: float
    main_body_ratio_median: float
    main_body_ratio_p10: float
    main_body_ratio_p90: float
    volume_ks: float
    phase_fractions: dict

    def to_dict(self) -> dict:
        return asdict(self)


def report(labels: np.ndarray, target_volume: np.ndarray, phase_of_grain: np.ndarray,
           voxel_volume: float, periodic: Sequence[bool]) -> RveReport:
    """Summarise how well a label field reproduces its grain targets.

    target_volume[i] and phase_of_grain[i] describe grain i (index 0 unused), in the same
    length unit as voxel_volume. volume_ratio_* compare all voxels of a grain with its target,
    main_body_ratio_* only its largest connected body, which is what a mesher can use.
    """
    n_grains = len(target_volume) - 1
    counts = grain_voxel_counts(labels, n_grains=n_grains)[:n_grains + 1]
    actual = counts[1:] * voxel_volume
    target = np.asarray(target_volume, dtype=float)[1:]
    ratio = actual / target
    components, main = _component_sizes(labels, periodic)
    pad = (0, max(0, n_grains + 1 - len(components)))
    components = np.pad(components, pad)[1:n_grains + 1]
    main_ratio = np.pad(main, pad)[1:n_grains + 1] * voxel_volume / target
    return RveReport(
        n_voxels=int(labels.size),
        n_grains=int(n_grains),
        fill_ratio=fill_ratio(labels),
        empty_grains=int(np.count_nonzero(counts[1:] == 0)),
        fragmented_grains=int(np.count_nonzero(components > 1)),
        volume_ratio_median=float(np.median(ratio)),
        volume_ratio_p10=float(np.percentile(ratio, 10)),
        volume_ratio_p90=float(np.percentile(ratio, 90)),
        main_body_ratio_median=float(np.median(main_ratio)),
        main_body_ratio_p10=float(np.percentile(main_ratio, 10)),
        main_body_ratio_p90=float(np.percentile(main_ratio, 90)),
        volume_ks=ks_distance(actual[counts[1:] > 0], target),
        phase_fractions=phase_fractions(labels, phase_of_grain),
    )
