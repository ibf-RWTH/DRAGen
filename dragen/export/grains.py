"""Grain table of an RVEResult (replaces the legacy sampled_grains.csv, finding 12: that file
held only the grains of the last phase)."""
from pathlib import Path

import numpy as np
import pandas as pd


def grain_table(result) -> pd.DataFrame:
    """One row per grain, all phases, bands and inclusions, in grain order (row g - 1 = grain g).

    grain_id, phase_id, phase, band_id (-1 outside bands), voxels, volume_m3 (voxel volume; 2D:
    area_m2), eq_diameter_m (diameter of the sphere, 2D: circle, with that volume), a_m, b_m, c_m, alpha_deg (sampled
    ellipsoid, if the result has it), parent_grain_id and packet_id (after a substructure; the
    rows are then blocks and grains without substructure), phi1, PHI, phi2 (Bunge, degrees), qw, qx, qy, qz
    (quaternion crystal -> sample, see dragen.api).
    """
    n = result.n_grains
    voxels = np.bincount(np.asarray(result.grain_ids).ravel(), minlength=n + 1)[1:]
    volume = voxels * float(np.prod(result.voxel_size_m))
    two_d = np.asarray(result.grain_ids).ndim == 2
    table = {
        'grain_id': np.arange(1, n + 1),
        'phase_id': result.phase_ids,
        'phase': [result.phase_names[int(p)] for p in result.phase_ids],
        'band_id': result.band_ids,
        'voxels': voxels,
        ('area_m2' if two_d else 'volume_m3'): volume,
        'eq_diameter_m': 2 * np.sqrt(volume / np.pi) if two_d else np.cbrt(6 * volume / np.pi),
    }
    if result.radii_m is not None:
        table.update(a_m=result.radii_m[:, 0], b_m=result.radii_m[:, 1], c_m=result.radii_m[:, 2],
                     alpha_deg=result.alpha_deg)
    if result.parent_grain_ids is not None:
        table.update(parent_grain_id=result.parent_grain_ids, packet_id=result.packet_ids)
    table.update(phi1=result.euler_bunge_deg[:, 0], PHI=result.euler_bunge_deg[:, 1],
                 phi2=result.euler_bunge_deg[:, 2])
    table.update({f'q{c}': result.orientations[:, i] for i, c in enumerate('wxyz')})
    return pd.DataFrame(table)


def export_grains(result, path: Path) -> Path:
    """<path>.csv with grain_table(result)."""
    path = Path(path).with_suffix('.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    grain_table(result).to_csv(path, index=False)
    return path
