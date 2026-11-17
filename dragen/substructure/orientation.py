"""Block orientations from the parent grain orientations.

'KS': every block gets one of the 24 Kurdjumov-Sachs variants of its parent grain, picked
hierarchically as in the legacy pipeline: a packet draws a habit plane class (0..3), a block a
pair class (0..2) and a choice (0..1), so the blocks of a packet share a habit plane.
'experimental': measured transformations T = g_child g_parent^T from parent/child EBSD pairs; all
blocks of a grain draw from the transformations of one template grain (weighted by its number of
pairs), which keeps the correlation of variants inside a real grain.

Matrices are Bunge orientation matrices g (sample -> crystal); a block gets g_block = T g_parent.
The variant table is the one of the legacy code, unchanged (see the FIXME on variant 0).
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def euler_bunge_to_matrix(phi1, PHI, phi2, degrees=True) -> np.ndarray:
    if degrees:
        phi1, PHI, phi2 = np.deg2rad(phi1), np.deg2rad(PHI), np.deg2rad(phi2)
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(PHI), np.sin(PHI)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    Rz1 = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    Rz2 = np.array([[c2, -s2, 0.0], [s2, c2, 0.0], [0.0, 0.0, 1.0]])
    return Rz2 @ Rx @ Rz1


def project_to_SO3(matrix: np.ndarray) -> np.ndarray:
    """Nearest proper rotation."""
    U, _, Vt = np.linalg.svd(matrix)
    rotation = U @ Vt
    if np.linalg.det(rotation) < 0:
        U[:, -1] *= -1.0
        rotation = U @ Vt
    return rotation


def matrix_to_euler_bunge(matrix: np.ndarray) -> Tuple[float, float, float]:
    """Bunge angles in degrees, each in [0, 360)."""
    matrix = project_to_SO3(np.asarray(matrix, dtype=float))
    PHI = float(np.arccos(float(np.clip(matrix[2, 2], -1.0, 1.0))))
    if np.sin(PHI) > 1e-8:
        phi1 = float(np.arctan2(matrix[2, 0], matrix[2, 1]))
        phi2 = float(np.arctan2(matrix[0, 2], -matrix[1, 2]))
    else:                                      # gimbal lock: only phi1 + phi2 is defined
        phi1 = 0.0
        phi2 = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    return tuple(float(np.rad2deg(a) % 360.0) for a in (phi1, PHI, phi2))


def rotation_angle_deg(matrix: np.ndarray) -> float:
    cosine = np.clip((np.trace(project_to_SO3(matrix)) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def apply_transformation(parent_euler_deg, transformation: np.ndarray) -> Tuple[float, float, float]:
    g_parent = euler_bunge_to_matrix(*parent_euler_deg)
    return matrix_to_euler_bunge(project_to_SO3(transformation @ g_parent))


# --------------------------------------------------------------------------- KS variants

T_LIST = np.array([
    # FIXME: not a proper rotation (det 0.119), see test_all_ks_variant_matrices_are_rotations.
    [[0.742, 0.667, 0.075], [0.650, 0.742, 0.167], [0.167, 0.075, 0.983]],
    [[0.075, 0.667, -0.742], [-0.167, 0.742, 0.650], [0.983, 0.075, 0.167]],
    [[-0.667, -0.075, 0.742], [0.742, -0.167, 0.650], [0.075, 0.983, 0.167]],
    [[0.667, -0.742, 0.075], [0.742, 0.650, -0.167], [0.075, 0.167, 0.983]],
    [[-0.075, 0.742, -0.667], [-0.167, 0.650, 0.742], [0.983, 0.167, 0.075]],
    [[-0.742, 0.075, 0.667], [0.650, -0.167, 0.742], [0.167, 0.983, 0.075]],
    [[-0.075, 0.667, 0.742], [-0.167, -0.742, 0.650], [0.983, -0.075, 0.167]],
    [[-0.742, -0.667, 0.075], [0.650, -0.742, -0.167], [0.167, -0.075, 0.983]],
    [[0.742, 0.075, -0.667], [0.650, 0.167, 0.742], [0.167, -0.983, 0.075]],
    [[0.075, 0.742, 0.667], [-0.167, -0.650, 0.742], [0.983, -0.167, 0.075]],
    [[-0.667, -0.742, -0.075], [0.742, -0.650, -0.167], [0.075, -0.167, 0.983]],
    [[0.667, -0.075, -0.742], [0.742, 0.167, 0.650], [0.075, -0.983, 0.167]],
    [[0.667, 0.742, -0.075], [-0.742, 0.650, -0.167], [-0.075, 0.167, 0.983]],
    [[-0.667, 0.075, -0.742], [-0.742, -0.167, 0.650], [-0.075, 0.983, 0.167]],
    [[0.075, -0.667, 0.742], [0.167, 0.742, 0.650], [-0.983, 0.075, 0.167]],
    [[0.742, 0.667, 0.075], [-0.650, 0.742, -0.167], [-0.167, 0.075, 0.983]],
    # (0,1) was 0.075 in the source table, which made this not a rotation.
    [[-0.742, -0.075, -0.667], [-0.650, -0.167, 0.742], [-0.167, 0.983, 0.075]],
    [[-0.075, -0.742, 0.667], [0.167, 0.650, 0.742], [-0.983, 0.167, 0.075]],
    [[0.742, -0.075, 0.667], [0.650, -0.167, -0.742], [0.167, 0.983, -0.075]],
    [[0.075, -0.742, -0.667], [-0.167, 0.650, -0.742], [0.983, 0.167, -0.075]],
    [[-0.667, 0.742, 0.075], [0.742, 0.650, 0.167], [0.075, 0.167, -0.983]],
    [[0.667, 0.075, 0.742], [0.742, -0.167, -0.650], [0.075, 0.983, -0.167]],
    [[-0.075, -0.667, -0.742], [-0.167, 0.742, -0.650], [0.983, 0.075, -0.167]],
    [[-0.742, 0.667, -0.075], [0.650, 0.742, 0.167], [0.167, 0.075, -0.983]],
], dtype=float)

# (PacketClass, PairClass) -> the two KS variants that form a Bain/CP pair.
KS_PAIR = {
    (0, 0): (0, 1), (0, 1): (2, 3), (0, 2): (4, 5),
    (1, 0): (6, 7), (1, 1): (8, 9), (1, 2): (10, 11),
    (2, 0): (12, 13), (2, 1): (14, 15), (2, 2): (16, 17),
    (3, 0): (18, 19), (3, 1): (20, 21), (3, 2): (22, 23),
}


def ks_orientations(parent_euler: np.ndarray, block_grain: np.ndarray, block_packet: np.ndarray,
                    n_packets: int, rng: np.random.Generator):
    """(block euler (B, 3), variant per block (B,), habit plane class per packet (P,)).
    block_grain and block_packet are 0-based indices; random draws in the order of the legacy
    code: all packet classes, then pair class and choice per block."""
    packet_class = rng.integers(0, 4, size=n_packets) if n_packets else np.zeros(0, int)
    pair = np.empty((len(block_grain), 2), dtype=int)
    for b in range(len(block_grain)):
        pair[b] = rng.integers(0, 3), rng.integers(0, 2)
    variant = np.array([KS_PAIR[(int(packet_class[p]), int(pc))][int(ch)]
                        for p, (pc, ch) in zip(block_packet, pair)], dtype=int)
    euler = np.array([apply_transformation(parent_euler[g], T_LIST[v]) for g, v in zip(block_grain, variant)])
    return euler.reshape(-1, 3), variant, packet_class


# --------------------------------------------------------------------------- measured OR

def _orientation_csv(path, require_grain_id: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if require_grain_id and 'grain_id' not in df:
        raise ValueError(f'{path}: column grain_id missing')
    if 'grain_id' not in df:
        df['grain_id'] = np.arange(1, len(df) + 1)
    return df


def experimental_library(parent_file, child_file) -> Dict[int, List[np.ndarray]]:
    """Measured transformations T = g_child g_parent^T, grouped by the measured parent grain."""
    parents = _orientation_csv(parent_file, require_grain_id=False)
    children = _orientation_csv(child_file, require_grain_id=True)
    g_parent = {int(r.grain_id): euler_bunge_to_matrix(r.phi1, r.PHI, r.phi2) for r in parents.itertuples()}
    library: Dict[int, List[np.ndarray]] = {}
    for r in children.itertuples():
        gid = int(r.grain_id)
        if gid in g_parent:
            library.setdefault(gid, []).append(
                project_to_SO3(euler_bunge_to_matrix(r.phi1, r.PHI, r.phi2) @ g_parent[gid].T))
    if not library:
        raise ValueError(f'no grain_id of {child_file} matches a grain of {parent_file}')
    return library


def experimental_orientations(parent_euler: np.ndarray, block_grain: np.ndarray, library, rng):
    """(block euler (B, 3), template grain per block, relation index per block)."""
    templates = np.array(sorted(library), dtype=int)
    weights = np.array([len(library[t]) for t in templates], dtype=float)
    grain_template: Dict[int, int] = {}
    euler = np.zeros((len(block_grain), 3))
    template = np.zeros(len(block_grain), dtype=int)
    relation = np.zeros(len(block_grain), dtype=int)
    for b, g in enumerate(block_grain):
        if int(g) not in grain_template:
            grain_template[int(g)] = int(rng.choice(templates, p=weights / weights.sum()))
        t = grain_template[int(g)]
        relation[b] = int(rng.integers(0, len(library[t])))
        template[b] = t
        euler[b] = apply_transformation(parent_euler[g], library[t][relation[b]])
    return euler, template, relation
