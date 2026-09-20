"""Block orientations from the parent-grain (PAG) orientations.

Two modes:

'KS'
    Each block gets one of the 24 Kurdjumov-Sachs variants of its parent grain. The variant is
    picked hierarchically -- a packet draws a PacketClass (0..3, the four {111}_gamma habit planes),
    a block draws a PairClass (0..2) and a PairChoice (0..1) -- so blocks in one packet share a
    habit plane, as they do in real lath martensite.

'experimental'
    Instead of the ideal KS matrices, the transformations are measured: for every parent/child
    orientation pair in the supplied EBSD files, T = R_child * R_parent^T. Blocks then draw from
    the transformations of one randomly assigned template grain, so the generated RVE reproduces
    the *actual* orientation relationship of the material rather than the idealised one.

Note on the DAMASK path: dragen/generation/spectral.py::write_material assigns
`Rotation.from_random(1)` to phases 2/3/4, so the PAG orientations transformed here are random
rather than sampled from the input texture. That is pre-existing DRAGen behaviour, not a defect of
this pipeline.
"""
import os

import numpy as np
import pandas as pd
import pyvista as pv
import damask
import matplotlib.pyplot as plt
import yaml

from dragen.substructure import validation
from dragen.substructure.config import SubsConfig
from dragen.utilities.InputInfo import RveInfo

OR_PLOT = 'or_misorientation_ebsd_vs_generated.png'


# --------------------------------------------------------------------------- #
# rotation helpers
# --------------------------------------------------------------------------- #

def wrap_0_360(angle_deg: float) -> float:
    angle = angle_deg % 360.0
    return angle + 360.0 if angle < 0 else angle


def project_to_SO3(matrix: np.ndarray) -> np.ndarray:
    """Nearest proper rotation matrix -- keeps accumulated float error from drifting off SO(3)."""
    U, _, Vt = np.linalg.svd(matrix)
    rotation = U @ Vt

    if np.linalg.det(rotation) < 0:
        U[:, -1] *= -1.0
        rotation = U @ Vt

    return rotation


def euler_bunge_to_matrix(phi1, PHI, phi2, degrees=True) -> np.ndarray:
    if degrees:
        phi1, PHI, phi2 = np.deg2rad(phi1), np.deg2rad(PHI), np.deg2rad(phi2)

    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(PHI), np.sin(PHI)
    c2, s2 = np.cos(phi2), np.sin(phi2)

    Rz1 = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)
    Rz2 = np.array([[c2, -s2, 0.0], [s2, c2, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    return Rz2 @ Rx @ Rz1


def matrix_to_euler_bunge(matrix: np.ndarray, degrees=True):
    matrix = project_to_SO3(np.asarray(matrix, dtype=float))

    PHI = float(np.arccos(float(np.clip(matrix[2, 2], -1.0, 1.0))))

    if np.sin(PHI) > 1e-8:
        phi1 = float(np.arctan2(matrix[2, 0], matrix[2, 1]))
        phi2 = float(np.arctan2(matrix[0, 2], -matrix[1, 2]))
    else:
        # Gimbal lock: phi1 and phi2 are degenerate, only their sum is defined.
        phi1 = 0.0
        phi2 = float(np.arctan2(matrix[1, 0], matrix[0, 0]))

    if degrees:
        return (wrap_0_360(np.rad2deg(phi1)),
                wrap_0_360(np.rad2deg(PHI)),
                wrap_0_360(np.rad2deg(phi2)))

    return phi1 % (2.0 * np.pi), PHI % (2.0 * np.pi), phi2 % (2.0 * np.pi)


def rotation_angle_deg(matrix: np.ndarray) -> float:
    matrix = project_to_SO3(matrix)
    cosine = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def apply_transformation(parent_euler_deg, transformation: np.ndarray):
    R_parent = euler_bunge_to_matrix(*parent_euler_deg, degrees=True)
    return matrix_to_euler_bunge(project_to_SO3(transformation @ R_parent), degrees=True)


# --------------------------------------------------------------------------- #
# KS variants
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# parent orientation sources
# --------------------------------------------------------------------------- #

def load_grain_euler_from_material_yaml(material_file: str) -> dict:
    """material[i] -> GrainID i+1, orientation stored as a quaternion."""
    with open(material_file, 'r', encoding='utf-8') as f:
        matdata = yaml.safe_load(f)

    grain_to_euler = {}

    for material_id, entry in enumerate(matdata.get('material', [])):
        quaternion = entry['constituents'][0]['O']

        if len(quaternion) != 4:
            raise ValueError(f'Material {material_id}: expected a quaternion with 4 values, '
                             f'got {quaternion}')

        euler = damask.Rotation.from_quaternion(quaternion).as_Euler_angles(degrees=True)
        grain_to_euler[material_id + 1] = (float(euler[0]), float(euler[1]), float(euler[2]))

    RveInfo.LOGGER.info(f'Loaded orientations for {len(grain_to_euler)} grains from {material_file}.')
    return grain_to_euler


def load_grain_euler_from_grain_data_output(csv_file: str) -> dict:
    """Read GrainID -> Euler angles from Generation_Data/grain_data_output.csv.

    This is the sampled grain table dragen/main3D.py writes for every run, so it covers every
    grain. graindata.inp does not: Mesher3D.write_grain_data() skips any phase marked isotropic in
    phase2iso_flag, which would leave those grains without an orientation here.
    """
    df = pd.read_csv(csv_file)

    for column in ('GrainID', 'phi1', 'PHI', 'phi2'):
        if column not in df.columns:
            raise ValueError(f"{csv_file}: missing required column '{column}'")

    grain_to_euler = {int(row['GrainID']): (float(row['phi1']), float(row['PHI']),
                                            float(row['phi2']))
                      for _, row in df.iterrows()}

    RveInfo.LOGGER.info(f'Loaded orientations for {len(grain_to_euler)} grains from {csv_file}.')
    return grain_to_euler


def load_grain_euler_from_graindata_inp(graindata_file: str) -> dict:
    """DRAGen/Abaqus graindata.inp lines: `Grain: <id>: <phi1>: <PHI>: <phi2>: <size>`."""
    grain_to_euler = {}

    with open(graindata_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('Grain:'):
                continue

            parts = [p.strip() for p in line.split(':')]
            if len(parts) < 5:
                raise ValueError(f'Invalid grain line: {line}')

            grain_to_euler[int(parts[1])] = (float(parts[2]), float(parts[3]), float(parts[4]))

    if not grain_to_euler:
        raise ValueError(f'No grain orientations found in {graindata_file}')

    RveInfo.LOGGER.info(f'Loaded orientations for {len(grain_to_euler)} grains '
                        f'from {graindata_file}.')
    return grain_to_euler


def _load_orientation_csv(file: str, require_grain_id: bool) -> pd.DataFrame:
    df = pd.read_csv(file)

    for column in ('phi1', 'PHI', 'phi2'):
        if column not in df.columns:
            raise ValueError(f"{file}: missing required column '{column}'")

    if require_grain_id and 'grain_id' not in df.columns:
        raise ValueError(
            f"{file}: missing required column 'grain_id'. For experimental orientation mode, "
            'parent and child files must use consistent grain_id numbering.')

    if 'grain_id' not in df.columns:
        df['grain_id'] = np.arange(1, len(df) + 1)

    return df


def build_experimental_or_library(parent_file: str, child_file: str) -> dict:
    """Measured parent -> child transformations, grouped by the parent grain they came from.

    Grouping matters: within one real grain the observed variants are correlated, so drawing a
    whole synthetic grain's blocks from a single template grain preserves that correlation.
    """
    parent_df = _load_orientation_csv(parent_file, require_grain_id=False)
    child_df = _load_orientation_csv(child_file, require_grain_id=True)

    parent_map = {
        int(row['grain_id']): euler_bunge_to_matrix(row['phi1'], row['PHI'], row['phi2'])
        for _, row in parent_df.iterrows()
    }

    library = {}
    used, skipped = 0, 0

    for _, row in child_df.iterrows():
        grain_id = int(row['grain_id'])

        if grain_id not in parent_map:
            skipped += 1
            continue

        R_child = euler_bunge_to_matrix(row['phi1'], row['PHI'], row['phi2'])
        transformation = project_to_SO3(R_child @ parent_map[grain_id].T)

        library.setdefault(grain_id, []).append(transformation)
        used += 1

    if used == 0:
        raise ValueError(
            'No experimental transformations could be created: no grain_id in '
            f'{child_file} matched a row of {parent_file}.')

    RveInfo.LOGGER.info(f'Experimental OR library: {used} parent-child pairs grouped into '
                        f'{len(library)} template grains.')

    if skipped:
        RveInfo.LOGGER.warning(f'Skipped {skipped} child orientations whose grain_id was not '
                               f'found in {parent_file}.')

    return library


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #

def assign_orientations(cfg: SubsConfig, mesh: pv.DataSet) -> pv.DataSet:
    """Write phi1/PHI/phi2 (degrees) plus the variant bookkeeping onto every cell."""
    for array in ('GrainID', 'PacketID', 'BlockID', 'SubstructureFlag'):
        if array not in mesh.cell_data:
            raise KeyError(f'Missing required cell_data array: {array}')

    grain_ids = np.asarray(mesh.cell_data['GrainID']).astype(np.int64)
    packet_ids = np.asarray(mesh.cell_data['PacketID']).astype(np.int64)
    block_ids = np.asarray(mesh.cell_data['BlockID']).astype(np.int64)
    sub_flag = np.asarray(mesh.cell_data['SubstructureFlag']).astype(np.int64)

    validation.check_positive_ids_continuous(packet_ids, 'PacketID')
    validation.check_positive_ids_continuous(block_ids, 'BlockID')
    validation.check_substructure_consistency(mesh)

    real_packet_ids = np.unique(packet_ids[packet_ids > 0])
    real_block_ids = np.unique(block_ids[block_ids > 0])

    grain_to_euler = _load_parent_orientations(cfg)

    missing = np.setdiff1d(np.unique(grain_ids), np.array(list(grain_to_euler), dtype=int))
    if missing.size > 0:
        raise ValueError(f'These GrainIDs exist in the mesh but have no orientation: {missing}')

    rng = np.random.default_rng(cfg.seed)

    packet_class = np.full(mesh.n_cells, -1, dtype=np.int32)
    pair_class = np.full(mesh.n_cells, -1, dtype=np.int32)
    pair_choice = np.full(mesh.n_cells, -1, dtype=np.int32)
    variant_id = np.full(mesh.n_cells, -1, dtype=np.int32)

    phi1 = np.zeros(mesh.n_cells, dtype=np.float64)
    PHI = np.zeros(mesh.n_cells, dtype=np.float64)
    phi2 = np.zeros(mesh.n_cells, dtype=np.float64)

    # One habit plane class per packet, one pair + choice per block.
    for pid in real_packet_ids:
        packet_class[packet_ids == pid] = rng.integers(0, 4)

    for bid in real_block_ids:
        mask = block_ids == bid
        pair_class[mask] = rng.integers(0, 3)
        pair_choice[mask] = rng.integers(0, 2)

    for cell_id in np.nonzero(sub_flag == 1)[0]:
        key = (int(packet_class[cell_id]), int(pair_class[cell_id]))

        if key not in KS_PAIR:
            raise ValueError(f'Invalid KS_PAIR key at cell {cell_id}: '
                             f'PacketClass={packet_class[cell_id]}, '
                             f'PairClass={pair_class[cell_id]}')

        variant_a, variant_b = KS_PAIR[key]
        variant_id[cell_id] = variant_a if int(pair_choice[cell_id]) == 0 else variant_b

    mesh.cell_data['PacketClass'] = packet_class
    mesh.cell_data['PairClass'] = pair_class
    mesh.cell_data['PairChoice'] = pair_choice
    mesh.cell_data['VariantID'] = variant_id

    def block_grain(block_id, mask):
        grains = np.unique(grain_ids[mask])
        if grains.size != 1:
            raise ValueError(f'BlockID {block_id} spans multiple GrainIDs: {grains}')
        return int(grains[0])

    if cfg.orientation_mode == 'KS':
        for bid in real_block_ids:
            mask = block_ids == bid

            variants = np.unique(variant_id[mask])
            if variants.size != 1:
                raise ValueError(f'BlockID {bid} has multiple VariantIDs: {variants}')

            variant = int(variants[0])
            if not 0 <= variant <= 23:
                raise ValueError(f'Invalid VariantID {variant} in BlockID {bid}')

            parent_euler = grain_to_euler[block_grain(bid, mask)]
            phi1[mask], PHI[mask], phi2[mask] = apply_transformation(parent_euler, T_LIST[variant])

    elif cfg.orientation_mode == 'experimental':
        library = build_experimental_or_library(cfg.parent_orientation_file,
                                                cfg.child_orientation_file)

        template_gids = np.array(list(library), dtype=int)
        weights = np.array([len(library[gid]) for gid in template_gids], dtype=float)
        probabilities = weights / weights.sum()

        relation_id_arr = np.full(mesh.n_cells, -1, dtype=np.int32)
        template_gid_arr = np.full(mesh.n_cells, -1, dtype=np.int32)

        measured_angles = [rotation_angle_deg(T) for transforms in library.values()
                           for T in transforms]
        generated_angles = []

        grain_to_template = {}

        for bid in real_block_ids:
            mask = block_ids == bid
            grain = block_grain(bid, mask)
            parent_euler = grain_to_euler[grain]

            # All blocks of one synthetic grain share a template grain.
            if grain not in grain_to_template:
                grain_to_template[grain] = int(rng.choice(template_gids, p=probabilities))
            template_gid = grain_to_template[grain]

            allowed = library[template_gid]
            relation_id = int(rng.integers(0, len(allowed)))
            transformation = allowed[relation_id]

            block_euler = apply_transformation(parent_euler, transformation)

            phi1[mask], PHI[mask], phi2[mask] = block_euler
            relation_id_arr[mask] = relation_id
            template_gid_arr[mask] = template_gid

            R_parent = euler_bunge_to_matrix(*parent_euler, degrees=True)
            R_child = euler_bunge_to_matrix(*block_euler, degrees=True)
            generated_angles.append(rotation_angle_deg(project_to_SO3(R_child @ R_parent.T)))

        mesh.cell_data['ExperimentalRelationID'] = relation_id_arr
        mesh.cell_data['ExperimentalTemplateGrainID'] = template_gid_arr

        _plot_or_misorientation(cfg, measured_angles, generated_angles)

    else:
        raise ValueError(f'Unknown orientation_mode: {cfg.orientation_mode}')

    # Cells outside a transformable phase keep their parent grain orientation.
    for grain in np.unique(grain_ids[sub_flag == 0]):
        mask = (grain_ids == grain) & (sub_flag == 0)
        phi1[mask], PHI[mask], phi2[mask] = grain_to_euler[int(grain)]

    mesh.cell_data['phi1'] = phi1
    mesh.cell_data['PHI'] = PHI
    mesh.cell_data['phi2'] = phi2

    if np.any(variant_id[sub_flag == 1] < 0) or np.any(variant_id[sub_flag == 1] > 23):
        raise ValueError('Some substructure cells have a VariantID outside 0..23.')

    if np.any(variant_id[sub_flag == 0] != -1):
        raise ValueError('Some non-substructure cells have VariantID != -1.')

    validation.cast_int32(mesh, [
        'PacketID', 'BlockID', 'GrainID', 'SubstructureFlag', 'PacketClass', 'PairClass',
        'PairChoice', 'VariantID', 'ExperimentalRelationID', 'ExperimentalTemplateGrainID',
    ])

    RveInfo.LOGGER.info(f"Assigned orientations to {len(real_block_ids)} blocks "
                        f"({cfg.orientation_mode} mode).")

    return mesh


def _load_parent_orientations(cfg: SubsConfig) -> dict:
    """PAG orientations come from whatever the run already wrote for this solver."""
    if cfg.solver != 'abaqus':
        return load_grain_euler_from_material_yaml(os.path.join(cfg.store_path, 'material.yaml'))

    grain_data = os.path.join(cfg.store_path, 'Generation_Data', 'grain_data_output.csv')
    if os.path.isfile(grain_data):
        return load_grain_euler_from_grain_data_output(grain_data)

    return load_grain_euler_from_graindata_inp(os.path.join(cfg.store_path, 'graindata.inp'))


def _plot_or_misorientation(cfg: SubsConfig, measured, generated) -> None:
    plt.figure(figsize=(11, 5))
    plt.hist(measured, bins=50, density=True, alpha=0.45, label='EBSD')
    plt.hist(generated, bins=50, density=True, alpha=0.45, label='Generated')
    plt.xlabel('Parent-child transformation angle [deg]')
    plt.ylabel('Density')
    plt.title('Actual OR: EBSD vs Generated RVE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.path(OR_PLOT), dpi=300)
    plt.close()

    RveInfo.LOGGER.info(f'Saved: {cfg.path(OR_PLOT)}')
