"""Block segmentation: slice each packet into parallel martensite blocks.

A packet is sliced by planes normal to a random direction through its centre. Slice spacing is
either a single user-given mean thickness ('user' mode) or sampled per slice from a measured EBSD
block-thickness distribution ('file' mode), which reproduces the measured spread rather than just
its mean.

Thicknesses are given in micrometres everywhere the user sees them and converted into mesh
coordinate units via `cfg.thickness_scale` -- see dragen/substructure/config.py for why that factor
differs between the DAMASK and the Abaqus mesh.
"""
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv

from dragen.substructure import validation
from dragen.substructure.config import SubsConfig
from dragen.substructure.validation import NO_SUBSTRUCTURE
from dragen.utilities.InputInfo import RveInfo

THICKNESS_COLUMN = 'block_thickness'
THICKNESS_PLOT = 'block_thickness_ebsd_vs_generated_pairplot_style.png'

_INT_ARRAYS = ['BlockID', 'BlockID_raw', 'BlockGenMode', 'OldBlockID',
               'PacketID', 'GrainID', 'SubstructureFlag']


def gen_blocks(cfg: SubsConfig, mesh: pv.DataSet) -> pv.DataSet:
    """Dispatch to the user-thickness or EBSD-distribution block generator."""
    if cfg.block_generation_mode == 'file':
        return gen_blocks_from_distribution(cfg, mesh)
    return gen_blocks_uniform(cfg, mesh)


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #

def _packet_setup(cfg: SubsConfig, mesh: pv.DataSet):
    packet_ids = np.asarray(mesh.cell_data['PacketID']).astype(int)

    if 'SubstructureFlag' not in mesh.cell_data:
        raise KeyError('SubstructureFlag is missing. Run gen_packets() first.')

    sub_flag = np.asarray(mesh.cell_data['SubstructureFlag']).astype(int)

    validation.check_positive_ids_continuous(packet_ids, 'PacketID')

    real_packet_ids = np.unique(packet_ids[packet_ids > 0])
    RveInfo.LOGGER.info(f'Generating blocks for {len(real_packet_ids)} packets.')

    return packet_ids, sub_flag, real_packet_ids, np.random.default_rng(cfg.seed)


def _slice_distances(packet: pv.DataSet, rng: np.random.Generator) -> np.ndarray:
    """Signed distance of every packet cell centre from a random plane through the packet centre."""
    normal = rng.normal(size=3)
    normal /= np.linalg.norm(normal)

    return np.dot(packet.cell_centers().points - packet.center, normal)


def _finalize(cfg: SubsConfig, mesh: pv.DataSet, block_ids: np.ndarray,
              sub_flag: np.ndarray, packet_ids: np.ndarray) -> np.ndarray:
    """Shared safety checks + 1-based gapless renumbering of the generated blocks."""
    if np.any(block_ids[sub_flag == 1] < 0):
        raise ValueError('Some transformable cells still have BlockID = -1.')

    if np.any(block_ids[sub_flag == 0] != NO_SUBSTRUCTURE):
        raise ValueError('Some non-transformable cells have BlockID != -1.')

    mesh.cell_data['BlockID_raw'] = block_ids.copy().astype(np.int32)

    block_ids = block_ids.copy()
    block_ids[block_ids >= 0] += 1
    block_ids = validation.renumber_positive_ids(block_ids)

    mesh.cell_data['BlockID'] = block_ids.astype(np.int32)
    validation.cast_int32(mesh, _INT_ARRAYS)

    validation.check_positive_ids_continuous(block_ids, 'BlockID')
    validation.check_ids_nested(block_ids, packet_ids, 'Block', 'Packet')
    validation.check_substructure_consistency(mesh)

    n_blocks = len(np.unique(block_ids[block_ids > 0]))
    RveInfo.LOGGER.info(f'Total number of blocks in the RVE: {n_blocks}')

    return block_ids


# --------------------------------------------------------------------------- #
# 'user' mode: one mean thickness for the whole RVE
# --------------------------------------------------------------------------- #

def gen_blocks_uniform(cfg: SubsConfig, mesh: pv.DataSet) -> pv.DataSet:
    if cfg.average_block_thickness is None:
        raise ValueError("Block generation mode 'user' requires an average block thickness (t_mu).")

    thickness = cfg.average_block_thickness * cfg.thickness_scale

    RveInfo.LOGGER.info(f'Average block thickness: {cfg.average_block_thickness} um '
                        f'= {thickness} {cfg.length_unit}')

    packet_ids, sub_flag, real_packet_ids, rng = _packet_setup(cfg, mesh)

    block_ids = np.full(mesh.n_cells, NO_SUBSTRUCTURE, dtype=int)
    num_tot_blocks = 0

    for pid in real_packet_ids:
        packet_mask = packet_ids == pid

        if np.any(sub_flag[packet_mask] != 1):
            raise ValueError(f'PacketID {pid} contains cells with SubstructureFlag != 1.')

        packet = mesh.extract_cells(packet_mask)
        if packet.n_cells == 0:
            raise ValueError(f'PacketID {pid} has zero cells.')

        if packet.n_cells > cfg.min_block_cells:
            distances = _slice_distances(packet, rng)
            local_ids = np.floor((distances - distances.min()) / thickness).astype(int)
            n_local = int(local_ids.max()) + 1
            block_ids[packet_mask] = local_ids + num_tot_blocks
        else:
            # Too few cells to slice meaningfully -- the packet is one block.
            n_local = 1
            block_ids[packet_mask] = num_tot_blocks

        num_tot_blocks += n_local

    _finalize(cfg, mesh, block_ids, sub_flag, packet_ids)
    return mesh


# --------------------------------------------------------------------------- #
# 'file' mode: thicknesses sampled from a measured EBSD distribution
# --------------------------------------------------------------------------- #

def _variable_blocks(distances: np.ndarray, thickness_pool: np.ndarray,
                     rng: np.random.Generator):
    """Walk the slicing axis, drawing each slice's thickness from the measured pool."""
    s_min, s_max = distances.min(), distances.max()

    local_ids = np.full(distances.size, -1, dtype=int)
    sampled = []

    start = s_min
    local_id = 0

    while start <= s_max:
        thickness = float(rng.choice(thickness_pool))
        if thickness <= 0:
            raise ValueError(f'Sampled non-positive block thickness: {thickness}')

        end = start + thickness

        if local_id == 0:
            mask = (distances >= start) & (distances <= end)
        else:
            mask = (distances > start) & (distances <= end)

        local_ids[mask] = local_id
        sampled.append(thickness)

        start = end
        local_id += 1

    # Floating point can leave the very last cells unassigned; fold them into the last slice.
    local_ids[local_ids < 0] = local_id - 1

    return local_ids, np.asarray(sampled, dtype=float)


def _thickness_pool(cfg: SubsConfig):
    """Read, filter and percentile-clip the measured block thicknesses (micrometres)."""
    ebsd_df = pd.read_csv(cfg.block_file)

    if THICKNESS_COLUMN not in ebsd_df.columns:
        raise KeyError(f'{cfg.block_file} must contain a column: {THICKNESS_COLUMN}')

    raw = ebsd_df[THICKNESS_COLUMN].dropna().to_numpy(dtype=float)
    raw = raw[raw > 0]

    if raw.size == 0:
        raise ValueError(f'No positive {THICKNESS_COLUMN} values found in {cfg.block_file}.')

    filtered = raw[raw >= cfg.min_physical_thickness]
    if filtered.size == 0:
        raise ValueError('No thickness values remain after physical minimum filtering.')

    t_low = np.percentile(filtered, cfg.lower_percentile)
    t_high = np.percentile(filtered, cfg.upper_percentile)
    filtered = filtered[(filtered >= t_low) & (filtered <= t_high)]

    if filtered.size == 0:
        raise ValueError('No thickness values remain after percentile clipping.')

    RveInfo.LOGGER.info(
        f'EBSD block thickness: {raw.size} values, mean {raw.mean():.4f} um; after clipping to '
        f'[{cfg.lower_percentile}%, {cfg.upper_percentile}%] = [{t_low:.4f}, {t_high:.4f}] um: '
        f'{filtered.size} values, mean {filtered.mean():.4f} um')

    return raw, filtered


def gen_blocks_from_distribution(cfg: SubsConfig, mesh: pv.DataSet) -> pv.DataSet:
    packet_ids, sub_flag, real_packet_ids, rng = _packet_setup(cfg, mesh)

    ebsd_thickness, filtered = _thickness_pool(cfg)
    thickness_pool = filtered * cfg.thickness_scale

    block_ids = np.full(mesh.n_cells, NO_SUBSTRUCTURE, dtype=int)
    block_thickness = np.zeros(mesh.n_cells, dtype=float)
    block_thickness_raw = np.zeros(mesh.n_cells, dtype=float)
    block_gen_mode = np.zeros(mesh.n_cells, dtype=np.int32)

    num_tot_blocks = 0

    for pid in real_packet_ids:
        packet_mask = packet_ids == pid

        if np.any(sub_flag[packet_mask] != 1):
            raise ValueError(f'PacketID {pid} contains cells with SubstructureFlag != 1.')

        packet = mesh.extract_cells(packet_mask)
        if packet.n_cells == 0:
            raise ValueError(f'PacketID {pid} has zero cells.')

        global_cells = np.nonzero(packet_mask)[0]

        if packet.n_cells > cfg.min_block_cells:
            distances = _slice_distances(packet, rng)
            local_ids, sampled = _variable_blocks(distances, thickness_pool, rng)
            n_local = int(local_ids.max()) + 1

            block_ids[global_cells] = local_ids + num_tot_blocks
            block_gen_mode[global_cells] = 1

            for local_id in range(n_local):
                cells = global_cells[local_ids == local_id]
                thickness = sampled[local_id] if local_id < len(sampled) else sampled[-1]
                block_thickness[cells] = thickness
                block_thickness_raw[cells] = thickness / cfg.thickness_scale
        else:
            n_local = 1
            thickness = float(rng.choice(thickness_pool))

            block_ids[packet_mask] = num_tot_blocks
            block_gen_mode[packet_mask] = 1
            block_thickness[packet_mask] = thickness
            block_thickness_raw[packet_mask] = thickness / cfg.thickness_scale

        num_tot_blocks += n_local

    if np.any(block_thickness[sub_flag == 1] <= 0):
        raise ValueError('Some transformable cells have non-positive BlockThickness.')

    if np.any(block_thickness[sub_flag == 0] != 0):
        raise ValueError('Some non-transformable cells have non-zero BlockThickness.')

    mesh.cell_data['BlockThickness'] = block_thickness
    mesh.cell_data['BlockThickness_raw'] = block_thickness_raw
    mesh.cell_data['BlockGenMode'] = block_gen_mode

    block_ids = _finalize(cfg, mesh, block_ids, sub_flag, packet_ids)

    _plot_thickness_comparison(cfg, mesh, block_ids, ebsd_thickness, rng)

    return mesh


# --------------------------------------------------------------------------- #
# merging
# --------------------------------------------------------------------------- #

def merge_small_blocks(cfg: SubsConfig, mesh: pv.DataSet) -> pv.DataSet:
    """Merge blocks below `min_cells_per_block` into their most common neighbouring block.

    Merging stays inside the packet, and other small blocks are not eligible as merge targets, so
    small blocks never chain into one another.
    """
    block_ids = np.asarray(mesh.cell_data['BlockID']).copy().astype(int)
    packet_ids = np.asarray(mesh.cell_data['PacketID']).astype(int)

    mesh.cell_data['OldBlockID'] = block_ids.copy().astype(np.int32)

    validation.check_positive_ids_continuous(block_ids, 'Initial BlockID')
    validation.check_ids_nested(block_ids, packet_ids, 'Block', 'Packet')
    validation.check_substructure_consistency(mesh)

    unique_blocks = np.unique(block_ids[block_ids > 0])
    small_blocks = [bid for bid in unique_blocks
                    if np.count_nonzero(block_ids == bid) < cfg.min_cells_per_block]
    small_set = set(small_blocks)

    RveInfo.LOGGER.info(f'{len(small_blocks)} of {len(unique_blocks)} blocks are below '
                        f'{cfg.min_cells_per_block} cells.')

    merged = 0

    for bid in small_blocks:
        cells = np.nonzero(block_ids == bid)[0]
        if cells.size == 0:
            continue

        packets = np.unique(packet_ids[cells])
        if packets.size != 1:
            raise ValueError(f'Block {bid} spans multiple packets: {packets}')

        packet = packets[0]
        if packet < 1:
            raise ValueError(f'Small block {bid} belongs to invalid PacketID {packet}')

        votes = Counter()

        for cell_id in cells:
            neighbors = mesh.cell_neighbors(int(cell_id), connections='points')
            if not neighbors:
                continue

            neighbors = np.asarray(neighbors, dtype=int)
            neighbors = neighbors[packet_ids[neighbors] == packet]
            if neighbors.size == 0:
                continue

            neighbor_bids = block_ids[neighbors]
            neighbor_bids = neighbor_bids[(neighbor_bids > 0) & (neighbor_bids != bid)]
            neighbor_bids = np.array([b for b in neighbor_bids if b not in small_set], dtype=int)
            if neighbor_bids.size == 0:
                continue

            votes.update(neighbor_bids.tolist())

        if not votes:
            RveInfo.LOGGER.info(f'Small block {bid}: no eligible neighbouring block in '
                                f'PacketID {packet}. Skipped.')
            continue

        target_bid, n_votes = votes.most_common(1)[0]
        RveInfo.LOGGER.info(f'  merged Block {bid} into Block {target_bid} '
                            f'(PacketID={packet}, votes={n_votes})')

        block_ids[cells] = target_bid
        merged += 1

    RveInfo.LOGGER.info(f'Total merged small blocks: {merged}')

    block_ids = validation.renumber_positive_ids(block_ids)
    mesh.cell_data['BlockID'] = block_ids.astype(np.int32)

    validation.check_positive_ids_continuous(block_ids, 'Final BlockID')
    validation.check_ids_nested(block_ids, packet_ids, 'Block', 'Packet')
    validation.check_substructure_consistency(mesh)
    validation.report_small_ids(block_ids, cfg.min_cells_per_block, 'block')

    validation.cast_int32(mesh, _INT_ARRAYS)

    return mesh


# --------------------------------------------------------------------------- #
# comparison plot
# --------------------------------------------------------------------------- #

def _kde_1d(data: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE with Silverman's rule of thumb."""
    data = np.asarray(data, dtype=float)

    if data.size < 2:
        return np.zeros_like(x_grid)

    std = np.std(data, ddof=1)
    if std <= 0:
        return np.zeros_like(x_grid)

    bandwidth = 1.06 * std * data.size ** (-1 / 5)
    if bandwidth <= 0:
        return np.zeros_like(x_grid)

    diff = (x_grid[:, None] - data[None, :]) / bandwidth
    density = np.exp(-0.5 * diff ** 2).sum(axis=1)

    return density / (data.size * bandwidth * np.sqrt(2 * np.pi))


def _plot_thickness_comparison(cfg: SubsConfig, mesh: pv.DataSet, block_ids: np.ndarray,
                               ebsd_thickness: np.ndarray, rng: np.random.Generator) -> None:
    """Measured vs. generated block thickness -- the check that the run reproduced the input."""
    thickness_raw = np.asarray(mesh.cell_data['BlockThickness_raw'])

    generated = []
    for bid in np.unique(block_ids[block_ids > 0]):
        # One sampled thickness per block by construction; mean() is a no-op safeguard in case a
        # slice boundary ever lands so that a block ends up spanning two sampled values.
        generated.append(np.unique(thickness_raw[block_ids == bid]).mean())

    generated = np.asarray(generated)

    x_max = np.percentile(np.concatenate([ebsd_thickness, generated]), 99.5)
    x_grid = np.linspace(0.0, x_max, 400)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={'width_ratios': [1.4, 1.0]})

    ax = axes[0]
    for values, label in ((ebsd_thickness, 'EBSD'), (generated, 'Generated')):
        density = _kde_1d(values, x_grid)
        ax.plot(x_grid, density, label=label)
        ax.fill_between(x_grid, density, alpha=0.25)
    ax.set_xlabel('Block thickness [um]')
    ax.set_ylabel('Density')
    ax.set_title('Distribution')
    ax.legend()

    ax = axes[1]
    ax.scatter(ebsd_thickness, rng.normal(0, 0.04, size=ebsd_thickness.size),
               s=4, alpha=0.35, label='EBSD')
    ax.scatter(generated, 1 + rng.normal(0, 0.04, size=generated.size),
               s=6, alpha=0.45, label='Generated')
    ax.set_xlim(0.0, x_max)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['EBSD', 'Generated'])
    ax.set_xlabel('Block thickness [um]')
    ax.set_title('Samples')

    plt.suptitle('Block thickness: EBSD vs Generated RVE', fontsize=16)
    plt.tight_layout()
    plt.savefig(cfg.path(THICKNESS_PLOT), dpi=300)
    plt.close(fig)

    RveInfo.LOGGER.info(
        f'Generated block thickness: {generated.size} blocks, mean {generated.mean():.4f} um, '
        f'median {np.median(generated):.4f} um (EBSD mean {ebsd_thickness.mean():.4f} um)')
