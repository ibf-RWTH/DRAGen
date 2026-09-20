"""Packet segmentation: split each transformable parent grain into martensite packets.

Two regimes, as in the original implementation:

* A grain that touches the RVE surface may have been split by periodicity into several disconnected
  pieces. Each connected component becomes its own packet -- a k-means split across disconnected
  pieces would produce packets that are not simply connected.
* A grain fully inside the RVE is split by k-means on the cell centres into at most 4 packets.

Cells of non-transformable phases keep PacketID = -1 and SubstructureFlag = 0.
"""
import os
from collections import Counter

import networkx as nx
import numpy as np
import pyvista as pv
from sklearn.cluster import KMeans

from dragen.substructure import validation
from dragen.substructure.config import SubsConfig
from dragen.substructure.validation import NO_SUBSTRUCTURE
from dragen.utilities.InputInfo import RveInfo

MAX_PACKETS_PER_GRAIN = 4


def kmeans_packets(grain: pv.DataSet, min_packet_cells: int = 100, num_logic_cores: int = 1):
    """Split one grain into at most MAX_PACKETS_PER_GRAIN packets by k-means on cell centres."""
    n_packets = grain.n_cells // min_packet_cells + 1
    n_packets = min(MAX_PACKETS_PER_GRAIN, max(1, n_packets))

    centers = grain.cell_centers().points

    os.environ['LOKY_MAX_CPU_COUNT'] = str(num_logic_cores)
    kmeans = KMeans(n_clusters=n_packets, init='k-means++', n_init=10, random_state=0)

    labels = kmeans.fit_predict(centers)

    return n_packets, labels, kmeans.cluster_centers_


def _connected_components(grain: pv.DataSet) -> np.ndarray:
    """Label the point-connected components of a grain, 0..k-1, in local cell order."""
    graph = nx.Graph()
    graph.add_nodes_from(range(grain.n_cells))

    for cell_id in range(grain.n_cells):
        for neighbor in grain.cell_neighbors(cell_id, connections='points'):
            graph.add_edge(cell_id, int(neighbor))

    labels = -np.ones(grain.n_cells, dtype=int)
    for label, component in enumerate(nx.connected_components(graph)):
        for cell_id in component:
            labels[cell_id] = label

    return labels


def gen_packets(cfg: SubsConfig, mesh: pv.DataSet) -> pv.DataSet:
    """Assign PacketID, ClusterID and SubstructureFlag to every cell of `mesh`."""
    if 'phaseID' not in mesh.cell_data:
        raise KeyError('phaseID is missing in mesh.cell_data')

    grain_ids = np.asarray(mesh.cell_data['GrainID']).astype(int)
    phase_ids = np.asarray(mesh.cell_data['phaseID']).astype(int)

    cluster_id = np.full(mesh.n_cells, NO_SUBSTRUCTURE, dtype=int)
    packet_id = np.full(mesh.n_cells, NO_SUBSTRUCTURE, dtype=int)
    sub_flag = np.zeros(mesh.n_cells, dtype=int)

    xmin, ymin, zmin = mesh.points.min(axis=0)
    xmax, ymax, zmax = mesh.points.max(axis=0)

    num_tot_packets = 0

    for grain in range(1, int(grain_ids.max()) + 1):

        mask = grain_ids == grain
        if not np.any(mask):
            continue

        phases_in_grain = np.unique(phase_ids[mask])
        if phases_in_grain.size != 1:
            raise ValueError(f'Grain {grain} contains multiple phaseIDs: {phases_in_grain}')

        phase_id = int(phases_in_grain[0])

        if phase_id not in cfg.transformable_phase_ids:
            # Ferrite, austenite, inclusions, bands: no packets, PacketID stays -1.
            continue

        sub_flag[mask] = 1
        grain_mesh = mesh.extract_cells(mask)

        g_min = grain_mesh.points.min(axis=0)
        g_max = grain_mesh.points.max(axis=0)

        touches_boundary = (
            np.isclose(g_min[0], xmin) or np.isclose(g_max[0], xmax) or
            np.isclose(g_min[1], ymin) or np.isclose(g_max[1], ymax) or
            np.isclose(g_min[2], zmin) or np.isclose(g_max[2], zmax)
        )

        if touches_boundary:
            local_ids = _connected_components(grain_mesh)
            n_local = int(local_ids.max()) + 1

            if np.any(local_ids < 0):
                raise ValueError(f'Grain {grain}: incomplete cluster labelling.')

            RveInfo.LOGGER.info(f'Grain {grain} touches the RVE surface: '
                                f'{n_local} connected component(s).')
        else:
            n_local, local_ids, _ = kmeans_packets(
                grain_mesh,
                min_packet_cells=cfg.min_packet_cells,
                num_logic_cores=cfg.num_logic_cores,
            )
            RveInfo.LOGGER.info(f'Grain {grain} is fully inside the RVE: {n_local} packet(s).')

        cluster_id[mask] = local_ids
        packet_id[mask] = local_ids + num_tot_packets
        num_tot_packets += n_local

    # Shift real packets to 1-based; non-transformable cells stay at -1.
    packet_id[packet_id >= 0] += 1

    mesh.cell_data['ClusterID'] = cluster_id.astype(np.int32)
    mesh.cell_data['PacketID'] = packet_id.astype(np.int32)
    mesh.cell_data['SubstructureFlag'] = sub_flag.astype(np.int32)

    RveInfo.LOGGER.info(f'Total number of packets in the RVE: {num_tot_packets}')

    validation.check_positive_ids_continuous(packet_id, 'PacketID')
    validation.check_ids_nested(packet_id, grain_ids, 'Packet', 'Grain')
    validation.check_substructure_consistency(mesh, id_names=('PacketID',))

    return mesh


def merge_small_packets(cfg: SubsConfig, mesh: pv.DataSet) -> pv.DataSet:
    """Merge packets below `min_cells_per_packet` into their most common neighbouring packet.

    Merging is restricted to neighbours in the same grain, so a packet never crosses a grain
    boundary. Packets with no eligible neighbour are left alone and reported.
    """
    packet_ids = np.asarray(mesh.cell_data['PacketID']).copy().astype(int)
    grain_ids = np.asarray(mesh.cell_data['GrainID']).astype(int)

    mesh.cell_data['OldPacketID'] = packet_ids.copy().astype(np.int32)

    validation.check_positive_ids_continuous(packet_ids, 'Initial PacketID')
    validation.check_ids_nested(packet_ids, grain_ids, 'Packet', 'Grain')
    validation.check_substructure_consistency(mesh, id_names=('PacketID',))

    for iteration in range(1, cfg.max_merge_iterations + 1):

        small_packets = [pid for pid in np.unique(packet_ids[packet_ids > 0])
                         if np.count_nonzero(packet_ids == pid) < cfg.min_cells_per_packet]

        RveInfo.LOGGER.info(f'[Iter {iteration}] small packets: {len(small_packets)}')

        if not small_packets:
            break

        reassign = {}

        for pid in small_packets:
            cells = np.nonzero(packet_ids == pid)[0]
            if cells.size == 0:
                continue

            grains = np.unique(grain_ids[cells])
            if grains.size != 1:
                raise ValueError(f'Packet {pid} spans multiple GrainIDs: {grains}')
            grain = int(grains[0])

            votes = Counter()

            for cell_id in cells:
                neighbors = mesh.cell_neighbors(int(cell_id), connections='points')
                if not neighbors:
                    continue

                neighbors = np.asarray(neighbors, dtype=int)
                neighbors = neighbors[grain_ids[neighbors] == grain]
                if neighbors.size == 0:
                    continue

                neighbor_pids = packet_ids[neighbors]
                neighbor_pids = neighbor_pids[(neighbor_pids > 0) & (neighbor_pids != pid)]
                if neighbor_pids.size == 0:
                    continue

                votes.update(neighbor_pids.tolist())

            if not votes:
                RveInfo.LOGGER.info(f'Packet {pid}: no valid neighbouring packet found. Skipped.')
                continue

            target_pid, n_votes = votes.most_common(1)[0]
            reassign[pid] = (target_pid, n_votes, grain)

        if not reassign:
            RveInfo.LOGGER.info(f'[Iter {iteration}] no merges possible under current rules. Stop.')
            break

        for pid, (target_pid, n_votes, grain) in reassign.items():
            cells = np.nonzero(packet_ids == pid)[0]
            if cells.size == 0:
                continue
            packet_ids[cells] = target_pid
            RveInfo.LOGGER.info(f'  merged Packet {pid} into Packet {target_pid} '
                                f'(GrainID={grain}, votes={n_votes})')

        RveInfo.LOGGER.info(f'[Iter {iteration}] merged packets: {len(reassign)}')

    packet_ids = validation.renumber_positive_ids(packet_ids)
    mesh.cell_data['PacketID'] = packet_ids.astype(np.int32)

    validation.check_positive_ids_continuous(packet_ids, 'Final PacketID')
    validation.check_ids_nested(packet_ids, grain_ids, 'Packet', 'Grain')
    validation.check_substructure_consistency(mesh, id_names=('PacketID',))
    validation.report_small_ids(packet_ids, cfg.min_cells_per_packet, 'packet')

    validation.cast_int32(mesh, ['PacketID', 'OldPacketID', 'GrainID', 'SubstructureFlag'])

    return mesh
