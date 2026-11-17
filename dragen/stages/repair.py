"""Safety net after tessellation: no grain may consist of several bodies.

Every piece of a grain that is not its largest body goes to the neighbouring grain it shares the
most faces with. Repeats until each grain is one body. With connected growth this should find
nothing; the count is reported so a regression shows up in the statistics.
"""
import numpy as np

from dragen.analysis.metrics import component_labels
from dragen.core.domain import VoxelDomain


def merge_detached_pieces(labels: np.ndarray, domain: VoxelDomain, max_rounds: int = 20) -> int:
    """Relabel detached pieces in place; returns the number of voxels that changed grain."""
    flat = labels.reshape(-1)
    moved = 0
    for _ in range(max_rounds):
        comp = component_labels(labels, domain.periodic)
        grain_voxels = np.flatnonzero(flat > 0)
        pairs, sizes = np.unique(np.stack([flat[grain_voxels], comp[grain_voxels]]), axis=1,
                                 return_counts=True)
        # the largest component of every grain is its main body
        order = np.lexsort((-sizes, pairs[0]))
        first = np.ones(order.size, dtype=bool)
        first[1:] = pairs[0, order[1:]] != pairs[0, order[:-1]]
        detached_comps = pairs[1, order[~first]]
        if detached_comps.size == 0:
            return moved

        voxels = grain_voxels[np.isin(comp[grain_voxels], detached_comps)]
        nb = domain.face_neighbors(voxels)
        own = flat[voxels][:, None]
        nb_label = np.where(nb >= 0, flat[np.maximum(nb, 0)], 0)
        foreign = (nb_label > 0) & (nb_label != own)
        piece = np.repeat(comp[voxels], nb.shape[1])[foreign.ravel()]
        neighbour = nb_label[foreign]
        keys, counts = np.unique(np.stack([piece, neighbour]), axis=1, return_counts=True)
        # per piece the neighbour with most shared faces
        order = np.lexsort((-counts, keys[0]))
        best = np.ones(order.size, dtype=bool)
        best[1:] = keys[0, order[1:]] != keys[0, order[:-1]]
        new_label = np.zeros(comp.max() + 1, dtype=flat.dtype)
        new_label[keys[0, order[best]]] = keys[1, order[best]]

        relabel = new_label[comp[voxels]]
        changed = relabel > 0
        flat[voxels[changed]] = relabel[changed]
        moved += int(changed.sum())
    raise RuntimeError(f'grains still split after {max_rounds} repair rounds')
