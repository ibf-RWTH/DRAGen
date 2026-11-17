"""Characterise grains that are split into several bodies in a stored label field.

A piece counts as detached when it has no face connection to the largest body of its grain,
with neighbours across the box boundary treated as connected (fully periodic box).

Example:
    venv/Scripts/python benchmarks/fragment_analysis.py benchmarks/results/legacy/lognormal_box30_res2_seed1

Writes fragments.json (summary) and fragments.csv (one row per detached piece) into that folder.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

FACE = ndimage.generate_binary_structure(3, 1)


def periodic_components(mask: np.ndarray) -> np.ndarray:
    """Label face-connected components of mask on a fully periodic box."""
    lab, n = ndimage.label(mask, structure=FACE)
    parent = np.arange(n + 1)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for axis in range(mask.ndim):
        first, last = np.take(lab, 0, axis=axis), np.take(lab, -1, axis=axis)
        both = (first > 0) & (last > 0)
        for i, j in zip(first[both], last[both]):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    roots = np.array([find(i) for i in range(n + 1)])
    return roots[lab] * (lab > 0)


def analyse(labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for grain in np.unique(labels[labels > 0]):
        comp = periodic_components(labels == grain)
        ids, counts = np.unique(comp[comp > 0], return_counts=True)
        if len(ids) < 2:
            continue
        main = ids[np.argmax(counts)]
        for piece, size in zip(ids, counts):
            if piece == main:
                continue
            m = comp == piece
            on_face = any(np.take(m, i, axis=a).any() for a in range(3) for i in (0, -1))
            ring = ndimage.binary_dilation(np.pad(m, 1, mode='wrap'), FACE)[1:-1, 1:-1, 1:-1] & ~m
            rows.append({'grain': int(grain), 'piece_voxels': int(size),
                         'main_body_voxels': int(counts.max()), 'touches_box_face': bool(on_face),
                         'surrounding_grains': len(set(np.unique(labels[ring]).tolist()) - {int(grain)})})
    return pd.DataFrame(rows, columns=['grain', 'piece_voxels', 'main_body_voxels',
                                       'touches_box_face', 'surrounding_grains'])


def main():
    folder = sys.argv[1]
    labels = np.load(os.path.join(folder, 'labels.npz'))['labels']
    pieces = analyse(labels)
    size = pieces['piece_voxels'] if len(pieces) else pd.Series(dtype=int)
    summary = {
        'n_voxels': int(labels.size),
        'n_grains': int(len(np.unique(labels[labels > 0]))),
        'fragmented_grains': int(pieces['grain'].nunique()) if len(pieces) else 0,
        'detached_pieces': int(len(pieces)),
        'detached_voxel_fraction': float(size.sum() / labels.size),
        'pieces_by_size': {'1': int((size == 1).sum()), '2-9': int(size.between(2, 9).sum()),
                           '10-99': int(size.between(10, 99).sum()), '>=100': int((size >= 100).sum())},
        'largest_piece': int(size.max()) if len(size) else 0,
        'pieces_touching_box_face': int(pieces['touches_box_face'].sum()) if len(pieces) else 0,
        'pieces_enclosed_by_2plus_grains': int((pieces['surrounding_grains'] >= 2).sum()) if len(pieces) else 0,
    }
    pieces.to_csv(os.path.join(folder, 'fragments.csv'), index=False)
    with open(os.path.join(folder, 'fragments.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
