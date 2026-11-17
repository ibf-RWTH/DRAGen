"""Mid-plane slices of two stored label fields side by side, detached grain pieces outlined.

Example:
    venv/Scripts/python benchmarks/plot_slices.py benchmarks/results/legacy/lognormal_box30_res2_seed1 \
        benchmarks/results/rework/lognormal_box30_res2_seed1 benchmarks/results/figures/slices_box30.png
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dragen.analysis.metrics import component_labels  # noqa: E402


def detached_mask(labels: np.ndarray) -> np.ndarray:
    """Voxels that do not belong to the largest body of their grain (fully periodic box)."""
    flat = labels.ravel()
    comp = component_labels(labels, (True,) * labels.ndim)
    pairs, sizes = np.unique(np.stack([flat, comp]), axis=1, return_counts=True)
    order = np.lexsort((-sizes, pairs[0]))
    first = np.ones(order.size, dtype=bool)
    first[1:] = pairs[0, order[1:]] != pairs[0, order[:-1]]
    main = np.zeros(comp.max() + 1, dtype=bool)
    main[pairs[1, order[first]]] = True
    return ~main[comp].reshape(labels.shape)


def panel(ax, labels, title):
    z = labels.shape[2] // 2
    sl = labels[:, :, z]
    colors = np.random.default_rng(0).random((labels.max() + 1, 3)) * 0.7 + 0.25
    ax.imshow(colors[sl].transpose(1, 0, 2), origin='lower', interpolation='nearest')
    edge = np.zeros(sl.shape, dtype=bool)
    edge[:-1] |= sl[:-1] != sl[1:]
    edge[:, :-1] |= sl[:, :-1] != sl[:, 1:]
    ax.contour(edge.T.astype(float), levels=[0.5], colors='white', linewidths=0.4)
    det = detached_mask(labels)[:, :, z]
    if det.any():
        ax.contour(det.T.astype(float), levels=[0.5], colors='black', linewidths=1.2)
    ax.set_title(f'{title}\n{int(det.sum())} voxels in detached pieces in this slice', fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])


def main():
    a, b, out = sys.argv[1:4]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    panel(axes[0], np.load(os.path.join(a, 'labels.npz'))['labels'], 'legacy: ' + os.path.basename(a))
    panel(axes[1], np.load(os.path.join(b, 'labels.npz'))['labels'], 'rework: ' + os.path.basename(b))
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(out)


if __name__ == '__main__':
    main()
