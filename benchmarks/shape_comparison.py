"""Compare grain shapes of stored RVEs: legacy, rework variants and a plain Voronoi reference.

Example:
    venv/Scripts/python benchmarks/shape_comparison.py --tag lognormal_box30_res2_seed1

Reads labels.npz / result.json from benchmarks/results/{legacy,rework_beta0,rework}/<tag>/, adds a
periodic Voronoi tessellation with the same number of grains, writes shapes_<tag>.csv and
shapes_<tag>.png to benchmarks/results/figures/ and prints quantiles and KS distances to legacy.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
from dragen.analysis.shapes import grain_shapes  # noqa: E402

RESULTS = os.path.join(REPO_ROOT, 'benchmarks', 'results')
VARIANTS = {'legacy': 'legacy', 'rework beta=0': 'rework_beta0', 'rework beta=0.3': 'rework'}
COLUMNS = ['solidity', 'elongation_2', 'elongation_3', 'slope']


def voronoi_labels(shape, n_grains, seed=0):
    rng = np.random.default_rng(seed)
    seeds = rng.random((n_grains, 3)) * np.asarray(shape)
    tree = cKDTree(seeds, boxsize=np.asarray(shape, dtype=float))
    centres = np.stack(np.meshgrid(*[np.arange(n) + 0.5 for n in shape], indexing='ij'), axis=-1)
    return tree.query(centres.reshape(-1, 3))[1].reshape(shape) + 1


def input_shapes(grains_csv: str) -> pd.DataFrame:
    """Elongation and slope the input ellipsoids would have (ratios are unaffected by shrinking)."""
    g = pd.read_csv(grains_csv)
    r = np.sort(g[['a', 'b', 'c']].to_numpy(), axis=1)[:, ::-1]
    slope = np.where(g['a'] >= g['b'], g['alpha'], g['alpha'] + 90) % 180
    return pd.DataFrame({'elongation_2': r[:, 1] / r[:, 0], 'elongation_3': r[:, 2] / r[:, 0], 'slope': slope})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', default='lognormal_box30_res2_seed1')
    args = parser.parse_args()

    frames = {}
    n_grains = None
    for name, folder in VARIANTS.items():
        d = os.path.join(RESULTS, folder, args.tag)
        labels = np.load(os.path.join(d, 'labels.npz'))['labels']
        meta = json.load(open(os.path.join(d, 'result.json')))
        spacing = np.asarray([meta['box']] * 3) / np.asarray(labels.shape)
        frames[name] = grain_shapes(labels, spacing, (True,) * 3)
        if folder == 'rework':
            n_grains = meta['n_grains']
            frames['input (rework grains)'] = input_shapes(os.path.join(d, 'grains.csv'))
    vor = voronoi_labels(labels.shape, n_grains)
    frames['voronoi'] = grain_shapes(vor, spacing, (True,) * 3)

    table = pd.concat([f.assign(variant=k) for k, f in frames.items()], ignore_index=True)
    os.makedirs(os.path.join(RESULTS, 'figures'), exist_ok=True)
    table.to_csv(os.path.join(RESULTS, 'figures', f'shapes_{args.tag}.csv'), index=False)

    print(table.groupby('variant')[COLUMNS].quantile([0.1, 0.5, 0.9]).unstack().round(3).to_string())
    print('\nKS distance to legacy:')
    for name, f in frames.items():
        if name == 'legacy':
            continue
        ks = {c: round(ks_2samp(f[c].dropna(), frames['legacy'][c].dropna()).statistic, 3)
              for c in COLUMNS if c in f}
        print(f'  {name:22} {ks}')

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    bins = {'solidity': np.linspace(0.3, 1, 36), 'elongation_2': np.linspace(0, 1, 26),
            'elongation_3': np.linspace(0, 1, 26), 'slope': np.linspace(0, 180, 19)}
    colors = dict(zip(frames, plt.rcParams['axes.prop_cycle'].by_key()['color']))
    for ax, col in zip(axes, COLUMNS):
        for name, f in frames.items():
            if col in f:
                ax.hist(f[col].dropna(), bins=bins[col], density=True, histtype='step', linewidth=1.5,
                        label=name, color=colors[name])
        ax.set_xlabel(col)
    axes[0].set_ylabel('density')
    handles = [plt.Line2D([], [], color=c, label=n) for n, c in colors.items()]
    axes[-1].legend(handles=handles, fontsize=8)
    fig.suptitle(args.tag)
    fig.tight_layout()
    out = os.path.join(RESULTS, 'figures', f'shapes_{args.tag}.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(out)


if __name__ == '__main__':
    main()
