"""Inputs shared by the benchmarks and tests of the new code."""
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_DIR = os.path.join(REPO_ROOT, 'benchmarks', 'inputs')


def lognormal_input(seed: int = 0, n: int = 20000) -> str:
    """Write (once) a synthetic single-phase grain file: lognormal radii, elongated in x-y.
    Same file as in the legacy baseline (tag legacy-baseline, benchmarks/legacy_baseline.py)."""
    path = os.path.join(INPUT_DIR, 'lognormal_ferrite.csv')
    if not os.path.isfile(path):
        rng = np.random.default_rng(seed)
        a = rng.lognormal(mean=np.log(2.5), sigma=0.35, size=n)
        df = pd.DataFrame({'a': a, 'b': a * rng.uniform(0.6, 1.0, n), 'c': a,
                           'alpha': rng.uniform(0, 180, n)})
        os.makedirs(INPUT_DIR, exist_ok=True)
        df.to_csv(path, index=False)
    return path
