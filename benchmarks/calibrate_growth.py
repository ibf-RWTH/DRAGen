"""Calibration of TessellationConfig.growth_exponent over several cases and seeds.

Example:
    venv/Scripts/python benchmarks/calibrate_growth.py --betas 0 0.3 0.35 0.4 0.45 0.5 --seeds 1 2 3 4 5

Writes one row per run to benchmarks/results/calibration/growth_exponent.csv and prints the mean
of every metric per exponent and case.
"""
import argparse
import contextlib
import functools
import io
import os
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

import dragen.pipeline as pipeline  # noqa: E402
from dragen.analysis import metrics  # noqa: E402
from dragen.analysis.acceptance import violations  # noqa: E402
from dragen.config import BoxConfig, PhaseConfig, RveConfig, TessellationConfig  # noqa: E402
from common import lognormal_input  # noqa: E402

OUT = os.path.join(REPO_ROOT, 'benchmarks', 'results', 'calibration', 'growth_exponent.csv')
GAN = os.path.join(REPO_ROOT, 'ExampleInput')

CASES = {
    'lognormal_60': (30, lambda: (PhaseConfig('ferrite', lognormal_input(), 1.0),)),
    'gan_dual_40': (20, lambda: (PhaseConfig('martensite', f'{GAN}/Martensite/TrainedData_Martensite.pkl', 0.5),
                                 PhaseConfig('austenite', f'{GAN}/Austenite/TrainedData_Austenite.pkl', 0.5))),
    'gan_ferrite_60': (30, lambda: (PhaseConfig('ferrite', f'{GAN}/Ferrite/TrainedData_Ferrite.pkl', 1.0),)),
}

# load every WGAN once instead of once per run
pipeline.open_source = functools.lru_cache(maxsize=None)(pipeline.open_source)


def run_one(case: str, beta: float, seed: int) -> dict:
    box, phases = CASES[case]
    config = RveConfig(box=BoxConfig((box,) * 3, 2), phases=phases(), seed=seed,
                       tessellation=replace(TessellationConfig(), growth_exponent=beta))
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        result = pipeline.generate(config)
    seconds = time.perf_counter() - start
    target = np.concatenate([[0.0], result.grains.volume])
    phase = np.concatenate([[0], result.grains.phase])
    rep = metrics.report(result.labels, target, phase, result.domain.voxel_volume, config.box.periodic)
    target_ratio = {int(p.phase): p.fraction for p in config.phases}
    found = violations(rep, target_ratio)
    phase_error = max(abs(rep.phase_fractions.get(k, 0.0) - v) for k, v in target_ratio.items())
    return {'beta': beta, 'case': case, 'seed': seed, 'n_grains': rep.n_grains, 'seconds': round(seconds, 2),
            'fragmented': rep.fragmented_grains, 'repaired_voxels': result.repaired_voxels,
            'p10_main': rep.main_body_ratio_p10, 'median': rep.volume_ratio_median,
            'p90': rep.volume_ratio_p90, 'ks': rep.volume_ks, 'phase_error': phase_error,
            'n_violations': len(found), 'violations': ' | '.join(found)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--betas', type=float, nargs='+', default=[0, 0.3, 0.35, 0.4, 0.45, 0.5])
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--cases', nargs='+', default=list(CASES))
    args = parser.parse_args()

    rows = []
    for beta in args.betas:
        for case in args.cases:
            for seed in args.seeds:
                rows.append(run_one(case, beta, seed))
                print(f"beta {beta:<5} {case:15} seed {seed}: {rows[-1]['n_violations']} violations", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    summary = df.groupby(['beta', 'case'])[['p10_main', 'median', 'p90', 'ks', 'phase_error', 'n_violations',
                                            'fragmented', 'seconds']].mean().round(3)
    print(summary.to_string())
    print('\nruns without any violation, per beta:')
    print(df.assign(ok=df.n_violations == 0).groupby('beta')['ok'].sum().to_string())


if __name__ == '__main__':
    main()
