"""Benchmark of the reworked generator on the same cases as benchmarks/legacy_baseline.py.

Example:
    venv/Scripts/python benchmarks/rework_benchmark.py --case lognormal --box 30 --resolution 2 --seed 1

Results go to benchmarks/results/rework/<case>_box<box>_res<res>_seed<seed>/ with the same
result.json layout as the legacy baseline, plus the acceptance check.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from common import lognormal_input  # noqa: E402
from dragen.analysis import metrics  # noqa: E402
from dragen.analysis.acceptance import feature_violations, violations  # noqa: E402
from dragen.analysis.features import band_report, inclusion_report  # noqa: E402
from dragen.config import BandConfig, BoxConfig, InclusionConfig, PhaseConfig, RveConfig  # noqa: E402
from dragen.pipeline import expected_phase_fractions, generate  # noqa: E402

RESULT_DIR = os.path.join(REPO_ROOT, 'benchmarks', 'results', 'rework')

CASES = {
    'lognormal': lambda: (PhaseConfig('ferrite', lognormal_input(), 1.0),),
    # same GAN pair as legacy_baseline.py
    'gan_dual': lambda: (
        PhaseConfig('martensite', os.path.join(REPO_ROOT, 'ExampleInput/Martensite/TrainedData_Martensite.pkl'), 0.5),
        PhaseConfig('austenite', os.path.join(REPO_ROOT, 'ExampleInput/Austenite/TrainedData_Austenite.pkl'), 0.5)),
    # same inputs as the legacy 'banded' and 'inclusions' cases (tests/scenarios/Case_033, Case_035)
    'banded': lambda: (
        PhaseConfig('martensite', os.path.join(REPO_ROOT, 'ExampleInput/Martensite/TrainedData_Martensite.pkl'), 1.0),),
    'inclusions': lambda: (
        PhaseConfig('martensite', os.path.join(REPO_ROOT, 'ExampleInput/Martensite/TrainedData_Martensite.pkl'), 1.0),),
    # coarse ferrite matrix whose grains can host the GAN inclusions
    'inclusions_ferrite_1pct': lambda: (PhaseConfig('ferrite', lognormal_input(), 1.0),),
    'inclusions_ferrite_5pct': lambda: (PhaseConfig('ferrite', lognormal_input(), 1.0),),
}

FEATURES = {
    'banded': lambda: dict(bands=BandConfig(
        count=1, phase='martensite', thickness=(2, 4), filling=0.5,
        source=os.path.join(REPO_ROOT, 'ExampleInput/Banding/TrainedData_Martensite_bands.pkl'))),
    'inclusions': lambda: dict(inclusions=InclusionConfig(
        source=os.path.join(REPO_ROOT, 'ExampleInput/Inclusions/TrainedData_Inclusions.pkl'), fraction=0.05)),
    'inclusions_ferrite_1pct': lambda: dict(inclusions=InclusionConfig(
        source=os.path.join(REPO_ROOT, 'ExampleInput/Inclusions/TrainedData_Inclusions.pkl'), fraction=0.01)),
    'inclusions_ferrite_5pct': lambda: dict(inclusions=InclusionConfig(
        source=os.path.join(REPO_ROOT, 'ExampleInput/Inclusions/TrainedData_Inclusions.pkl'), fraction=0.05)),
}


def code_state() -> dict:
    """Commit of the working tree and whether library code has uncommitted changes."""
    def git(*args):
        return subprocess.run(['git', *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return {'commit': git('rev-parse', '--short', 'HEAD'),
            'library_dirty': bool(git('status', '--porcelain', '--', 'dragen', 'InputGenerator'))}


def run(case: str, box: float, resolution: float, seed: int, out_dir: str = None) -> dict:
    config = RveConfig(box=BoxConfig((box,) * 3, resolution), phases=CASES[case](), seed=seed,
                       **FEATURES.get(case, dict)())
    if out_dir is None:
        out_dir = os.path.join(RESULT_DIR, f'{case}_box{box:g}_res{resolution:g}_seed{seed}')
    os.makedirs(out_dir, exist_ok=True)

    start = time.perf_counter()
    result = generate(config)
    total = time.perf_counter() - start

    target = np.concatenate([[0.0], result.grains.volume])
    phase = np.concatenate([[0], result.grains.phase])
    rep = metrics.report(result.labels, target, phase, result.domain.voxel_volume, config.box.periodic)
    stats = result.tessellation
    n = result.labels.size
    target_ratio = expected_phase_fractions(config, result.bands, result.domain)
    bands = band_report(result.labels, result.grains, result.bands, result.domain, result.band_grains)
    requested = None if result.inclusions is None else result.inclusions.placed + result.inclusions.not_placed
    inclusions = inclusion_report(result.labels, result.grains, result.domain, requested)

    out = {
        'case': case, 'box': box, 'resolution': resolution, 'seed': seed, 'code': code_state(),
        'shape': list(result.labels.shape), 'n_grains': len(result.grains),
        'seconds': {**{k: round(v, 3) for k, v in result.seconds.items()}, 'total': round(total, 3)},
        'report': rep.to_dict(),
        'target_phase_ratio': target_ratio,
        'acceptance_violations': violations(rep, target_ratio) + feature_violations(
            bands, inclusions, config.bands.filling, config.inclusions.fraction),
        'bands': bands,
        'inclusions': inclusions,
        'rsa_attempts': result.rsa_attempts,
        'repaired_voxels': result.repaired_voxels,
        'tessellation': {
            'epochs': stats.epochs,
            'reset_epoch': stats.reset_epoch if stats.reset_epoch >= 0 else None,
            'fraction_after_rsa': round(stats.filled_after_rsa / n, 4),
            'fraction_from_regular_growth': round(stats.filled_regular / n, 4),
            'fraction_from_reset_phase': round(stats.filled_reset / n, 4),
            'fraction_from_dilation': round(stats.filled_by_dilation / n, 4),
            'filled_per_epoch': [round(f / n, 4) for f in stats.filled_per_epoch],
        },
        'config': config.to_dict(),
    }
    np.savez_compressed(os.path.join(out_dir, 'labels.npz'), labels=result.labels)
    result.grains.to_frame().to_csv(os.path.join(out_dir, 'grains.csv'), index=False)
    config.save(os.path.join(out_dir, 'config.yaml'))
    with open(os.path.join(out_dir, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case', choices=sorted(CASES), default='lognormal')
    parser.add_argument('--box', type=float, default=20.0, help='edge length in micrometres')
    parser.add_argument('--resolution', type=float, default=2.0, help='voxels per micrometre')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    out = run(args.case, args.box, args.resolution, args.seed)
    print(json.dumps({k: v for k, v in out.items() if k not in ('config',)}, indent=2))


if __name__ == '__main__':
    main()
