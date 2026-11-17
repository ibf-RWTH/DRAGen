"""Run every benchmark case of the reworked generator (the set reported in docs/rework/process.md).

Example:
    venv/Scripts/python benchmarks/run_rework_suite.py

Each run is a separate process (clean timings, no state carried over) and writes to
benchmarks/results/rework/<case>_box<box>_res<res>_seed<seed>/.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

RUNS = (
    [('lognormal', box, 2, 1) for box in (15, 20, 25, 30, 40, 50, 64)]
    + [('lognormal', box, 2, seed) for box in (30, 50) for seed in (2, 3)]
    + [('gan_dual', 20, 2, seed) for seed in (1, 2, 3)]
    + [('banded', 25, res, seed) for res in (1, 2) for seed in (1, 2, 3)]
    + [('inclusions', 25, 1, seed) for seed in (1, 2, 3)]
    + [(case, 25, 2, seed) for case in ('inclusions_ferrite_1pct', 'inclusions_ferrite_5pct') for seed in (1, 2, 3)]
)


def main():
    failed = []
    for case, box, res, seed in RUNS:
        cmd = [sys.executable, '-W', 'ignore', os.path.join(HERE, 'rework_benchmark.py'),
               '--case', case, '--box', str(box), '--resolution', str(res), '--seed', str(seed)]
        ok = subprocess.run(cmd, capture_output=True).returncode == 0
        print(f'{case} box {box} res {res} seed {seed}: {"ok" if ok else "FAILED"}', flush=True)
        if not ok:
            failed.append((case, box, res, seed))
    print(f'{len(RUNS) - len(failed)} of {len(RUNS)} runs finished')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
