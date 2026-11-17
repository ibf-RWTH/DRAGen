import os
import sys

# Repo root on sys.path so `import dragen...` resolves regardless of how pytest is invoked, and
# benchmarks/ for the shared inputs (benchmarks/common.py).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, 'benchmarks'))
