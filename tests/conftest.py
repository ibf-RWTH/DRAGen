import sys
import os

# Repo root on sys.path so `import dragen...` and `import tests.scenarios...` resolve regardless
# of how pytest is invoked (bare `pytest`, `python -m pytest`, from any cwd).
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from dragen.utilities.InputInfo import RveInfo


@pytest.fixture(autouse=True)
def _isolate_rve_info():
    """Snapshot every RveInfo class attribute before a test and restore it afterward.

    RveInfo is a bare class used as global config/state (dragen/utilities/InputInfo.py):
    Run.__init__ doesn't reset every attribute it defines (e.g. n_pts_y/n_pts_z can leak from a
    prior Run() - see tests/unit/test_rveinfo_leak.py), so without this, running multiple tests
    or scenarios in one pytest session could have one test's leftover config silently affect the
    next. Applies to everything under tests/ (both tests/unit and tests/scenarios).
    """
    snapshot = dict(vars(RveInfo))
    yield
    for key, value in snapshot.items():
        if key.startswith('__'):
            continue
        setattr(RveInfo, key, value)
