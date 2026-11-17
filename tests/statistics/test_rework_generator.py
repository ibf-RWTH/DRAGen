"""The reworked generator against the acceptance criteria, on the same input as the legacy test."""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'benchmarks'))

from dragen.analysis import metrics  # noqa: E402
from dragen.analysis.acceptance import violations  # noqa: E402
from dragen.config import BoxConfig, PhaseConfig, RveConfig  # noqa: E402
from dragen.pipeline import generate  # noqa: E402


def lognormal_config(box, seed, periodic=None):
    import common
    return RveConfig(box=BoxConfig((box,) * 3, 2, periodic=periodic),
                     phases=(PhaseConfig('ferrite', common.lognormal_input(), 1.0),), seed=seed)


def report_of(result, periodic):
    target = np.concatenate([[0.0], result.grains.volume])
    phase = np.concatenate([[0], result.grains.phase])
    return metrics.report(result.labels, target, phase, result.domain.voxel_volume, periodic)


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_rework_meets_acceptance_criteria(seed):
    result = generate(lognormal_config(15, seed))
    rep = report_of(result, (True,) * 3)
    assert violations(rep, {1: 1.0}) == []
    assert result.repaired_voxels == 0


def test_rework_with_free_surface():
    periodic = (True, True, False)
    result = generate(lognormal_config(15, 1, periodic))
    rep = report_of(result, periodic)
    assert rep.fill_ratio == 1.0
    assert rep.fragmented_grains == 0


@pytest.fixture(scope='module')
def small_inclusions(tmp_path_factory):
    import pandas as pd
    rng = np.random.default_rng(0)
    a = rng.uniform(0.8, 1.4, 3000)
    path = tmp_path_factory.mktemp('incl') / 'inclusions.csv'
    pd.DataFrame({'a': a, 'b': a * rng.uniform(0.6, 1, 3000), 'c': a}).to_csv(path, index=False)
    return str(path)


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_rework_with_bands_and_inclusions(seed, small_inclusions):
    import common
    from dragen.analysis.acceptance import feature_violations
    from dragen.analysis.features import band_report, inclusion_report
    from dragen.config import BandConfig, InclusionConfig
    source = common.lognormal_input()
    config = RveConfig(box=BoxConfig((20,) * 3, 2), phases=(PhaseConfig('ferrite', source, 1.0),), seed=seed,
                       bands=BandConfig(count=2, phase='martensite', source=source, thickness=(3, 4), filling=0.5),
                       inclusions=InclusionConfig(source=small_inclusions, fraction=0.01))
    result = generate(config)
    rep = report_of(result, (True,) * 3)
    assert rep.fill_ratio == 1.0
    assert rep.fragmented_grains == 0
    assert result.repaired_voxels == 0
    bands = band_report(result.labels, result.grains, result.bands, result.domain, result.band_grains)
    inclusions = inclusion_report(result.labels, result.grains, result.domain)
    assert feature_violations(bands, inclusions, 0.5, 0.01) == []
