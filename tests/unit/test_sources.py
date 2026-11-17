import numpy as np
import pandas as pd
import pytest

from dragen.config import Phase
from dragen.sources import CsvSource, random_euler


@pytest.fixture
def csv(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.uniform(1, 3, 500)
    path = tmp_path / 'grains.csv'
    pd.DataFrame({'a': a, 'b': a * 0.5}).to_csv(path, index=False)
    return str(path)


def test_missing_columns_get_legacy_defaults(csv):
    src = CsvSource(csv, Phase.FERRITE)
    table = src.sample(200.0, np.random.default_rng(1))
    assert np.allclose(table.radii[:, 2], table.radii[:, 0])        # c = a
    assert np.all(table.alpha == 0)
    assert table.euler.shape == (len(table), 3)
    assert np.all(table.phase == 1)


def test_sampled_volume_reaches_target(csv):
    table = CsvSource(csv, 'ferrite').sample(500.0, np.random.default_rng(2), tolerance=0.05)
    assert 500.0 <= table.volume.sum() <= 525.0


def test_radius_filters(csv):
    table = CsvSource(csv, 'ferrite').sample(100.0, np.random.default_rng(3), min_radius=0.8, max_radius=2.0)
    assert table.radii.min() >= 0.8
    assert table.radii[:, 0].max() < 2.0


def test_not_enough_grains_raises(csv):
    with pytest.raises(ValueError):
        CsvSource(csv, 'ferrite').sample(1e9, np.random.default_rng(0))


def test_column_a_required(tmp_path):
    path = tmp_path / 'bad.csv'
    pd.DataFrame({'b': [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        CsvSource(str(path), 'ferrite')


def test_random_euler_ranges():
    e = random_euler(1000, np.random.default_rng(0))
    assert e[:, [0, 2]].min() >= 0 and e[:, [0, 2]].max() < 360
    assert e[:, 1].min() >= 0 and e[:, 1].max() <= 180


def test_band_narrower_than_one_grain_still_reaches_target(tmp_path):
    path = tmp_path / 'big.csv'
    pd.DataFrame({'a': [2.0] * 20}).to_csv(path, index=False)     # 33.5 per grain, band 50..52.5
    table = CsvSource(str(path), 'ferrite').sample(50.0, np.random.default_rng(0))
    assert len(table) == 2
    assert table.volume.sum() >= 50.0


def test_max_extent_limits_size_along_band_normal(tmp_path):
    path = tmp_path / 'g.csv'
    rng = np.random.default_rng(0)
    a = rng.uniform(0.5, 3, 2000)
    pd.DataFrame({'a': a, 'b': a * 0.6, 'c': a * 0.5, 'alpha': rng.uniform(0, 180, 2000)}).to_csv(path, index=False)
    src = CsvSource(str(path), 'martensite')
    z = src.sample(200.0, np.random.default_rng(1), max_extent=(2, 2.0))
    assert (2 * z.radii[:, 2]).max() <= 2.0
    x = src.sample(200.0, np.random.default_rng(1), max_extent=(0, 2.0))
    t = np.deg2rad(x.alpha)
    extent_x = 2 * np.sqrt((x.radii[:, 0] * np.cos(t)) ** 2 + (x.radii[:, 1] * np.sin(t)) ** 2)
    assert extent_x.max() <= 2.0 + 1e-9
