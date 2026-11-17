import numpy as np
import pytest

from dragen.core.grains import GrainTable
from dragen.pipeline import growth_rates


@pytest.fixture
def grains():
    radii = np.array([[4.0, 2.0, 4.0], [1.0, 0.5, 1.0], [2.0, 3.0, 2.0]])
    return GrainTable(radii=radii, alpha=np.zeros(3), euler=np.zeros((3, 3)), phase=np.ones(3, int))


def test_exponent_zero_is_the_legacy_law(grains):
    shrink, bin_size = 0.5, 0.25
    rate, _ = growth_rates(grains, shrink, bin_size, margin=2.0, exponent=0.0)
    legacy = bin_size / (grains.radii.max(axis=0) * shrink)       # a_max, b_max, c_max of the shrunk radii
    assert np.allclose(rate, np.broadcast_to(legacy, rate.shape))


def test_exponent_one_moves_every_front_one_bin_per_epoch(grains):
    shrink, bin_size = 0.5, 0.25
    rate, _ = growth_rates(grains, shrink, bin_size, margin=2.0, exponent=1.0)
    longest = grains.radii.max(axis=1) * shrink
    assert np.allclose(rate, (bin_size / longest)[:, None])
    assert np.allclose(rate * longest[:, None], bin_size)          # front step along the longest axis


def test_every_kernel_reaches_the_margin(grains):
    shrink, margin = 0.5, 2.0
    for exponent in (0.0, 0.4, 1.0):
        rate, max_epoch = growth_rates(grains, shrink, 0.25, margin=margin, exponent=exponent)
        reach = shrink * (1 + max_epoch[:, None] * rate)
        assert np.all(reach.min(axis=1) >= margin - 1e-9)
