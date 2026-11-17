"""The GAN source must turn WGAN output into the same ellipsoids as the legacy reader (reference
data frozen from tag legacy-baseline)."""
import os

import numpy as np
import pytest

from dragen.sources import GanSource, ellipses_to_ellipsoids

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
GAN_FILES = ['ExampleInput/Ferrite/TrainedData_Ferrite.pkl', 'ExampleInput/Martensite/TrainedData_Martensite.pkl']


@pytest.mark.parametrize('relpath', GAN_FILES)
def test_conversion_matches_legacy_reader(relpath):
    """Reference: HelperFunctions.read_input_gan(path, 3, size=400) after torch.manual_seed(5),
    computed on tag legacy-baseline (tests/data/legacy_gan_<phase>.npz)."""
    torch = pytest.importorskip('torch')
    name = os.path.basename(relpath).split('_')[1].split('.')[0].lower()
    legacy = np.load(os.path.join(REPO_ROOT, 'tests', 'data', f'legacy_gan_{name}.npz'))

    source = GanSource(os.path.join(REPO_ROOT, relpath), 'ferrite')
    torch.manual_seed(5)
    new = ellipses_to_ellipsoids(source._load().sample_batch(label=0, size=400).dropna(axis=0))

    for col in ('a', 'b', 'c', 'alpha'):
        assert np.allclose(new[col].to_numpy(), legacy[col]), col
    if 'phi1' in new:
        assert np.allclose(new[['phi1', 'PHI', 'phi2']].to_numpy(), np.column_stack([legacy[c] for c in ('phi1', 'PHI', 'phi2')]))


def test_gan_source_is_reproducible():
    pytest.importorskip('torch')
    source = GanSource(os.path.join(REPO_ROOT, GAN_FILES[1]), 'martensite', batch_size=2000)
    a = source.sample(500.0, np.random.default_rng(1))
    b = source.sample(500.0, np.random.default_rng(1))
    assert np.array_equal(a.radii, b.radii)
    assert np.all(a.phase == 2)
