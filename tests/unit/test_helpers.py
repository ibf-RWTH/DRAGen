"""Unit tests for HelperFunctions methods that don't need a full Run()-configured RveInfo."""
import numpy as np
import pandas as pd

from dragen.utilities.Helpers import HelperFunctions
from dragen.utilities.InputInfo import RveInfo


def test_upsampling_rsa_conserves_volume_per_label():
    rsa = np.array([
        [[0, 1], [1, 2]],
        [[2, 0], [1, 1]],
    ])

    upsampled = HelperFunctions().upsampling_rsa(rsa)

    assert upsampled.shape == (4, 4, 4)
    for label in np.unique(rsa):
        assert np.count_nonzero(upsampled == label) == 8 * np.count_nonzero(rsa == label)


def test_rearange_grain_ids_bands_remaps_band_cells_only():
    # cells -1001/-1002 belong to bands 0/1; 5 and 7 are plain grain IDs that must stay untouched
    rsa = np.array([-1001, -1002, 5, 7])
    bands_df = pd.DataFrame({'GrainID': [0, 1]})
    grains_df = pd.DataFrame({'GrainID': [10, 11]})  # max GrainID is 11 -> start = 12

    result = HelperFunctions().rearange_grain_ids_bands(bands_df, grains_df, rsa)

    assert result.tolist() == [13, 14, 5, 7]


def test_sample_input_3D_hits_target_volume_without_oversized_grains():
    RveInfo.box_size = 20.0
    RveInfo.bin_size = 1.0

    rng = np.random.default_rng(0)
    radii = rng.uniform(low=3.0, high=6.0, size=500)
    data = pd.DataFrame({'a': radii, 'b': radii, 'c': radii})

    sampled = HelperFunctions().sample_input_3D(data.copy(), bs=20.0, phase_id=1)

    target_volume = 20.0 ** 3
    assert target_volume <= sampled['volume'].sum() <= 1.05 * target_volume
    assert (sampled['a'] < RveInfo.box_size / 2).all()
