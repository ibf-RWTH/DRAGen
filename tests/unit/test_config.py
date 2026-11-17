import pytest

from dragen.config import BandConfig, BoxConfig, InclusionConfig, Phase, PhaseConfig, RsaConfig, RveConfig


def ferrite(fraction=1.0):
    return PhaseConfig(phase='ferrite', source='ferrite.csv', fraction=fraction)


def test_box_shape_spacing_and_default_periodicity():
    box = BoxConfig(size=(15, 15, 15), resolution=2)
    assert box.shape == (30, 30, 30)
    assert box.spacing == (0.5, 0.5, 0.5)
    assert box.periodic == (True, True, True)
    assert box.volume == 3375


def test_box_shape_does_not_suffer_from_float_rounding():
    # 0.1 * 30 is 3.0000000000000004 in floating point; ceil() would give 4 voxels
    assert BoxConfig(size=(0.1 * 30, 3), resolution=1).shape == (3, 3)


def test_non_cubic_sheet_box():
    box = BoxConfig(size=(600, 600, 300), resolution=0.1, periodic=(True, True, False))
    assert box.shape == (60, 60, 30)
    assert box.periodic == (True, True, False)


@pytest.mark.parametrize('kwargs', [
    dict(size=(10,), resolution=1),
    dict(size=(10, 10, 10, 10), resolution=1),
    dict(size=(10, -1, 10), resolution=1),
    dict(size=(10, 10, 10), resolution=0),
    dict(size=(10, 10, 10), resolution=1, periodic=(True, True)),
    dict(size=(10, 10, 10), resolution=0.1),
])
def test_box_rejects_invalid_input(kwargs):
    with pytest.raises(ValueError):
        BoxConfig(**kwargs)


def test_phase_parsing():
    assert Phase.parse('Martensite') is Phase.MARTENSITE
    assert Phase.parse(5) is Phase.AUSTENITE
    assert PhaseConfig(phase=2, source='x.pkl', fraction=0.5).phase is Phase.MARTENSITE
    with pytest.raises(ValueError):
        Phase.parse('cementite')


def test_rsa_shrink_factor_matches_legacy_constant():
    assert RsaConfig().shrink_factor == pytest.approx(0.3 ** (1 / 3))


def test_rve_requires_fractions_adding_up_to_one():
    with pytest.raises(ValueError):
        RveConfig(box=BoxConfig((10, 10, 10), 2), phases=(ferrite(0.6),))
    RveConfig(box=BoxConfig((10, 10, 10), 2),
              phases=(ferrite(0.6), PhaseConfig('martensite', 'm.pkl', 0.4)))


def test_rve_rejects_duplicate_phase():
    with pytest.raises(ValueError):
        RveConfig(box=BoxConfig((10, 10, 10), 2), phases=(ferrite(0.5), ferrite(0.5)))


def test_band_and_inclusion_validation():
    with pytest.raises(ValueError):
        BandConfig(count=1, source=None, thickness=(1, 2))
    with pytest.raises(ValueError):
        BandConfig(count=1, source='b.pkl', thickness=(3, 2))
    with pytest.raises(ValueError):
        InclusionConfig(fraction=0.05)
    with pytest.raises(ValueError):
        RveConfig(box=BoxConfig((10, 10), 2), phases=(ferrite(),),
                  bands=BandConfig(count=1, source='b.pkl', thickness=(1, 2), normal_axis=2))


def test_yaml_round_trip(tmp_path):
    cfg = RveConfig(
        box=BoxConfig(size=(40, 40, 20), resolution=2, periodic=(True, True, False)),
        phases=(ferrite(0.8), PhaseConfig('martensite', 'm.pkl', 0.2)),
        rsa=RsaConfig(max_overlap=0.0),
        bands=BandConfig(count=2, source='b.pkl', thickness=(2, 4), filling=0.5),
        inclusions=InclusionConfig(source='inc.pkl', fraction=0.01),
        seed=7,
    )
    path = tmp_path / 'rve.yaml'
    cfg.save(path)
    assert 'martensite' in path.read_text(encoding='utf-8')
    assert RveConfig.load(path) == cfg
