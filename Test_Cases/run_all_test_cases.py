import importlib
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

scenarios = [
    'Case_001',
    'Case_002',
    'Case_003',
    'Case_004',
    'Case_005',
    'Case_006',
    'Case_007',
    'Case_008',
    'Case_009',
    'Case_010',
    'Case_011',
    'Case_012',
    'Case_013',
    'Case_014',
    'Case_015',
    'Case_016',
    'Case_017',
    'Case_018',
    'Case_019',
    'Case_020',
    'Case_021',
    'Case_022',
    'Case_023',
    'Case_024',
    'Case_025',
    'Case_026',
    'Case_027',
    'Case_028',
    'Case_029',
    'Case_030',
    'Case_031',
    'Case_032',
    'Case_033',
    'Case_034',
    'Case_035',
    'Case_036',
    'Case_037',
    'Case_038',
    'Case_039',
    'Case_040',
    'Case_041',
    'Case_042',
    'Case_043',
    'Case_044',
    'Case_045',
    'Case_046',
    'Case_047',
    'Case_048',
    'Case_049',
    'Case_050',
    'Case_051',
    'Case_052',
]

@pytest.fixture(params=scenarios)
def scenario(request):
    return request.param

def test_scenario(scenario):
    module = importlib.import_module(f'Test_Cases.{scenario}')

if __name__ == "__main__":
    pytest.main([__file__, "-v"])