import importlib
import pytest

from tests.scenarios.assertions import verify_rve_output

# Curated subset covering every distinct feature area at its cheapest parameterization: each
# solver (Abaqus/DAMASK/MOOSE), banding, inclusions, substructure, and a feature-stacking combo.
# These were selected out of the original 52 numbered Case_*.py scenarios this repo used to carry
# (dimension=3, single-RVE, cubic-box in every one -- no 2D or non-cubic case exists in the suite
# at all; see tests/unit/test_periodicity_regression.py, which covers the non-cubic code path
# directly and much more cheaply than a full RVE run). The other 44 were deleted: they were
# largely redundant reparameterizations of these same feature combinations (e.g. DAMASK+box25
# appeared 8+ times with only inclusion/band/subs toggled), never exercised by anything, and
# several were already broken (see the pag-file note below) -- see git history if one is needed
# again as a reference.
#
# NOTE: Case_031/038/039/040/047/048/049 (the deleted substructure scenarios other than 029/030)
# pointed their phase input at './ExampleInput/Substructure/example_pag_inp.csv', which does not
# exist anywhere in this repo (confirmed while wiring up these assertions -- every one of those 7
# scenarios failed with FileNotFoundError before ever reaching RSA/tessellation). Case_029 and
# Case_030 are used here instead since they exercise substructure generation against input files
# that do exist.
scenarios = [
    'Case_016',  # Abaqus baseline
    'Case_002',  # DAMASK baseline
    'Case_003',  # MOOSE baseline
    'Case_035',  # inclusions
    # Case_029/030 are 100% ferrite, so nothing is transformable: they cover the inert path, where
    # subs_flag=True must run through and leave every cell unsubstructured. Case_053 is the case
    # where packets and blocks are actually generated.
    'Case_029',  # substructure + DAMASK, no transformable phase
    'Case_033',  # banding
    'Case_030',  # substructure + MOOSE (MOOSE has no substructure hook at all -- must not crash)
    'Case_053',  # substructure + Abaqus, half martensite -> real packets, blocks and deck files
    'Case_009',  # box25/res1 tier sanity check
]


@pytest.fixture(params=scenarios)
def scenario(request):
    return request.param


def test_scenario(scenario, request):
    module = importlib.import_module(f'tests.scenarios.{scenario}')
    verify_rve_output(module)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
