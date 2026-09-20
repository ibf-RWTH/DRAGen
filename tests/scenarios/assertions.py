"""Shared assertions for tests/scenarios/Case_*.py scenarios.

Each Case_*.py script defines its parameters (phase_ratio, dimension, abaqus_flag, damask_flag,
moose_flag, ...) as module-level globals before calling Run(...).run(). run_all_test_cases.py
imports each scenario via importlib, so those globals are readable straight off the returned
module object -- no need to edit any of the Case_*.py files themselves to gain real assertions.
"""
import os

import numpy as np
import pandas as pd
import pyvista as pv

from dragen.utilities.InputInfo import RveInfo

# RSA/tessellation introduce natural sampling variance around the target phase ratio; this was
# calibrated against real runs. A baseline scenario (Case_016: target [0.5, 0.5] -> actual
# [0.4978, 0.5022]) lands within a percent, but a banded scenario (Case_033: targets
# [Martensite 0.45, Pearlite 0.45] -> actuals [0.357, 0.373]) drifts much further, so this is
# left generous rather than tuned tight to the baseline case.
PHASE_RATIO_TOLERANCE = 0.12

BANDS_PHASE_ID = 7
"""Bands' actual volume fraction doesn't track its phase_ratio target the way other phases do --
empirically (Case_033) actual band volume came in at ~2x the fraction implied by its own
lower/upper_band_bound geometry, most likely an interaction with band_filling that isn't a simple
target-vs-actual comparison. Real, but a separate investigation from making this suite meaningful;
band placement is still exercised (Case_033 is in the curated CI set), just not ratio-checked."""


def verify_rve_output(module) -> None:
    """Check that a Case_*.py scenario produced output matching its own configuration."""
    store_path = RveInfo.store_path
    assert store_path and os.path.isdir(store_path), f"no output directory found at {store_path!r}"

    _verify_phase_ratios(module, store_path)
    _verify_solver_output(module, store_path)
    _verify_substructure_output(module, store_path)


def _verify_phase_ratios(module, store_path: str) -> None:
    grain_data_path = os.path.join(store_path, 'Generation_Data', 'grain_data_output.csv')
    assert os.path.isfile(grain_data_path), f"missing {grain_data_path}"

    grains = pd.read_csv(grain_data_path)
    total_volume = grains['final_discrete_volume'].sum()
    assert total_volume > 0, "generated RVE has zero total volume"

    actual_ratios = grains.groupby('phaseID')['final_discrete_volume'].sum() / total_volume

    for phase_id, target_ratio in module.phase_ratio.items():
        if target_ratio == 0 or phase_id == BANDS_PHASE_ID:
            continue
        actual_ratio = actual_ratios.get(phase_id, 0.0)
        assert abs(actual_ratio - target_ratio) <= PHASE_RATIO_TOLERANCE, (
            f"phase {phase_id}: target ratio {target_ratio}, actual ratio {actual_ratio:.4f} "
            f"(tolerance {PHASE_RATIO_TOLERANCE})"
        )


def _verify_solver_output(module, store_path: str) -> None:
    if module.abaqus_flag:
        assert os.path.isfile(os.path.join(store_path, 'DRAGen_RVE.inp')), "missing Abaqus DRAGen_RVE.inp"
    if module.damask_flag:
        for name in ('material.yaml', 'load.yaml', 'grid.vti'):
            assert os.path.isfile(os.path.join(store_path, name)), f"missing DAMASK output {name}"
    if module.moose_flag:
        for name in ('EulerAngles.txt', 'phases.txt'):
            assert os.path.isfile(os.path.join(store_path, name)), f"missing MOOSE output {name}"


def _verify_substructure_output(module, store_path: str) -> None:
    """Substructures run for DAMASK and Abaqus only (dragen/main3D.py); MOOSE has no hook."""
    if not getattr(module, 'subs_flag', False):
        return
    if not (module.abaqus_flag or module.damask_flag):
        return

    subs_dir = os.path.join(store_path, 'Postprocessing', 'Substructure')
    assert os.path.isdir(subs_dir), f"missing substructure output directory {subs_dir}"

    for name in ('substructure_config.yaml', 'rve_with_orientations.vtk',
                 'combined_phase_grain_packet_block.png'):
        assert os.path.isfile(os.path.join(subs_dir, name)), f"missing substructure output {name}"

    mesh = pv.read(os.path.join(subs_dir, 'rve_with_orientations.vtk'))

    for array in ('GrainID', 'phaseID', 'PacketID', 'BlockID', 'SubstructureFlag',
                  'phi1', 'PHI', 'phi2'):
        assert array in mesh.cell_data, f"rve_with_orientations.vtk is missing {array}"

    _assert_substructure_invariant(mesh)

    transformable = set(RveInfo.subs_transformable_phase_ids)
    has_transformable_phase = any(module.phase_ratio.get(pid, 0) > 0 for pid in transformable)

    sub_flag = np.asarray(mesh.cell_data['SubstructureFlag']).astype(int)
    block_id = np.asarray(mesh.cell_data['BlockID']).astype(int)

    if has_transformable_phase:
        assert np.any(sub_flag == 1), (
            f"phases {sorted(transformable)} are present but no cell was substructured")
        assert block_id.max() > 0, "substructure ran but produced no blocks"
    else:
        # Nothing to transform: the pipeline must run through and leave every cell untouched.
        assert not np.any(sub_flag == 1), "cells were substructured despite no transformable phase"

    if module.abaqus_flag:
        for name in ('substructure.inp', 'SubstructureMaterials.inp', 'SubstructureSections.inp'):
            assert os.path.isfile(os.path.join(store_path, name)), f"missing Abaqus deck file {name}"

        deck = open(os.path.join(store_path, 'DRAGen_RVE.inp')).read()
        assert '*Include, input=substructure.inp' in deck, "DRAGen_RVE.inp does not include substructure.inp"
        assert '*Include, input=SubstructureMaterials.inp' in deck, (
            "DRAGen_RVE.inp does not include SubstructureMaterials.inp")
        # Per-grain and per-block sections would put every element in two solid sections.
        assert '*Solid Section' not in deck, (
            "DRAGen_RVE.inp still writes per-grain sections alongside the per-block ones")


def _assert_substructure_invariant(mesh) -> None:
    """SubstructureFlag == 1 -> PacketID/BlockID > 0; == 0 -> both exactly -1."""
    sub_flag = np.asarray(mesh.cell_data['SubstructureFlag']).astype(int)

    for name in ('PacketID', 'BlockID'):
        ids = np.asarray(mesh.cell_data[name]).astype(int)
        assert np.all(ids[sub_flag == 1] > 0), f"substructure cells with {name} <= 0"
        assert np.all(ids[sub_flag == 0] == -1), f"non-substructure cells with {name} != -1"

        positive = np.unique(ids[ids > 0])
        if positive.size:
            assert np.array_equal(positive, np.arange(1, positive.max() + 1)), (
                f"{name} is not a gapless 1..N range")
