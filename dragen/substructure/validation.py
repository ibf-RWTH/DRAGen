"""Shared ID bookkeeping and invariant checks for the substructure pipeline.

The pipeline's central invariant is:

    SubstructureFlag == 1  ->  PacketID > 0  and  BlockID > 0
    SubstructureFlag == 0  ->  PacketID == -1 and BlockID == -1

i.e. only cells of a transformable phase carry packets and blocks; everything else stays at -1 and
keeps its parent grain orientation. Every stage re-checks this, which is what keeps a partially
labelled mesh from silently reaching the solver export.
"""
import numpy as np

from dragen.utilities.InputInfo import RveInfo

NO_SUBSTRUCTURE = -1


def _log(message: str) -> None:
    RveInfo.LOGGER.info(message)


def renumber_positive_ids(ids: np.ndarray) -> np.ndarray:
    """Renumber positive IDs to a gapless 1..N, leaving -1 (no substructure) untouched."""
    ids = np.asarray(ids).copy().astype(int)

    unique_ids = np.unique(ids[ids > 0])
    mapping = {old: new for new, old in enumerate(unique_ids, start=1)}

    renumbered = ids.copy()
    for old, new in mapping.items():
        renumbered[ids == old] = new

    return renumbered


def check_positive_ids_continuous(ids: np.ndarray, name: str) -> None:
    """Assert the positive IDs form a gapless 1..max range. -1 is allowed and ignored."""
    positive_ids = np.unique(ids[ids > 0])

    if positive_ids.size == 0:
        _log(f'{name}: no positive IDs found.')
        return

    expected = np.arange(1, positive_ids.max() + 1)

    if not np.array_equal(positive_ids, expected):
        missing = np.setdiff1d(expected, positive_ids)
        raise ValueError(f'{name}: missing positive IDs: {missing}')

    _log(f'{name}: OK | min={positive_ids.min()} | max={positive_ids.max()} | '
         f'count={len(positive_ids)}')


def check_ids_nested(child_ids: np.ndarray, parent_ids: np.ndarray,
                     child_name: str, parent_name: str) -> None:
    """Assert every real child ID lives inside exactly one real parent ID."""
    for cid in np.unique(child_ids[child_ids > 0]):
        parents = np.unique(parent_ids[child_ids == cid])

        if parents.size != 1:
            raise ValueError(f'{child_name} {cid} spans multiple {parent_name}s: {parents}')

        if parents[0] < 1:
            raise ValueError(f'{child_name} {cid} belongs to invalid {parent_name}: {parents[0]}')

    _log(f'{child_name}-{parent_name} validity: OK')


def check_substructure_consistency(mesh, id_names=('PacketID', 'BlockID')) -> None:
    """Assert the SubstructureFlag invariant for each of the given ID arrays."""
    if 'SubstructureFlag' not in mesh.cell_data:
        raise KeyError('SubstructureFlag is missing. Run gen_packets() first.')

    sub_flag = np.asarray(mesh.cell_data['SubstructureFlag']).astype(int)

    for name in id_names:
        if name not in mesh.cell_data:
            continue

        ids = np.asarray(mesh.cell_data[name]).astype(int)

        if np.any(ids[sub_flag == 1] < 1):
            raise ValueError(f'Some SubstructureFlag=1 cells have {name} < 1.')

        if np.any(ids[sub_flag == 0] != NO_SUBSTRUCTURE):
            bad = np.unique(ids[sub_flag == 0])
            raise ValueError(f'Some SubstructureFlag=0 cells have {name} != -1: {bad}')

    _log(f'SubstructureFlag consistency for {list(id_names)}: OK')


def cast_int32(mesh, names) -> None:
    """Cast the given cell_data arrays to int32 so meshio/vtk round-trips them losslessly."""
    for name in names:
        if name in mesh.cell_data:
            mesh.cell_data[name] = np.asarray(mesh.cell_data[name]).astype(np.int32)


def report_small_ids(ids: np.ndarray, min_cells: int, name: str) -> None:
    """Warn about real IDs that are still below the minimum cell count after merging."""
    remaining = [(int(i), int(np.count_nonzero(ids == i)))
                 for i in np.unique(ids[ids > 0])
                 if np.count_nonzero(ids == i) < min_cells]

    if remaining:
        RveInfo.LOGGER.warning(f'{len(remaining)} {name}(s) remain below {min_cells} cells: '
                               f'{remaining[:10]}{" ..." if len(remaining) > 10 else ""}')
    else:
        _log(f'No {name}s below {min_cells} cells remain.')
