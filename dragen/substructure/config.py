"""Configuration object for the substructure pipeline.

Every knob the pipeline uses lives here. `SubsConfig.from_rve_info()` is the only place that reads
`RveInfo`, so the pipeline modules themselves stay free of global state and are directly testable.
"""
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import pandas as pd
import yaml

from dragen.utilities.InputInfo import RveInfo

# Length unit of the mesh coordinates the pipeline operates on, per solver. Block thicknesses are
# always given in micrometres (GUI value / EBSD file) and are converted into these units.
#   damask: dragen/generation/spectral.py::write_grid builds grid.vti with size=[RveInfo.box_size,...],
#           and box_size is in micrometres.
#   abaqus: dragen/utilities/PvGridGeneration.py::gen_blocks builds the grid on box_size/1000, i.e. mm.
SOLVER_LENGTH_UNIT = {
    'damask': 'micrometer',
    'abaqus': 'millimeter',
}

THICKNESS_SCALE = {
    'meter': 1e-6,
    'millimeter': 1e-3,
    'micrometer': 1.0,
}

ORIENTATION_MODES = ('KS', 'experimental')


@dataclass
class SubsConfig:
    """All parameters of one substructure run."""

    store_path: str
    solver: str                       # 'damask' | 'abaqus'
    length_unit: str

    # --- block generation ---------------------------------------------------
    block_generation_mode: str        # 'user' (mean thickness) | 'file' (EBSD distribution)
    average_block_thickness: Optional[float] = None   # micrometres, 'user' mode
    block_file: Optional[str] = None                  # EBSD csv with a 'block_thickness' column
    lower_percentile: float = 5.0
    upper_percentile: float = 95.0
    min_physical_thickness: float = 0.1               # micrometres, drops EBSD noise

    # --- segmentation ------------------------------------------------------
    transformable_phase_ids: List[int] = field(default_factory=lambda: [2, 3, 4])
    min_packet_cells: int = 100        # cells per packet targeted by the k-means split
    min_block_cells: int = 10          # packets below this are kept as a single block
    min_cells_per_packet: int = 5      # packets below this get merged into a neighbour
    min_cells_per_block: int = 5       # blocks below this get merged into a neighbour
    max_merge_iterations: int = 50
    num_logic_cores: int = 1
    seed: int = 42

    # --- orientation --------------------------------------------------------
    orientation_mode: str = 'KS'
    parent_orientation_file: Optional[str] = None
    child_orientation_file: Optional[str] = None

    # --- abaqus deck --------------------------------------------------------
    depvar_number: int = 176
    user_material_constants: str = '1.,3.'
    section_controls: str = 'EC-1'
    grain_size_value: float = 1.8

    # ------------------------------------------------------------------ #

    @property
    def thickness_scale(self) -> float:
        """Factor converting a thickness in micrometres into mesh coordinate units."""
        try:
            return THICKNESS_SCALE[self.length_unit]
        except KeyError:
            raise ValueError(f'Unknown length_unit: {self.length_unit}')

    @property
    def subs_dir(self) -> str:
        """Directory all substructure artefacts (vtk dumps, plots, config) are written to."""
        return os.path.join(self.store_path, 'Postprocessing', 'Substructure')

    def path(self, filename: str) -> str:
        return os.path.join(self.subs_dir, filename)

    def dump(self) -> str:
        """Write the config next to the results as a record of what produced them."""
        os.makedirs(self.subs_dir, exist_ok=True)
        config_file = self.path('substructure_config.yaml')
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)
        return config_file

    # ------------------------------------------------------------------ #

    @classmethod
    def from_rve_info(cls, store_path: str, solver: str) -> 'SubsConfig':
        if solver not in SOLVER_LENGTH_UNIT:
            raise ValueError(f"Unknown solver '{solver}', expected one of {list(SOLVER_LENGTH_UNIT)}")

        use_file_mode = bool(RveInfo.subs_file_flag)
        if use_file_mode and not RveInfo.subs_file:
            raise ValueError(
                'subs_file_flag is set but no subs_file was given. A csv with a '
                "'block_thickness' column is required for file-based block generation."
            )

        transformable = list(RveInfo.subs_transformable_phase_ids or [2, 3, 4])

        orientation_mode = RveInfo.subs_orientation_mode or 'KS'
        if orientation_mode not in ORIENTATION_MODES:
            raise ValueError(
                f"Unknown subs_orientation_mode '{orientation_mode}', expected one of "
                f'{list(ORIENTATION_MODES)}'
            )

        parent_file = RveInfo.subs_parent_orientation_file
        child_file = RveInfo.subs_child_orientation_file
        if orientation_mode == 'experimental':
            parent_file = parent_file or cls._parent_file_from_phase_input(transformable)
            _validate_orientation_file(parent_file, 'parent_orientation_file', require_grain_id=False)
            if not child_file:
                raise ValueError(
                    "orientation_mode 'experimental' requires subs_child_orientation_file: a csv of "
                    'measured block/child orientations with grain_id, phi1, PHI, phi2.'
                )
            _validate_orientation_file(child_file, 'child_orientation_file', require_grain_id=True)

        return cls(
            store_path=store_path,
            solver=solver,
            length_unit=SOLVER_LENGTH_UNIT[solver],
            block_generation_mode='file' if use_file_mode else 'user',
            average_block_thickness=float(RveInfo.t_mu) if RveInfo.t_mu is not None else None,
            block_file=RveInfo.subs_file if use_file_mode else None,
            lower_percentile=float(RveInfo.subs_lower_percentile),
            upper_percentile=float(RveInfo.subs_upper_percentile),
            transformable_phase_ids=transformable,
            min_packet_cells=int(RveInfo.subs_min_packet_cells),
            min_block_cells=int(RveInfo.subs_min_block_cells),
            min_cells_per_packet=int(RveInfo.subs_min_cells_per_packet),
            min_cells_per_block=int(RveInfo.subs_min_cells_per_block),
            num_logic_cores=int(RveInfo.num_cores),
            orientation_mode=orientation_mode,
            parent_orientation_file=parent_file,
            child_orientation_file=child_file,
        )

    @staticmethod
    def _parent_file_from_phase_input(transformable_phase_ids: List[int]) -> str:
        """Fall back to the phase input file of the first transformable phase as PAG orientations."""
        file_dict = RveInfo.file_dict or {}
        for phase_id in transformable_phase_ids:
            if file_dict.get(phase_id):
                return file_dict[phase_id]
        raise KeyError(
            'No parent orientation file could be determined. Set subs_parent_orientation_file '
            f'explicitly, or provide a phase input file for one of {transformable_phase_ids}. '
            f'Available file_dict keys with a file: '
            f'{[k for k, v in file_dict.items() if v]}'
        )


def _validate_orientation_file(file: str, label: str, require_grain_id: bool) -> None:
    """Fail early and legibly instead of deep inside pandas."""
    if not file:
        raise ValueError(f"orientation_mode 'experimental' requires {label}.")

    if not os.path.isfile(file):
        raise FileNotFoundError(f'{label} not found: {file}')

    if not file.lower().endswith('.csv'):
        raise ValueError(
            f'{label} must be a csv with phi1/PHI/phi2 columns, got: {file}. '
            'GAN .pkl inputs carry no Euler angles and cannot be used as an orientation source.'
        )

    columns = pd.read_csv(file, nrows=0).columns
    missing = [c for c in ('phi1', 'PHI', 'phi2') if c not in columns]
    if require_grain_id and 'grain_id' not in columns:
        missing.append('grain_id')
    if missing:
        raise ValueError(f'{label} ({file}) is missing required column(s): {missing}')
