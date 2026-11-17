"""Martensite substructure: parent grains -> packets -> blocks -> orientations, on the voxels of an
RVEResult (dragen.api).

    sub = generate_substructure(rve, SubstructureConfig(block_thickness=1.0))
    rve_blocks = apply_substructure(rve, sub)      # every block a region; exports as usual
"""
from dragen.substructure.config import SubstructureConfig
from dragen.substructure.core import Substructure, apply_substructure, generate_substructure

__all__ = ['SubstructureConfig', 'Substructure', 'generate_substructure', 'apply_substructure']
