"""Typed configuration of one RVE generation run.

Replaces the global, mutable RveInfo class for the reworked pipeline: every stage receives the
part of the config it needs as an immutable value, and a config can be saved to and loaded from
YAML next to its results.
"""
from dragen.config.phases import Phase
from dragen.config.rve import (BandConfig, BoxConfig, InclusionConfig, PhaseConfig, RsaConfig, RveConfig,
                               TessellationConfig)

__all__ = ['Phase', 'BoxConfig', 'PhaseConfig', 'RsaConfig', 'TessellationConfig', 'BandConfig',
           'InclusionConfig', 'RveConfig']
