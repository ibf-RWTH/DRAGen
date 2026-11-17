from enum import IntEnum


class Phase(IntEnum):
    """Phase IDs as written into solver input files (same numbering as the legacy PHASENUM)."""

    FERRITE = 1
    MARTENSITE = 2
    PEARLITE = 3
    BAINITE = 4
    AUSTENITE = 5
    INCLUSIONS = 6
    BANDS = 7

    @classmethod
    def parse(cls, value) -> 'Phase':
        """Accept a Phase, its ID or its name in any case ('ferrite', 'Ferrite', 1)."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls[value.upper()]
            except KeyError:
                raise ValueError(f"unknown phase '{value}', expected one of {[p.name.lower() for p in cls]}")
        return cls(int(value))
