from dataclasses import dataclass


@dataclass(frozen=True)
class Electrode:
    """Represents an electrode with shank and electrode ID."""

    shank_id: int
    electrode_id: int


@dataclass(frozen=True)
class ElectrodePosition:
    """Represents an electrode position on a probe."""

    electrode: Electrode
    x: float
    y: float
