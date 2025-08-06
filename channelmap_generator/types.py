"""Type definitions for the channelmap generator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class ProbeType(StrEnum):
    """Supported Neuropixels probe types."""

    NEUROPIXELS_1_0 = "1.0"
    NEUROPIXELS_2_0_1SHANK = "2.0-1shank"
    NEUROPIXELS_2_0_4SHANKS = "2.0-4shanks"
    NEUROPIXELS_NXT = "NXT"

    @classmethod
    def from_subtype(cls, subtype: int) -> "ProbeType":
        """Get probe type from SpikeGLX subtype number."""
        from .constants import PROBE_TYPE_MAP

        for probe_type, subtypes in PROBE_TYPE_MAP.items():
            if subtype in subtypes:
                return cls(probe_type)
        raise ValueError(f"Unknown probe subtype: {subtype}")


class ReferenceType(StrEnum):
    """Reference electrode types."""

    EXTERNAL = "ext"
    TIP = "tip"
    GROUND = "gnd"


@dataclass(frozen=True)
class ElectrodePosition:
    """Represents an electrode position on a probe."""

    electrode: Electrode
    x: float
    y: float


@dataclass
class ChannelEntry:
    """Represents a single channel entry in IMRO format."""

    channel: int
    shank_id: int | None = None
    bank: int | None = None
    bank_mask: int | None = None
    ref: int | None = None
    ap_gain: int | None = None
    lf_gain: int | None = None
    hp_filter: int | None = None
    electrode_id: int | None = None


@dataclass
class ParsedIMRO:
    """Parsed IMRO data structure."""

    selected_electrodes: np.ndarray  # Shape: (n_electrodes, 2) - [[shank_id, electrode_id], ...]
    probe_type: ProbeType
    probe_subtype: int
    reference_id: str
    ap_gain: int | None = None
    lf_gain: int | None = None
    hp_filter: int | None = None


@dataclass(frozen=True)
class Electrode:
    """Represents an electrode with shank and electrode ID."""

    shank_id: int
    electrode_id: int


# Set-based types for efficient operations
ElectrodeSet = set[Electrode]
ForbiddenElectrodes = set[Electrode]
WiringConstraints = dict[Electrode, set[Electrode]]
