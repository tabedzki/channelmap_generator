import numpy as np
import pytest

from channelmap_generator.constants import PROBE_TYPE_MAP
from channelmap_generator.types import (
    ChannelEntry,
    Electrode,
    ElectrodePosition,
    ParsedIMRO,
    ProbeType,
    ReferenceType,
)


def test_probe_type_from_subtype():
    for probe_type, subtypes in PROBE_TYPE_MAP.items():
        for subtype in subtypes:
            assert ProbeType.from_subtype(subtype) == ProbeType(probe_type)

    with pytest.raises(ValueError, match="Unknown probe subtype: 999"):
        ProbeType.from_subtype(999)


def test_reference_type():
    assert ReferenceType.EXTERNAL == "ext"
    assert ReferenceType.TIP == "tip"
    assert ReferenceType.GROUND == "gnd"


def test_electrode_position():
    electrode = Electrode(shank_id=1, electrode_id=10)
    position = ElectrodePosition(electrode=electrode, x=15.0, y=20.0)

    assert position.electrode == electrode
    assert position.x == 15.0
    assert position.y == 20.0


def test_channel_entry():
    entry = ChannelEntry(
        channel=1,
        shank_id=0,
        bank=2,
        bank_mask=3,
        ref=4,
        ap_gain=5,
        lf_gain=6,
        hp_filter=7,
        electrode_id=8,
    )

    assert entry.channel == 1
    assert entry.shank_id == 0
    assert entry.bank == 2
    assert entry.bank_mask == 3
    assert entry.ref == 4
    assert entry.ap_gain == 5
    assert entry.lf_gain == 6
    assert entry.hp_filter == 7
    assert entry.electrode_id == 8


def test_parsed_imro():
    selected_electrodes = np.array([[0, 1], [1, 2]])
    parsed = ParsedIMRO(
        selected_electrodes=selected_electrodes,
        probe_type=ProbeType.NEUROPIXELS_1_0,
        probe_subtype=1,
        reference_id="ext",
        ap_gain=500,
        lf_gain=250,
        hp_filter=1,
    )

    assert np.array_equal(parsed.selected_electrodes, selected_electrodes)
    assert parsed.probe_type == ProbeType.NEUROPIXELS_1_0
    assert parsed.probe_subtype == 1
    assert parsed.reference_id == "ext"
    assert parsed.ap_gain == 500
    assert parsed.lf_gain == 250
    assert parsed.hp_filter == 1


def test_electrode():
    electrode = Electrode(shank_id=2, electrode_id=15)

    assert electrode.shank_id == 2
    assert electrode.electrode_id == 15
