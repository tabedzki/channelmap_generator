"""Frozen dev IMRO oracle: 25 random full-readout selections per supported group."""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pixelmap.backend import find_electrode_coordinates, find_forbidden_electrodes
from pixelmap.constants import PROBE_FEATURES, PROBE_TYPE_MAP, REF_ELECTRODES, WIRING_FILE_MAP
from pixelmap.utils.imro import (
    generate_imro_channelmap,
    parse_imro_file,
    parse_imro_list,
    read_imro_file,
    save_to_imro_file,
)

REFERENCE_DIR = Path(__file__).parent / "fixtures" / "random_imro"
MANIFEST = json.loads((REFERENCE_DIR / "manifest.json").read_text())


@pytest.fixture(scope="module")
def reference_wiring():
    """Read raw CSV cells independently of backend wiring helpers."""
    result = {}
    for group, (_, filename) in WIRING_FILE_MAP.items():
        path = Path(__file__).parents[1] / "pixelmap" / "wiring_maps" / filename
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            rows = list(reader)
        coordinates = {}
        channels = []
        for row_index, row in enumerate(rows):
            cells = set()
            for col_index, column in enumerate(columns[1:], start=1):
                if row[column]:
                    shank = int(column.split("-")[0].removeprefix("shank"))
                    electrode = (shank, int(float(row[column])))
                    coordinates[electrode] = (row_index, col_index)
                    cells.add(electrode)
            channels.append(cells)
        result[group] = (path, pd.read_csv(path), coordinates, channels)
    return result


def test_reference_corpus_coverage():
    cases = MANIFEST["cases"]
    assert len(cases) == 100
    assert len({case["id"] for case in cases}) == 100
    assert Counter(case["probe_type"] for case in cases) == {group: 25 for group in WIRING_FILE_MAP}
    assert {case["probe_subtype"] for case in cases} == {
        subtype for group in WIRING_FILE_MAP for subtype in PROBE_TYPE_MAP[group]
    }
    for fmt, references in REF_ELECTRODES.items():
        assert {
            case["reference_id"]
            for case in cases
            if PROBE_FEATURES[case["probe_subtype"]]["IMRO_format"] == fmt
        } == set(references)
    assert len({tuple(map(tuple, case["electrodes"])) for case in cases}) == 100
    assert {p.stem for p in REFERENCE_DIR.glob("*.imro")} == {c["id"] for c in cases}


@pytest.mark.parametrize("case", MANIFEST["cases"], ids=lambda case: case["id"])
def test_random_imro_matches_dev_reference(case, reference_wiring, tmp_path):
    path, wiring, coordinates, channels = reference_wiring[case["probe_type"]]
    features = PROBE_FEATURES[case["probe_subtype"]]
    electrodes = case["electrodes"]
    selected = set(map(tuple, electrodes))
    # Validate the frozen inputs directly against the CSV, without using the
    # same backend conflict/coordinate helpers that the performance PR replaces.
    assert len(selected) == len(electrodes) == features["n_readouts_total"]
    assert selected <= coordinates.keys()
    assert all(len(selected & cells) == 1 for cells in channels)
    assert all(
        count <= features["n_readouts_per_shank"]
        for count in Counter(shank for shank, _ in selected).values()
    )
    assert all(
        0 <= shank < features["num_shanks"] and 0 <= electrode < features["n_sites_per_shank"]
        for shank, electrode in selected
    )
    np.testing.assert_array_equal(
        find_electrode_coordinates(np.array(electrodes), wiring),
        [coordinates[tuple(electrode)] for electrode in electrodes],
    )
    forbidden = find_forbidden_electrodes(np.array(electrodes), wiring)
    assert set(map(tuple, forbidden)) == coordinates.keys() - selected

    expected_bytes = (REFERENCE_DIR / f"{case['id']}.imro").read_bytes()
    expected = parse_imro_file(expected_bytes.decode().strip())
    params = {
        key: case[key]
        for key in (
            "probe_type",
            "probe_subtype",
            "reference_id",
            "ap_gain",
            "lf_gain",
            "hp_filter",
        )
    }
    actual = generate_imro_channelmap(
        **params, custom_electrodes=np.array(electrodes), wiring_file=str(path)
    )
    assert actual == expected
    if case["probe_type"] == "2.0-1shank":
        # The bank mask must select exactly the bank containing the electrode.
        assert all(entry[1] == 1 << (entry[3] // 384) for entry in actual[1:])
    assert actual[0] == (case["probe_subtype"], features["n_readouts_total"])
    assert [entry[0] for entry in actual[1:]] == list(range(features["n_readouts_total"]))
    destination = tmp_path / "roundtrip.imro"
    save_to_imro_file(actual, str(destination))
    assert destination.read_bytes() == expected_bytes
    assert read_imro_file(destination) == expected
    parsed, group, subtype, reference, ap, lf, hp = parse_imro_list(actual)
    assert set(map(tuple, parsed)) == selected
    assert (group, subtype, reference) == (
        case["probe_type"],
        case["probe_subtype"],
        case["reference_id"],
    )
    legacy = [(features["SpikeGLX_probe_type"], actual[0][1]), *actual[1:]]
    legacy_parsed = parse_imro_list(legacy)
    np.testing.assert_array_equal(legacy_parsed[0], parsed)
    assert legacy_parsed[1] == group
    assert legacy_parsed[3:] == (reference, ap, lf, hp)
    if group == "1.0":
        assert (ap, lf, hp) == (case["ap_gain"], case["lf_gain"], case["hp_filter"])
    else:
        assert (ap, lf, hp) == (None, None, None)
