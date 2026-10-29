"""Regression tests for the package's public imports and documented API."""

import re
from pathlib import Path

import pytest

import pixelmap
from pixelmap.constants import PROBE_TYPE_MAP
from pixelmap.types import Electrode

PUBLIC_NAMES = {
    "get_electrodes",
    "find_forbidden_electrodes",
    "make_wiring_maps",
    "format_wiring_df",
    "format_imro_string",
    "find_electrode_coordinates",
    "get_preset_candidates",
    "find_selected_electrodes",
    "generate_kilosort_channelmap_dict",
    "plot_probe_layout",
    "IMRO_FORMAT_TO_PROBE_TYPE",
    "LEGACY_INT_TO_IMRO_FORMAT",
    "LEGACY_PROBE_TYPE_MAP",
    "PROBE_FEATURES",
    "PROBE_TYPE_MAP",
    "REF_BANKS",
    "REF_ELECTRODES",
    "WIRING_FILE_MAP",
    "SUPPORTED_1shank_PRESETS",
    "SUPPORTED_4shanks_PRESETS",
    "SUPPORTED_QuadBase_PRESETS",
    "Electrode",
    "generate_imro_channelmap",
    "parse_imro_file",
    "parse_imro_list",
    "read_imro_file",
    "save_to_imro_file",
}


def test_documented_api_names_resolve():
    root = Path(__file__).resolve().parents[1]
    paths = [
        path for path in (root / "docs").rglob("*")
        if path.suffix in {".md", ".rst", ".py"} and "_build" not in path.parts
    ]
    paths.extend(root.glob("README*"))
    references = set()
    for path in paths:
        # Exclude URLs and paths such as github.com/m-beau/pixelmap.git.
        names = re.findall(r"(?<![\w/.-])(?:cmg|pixelmap)\.(\w+)", path.read_text())
        for name in names:
            references.add(name)
            assert hasattr(pixelmap, name), f"{path.relative_to(root)} references missing {name}"
    assert "generate_imro_channelmap" in references


def test_all_lists_exact_public_api():
    assert set(pixelmap.__all__) == PUBLIC_NAMES
    assert len(pixelmap.__all__) == len(PUBLIC_NAMES)
    for name in pixelmap.__all__:
        assert getattr(pixelmap, name) is not None


def test_star_import_matches_all():
    namespace = {}
    exec("from pixelmap import *", namespace)  # noqa: S102 - exercise Python star-import semantics
    namespace.pop("__builtins__")
    assert set(namespace) == set(pixelmap.__all__)
    for name, value in namespace.items():
        assert value is getattr(pixelmap, name)


@pytest.mark.parametrize("name", ["np", "pd", "plt", "pickle", "Path", "cm", "mcolors", "patches"])
def test_incidental_imports_are_not_public(name):
    assert name not in pixelmap.__all__
    assert not hasattr(pixelmap, name)


def test_public_constant_and_type_identity():
    assert pixelmap.PROBE_TYPE_MAP is PROBE_TYPE_MAP
    assert pixelmap.Electrode is Electrode
