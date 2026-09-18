"""Tests for the brainglobe atlas wrapper.

We don't want the test suite to download a real atlas — that's tens of MB
and CI-hostile. We patch atlas_module.BrainGlobeAtlas with a tiny fake atlas
that exercises the indexing logic without touching the network.
"""

from __future__ import annotations

import gc
import weakref
from pathlib import Path

import numpy as np
import pytest

from pixelmap.anatomy import atlas as atlas_module
from pixelmap.anatomy import regions as regions_module
from pixelmap.anatomy._registry_snapshot import REGISTRY_SNAPSHOT


class _FakeAtlas:
    """A 3-voxel-cube fake atlas: half is region 1, half is region 2."""

    def __init__(self, name: str, **_kwargs):
        self.name = name
        self.orientation = "asr"
        self.resolution = (25.0, 25.0, 25.0)  # µm per voxel
        # 4×4×4 volume; left ML half = region 1, right half = region 2; 0 = outside
        ann = np.zeros((4, 4, 4), dtype=np.int32)
        ann[:, :, :2] = 1
        ann[:, :, 2:] = 2
        self.annotation = ann
        self.structures = {
            1: {"acronym": "LEFT", "name": "Left hemisphere", "rgb_triplet": [200, 0, 0]},
            2: {"acronym": "RIGHT", "name": "Right hemisphere", "rgb_triplet": [0, 200, 0]},
        }


def _must_not_be_called(*args, **kwargs):
    raise AssertionError("this call must never happen on the session path")


@pytest.fixture(autouse=True)
def _reset_atlas_cache():
    """Make sure no real atlas leaks across tests via the lru_cache."""
    atlas_module.get_atlas.cache_clear()
    atlas_module.canonical_annotation.cache_clear()
    atlas_module.derive_origin_from_ac.cache_clear()
    atlas_module._region_info_from_id.cache_clear()
    yield
    atlas_module.get_atlas.cache_clear()
    atlas_module.canonical_annotation.cache_clear()
    atlas_module.derive_origin_from_ac.cache_clear()
    atlas_module._region_info_from_id.cache_clear()


@pytest.fixture
def fake_brainglobe(monkeypatch):
    """Patch BrainGlobeAtlas in the atlas module with a tiny fake."""
    monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _FakeAtlas)
    yield


class TestLookup:
    def test_voxel_indexing_resolves_left_vs_right(self, fake_brainglobe):
        # ML=10 (voxel 0, in left half) vs ML=60 (voxel 2, in right half)
        coords = np.array([
            [0.0,  10.0, 0.0],   # left
            [0.0,  60.0, 0.0],   # right
        ])
        out = atlas_module.lookup_regions("fake", coords)
        assert out[0].acronym == "LEFT"
        assert out[1].acronym == "RIGHT"
        assert out[0].rgb == (200, 0, 0)

    def test_out_of_bounds_returns_none(self, fake_brainglobe):
        coords = np.array([
            [-100.0, 0.0, 0.0],
            [9999.0, 9999.0, 9999.0],
        ])
        out = atlas_module.lookup_regions("fake", coords)
        assert out == [None, None]

    def test_zero_label_returns_none(self, monkeypatch):
        # Override the fake atlas to have all zeros (outside-brain everywhere).
        class Zeros(_FakeAtlas):
            def __init__(self, name, **kwargs):
                super().__init__(name, **kwargs)
                self.annotation = np.zeros((4, 4, 4), dtype=np.int32)

        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", Zeros)
        atlas_module.get_atlas.cache_clear()
        atlas_module.canonical_annotation.cache_clear()

        coords = np.array([[0.0, 0.0, 0.0]])
        assert atlas_module.lookup_regions("zeros", coords) == [None]


class TestRegionsForPositions:
    def test_end_to_end_lookup_with_pose(self, fake_brainglobe):
        # Place the tip at the very corner of the volume; both electrodes are
        # at probe (0, 0) and (xp=50, 0). With default pose (vertical, +xp=+ML),
        # the second electrode is 50 µm to the right — voxel 2 → "RIGHT".
        electrode_xy = np.array([[0.0, 0.0], [50.0, 0.0]])
        regions = regions_module.regions_for_positions(
            electrode_xy,
            tip_atlas=(0.0, 0.0, 0.0),
            atlas_name="fake",
        )
        assert regions[0].acronym == "LEFT"
        assert regions[1].acronym == "RIGHT"


def _atlas_cls(orientation, annotation, structures=None):
    """Build a fake BrainGlobeAtlas class with a given orientation/annotation."""
    structures = structures or {
        1: {"acronym": "ONE", "name": "One", "rgb_triplet": [1, 2, 3]},
        2: {"acronym": "TWO", "name": "Two", "rgb_triplet": [4, 5, 6]},
    }

    class _A:
        def __init__(self, name, **_kwargs):
            self.name = name
            self.orientation = orientation
            self.resolution = (25.0, 25.0, 25.0)
            self.annotation = annotation
            self.structures = structures

    return _A


class TestOrientation:
    def test_anatomical_axes_asr_is_identity(self):
        atlas = _atlas_cls("asr", np.zeros((4, 5, 6), np.int32))("x")
        axes = atlas_module.anatomical_axes(atlas)
        assert (axes["AP"].array_axis, axes["AP"].flip) == (0, False)
        assert (axes["DV"].array_axis, axes["DV"].flip) == (1, False)
        assert (axes["ML"].array_axis, axes["ML"].flip) == (2, False)

    def test_anatomical_axes_permuted_and_flipped(self):
        # "sla": axis0=DV(s), axis1=ML(l → flipped vs canonical right), axis2=AP(a)
        axes = atlas_module.anatomical_axes(_atlas_cls("sla", np.zeros((4, 5, 6), np.int32))("x"))
        assert (axes["DV"].array_axis, axes["DV"].flip) == (0, False)
        assert (axes["ML"].array_axis, axes["ML"].flip) == (1, True)
        assert (axes["AP"].array_axis, axes["AP"].flip) == (2, False)
        assert axes["DV"].n == 4 and axes["ML"].n == 5 and axes["AP"].n == 6

    def test_canonical_annotation_reorients_to_asr(self, monkeypatch):
        # Native "sla" volume is indexed (DV, ML, AP); canonical must be (AP, DV, ML).
        native = np.arange(2 * 3 * 4, dtype=np.int32).reshape(2, 3, 4)
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _atlas_cls("sla", native))
        atlas_module.get_atlas.cache_clear()
        atlas_module.canonical_annotation.cache_clear()
        arr, _ = atlas_module.canonical_annotation("x")
        expected = np.flip(np.transpose(native, (2, 0, 1)), axis=2)  # AP from axis2, flip ML
        assert arr.shape == (4, 2, 3)
        np.testing.assert_array_equal(arr, expected)

    def test_derive_origin_from_ac_finds_midline_crossing(self, monkeypatch):
        # AC at AP voxel 2, DV voxel 2, ML voxels 3-5 (midline 4) in an asr volume.
        ann = np.zeros((6, 4, 8), dtype=np.int32)
        ann[2, 2, 3:6] = 5
        structs = {5: {"id": 5, "acronym": "ac", "name": "anterior commissure",
                       "rgb_triplet": [1, 2, 3]}}
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _atlas_cls("asr", ann, structs))
        atlas_module.get_atlas.cache_clear()
        atlas_module.canonical_annotation.cache_clear()
        atlas_module.derive_origin_from_ac.cache_clear()
        # (AP, ML, DV) µm = (2*25, 4*25, 2*25)
        assert atlas_module.derive_origin_from_ac("x") == (50.0, 100.0, 50.0)

    def test_derive_origin_from_ac_none_without_ac(self, monkeypatch):
        ann = np.zeros((4, 4, 4), dtype=np.int32)
        ann[1, 1, 1] = 9
        structs = {9: {"id": 9, "acronym": "x", "name": "some nucleus",
                       "rgb_triplet": [0, 0, 0]}}
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _atlas_cls("asr", ann, structs))
        atlas_module.get_atlas.cache_clear()
        atlas_module.canonical_annotation.cache_clear()
        atlas_module.derive_origin_from_ac.cache_clear()
        assert atlas_module.derive_origin_from_ac("y") is None

    def test_lookup_resolves_through_ap_flip(self, monkeypatch):
        # "psr" reverses AP: a marker at the native posterior pole (index 0) must
        # be read at large canonical AP, not at AP=0 (the anterior pole).
        native = np.zeros((4, 2, 2), dtype=np.int32)
        native[0, :, :] = 2     # native index 0 = posterior
        native[1:, :, :] = 1
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _atlas_cls("psr", native))
        atlas_module.get_atlas.cache_clear()
        atlas_module.canonical_annotation.cache_clear()
        anterior = atlas_module.lookup_regions("x", np.array([[0.0, 0.0, 0.0]]))
        posterior = atlas_module.lookup_regions("x", np.array([[75.0, 0.0, 0.0]]))
        assert anterior[0].acronym == "ONE"
        assert posterior[0].acronym == "TWO"


class TestMetadataOnlyQueries:
    """Shape/extent queries must not touch ``annotation``.

    brainglobe v3 fetches the annotation array lazily from S3, so reading it
    just to learn the volume's size would trigger the very download these
    queries let the GUI avoid.
    """

    @staticmethod
    def _metadata_atlas_cls(shape, orientation="asr"):
        class _A:
            def __init__(self, name, **_kwargs):
                self.name = name
                self.orientation = orientation
                self.resolution = (25.0, 25.0, 25.0)
                self.shape = shape
                self.structures = {}

            @property
            def annotation(self):
                raise AssertionError("annotation must not be loaded here")

        return _A

    def test_volume_center_reads_shape_not_the_array(self, monkeypatch):
        monkeypatch.setattr(
            atlas_module, "BrainGlobeAtlas", self._metadata_atlas_cls((4, 8, 16))
        )
        # asr: shape is (AP, DV, ML); result is (AP, ML, DV) µm half-extents.
        assert atlas_module.volume_center_um("x") == (50.0, 200.0, 100.0)

    def test_anatomical_axes_reads_shape_not_the_array(self, monkeypatch):
        atlas = self._metadata_atlas_cls((4, 8, 16), "sla")("x")
        axes = atlas_module.anatomical_axes(atlas)
        assert (axes["DV"].n, axes["ML"].n, axes["AP"].n) == (4, 8, 16)

    def test_falls_back_to_the_array_when_shape_is_absent(self, monkeypatch):
        # Older brainglobe objects and our other test doubles carry no `shape`.
        atlas = _atlas_cls("asr", np.zeros((3, 5, 7), np.int32))("x")
        axes = atlas_module.anatomical_axes(atlas)
        assert (axes["AP"].n, axes["DV"].n, axes["ML"].n) == (3, 5, 7)


class TestIsDownloaded:
    """``is_downloaded`` answers "is reading the annotation free?".

    Not "is the atlas present": the atlas directory appears as soon as the tiny
    manifest lands, while the annotation is still remote, so the two questions
    come apart.
    """

    @pytest.fixture
    def listed(self, monkeypatch):
        monkeypatch.setattr(atlas_module, "get_downloaded_atlases", lambda: ["x"])

    def test_absent_atlas_is_never_downloaded(self, monkeypatch):
        monkeypatch.setattr(atlas_module, "get_downloaded_atlases", list)
        assert atlas_module.is_downloaded("x") is False

    def test_manifest_without_chunks_is_not_downloaded(
        self, monkeypatch, listed, tmp_path
    ):
        monkeypatch.setattr(
            atlas_module, "get_atlas", lambda name: _v3_atlas(tmp_path, chunks=False)
        )
        assert atlas_module.is_downloaded("x") is False

    def test_cached_chunks_are_downloaded(self, monkeypatch, listed, tmp_path):
        monkeypatch.setattr(
            atlas_module, "get_atlas", lambda name: _v3_atlas(tmp_path, chunks=True)
        )
        assert atlas_module.is_downloaded("x") is True

    def test_another_resolution_of_the_family_does_not_count(
        self, monkeypatch, listed, tmp_path
    ):
        """allen_mouse_25um and allen_mouse_10um share one annotation set:
        25 µm is level s1, 10 µm is s0. With only s1 local, the 10 µm atlas
        is *not* downloaded -- its level is a 4.8 GB fetch away."""
        monkeypatch.setattr(
            atlas_module, "get_atlas",
            lambda name: _v3_atlas(tmp_path, chunks=True, resolution=(10.0,) * 3),
        )
        assert atlas_module.is_downloaded("x") is False

    def test_unreadable_metadata_reports_not_downloaded(self, monkeypatch, listed):
        """A half-written manifest must send callers down the "this will cost
        you" path rather than crashing the GUI."""

        def _boom(name):
            raise RuntimeError("corrupt manifest")

        monkeypatch.setattr(atlas_module, "get_atlas", _boom)
        assert atlas_module.is_downloaded("x") is False


def _v3_atlas(root, *, chunks: bool, resolution=(25.0, 25.0, 25.0)):
    """A stand-in for a v3 atlas over a two-level pyramid (s0 @ 10 µm, s1 @
    25 µm) whose 25 µm chunks may or may not be local."""
    import shutil

    atlas = _DiskAtlas(root, resolution=resolution)
    _write_pyramid(atlas.pyramid_root, {
        "s0": ((10.0,) * 3, np.zeros((1, 1, 1), np.uint32)),
        "s1": ((25.0,) * 3, np.ones((2, 2, 2), np.uint32)),
    })
    # s0's chunks are never local (all-zero chunks aren't even written); drop
    # s1's unless the test wants them.
    shutil.rmtree(atlas.pyramid_root / "s0" / "c", ignore_errors=True)
    if not chunks:
        shutil.rmtree(atlas.pyramid_root / "s1" / "c")
    return atlas


class TestRegistryCachePath:
    def test_points_at_the_v3_layout(self, tmp_path):
        assert atlas_module._registry_cache_path(tmp_path) == (
            tmp_path / "brainglobe-atlasapi" / "atlases" / "last_versions.conf"
        )


class TestListAtlasesNeverBlocks:
    """The regression that took the server down.

    ``list_atlases`` runs while a GUI session is being built — on the deployed
    server, on the Bokeh event loop. brainglobe's registry fetch is a
    ``requests.get`` with no timeout, so doing it inline turned an outage at
    the atlas host into a hung server for every connected user.
    """

    @pytest.fixture(autouse=True)
    def _clear(self):
        atlas_module.list_atlases.cache_clear()
        yield
        atlas_module.list_atlases.cache_clear()

    def test_prefers_the_on_disk_registry_cache(self, monkeypatch):
        monkeypatch.setattr(
            atlas_module, "_registry_from_disk", lambda: ["b_atlas", "a_atlas"]
        )
        monkeypatch.setattr(
            atlas_module, "_refresh_registry_in_background", _must_not_be_called
        )
        assert atlas_module.list_atlases() == ["b_atlas", "a_atlas"]

    def test_cold_cache_returns_the_snapshot_without_network(self, monkeypatch):
        monkeypatch.setattr(atlas_module, "_registry_from_disk", lambda: None)
        monkeypatch.setattr(
            atlas_module, "get_all_atlases_lastversions", _must_not_be_called
        )
        refreshed = []
        monkeypatch.setattr(
            atlas_module,
            "_refresh_registry_in_background",
            lambda: refreshed.append(True),
        )

        out = atlas_module.list_atlases()

        assert out == sorted(REGISTRY_SNAPSHOT)
        assert "allen_mouse_25um" in out
        assert refreshed == [True], "a cold cache should schedule a background refresh"

    def test_result_is_memoised(self, monkeypatch):
        calls = []

        def _once():
            calls.append(True)
            return ["only_atlas"]

        monkeypatch.setattr(atlas_module, "_registry_from_disk", _once)
        for _ in range(5):
            assert atlas_module.list_atlases() == ["only_atlas"]
        assert len(calls) == 1

    def test_unreadable_cache_falls_back_instead_of_raising(self, monkeypatch, tmp_path):
        """A truncated conf file must not take the dropdown down with it."""
        cache = atlas_module._registry_cache_path(tmp_path)
        cache.parent.mkdir(parents=True)
        cache.write_text("this is not a conf file")

        import brainglobe_atlasapi.config as bg_config

        monkeypatch.setattr(bg_config, "get_brainglobe_dir", lambda: tmp_path)
        assert atlas_module._registry_from_disk() is None


class TestEnsureDownloaded:
    def test_materialises_the_annotation(self, monkeypatch):
        reads = []

        class _A:
            def __init__(self, name, **_kwargs):
                self.name = name
                self.orientation = "asr"
                self.resolution = (25.0, 25.0, 25.0)
                self.shape = (2, 2, 2)
                self.structures = {}

            @property
            def annotation(self):
                reads.append(self.name)
                return np.zeros((2, 2, 2), np.int32)

        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _A)
        atlas_module.ensure_downloaded("some_atlas")
        assert reads == ["some_atlas"]
        # ...and the decoded volume is the one the overlay will read: no
        # second decode when the compute follows.
        atlas_module.canonical_annotation("some_atlas")
        assert reads == ["some_atlas"]


def _write_pyramid(root, levels, chunks=(2, 2, 2)):
    """Build a tiny OME-Zarr pyramid like brainglobe v3 ships: one group with
    ``ome.multiscales`` and a uint32 array per level. ``levels`` maps a dataset
    path to ``(scale_um, array)``."""
    import zarr

    group = zarr.open_group(root, mode="w")
    datasets = []
    for path, (scale_um, arr) in levels.items():
        z = group.create_array(path, shape=arr.shape, chunks=chunks, dtype="uint32")
        z[:] = arr
        datasets.append({
            "path": path,
            "coordinateTransformations": [
                {"type": "scale", "scale": [s / 1000.0 for s in scale_um]},
                {"type": "translation", "translation": [0.0, 0.0, 0.0]},
            ],
        })
    group.attrs["ome"] = {"multiscales": [{"datasets": datasets}]}
    return root


class _DiskAtlas:
    """A brainglobe-v3-shaped atlas over a pyramid written by ``_write_pyramid``.

    ``annotation`` raises so the tests prove the loader never takes dask's
    path; ``fs`` records what a download would fetch."""

    def __init__(self, tmp_path, resolution=(25.0, 25.0, 25.0)):
        self.root_dir = tmp_path
        self.metadata = {"annotation_set": {"location": "/annotation-sets/x"}}
        self.resolution = resolution
        self.orientation = "asr"
        self.structures = {}
        self.fetched = []
        self.fs = self

    def get(self, remote, local, recursive=False):  # fsspec-shaped
        self.fetched.append((remote, local))

    @property
    def annotation(self):
        raise AssertionError("must decode through zarr, not brainglobe's dask path")

    @property
    def pyramid_root(self):
        return self.root_dir / "annotation-sets" / "x" / atlas_module.V3_ANNOTATION_NAME


class TestLoadAnnotation:
    """``load_annotation`` decodes straight from the zarr chunks.

    brainglobe's ``annotation`` property computes a dask graph that holds every
    decompressed chunk alongside the assembled array (2x the volume at peak:
    2.3 GB for the 1.07 GB rat atlas). Reading the level through zarr
    indexing keeps the peak at the array itself.
    """

    def test_reads_the_level_matching_the_atlas_resolution(self, tmp_path):
        atlas = _DiskAtlas(tmp_path, resolution=(50.0, 50.0, 50.0))
        fine = np.arange(64, dtype=np.uint32).reshape(4, 4, 4)
        coarse = np.full((2, 2, 2), 7, dtype=np.uint32)
        _write_pyramid(atlas.pyramid_root, {"s0": ((25.0,) * 3, fine), "s1": ((50.0,) * 3, coarse)})
        out = atlas_module.load_annotation(atlas)
        assert out.dtype == np.uint32 and out.shape == (2, 2, 2)
        assert (out == 7).all()
        assert atlas.fetched == [], "chunks were local: no download"

    def test_returns_a_plain_array_not_a_lazy_view(self, tmp_path):
        atlas = _DiskAtlas(tmp_path)
        _write_pyramid(atlas.pyramid_root, {"s0": ((25.0,) * 3, np.ones((4, 4, 4), np.uint32))})
        out = atlas_module.load_annotation(atlas)
        assert isinstance(out, np.ndarray) and out.flags.owndata

    def test_downloads_the_level_when_chunks_are_missing(self, tmp_path):
        import shutil

        atlas = _DiskAtlas(tmp_path)
        arr = np.arange(64, dtype=np.uint32).reshape(4, 4, 4)
        _write_pyramid(atlas.pyramid_root, {"s0": ((25.0,) * 3, arr)})
        level = atlas.pyramid_root / "s0"
        # Stash the chunks so a "download" can restore them.
        stash = tmp_path / "remote"
        shutil.move(level / "c", stash)

        def _fetch(remote, local, recursive=False):
            atlas.fetched.append((remote, local))
            shutil.copytree(stash, Path(local) / "c")
            shutil.copy(level / "zarr.json", Path(local) / "zarr.json")

        atlas.get = _fetch
        out = atlas_module.load_annotation(atlas)
        name = atlas_module.V3_ANNOTATION_NAME
        staging = level.with_name("s0.partial")
        assert atlas.fetched == [
            (f"s3://brainglobe/atlas/annotation-sets/x/{name}/s0/", str(staging))
        ]
        np.testing.assert_array_equal(out, arr)
        assert (level / "c").is_dir() and not staging.exists()

    def test_a_killed_download_leaves_no_chunk_dir_behind(self, tmp_path):
        """The `c` directory is the only "downloaded" signal; a partial one
        would be trusted forever and read back as a truncated atlas."""
        import shutil

        atlas = _DiskAtlas(tmp_path)
        arr = np.arange(64, dtype=np.uint32).reshape(4, 4, 4)
        _write_pyramid(atlas.pyramid_root, {"s0": ((25.0,) * 3, arr)})
        level = atlas.pyramid_root / "s0"
        stash = tmp_path / "remote"
        shutil.move(level / "c", stash)
        attempts = []

        def _fetch(remote, local, recursive=False):
            attempts.append(local)
            if len(attempts) == 1:
                # Half the chunks land, then the process dies.
                (Path(local) / "c" / "0" / "0").mkdir(parents=True)
                (Path(local) / "c" / "0" / "0" / "0").write_bytes(b"partial")
                raise KeyboardInterrupt("OOM-killed mid-transfer")
            shutil.copytree(stash, Path(local) / "c")

        atlas.get = _fetch
        with pytest.raises(KeyboardInterrupt):
            atlas_module.load_annotation(atlas)
        assert not (level / "c").exists(), "partial download must not look complete"
        # Next attempt discards the stale staging dir and fetches again.
        np.testing.assert_array_equal(atlas_module.load_annotation(atlas), arr)
        assert len(attempts) == 2

    def test_unknown_resolution_is_an_error_not_a_silent_level(self, tmp_path):
        atlas = _DiskAtlas(tmp_path, resolution=(10.0, 10.0, 10.0))
        _write_pyramid(atlas.pyramid_root, {"s0": ((25.0,) * 3, np.ones((2, 2, 2), np.uint32))})
        with pytest.raises(ValueError, match="10.0"):
            atlas_module.load_annotation(atlas)

    def test_doubles_without_a_zarr_are_read_as_is(self):
        out = atlas_module.load_annotation(_FakeAtlas("x"))
        assert out.shape == (4, 4, 4)


class TestAtlasesAreReleased:
    """Atlas objects must not outlive ``get_atlas``'s cache.

    Regression test for a production OOM: ``_region_info_from_id`` used to take
    the atlas *object* as its cache key. ``lru_cache`` holds keys alive, so
    every atlas ever looked up stayed resident with its annotation volume —
    hundreds of MB each — and eviction from ``get_atlas`` leaked a fresh copy
    per lookup instead of reusing one.
    """

    @staticmethod
    def _tracked_atlas_cls(built):
        """A fake atlas class that weakly records every instance it builds."""

        class _A:
            def __init__(self, name, **_kwargs):
                self.name = name
                self.orientation = "asr"
                self.resolution = (25.0, 25.0, 25.0)
                self.annotation = np.ones((2, 2, 2), dtype=np.int32)
                self.structures = {
                    1: {"id": 1, "acronym": "R", "name": "Region",
                        "rgb_triplet": [1, 2, 3]}
                }
                built.append(weakref.ref(self))

        return _A

    @staticmethod
    def _live(built):
        gc.collect()
        return sum(ref() is not None for ref in built)

    def test_atlases_beyond_the_cache_bound_are_freed(self, monkeypatch):
        built: list = []
        monkeypatch.setattr(
            atlas_module, "BrainGlobeAtlas", self._tracked_atlas_cls(built)
        )
        coords = np.array([[0.0, 0.0, 0.0]])

        bound = atlas_module.get_atlas.cache_info().maxsize
        for i in range(bound * 3):
            atlas_module.lookup_regions(f"atlas_{i}", coords)
            atlas_module.canonical_annotation.cache_clear()  # this cache holds views

        assert len(built) == bound * 3, "expected one atlas per distinct name"
        assert self._live(built) <= bound

    def test_rotating_past_the_bound_does_not_accumulate(self, monkeypatch):
        """The case that took the server down: one more atlas in rotation than
        ``get_atlas`` can hold, so every lookup rebuilds an evicted atlas."""
        built: list = []
        monkeypatch.setattr(
            atlas_module, "BrainGlobeAtlas", self._tracked_atlas_cls(built)
        )
        coords = np.array([[0.0, 0.0, 0.0]])

        bound = atlas_module.get_atlas.cache_info().maxsize
        names = [f"atlas_{i}" for i in range(bound + 1)]
        for i in range(bound * 8):
            atlas_module.lookup_regions(names[i % len(names)], coords)
            atlas_module.canonical_annotation.cache_clear()

        # Rebuilding on eviction is expected; *retaining* the rebuilds is not.
        assert len(built) > bound, "expected evictions to force rebuilds"
        assert self._live(built) <= bound
