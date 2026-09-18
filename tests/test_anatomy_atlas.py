"""Tests for the brainglobe atlas wrapper.

We don't want the test suite to download a real atlas — that's tens of MB
and CI-hostile. We patch atlas_module.BrainGlobeAtlas with a tiny fake atlas
that exercises the indexing logic without touching the network.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from pixelmap.anatomy import _memory
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

    def test_unreadable_metadata_reports_not_downloaded(self, monkeypatch, listed):
        """A half-written manifest must send callers down the "this will cost
        you" path rather than crashing the GUI."""

        def _boom(name):
            raise RuntimeError("corrupt manifest")

        monkeypatch.setattr(atlas_module, "get_atlas", _boom)
        assert atlas_module.is_downloaded("x") is False


def _v3_atlas(root, *, chunks: bool):
    """A stand-in for a v3 atlas whose OME-Zarr chunks may or may not be local."""
    annotation_dir = root / "annotation-sets" / "some-annotation" / "1_0"
    # brainglobe names pyramid levels s0, s1, ...; the level a given atlas
    # pulls depends on its resolution, so the check accepts any of them.
    scale = annotation_dir / atlas_module.V3_ANNOTATION_NAME / "s1"
    (scale / "c" if chunks else scale).mkdir(parents=True)

    class _A:
        root_dir = root
        metadata = {
            # brainglobe stores locations with a leading "/" that it strips.
            "annotation_set": {"location": "/annotation-sets/some-annotation/1_0"}
        }

    return _A()


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


def _lazy_atlas_cls(shape=(2, 2, 2), reads=None):
    """A fake that mimics brainglobe v3: ``annotation`` is a property that
    materialises the array on first read and caches it on ``_annotation``."""
    reads = [] if reads is None else reads

    class _A:
        def __init__(self, name, **_kwargs):
            self.name = name
            self.orientation = "asr"
            self.resolution = (25.0, 25.0, 25.0)
            self.shape = shape
            self.structures = {}
            self._annotation = None

        @property
        def annotation(self):
            if self._annotation is None:
                reads.append(self.name)
                self._annotation = np.ones(shape, np.uint32)
            return self._annotation

    return _A


class TestEnsureDownloaded:
    def test_materialises_the_annotation(self, monkeypatch):
        reads = []
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _lazy_atlas_cls(reads=reads))
        atlas_module.ensure_downloaded("some_atlas")
        assert reads == ["some_atlas"]

    def test_loads_the_copy_the_overlay_will_use(self, monkeypatch):
        # The worker-thread download and the later compute must share one
        # array, not decode the volume twice.
        reads = []
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _lazy_atlas_cls(reads=reads))
        atlas_module.ensure_downloaded("some_atlas")
        atlas_module.canonical_annotation("some_atlas")
        atlas_module.lookup_regions("some_atlas", np.array([[25.0, 25.0, 25.0]]))
        assert reads == ["some_atlas"]


class TestVolumeCache:
    """:data:`_VOLUMES` is the only long-lived owner of annotation arrays.

    Regression tests for the second production OOM: ``get_atlas`` and
    ``canonical_annotation`` were separate ``lru_cache``s with separate
    eviction orders, and brainglobe caches the array on the atlas object too.
    Evict one, and the other still pinned the volume; reload, and a second
    copy appeared.  Meanwhile ``maxsize=4`` bounded a *count*, not bytes.
    """

    @pytest.fixture(autouse=True)
    def _no_trim(self, monkeypatch):
        # malloc_trim is real and harmless, but keep the tests about ownership.
        monkeypatch.setattr(_memory, "release_freed_memory", lambda: None)

    @staticmethod
    def _budget(monkeypatch, nbytes):
        monkeypatch.setattr(atlas_module._VOLUMES, "_budget", nbytes)

    def test_annotation_is_detached_from_the_atlas_object(self, monkeypatch):
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _lazy_atlas_cls())
        arr, _ = atlas_module.canonical_annotation("x")
        atlas = atlas_module.get_atlas("x")
        assert atlas._annotation is None, "get_atlas must hold metadata only"
        assert arr.base is not None and arr.base.shape == (2, 2, 2)

    def test_evicting_the_atlas_object_does_not_reload_the_volume(self, monkeypatch):
        reads = []
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _lazy_atlas_cls(reads=reads))
        before, _ = atlas_module.canonical_annotation("x")
        atlas_module.get_atlas.cache_clear()          # the old duplicate-copy trigger
        atlas_module.ensure_downloaded("x")
        after, _ = atlas_module.canonical_annotation("x")
        assert after is before
        assert reads == ["x"]

    def test_evicts_by_bytes_not_by_count(self, monkeypatch):
        cls = _lazy_atlas_cls(shape=(4, 4, 4))       # 256 B per volume
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", cls)
        self._budget(monkeypatch, 3 * 256)
        for name in "abcde":                          # 5 loads, budget fits 3
            atlas_module.canonical_annotation(name)
        assert list(atlas_module._VOLUMES._entries) == ["c", "d", "e"]
        assert atlas_module._VOLUMES.resident_bytes() == 3 * 256

    def test_recent_use_protects_from_eviction(self, monkeypatch):
        cls = _lazy_atlas_cls(shape=(4, 4, 4))
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", cls)
        self._budget(monkeypatch, 2 * 256)
        atlas_module.canonical_annotation("a")
        atlas_module.canonical_annotation("b")
        atlas_module.canonical_annotation("a")        # touch a: b is now oldest
        atlas_module.canonical_annotation("c")
        assert list(atlas_module._VOLUMES._entries) == ["a", "c"]

    def test_a_volume_larger_than_the_budget_still_loads(self, monkeypatch):
        cls = _lazy_atlas_cls(shape=(4, 4, 4))
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", cls)
        self._budget(monkeypatch, 10)
        atlas_module.canonical_annotation("a")
        atlas_module.canonical_annotation("b")
        # Refusing an oversized atlas is the GUI's call; the cache just keeps
        # only the one that's in use.
        assert list(atlas_module._VOLUMES._entries) == ["b"]

    def test_eviction_releases_freed_memory(self, monkeypatch):
        calls = []
        monkeypatch.setattr(_memory, "release_freed_memory", lambda: calls.append(1))
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _lazy_atlas_cls())
        atlas_module.canonical_annotation("a")        # trim after every load
        atlas_module.canonical_annotation("a")        # cache hit: no trim
        n = len(calls)
        atlas_module._VOLUMES.clear()
        assert n == 1 and len(calls) == 2

    def test_failed_load_does_not_wedge_later_callers(self, monkeypatch):
        calls = []

        def _flaky(name):
            calls.append(name)
            if len(calls) == 1:
                raise RuntimeError("s3 hiccup")
            return np.ones((2, 2, 2), np.uint32), np.array([25.0, 25.0, 25.0])

        monkeypatch.setattr(atlas_module, "_load_canonical", _flaky)
        with pytest.raises(RuntimeError):
            atlas_module.canonical_annotation("x")
        assert "x" not in atlas_module._VOLUMES._loading
        arr, _ = atlas_module.canonical_annotation("x")
        assert arr.shape == (2, 2, 2)

    def test_concurrent_requests_decode_once(self, monkeypatch):
        import threading

        started = threading.Event()
        release = threading.Event()
        loads = []

        def _slow(name):
            loads.append(name)
            started.set()
            release.wait(5)
            return np.ones((2, 2, 2), np.uint32), np.array([25.0, 25.0, 25.0])

        monkeypatch.setattr(atlas_module, "_load_canonical", _slow)
        results = []
        t1 = threading.Thread(target=lambda: results.append(atlas_module.canonical_annotation("x")[0]))
        t1.start()
        started.wait(5)
        t2 = threading.Thread(target=lambda: results.append(atlas_module.canonical_annotation("x")[0]))
        t2.start()
        release.set()
        t1.join(5); t2.join(5)
        assert loads == ["x"]
        assert len(results) == 2 and results[0] is results[1]


class TestBudgetDefaults:
    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("PIXELMAP_ATLAS_CACHE_MB", "512")
        assert atlas_module._default_budget_bytes() == 512 * 2**20

    def test_half_the_cgroup_limit(self, monkeypatch, tmp_path):
        monkeypatch.delenv("PIXELMAP_ATLAS_CACHE_MB", raising=False)
        f = tmp_path / "memory.max"
        f.write_text("3221225472\n")                 # a 3 GB `deploy` limit
        monkeypatch.setattr(_memory, "_CGROUP_LIMIT_FILES", (str(f),))
        assert atlas_module._default_budget_bytes() == 3221225472 // 2

    def test_unlimited_cgroup_means_2gb(self, monkeypatch, tmp_path):
        monkeypatch.delenv("PIXELMAP_ATLAS_CACHE_MB", raising=False)
        v2 = tmp_path / "memory.max"; v2.write_text("max\n")
        v1 = tmp_path / "limit"; v1.write_text(str(2**63 - 4096) + "\n")
        assert _memory.cgroup_memory_limit_bytes((str(v2),)) is None
        assert _memory.cgroup_memory_limit_bytes((str(v1),)) is None
        assert _memory.cgroup_memory_limit_bytes((str(tmp_path / "missing"),)) is None
        monkeypatch.setattr(_memory, "_CGROUP_LIMIT_FILES", (str(v2), str(v1)))
        assert atlas_module._default_budget_bytes() == 2 * 2**30


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
