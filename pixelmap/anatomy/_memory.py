"""Process-memory helpers for the atlas layer.

Two facts about the deployed server that the atlas code has to reason
about, kept here so that every caller reasons about them the same way:

* The container runs under a cgroup memory cap (Compose's
  ``deploy.resources.limits.memory``), and that cap is enforced on RSS.
  :func:`cgroup_memory_limit_bytes` reads it, so cache budgets and size
  gates can be derived from the real limit instead of guessed.
* Dropping the last reference to a large array does not lower RSS.  glibc
  keeps freed pages in its arenas for reuse (``MALLOC_ARENA_MAX`` caps how
  many arenas there are, not what they retain).  Measured after clearing
  every atlas cache: 1,188 MB RSS, then 211 MB after ``malloc_trim``.  An
  eviction that doesn't trim is one the OOM killer never sees, which is
  what :func:`release_freed_memory` is for.
"""

from __future__ import annotations

import ctypes
import gc
from pathlib import Path

# Where a cgroup exposes the memory cap this process runs under -- v2 first,
# then v1.
_CGROUP_LIMIT_FILES = (
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
)


def cgroup_memory_limit_bytes(paths=None) -> int | None:
    """The container's memory cap, or ``None`` when there isn't one."""
    for path in _CGROUP_LIMIT_FILES if paths is None else paths:
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            limit = int(raw)
        except ValueError:
            continue
        # cgroup v1 reports a huge number (~2**63) for "unlimited".
        if limit <= 0 or limit >= 1 << 60:
            return None
        return limit
    return None


_libc = None


def release_freed_memory() -> None:
    """Return freed heap pages to the OS.

    Collects garbage first so that reference cycles holding large arrays are
    actually freed, then asks glibc to trim.  A no-op on platforms without
    ``malloc_trim`` (macOS, musl).
    """
    global _libc
    gc.collect()
    try:
        if _libc is None:
            _libc = ctypes.CDLL("libc.so.6")
        _libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass
