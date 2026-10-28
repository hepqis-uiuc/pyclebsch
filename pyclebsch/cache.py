"""Two-tier cache for Clebsch-Gordan coefficient tables.

Lookups go memory -> disk (only if a directory is configured) -> compute.
The memory tier is always on and lives for the process; set_cgc_cache_dir()
clears it. The disk tier is opt-in, via set_cgc_cache_dir() or the
PYCLEBSCH_CACHE_DIR environment variable, and is only ever used at a path the
user names: there is no default location. set_cgc_cache_dir() also writes that
variable, so worker processes started afterwards (including under spawn and
forkserver) use the same directory; pool_kwargs() passes the settings to a
multiprocessing.Pool explicitly.

This module also holds the in-memory caches of the plaquette tables that
pyclebsch.matrix_elements computes once per plaquette signature (see
_PlaquetteCache), so that clear_memory_cache() clears them with the CGC
tables. They have no disk tier. set_physical_plaquette_states_cache_limit()
and set_plaquette_matrix_elements_cache_limit() cap them, separately from
each other and from the CGC tables.

The public interface is the section headed "Public interface" at the end of
this module, and the names in __all__. The cache classes are internal: they
return the stored objects themselves, which must not be modified, while the
public ways to get tables (calc_cgcs and the by-signature functions of
pyclebsch.matrix_elements) return new objects.

Disk entry format
-----------------
Each file under <cache dir>/v<CGC_CACHE_VERSION>/ is a pickled dict with keys:
  format_version     envelope layout version (CACHE_FORMAT_VERSION)
  cgc_cache_version  CGC convention version (CGC_CACHE_VERSION)
  key                kind, product, sum_irrep, mult_idx, eps of the table
  pyclebsch_version  installed pyclebsch version string, or None when
                     pyclebsch runs without being installed (e.g. from a
                     source tree on PYTHONPATH)
  pyclebsch_commit   git commit of the installed pyclebsch, or None. It is
                     only known for installs made directly from a git
                     repository (e.g. a uv/pip git source). It is None for
                     installs from a package index such as PyPI and for
                     editable installs, because neither records a commit.
                     None is expected, not an error.
  data               the CGC table
Only format_version, cgc_cache_version and key are checked when an entry is
read. pyclebsch_version and pyclebsch_commit are informational. An entry
that cannot be read or fails the check is deleted, with a CGCCacheWarning,
and recomputed. Entries are written atomically, so a reader never sees a
partly written file.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import pickle
import tempfile
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib import metadata
from numbers import Integral
from pathlib import Path
from typing import Literal, TypedDict

# su_n_operators imports neither cgc nor cache, so there is no import cycle.
from pyclebsch.su_n_operators import IrrepWeight, normalize_iweight, standardize_iweight_type

__all__ = [
    "CACHE_FORMAT_VERSION",
    "CGC_CACHE_VERSION",
    "ENV_VAR",
    "CGCCacheWarning",
    "PoolKwargs",
    "cgc_cache_stats",
    "clear_memory_cache",
    "get_cgc_cache_dir",
    "init_worker",
    "physical_plaquette_states_cache_stats",
    "plaquette_matrix_elements_cache_stats",
    "pool_kwargs",
    "set_cgc_cache_dir",
    "set_cgc_memory_cache_limit",
    "set_physical_plaquette_states_cache_limit",
    "set_plaquette_matrix_elements_cache_limit",
]

# Version of the CGC *conventions* baked into cached tables. Bump it whenever a
# change alters any stored table for the same key. That covers:
#   - the ordering of Gelfand-Tsetlin patterns (stored states are indices into it);
#   - the labeling of multiplicity copies (S_n symmetrization, RREF/QR basis choice);
#   - the phase convention;
#   - the normalization / trivial-irrep rules that produce the key.
# tests/test_cgc_golden.py fails when any of these change; the fix is to bump
# this number and add golden values for the new version. Disk entries live
# under v<CGC_CACHE_VERSION>/, so a bump orphans every older entry. Version 1
# is the first versioned layout; unversioned entries from pyclebsch <= 0.1 are
# never read.
CGC_CACHE_VERSION: int = 1

# Version of the on-disk envelope format (the dict around the table), which
# is independent of the CGC conventions above. Bump it whenever the layout
# written by _CGCCache._write_disk changes: a field added, removed or renamed,
# a field of the stored key (_CGCCacheKey.as_header) changed, or a change in the
# type of any value, including the structure of CGCTable itself.
# tests/test_cgc_golden.py fails when the layout changes; the fix is to bump
# this number and add the new layout there. Unlike a CGC_CACHE_VERSION bump,
# this does not change the directory: old entries fail the format_version
# check when read, and are deleted and recomputed.
CACHE_FORMAT_VERSION: int = 1

# Environment variable that enables the disk tier and carries the setting to
# worker processes started with spawn or forkserver. Unset or empty means the
# disk tier is off.
ENV_VAR: str = "PYCLEBSCH_CACHE_DIR"

# Longest readable file or directory name used before switching to a hashed
# name. Common filesystems (ext4, APFS, NTFS) cap a name at 255 bytes; the
# rest is a safety margin, not a derived bound. Too high a value produces
# OSError(ENAMETOOLONG) on large products; too low merely makes more names
# hashed.
MAX_ENTRY_NAME_BYTES: int = 200

# A name too long to keep readable becomes this prefix plus the SHA-256 hex
# digest of the readable name (71 bytes in all).
HASHED_NAME_PREFIX: str = "sha256-"

# Prefix of the temporary files that atomic writes create next to their
# target. A file with this prefix can only be left behind by a process killed
# mid-write; it is never read and can be deleted.
TEMP_FILE_PREFIX: str = ".tmp-"

type ProductState = tuple[int, ...]
type CGCTable = dict[int, dict[ProductState, float]]


class CGCCacheWarning(UserWarning):
    """A disk entry was unreadable or did not match its key and was recomputed."""


@dataclass(frozen=True, slots=True)
class _CGCCacheKey:
    """Identity of one cached CGC table. Internal to this module and to
    pyclebsch.cgc.

    product keeps the caller's order (not re-sorted), because the computation
    depends on it; calc_cgcs sorts before calling in.
    """
    kind: Literal["highest", "lower"]
    product: tuple[IrrepWeight, ...]
    sum_irrep: IrrepWeight
    mult_idx: int | None
    eps: float

    @classmethod
    def highest(cls, product_iweights: Iterable[Sequence[int]],
                sum_iweight: Sequence[int], eps: float) -> _CGCCacheKey:
        """Key for the highest-weight table of sum_iweight in product_iweights."""
        return cls("highest", _key_product(product_iweights), _key_iweight(sum_iweight), None, eps)

    @classmethod
    def lower(cls, product_iweights: Iterable[Sequence[int]],
              sum_iweight: Sequence[int], mult_idx: int, eps: float) -> _CGCCacheKey:
        """Key for all states of copy mult_idx of sum_iweight in product_iweights.

        numpy integers are accepted wherever int is annotated, and converted.
        """
        if not isinstance(mult_idx, Integral):
            raise TypeError(f"mult_idx must be an integer; got {mult_idx!r}")
        return cls("lower", _key_product(product_iweights), _key_iweight(sum_iweight),
                   int(mult_idx), eps)

    def relative_path(self) -> Path:
        """<product dir>/<entry file>, relative to <disk dir>/v<CGC_CACHE_VERSION>/.

        For example [(1, 0, 0), (1, 1, 0)]/lower_weight_CGC_((2, 1, 0), 1).
        A component longer than MAX_ENTRY_NAME_BYTES is replaced by
        HASHED_NAME_PREFIX plus the SHA-256 digest of its readable form. eps is
        not part of the path; the header check handles it.
        """
        product_dir = str(list(self.product))
        if self.kind == "highest":
            entry_file = f"highest_weight_CGC_{self.sum_irrep}"
        else:
            entry_file = f"lower_weight_CGC_{(self.sum_irrep, self.mult_idx)}"
        return Path(_name_or_hash(product_dir)) / _name_or_hash(entry_file)

    def as_header(self) -> dict[str, object]:
        """Plain-data form stored in, and compared against, the disk envelope."""
        return {"kind": self.kind, "product": self.product, "sum_irrep": self.sum_irrep,
                "mult_idx": self.mult_idx, "eps": self.eps}


def _key_iweight(iweight: Sequence[int]) -> IrrepWeight:
    """Plain-int, normalized form of an i-weight, as stored in a key."""
    return normalize_iweight(standardize_iweight_type(iweight))


def _key_product(product_iweights: Iterable[Sequence[int]]) -> tuple[IrrepWeight, ...]:
    return tuple(_key_iweight(iweight) for iweight in product_iweights)


def _name_or_hash(name: str) -> str:
    """name itself if it fits in MAX_ENTRY_NAME_BYTES, else a hashed name."""
    encoded = name.encode()
    if len(encoded) <= MAX_ENTRY_NAME_BYTES:
        return name
    return HASHED_NAME_PREFIX + hashlib.sha256(encoded).hexdigest()


class _LRUMemoryCache[K, V]:
    """In-memory map with an optional cap on its number of entries. Reading or
    storing an entry makes it the most recently used; over the cap, the least
    recently used entries are evicted. Values must not be None, which get()
    returns for a missing key.

    Counts, since creation or the last reset_counters(): hits (values found
    by get, including through get_or_compute), computed (values computed by
    get_or_compute) and evictions."""

    def __init__(self, max_entries: int | None = None) -> None:
        # OrderedDict so a hit can move its entry to the end (most recently
        # used) and eviction can pop from the front (least recently used).
        self._entries: OrderedDict[K, V] = OrderedDict()
        self._max_entries: int | None = None
        self.hits = 0
        self.computed = 0
        self.evictions = 0
        self.set_max_entries(max_entries)

    def reset_counters(self) -> None:
        self.hits = 0
        self.computed = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def max_entries(self) -> int | None:
        return self._max_entries

    def set_max_entries(self, max_entries: int | None) -> None:
        """None means unbounded; otherwise an int >= 1 (ValueError if not).
        Evicts least recently used entries down to the new limit."""
        if max_entries is not None and (
            isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1
        ):
            raise ValueError(f"max_entries must be None or an integer >= 1; got {max_entries!r}")
        self._max_entries = max_entries
        self._evict_over_limit()

    def clear(self) -> None:
        """Drop every entry."""
        self._entries.clear()

    def get(self, key: K) -> V | None:
        """The value stored for key, now the most recently used, or None if
        there is none."""
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
            self.hits += 1
        return value

    def put(self, key: K, value: V) -> None:
        """Store value for key as the most recently used entry, then evict
        down to the cap."""
        self._entries[key] = value
        self._entries.move_to_end(key)
        self._evict_over_limit()

    def get_or_compute(self, key: K, compute: Callable[[], V]) -> V:
        """The value for key, computed with compute() if it is not stored.

        The returned object is shared with the cache and must not be mutated.
        """
        value = self.get(key)
        if value is None:
            value = compute()
            self.computed += 1
            self.put(key, value)
        return value

    def _evict_over_limit(self) -> None:
        if self._max_entries is None:
            return
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
            self.evictions += 1


# Errors that loading a damaged or foreign pickle can raise. A truncated file
# raises EOFError or UnpicklingError; a pickle naming a class or module that no
# longer exists raises AttributeError or ModuleNotFoundError; malformed opcode
# arguments raise ValueError, TypeError or KeyError.
_UNREADABLE_ENTRY_ERRORS = (EOFError, pickle.UnpicklingError, AttributeError,
                            ModuleNotFoundError, ValueError, TypeError, KeyError)


class _CGCCache:
    """Memory tier (always on) in front of an optional disk tier.

    Internal: the module-level functions configure, clear and inspect the
    process-wide instance. get_or_compute returns the stored table itself,
    which must not be modified; calc_cgcs, the public way to get CGCs,
    returns new dicts."""

    def __init__(self, disk_dir: Path | None = None,
                 max_memory_entries: int | None = None) -> None:
        self._memory: _LRUMemoryCache[_CGCCacheKey, CGCTable] = _LRUMemoryCache()
        self._disk_dir: Path | None = disk_dir
        # Counters of this cache's own events. The memory tier counts its
        # hits and evictions itself (see stats()).
        self._counts = dict.fromkeys(("disk_hits", "computed", "disk_writes", "disk_rejected"), 0)
        self.set_max_memory_entries(max_memory_entries)

    @property
    def max_memory_entries(self) -> int | None:
        return self._memory.max_entries

    def set_max_memory_entries(self, max_entries: int | None) -> None:
        """None means unbounded; otherwise an int >= 1 (ValueError if not).
        Evicts least recently used entries down to the new limit."""
        self._memory.set_max_entries(max_entries)

    @property
    def disk_dir(self) -> Path | None:
        """Resolved disk-tier directory, or None if the disk tier is off."""
        return self._disk_dir

    def set_disk_dir(self, path: str | os.PathLike[str] | None) -> None:
        """Store _resolve_disk_dir(path), or None to turn the disk tier off, and
        clear memory, so that no table read from or destined for the old
        directory is served. ValueError for the empty string, with nothing
        changed (see _resolve_disk_dir for Path(""))."""
        self._disk_dir = _resolve_disk_dir(path) if path is not None else None
        self.clear_memory()

    def clear_memory(self) -> None:
        """Drop every in-memory entry."""
        self._memory.clear()

    def get_or_compute(self, key: _CGCCacheKey, compute: Callable[[], CGCTable]) -> CGCTable:
        """Return the table for key, computing it at most once per process
        (unless the memory cap evicts it).

        The returned object is shared with the cache and must not be mutated.
        """
        table = self._memory.get(key)
        if table is not None:
            return table
        if self._disk_dir is not None:
            table = self._read_disk(key)
            if table is not None:
                self._remember(key, table)
                self._counts["disk_hits"] += 1
                return table
        table = compute()
        self._counts["computed"] += 1
        self._remember(key, table)
        if self._disk_dir is not None:
            self._write_disk(key, table)
        return table

    def _remember(self, key: _CGCCacheKey, table: CGCTable) -> None:
        """Store in memory as the most recently used entry, then evict down
        to the cap."""
        self._memory.put(key, table)

    def stats(self) -> dict[str, int]:
        """The counters since the last reset, as a new dict (see
        cgc_cache_stats for the keys)."""
        return {"memory_hits": self._memory.hits, **self._counts,
                "evictions": self._memory.evictions}

    def reset_stats(self) -> None:
        self._counts = dict.fromkeys(self._counts, 0)
        self._memory.reset_counters()

    def _entry_path(self, key: _CGCCacheKey) -> Path:
        """<disk dir>/v<CGC_CACHE_VERSION>/<key.relative_path()>."""
        assert self._disk_dir is not None
        return self._disk_dir / f"v{CGC_CACHE_VERSION}" / key.relative_path()

    def _read_disk(self, key: _CGCCacheKey) -> CGCTable | None:
        """The stored table for key, or None if there is none. An entry that
        cannot be unpickled, or whose envelope does not match the current
        versions and key, is rejected: warn with CGCCacheWarning, delete the
        file, count it in disk_rejected, and return None."""
        path = self._entry_path(key)
        try:
            with open(path, "rb") as fp:
                envelope = pickle.load(fp)
        except FileNotFoundError:
            return None
        except _UNREADABLE_ENTRY_ERRORS as error:
            self._reject(path, f"unreadable ({type(error).__name__}: {error})")
            return None
        mismatch = _envelope_mismatch(envelope, key)
        if mismatch is not None:
            self._reject(path, mismatch)
            return None
        return envelope["data"]

    def _reject(self, path: Path, reason: str) -> None:
        warnings.warn(f"Discarding CGC cache entry {path}: {reason}. It will be recomputed.",
                      CGCCacheWarning, stacklevel=2)
        path.unlink(missing_ok=True)
        self._counts["disk_rejected"] += 1

    def _write_disk(self, key: _CGCCacheKey, table: CGCTable) -> None:
        """Write the envelope for key atomically: pickle it to a temporary
        file in the target's directory, then rename that onto the target.
        os.replace is atomic within one filesystem, so a concurrent reader sees
        either no file or a complete one. If anything fails, the temporary
        file is removed and the exception re-raised."""
        path = self._entry_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Any change to this layout requires a CACHE_FORMAT_VERSION bump.
        envelope = {
            "format_version": CACHE_FORMAT_VERSION,
            "cgc_cache_version": CGC_CACHE_VERSION,
            "key": key.as_header(),
            "pyclebsch_version": _installed_version(),
            "pyclebsch_commit": _installed_commit(),
            "data": table,
        }
        temp = tempfile.NamedTemporaryFile(dir=path.parent, prefix=TEMP_FILE_PREFIX, delete=False)
        temp_path = Path(temp.name)
        try:
            with temp:
                pickle.dump(envelope, temp)
            os.replace(temp_path, path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
        self._counts["disk_writes"] += 1


class _PlaquetteCache:
    """In-memory caches of the plaquette tables that pyclebsch.matrix_elements
    computes once per plaquette signature: physical plaquette states, and
    plaquette matrix elements. Each has its own cap on its number of entries;
    neither has a disk tier.

    The keys are built where the tables are computed:
    _physical_plaquette_states_cache_key in lattice_data, and
    _plaquette_matrix_elements_cache_key in plaquette_matrix_elements.

    Internal: the module-level functions cap and clear the process-wide
    instance. Its caches return the stored tables themselves, which must not
    be modified; the by-signature functions, the public way to get the
    tables, return copies.
    """

    def __init__(self) -> None:
        self._physical_states: _LRUMemoryCache[tuple, list] = _LRUMemoryCache()
        self._matrix_elements: _LRUMemoryCache[tuple, dict] = _LRUMemoryCache()

    @property
    def physical_states(self) -> _LRUMemoryCache[tuple, list]:
        """Each entry the list of physical plaquette states of one signature."""
        return self._physical_states

    @property
    def matrix_elements(self) -> _LRUMemoryCache[tuple, dict]:
        """Each entry the plaquette matrix elements of one signature, as
        {(final state, initial state): value}."""
        return self._matrix_elements

    def clear(self) -> None:
        """Drop every entry of both caches."""
        self._physical_states.clear()
        self._matrix_elements.clear()


def _envelope_mismatch(envelope: object, key: _CGCCacheKey) -> str | None:
    """The reason envelope cannot be used for key, as a message for the
    warning that rejects it; None if it can be used."""
    if not isinstance(envelope, dict) or "data" not in envelope:
        return "not a cache entry envelope"
    if envelope.get("format_version") != CACHE_FORMAT_VERSION:
        return (f"format_version {envelope.get('format_version')!r}, "
                f"expected {CACHE_FORMAT_VERSION}")
    if envelope.get("cgc_cache_version") != CGC_CACHE_VERSION:
        return (f"cgc_cache_version {envelope.get('cgc_cache_version')!r}, "
                f"expected {CGC_CACHE_VERSION}")
    if envelope.get("key") != key.as_header():
        return f"stored key {envelope.get('key')!r} does not match {key.as_header()!r}"
    return None


def _resolve_disk_dir(path: str | os.PathLike[str]) -> Path:
    """Path(path).expanduser().resolve(); ValueError for the empty string.

    The empty string would resolve to the working directory, a location the
    user did not name. Path("") cannot be caught here: pathlib turns it into
    Path("."), indistinguishable from an explicit request for the working
    directory. expanduser comes first because Path does not expand
    "~" on its own: without it, "~/cgc" would resolve to
    <working directory>/~/cgc.
    """
    if os.fspath(path) == "":
        raise ValueError("the CGC cache directory must be a non-empty path")
    return Path(path).expanduser().resolve()


def _disk_dir_from_env() -> Path | None:
    """_resolve_disk_dir(PYCLEBSCH_CACHE_DIR), or None if unset or empty."""
    value = os.environ.get(ENV_VAR, "")
    return _resolve_disk_dir(value) if value else None


@functools.cache
def _installed_commit() -> str | None:
    """Git commit of the installed pyclebsch, or None when it is not known.

    Read from the distribution's direct_url.json (PEP 610):
      - git install:      {"vcs_info": {"commit_id": "<40 hex chars>", ...}} -> that hash
      - editable install: {"dir_info": {"editable": true}}                   -> None
      - index install (PyPI wheel or sdist): no direct_url.json              -> None
      - pyclebsch not installed, unreadable file, malformed JSON, missing keys -> None
    Never raises. Computed once per process. Informational only; never used
    to validate cache entries.
    """
    try:
        direct_url = metadata.distribution("pyclebsch").read_text("direct_url.json")
        if direct_url is None:
            return None
        commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    # The commit only annotates cache entries, so any failure to read it,
    # whatever its type, must not stop a computation.
    except Exception:
        return None
    return commit if isinstance(commit, str) else None


@functools.cache
def _installed_version() -> str | None:
    """importlib.metadata.version("pyclebsch"), or None if pyclebsch is not
    installed (PackageNotFoundError). Informational only."""
    try:
        return metadata.version("pyclebsch")
    except metadata.PackageNotFoundError:
        return None


# Process-wide cache, configured from the environment at import time.
_cgc_cache: _CGCCache = _CGCCache(disk_dir=_disk_dir_from_env())

# Process-wide plaquette caches, filled by
# physical_plaquette_states_by_signature and
# plaquette_matrix_elements_by_signature. Only the process that calls those
# functions uses them: worker processes compute site factors, not whole
# tables, so pool_kwargs() does not pass the plaquette cache limits on.
_plaquette_cache: _PlaquetteCache = _PlaquetteCache()


# Public interface: the functions a user calls to configure, clear and
# inspect the caches. __all__ lists every public name of this module.

# The disk tier.

def set_cgc_cache_dir(path: str | os.PathLike[str] | None) -> None:
    """Enable the disk tier at path (a leading "~" expanded, then resolved to
    an absolute path), or disable it with None. There is no default location:
    path must be given, and the empty string raises ValueError. Note that
    pathlib.Path("") is the same object as Path("."), so it is accepted and
    means the current working directory. Clears the in-memory cache and
    updates PYCLEBSCH_CACHE_DIR so worker processes started afterwards use
    the same setting."""
    _cgc_cache.set_disk_dir(path)
    if _cgc_cache.disk_dir is None:
        os.environ.pop(ENV_VAR, None)
    else:
        os.environ[ENV_VAR] = str(_cgc_cache.disk_dir)


def get_cgc_cache_dir() -> Path | None:
    """Absolute disk-tier directory, or None when disk caching is off (the default)."""
    return _cgc_cache.disk_dir


# The in-memory caches: clearing, statistics and limits.

def clear_memory_cache() -> None:
    """Drop all in-memory CGC tables and plaquette tables (disk entries are
    untouched)."""
    _cgc_cache.clear_memory()
    _plaquette_cache.clear()


def cgc_cache_stats() -> dict[str, int]:
    """Counters of the CGC cache for this process, as a new dict:
      memory_hits    tables found in memory
      disk_hits      tables read from disk
      computed       tables computed
      disk_writes    tables written to disk
      disk_rejected  disk entries discarded as unreadable, of the wrong
                     version, or with a mismatched key
      evictions      tables dropped from memory because of the cap
    The plaquette caches' statistics use the same keys for the counts they
    share (memory_hits, computed, evictions)."""
    return _cgc_cache.stats()


def set_cgc_memory_cache_limit(max_entries: int | None) -> None:
    """Cap the in-memory tier at max_entries tables (least recently used are
    evicted), or remove the cap with None (the default)."""
    _cgc_cache.set_max_memory_entries(max_entries)


def set_physical_plaquette_states_cache_limit(max_entries: int | None) -> None:
    """Cap the in-memory cache of physical plaquette states at max_entries
    tables (least recently used are evicted), or remove the cap with None
    (the default). A table is the list of physical states of one plaquette
    signature, for one set of the other inputs in its cache key (see
    physical_plaquette_states_by_signature).

    The limits of the CGC cache (set_cgc_memory_cache_limit) and of the plaquette
    matrix elements cache (set_plaquette_matrix_elements_cache_limit) are
    separate settings."""
    _plaquette_cache.physical_states.set_max_entries(max_entries)


def set_plaquette_matrix_elements_cache_limit(max_entries: int | None) -> None:
    """Cap the in-memory cache of plaquette matrix elements at max_entries
    tables (least recently used are evicted), or remove the cap with None
    (the default). A table is the matrix elements of one plaquette
    signature, for one set of the other inputs in its cache key (see
    plaquette_matrix_elements_by_signature); at high cutoffs, one table can
    hold millions of matrix elements.

    The limits of the CGC cache (set_cgc_memory_cache_limit) and of the physical
    plaquette states cache (set_physical_plaquette_states_cache_limit) are
    separate settings."""
    _plaquette_cache.matrix_elements.set_max_entries(max_entries)


def physical_plaquette_states_cache_stats() -> dict[str, int]:
    """Counters of the physical plaquette states cache since the process
    started, as a new dict:
      memory_hits  tables found in the cache
      computed     tables computed and stored
      evictions    tables dropped because of the cap
    The keys mean the same as the keys of the same names in
    cgc_cache_stats(); this cache has no disk tier, so it has no disk keys."""
    return _plaquette_cache_stats(_plaquette_cache.physical_states)


def plaquette_matrix_elements_cache_stats() -> dict[str, int]:
    """Counters of the plaquette matrix elements cache since the process
    started, as a new dict:
      memory_hits  tables found in the cache
      computed     tables computed and stored
      evictions    tables dropped because of the cap
    The keys mean the same as the keys of the same names in
    cgc_cache_stats(); this cache has no disk tier, so it has no disk keys."""
    return _plaquette_cache_stats(_plaquette_cache.matrix_elements)


def _plaquette_cache_stats(cache: _LRUMemoryCache) -> dict[str, int]:
    return {"memory_hits": cache.hits, "computed": cache.computed,
            "evictions": cache.evictions}


# Worker processes.

def init_worker(disk_dir: Path | None, max_memory_entries: int | None) -> None:
    """multiprocessing.Pool initializer: apply the parent's cache settings.

    The directory is applied only if it differs from the worker's own
    setting. A fork worker already has the parent's setting, and its copy of
    the parent's memory tier; calling set_cgc_cache_dir would needlessly clear it.
    """
    if get_cgc_cache_dir() != disk_dir:
        set_cgc_cache_dir(disk_dir)
    set_cgc_memory_cache_limit(max_memory_entries)


class PoolKwargs(TypedDict):
    """The keyword arguments pool_kwargs() returns, typed so that
    Pool(n, **pool_kwargs()) can be type-checked."""
    initializer: Callable[[Path | None, int | None], None]
    initargs: tuple[Path | None, int | None]


def pool_kwargs() -> PoolKwargs:
    """Keyword arguments that make a Pool's workers use this process's cache
    settings, e.g. multiprocessing.Pool(5, **pool_kwargs())."""
    return {"initializer": init_worker,
            "initargs": (get_cgc_cache_dir(), _cgc_cache.max_memory_entries)}
