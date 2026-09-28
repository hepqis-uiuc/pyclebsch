"""Two-tier cache for Clebsch-Gordan coefficient tables.

Lookups go memory -> disk (unless the disk tier is turned off with
set_cache_dir(None)) -> compute. The memory tier is always on and lives for
the process; set_cache_dir() clears it. The disk tier stores one pickle file
per table. It is on by default, at ./CGC_Data relative to the working
directory at the time of each read or write.

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

import copy
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
from typing import Literal

# su_n_operators imports neither cgc nor cache, so there is no import cycle.
from pyclebsch.su_n_operators import IrrepWeight, normalize_iweight, standardize_iweight_type

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
# is independent of the CGC conventions above.
CACHE_FORMAT_VERSION: int = 1

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
class CacheKey:
    """Identity of one cached CGC table.

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
                sum_iweight: Sequence[int], eps: float) -> CacheKey:
        """Key for the highest-weight table of sum_iweight in product_iweights."""
        return cls("highest", _key_product(product_iweights), _key_iweight(sum_iweight), None, eps)

    @classmethod
    def lower(cls, product_iweights: Iterable[Sequence[int]],
              sum_iweight: Sequence[int], mult_idx: int, eps: float) -> CacheKey:
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


@dataclass(slots=True)
class CacheStats:
    """Counters since the last reset (per process)."""
    memory_hits: int = 0
    disk_hits: int = 0
    computed: int = 0
    disk_writes: int = 0
    disk_rejected: int = 0     # unreadable, wrong version, or key mismatch
    evictions: int = 0         # memory entries dropped because of the cap


# Errors that loading a damaged or foreign pickle can raise. A truncated file
# raises EOFError or UnpicklingError; a pickle naming a class or module that no
# longer exists raises AttributeError or ModuleNotFoundError; malformed opcode
# arguments raise ValueError, TypeError or KeyError.
_UNREADABLE_ENTRY_ERRORS = (EOFError, pickle.UnpicklingError, AttributeError,
                            ModuleNotFoundError, ValueError, TypeError, KeyError)


class CGCCache:
    """Memory tier (always on) in front of an optional disk tier."""

    def __init__(self, disk_dir: Path | None = None,
                 max_memory_entries: int | None = None) -> None:
        # OrderedDict so a hit can move its entry to the end (most recently
        # used) and eviction can pop from the front (least recently used).
        self._memory: OrderedDict[CacheKey, CGCTable] = OrderedDict()
        self._disk_dir: Path | None = disk_dir
        self._max_memory_entries: int | None = None
        self._stats = CacheStats()
        self.set_max_memory_entries(max_memory_entries)

    @property
    def max_memory_entries(self) -> int | None:
        return self._max_memory_entries

    def set_max_memory_entries(self, max_entries: int | None) -> None:
        """None means unbounded; otherwise an int >= 1 (ValueError if not).
        Evicts least recently used entries down to the new limit."""
        if max_entries is not None and (
            isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1
        ):
            raise ValueError(f"max_entries must be None or an integer >= 1; got {max_entries!r}")
        self._max_memory_entries = max_entries
        self._evict_over_limit()

    @property
    def disk_dir(self) -> Path | None:
        """Disk-tier directory, or None if the disk tier is off."""
        return self._disk_dir

    def set_disk_dir(self, path: str | os.PathLike[str] | None) -> None:
        """Store path (None turns the disk tier off) and clear memory, so that
        no table read from or destined for the old directory is served."""
        self._disk_dir = Path(path) if path is not None else None
        self.clear_memory()

    def clear_memory(self) -> None:
        """Drop every in-memory entry."""
        self._memory.clear()

    def get_or_compute(self, key: CacheKey, compute: Callable[[], CGCTable]) -> CGCTable:
        """Return the table for key, computing it at most once per process
        (unless the memory cap evicts it).

        The returned object is shared with the cache and must not be mutated.
        """
        table = self._memory.get(key)
        if table is not None:
            self._memory.move_to_end(key)
            self._stats.memory_hits += 1
            return table
        if self._disk_dir is not None:
            table = self._read_disk(key)
            if table is not None:
                self._remember(key, table)
                self._stats.disk_hits += 1
                return table
        table = compute()
        self._stats.computed += 1
        self._remember(key, table)
        if self._disk_dir is not None:
            self._write_disk(key, table)
        return table

    def _remember(self, key: CacheKey, table: CGCTable) -> None:
        """Store in memory as the most recently used entry, then evict down
        to the cap."""
        self._memory[key] = table
        self._memory.move_to_end(key)
        self._evict_over_limit()

    def _evict_over_limit(self) -> None:
        if self._max_memory_entries is None:
            return
        while len(self._memory) > self._max_memory_entries:
            self._memory.popitem(last=False)
            self._stats.evictions += 1

    def stats(self) -> CacheStats:
        """Copy of the current counters."""
        return copy.copy(self._stats)

    def reset_stats(self) -> None:
        self._stats = CacheStats()

    def _entry_path(self, key: CacheKey) -> Path:
        """<disk dir>/v<CGC_CACHE_VERSION>/<key.relative_path()>."""
        assert self._disk_dir is not None
        return self._disk_dir / f"v{CGC_CACHE_VERSION}" / key.relative_path()

    def _read_disk(self, key: CacheKey) -> CGCTable | None:
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
        self._stats.disk_rejected += 1

    def _write_disk(self, key: CacheKey, table: CGCTable) -> None:
        """Write the envelope for key atomically: pickle it to a temporary
        file in the target's directory, then rename that onto the target.
        os.replace is atomic within one filesystem, so a concurrent reader sees
        either no file or a complete one. If anything fails, the temporary
        file is removed and the exception re-raised."""
        path = self._entry_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
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
        self._stats.disk_writes += 1


def _envelope_mismatch(envelope: object, key: CacheKey) -> str | None:
    """Why envelope cannot serve key, or None if it can."""
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


# Process-wide cache.
_cache: CGCCache = CGCCache(disk_dir=Path("./CGC_Data"))


def set_cache_dir(path: str | os.PathLike[str] | None) -> None:
    """Set the directory where computed CGCs are cached on disk, or disable
    the disk tier with None. Clears the in-memory cache."""
    _cache.set_disk_dir(path)


def get_cache_dir() -> Path | None:
    """Disk-tier directory, or None when disk caching is off."""
    return _cache.disk_dir


def clear_memory_cache() -> None:
    """Drop all in-memory CGC tables (disk entries are untouched)."""
    _cache.clear_memory()


def cache_stats() -> CacheStats:
    """Hit/compute counters for this process since the last reset."""
    return _cache.stats()


def set_memory_cache_limit(max_entries: int | None) -> None:
    """Cap the in-memory tier at max_entries tables (least recently used are
    evicted), or remove the cap with None (the default)."""
    _cache.set_max_memory_entries(max_entries)
