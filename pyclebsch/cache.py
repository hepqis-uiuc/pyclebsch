"""Two-tier cache for Clebsch-Gordan coefficient tables.

Lookups go memory -> disk (unless the disk tier is turned off with
set_cache_dir(None)) -> compute. The memory tier is always on and lives for
the process; set_cache_dir() clears it. The disk tier stores one pickle file
per table. It is on by default, at ./CGC_Data relative to the working
directory at the time of each read or write.
"""
from __future__ import annotations

import copy
import os
import pickle
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
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
# this number and add golden values for the new version.
CGC_CACHE_VERSION: int = 1

type ProductState = tuple[int, ...]
type CGCTable = dict[int, dict[ProductState, float]]


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
        """<product dir>/<entry file>, relative to the disk-tier directory.

        For example [(1, 0, 0), (1, 1, 0)]/lower_weight_CGC_((2, 1, 0), 1).
        eps is not part of the path.
        """
        product_dir = str(list(self.product))
        if self.kind == "highest":
            entry_file = f"highest_weight_CGC_{self.sum_irrep}"
        else:
            entry_file = f"lower_weight_CGC_{(self.sum_irrep, self.mult_idx)}"
        return Path(product_dir) / entry_file


def _key_iweight(iweight: Sequence[int]) -> IrrepWeight:
    """Plain-int, normalized form of an i-weight, as stored in a key."""
    return normalize_iweight(standardize_iweight_type(iweight))


def _key_product(product_iweights: Iterable[Sequence[int]]) -> tuple[IrrepWeight, ...]:
    return tuple(_key_iweight(iweight) for iweight in product_iweights)


@dataclass(slots=True)
class CacheStats:
    """Counters since the last reset (per process)."""
    memory_hits: int = 0
    disk_hits: int = 0
    computed: int = 0
    disk_writes: int = 0
    evictions: int = 0         # memory entries dropped because of the cap


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
        """<disk dir>/<key.relative_path()>."""
        assert self._disk_dir is not None
        return self._disk_dir / key.relative_path()

    def _read_disk(self, key: CacheKey) -> CGCTable | None:
        """The stored table for key, or None if there is no file."""
        path = self._entry_path(key)
        if not path.exists():
            return None
        with open(path, "rb") as fp:
            return pickle.load(fp)

    def _write_disk(self, key: CacheKey, table: CGCTable) -> None:
        path = self._entry_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fp:
            pickle.dump(table, fp)
        self._stats.disk_writes += 1


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
