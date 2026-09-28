"""The CGC cache settings in multiprocessing workers.

Each test writes a short script and runs it in a fresh interpreter, because
the multiprocessing start method can be chosen only once per process. The
script prints one JSON object with what it observed.

Under spawn and forkserver, workers start from a fresh interpreter and
re-import pyclebsch, so they see only what reaches them through the
environment (PYCLEBSCH_CACHE_DIR) or through a Pool initializer
(pyclebsch.cache.pool_kwargs()). Under fork, they inherit the parent's memory.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

CACHE_DIR_ENV_VAR = "PYCLEBSCH_CACHE_DIR"

# Largest allowed difference between site factors computed in parallel and
# serially. Both runs compute the same tables with the same code; repeated
# computations agree to ~1e-15, while a wrong table differs by O(0.1-1).
PARALLEL_MATCH_TOL = 1e-12

# Builds the lattice objects for one plaquette and defines site_factors().
# L1 is the 2x2x1 lattice, periodic in x and y, T truncation cutoff 1; L3 is
# 2x2x2, periodic in x, y and z, T truncation cutoff 1.
PRELUDE = """
import json, multiprocessing as mp, sys
from pathlib import Path
import pyclebsch.cgc as cgc
from pyclebsch.cache import get_cache_dir
from pyclebsch.matrix_elements.lattice_data import irreps_and_singlets, sites_links_and_plaquettes
from pyclebsch.matrix_elements.plaquette_matrix_elements import calc_plaquette_site_factors

FORDER = [1, 2, 3, -1, -2, -3]
LATTICES = {"L1": ([2, 2, 1], [True, True, False]), "L3": ([2, 2, 2], [True, True, True])}

def site_factors(lattice, parallelize):
    num_sites, pbcs = LATTICES[lattice]
    sites, _links, plaquettes = sites_links_and_plaquettes(num_sites, pbcs, FORDER)
    trunc, singlets, conj = irreps_and_singlets(3, sites, "T", 1)
    P = sorted(plaquettes)[0]
    result = calc_plaquette_site_factors(3, P, sites, plaquettes, trunc, singlets, conj,
                                         FORDER, 1e-10, parallelize=parallelize)
    return {(s, info): value for s, per_site in result.items() for info, value in per_site.items()}

def max_difference(a, b):
    return max((abs(a.get(k, 0) - b.get(k, 0)) for k in a.keys() | b.keys()), default=0.0)

def worker_cache_dir(_):
    return str(get_cache_dir())

def files_below(path):
    return sorted(str(f.relative_to(path)) for f in Path(path).rglob("*") if f.is_file())
"""


def _run_script(body: str, cwd: Path, env_updates: dict[str, str | None]) -> dict:
    """Run PRELUDE + body as a script in a fresh interpreter, from cwd, and
    return the JSON object it prints last."""
    script = cwd.parent / f"{cwd.name}_script.py"
    script.write_text(PRELUDE + "\nif __name__ == '__main__':\n" + textwrap.indent(textwrap.dedent(body), "    "))
    env = dict(os.environ)
    for name, value in env_updates.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    completed = subprocess.run([sys.executable, str(script)], cwd=cwd, env=env,
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr[-3000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _dirs(tmp_path: Path) -> tuple[Path, Path]:
    disk, work = tmp_path / "disk", tmp_path / "cwd"
    work.mkdir()
    return disk, work


def test_spawn_site_factor_workers_use_parent_disk_dir(tmp_path):
    """Under spawn, pyclebsch's own site-factor workers write to the parent's
    disk directory, not to ./CGC_Data in the working directory, and the
    parallel result equals the serial one."""
    disk, work = _dirs(tmp_path)
    out = _run_script(f"""
        mp.set_start_method("spawn")
        cgc.set_cache_dir({str(disk)!r})
        parallel = site_factors("L1", parallelize=True)
        cgc.set_cache_dir(None)
        serial = site_factors("L1", parallelize=False)
        print(json.dumps({{"disk_files": files_below({str(disk)!r}) if Path({str(disk)!r}).exists() else [],
                          "cwd_files": files_below("."), "diff": max_difference(parallel, serial)}}))
    """, work, {CACHE_DIR_ENV_VAR: None})
    assert out["disk_files"] and all(f.startswith("v1/") for f in out["disk_files"])
    assert out["cwd_files"] == []
    assert out["diff"] <= PARALLEL_MATCH_TOL


def test_spawn_workers_keep_disk_disabled(tmp_path):
    """Under spawn with default settings, neither the parent nor the workers
    write anything."""
    _disk, work = _dirs(tmp_path)
    out = _run_script("""
        mp.set_start_method("spawn")
        site_factors("L1", parallelize=True)
        print(json.dumps({"cache_dir": str(get_cache_dir()), "cwd_files": files_below(".")}))
    """, work, {CACHE_DIR_ENV_VAR: None})
    assert out == {"cache_dir": "None", "cwd_files": []}


def test_user_pool_workers_see_cache_dir_via_env(tmp_path):
    """A user's own spawn Pool, created without pool_kwargs(), sees the
    parent's disk directory through PYCLEBSCH_CACHE_DIR."""
    disk, work = _dirs(tmp_path)
    out = _run_script(f"""
        mp.set_start_method("spawn")
        cgc.set_cache_dir({str(disk)!r})
        with mp.Pool(2) as pool:
            seen = sorted(set(pool.map(worker_cache_dir, range(4))))
        print(json.dumps({{"parent": str(get_cache_dir()), "workers": seen}}))
    """, work, {CACHE_DIR_ENV_VAR: None})
    assert out["workers"] == [out["parent"]] == [str(disk.resolve())]


def test_forkserver_user_pool_sees_later_setting(tmp_path):
    """Records whether a user's forkserver Pool (no initializer), created after
    the fork server started, sees a set_cache_dir call made in between.

    This records behavior rather than a preference. Forkserver workers are
    forked from the server process, which keeps the environment it started
    with, so they see the stale setting. The documented remedy is to
    configure the cache before creating any forkserver Pool, or to pass
    pool_kwargs(). pyclebsch's own site-factor Pool does the latter, and this
    test checks that a Pool built with pool_kwargs() sees the new setting."""
    disk, work = _dirs(tmp_path)
    out = _run_script(f"""
        from pyclebsch.cache import pool_kwargs
        mp.set_start_method("forkserver")
        with mp.Pool(1) as pool:            # starts the fork server
            before = pool.map(worker_cache_dir, range(1))
        cgc.set_cache_dir({str(disk)!r})
        with mp.Pool(2) as pool:
            plain = sorted(set(pool.map(worker_cache_dir, range(4))))
        with mp.Pool(2, **pool_kwargs()) as pool:
            configured = sorted(set(pool.map(worker_cache_dir, range(4))))
        print(json.dumps({{"before": before, "plain": plain, "configured": configured}}))
    """, work, {CACHE_DIR_ENV_VAR: None})
    assert out["before"] == ["None"]
    assert out["plain"] == ["None"]
    assert out["configured"] == [str(disk.resolve())]


def test_parallel_cold_disk_run_completes(tmp_path):
    """Under fork, with the disk tier on and empty, the parallel site factors
    of L3 plaquette 0 complete and equal the serial ones. Before writes were
    atomic, this run crashed with EOFError when one worker read an entry
    another was still writing."""
    disk, work = _dirs(tmp_path)
    out = _run_script(f"""
        mp.set_start_method("fork")
        cgc.set_cache_dir({str(disk)!r})
        parallel = site_factors("L3", parallelize=True)
        cgc.set_cache_dir(None)
        serial = site_factors("L3", parallelize=False)
        leftover = [f for f in files_below({str(disk)!r}) if Path(f).name.startswith(".tmp-")]
        print(json.dumps({{"entries": len(parallel), "diff": max_difference(parallel, serial),
                          "leftover_temp_files": leftover}}))
    """, work, {CACHE_DIR_ENV_VAR: None})
    assert out["entries"] > 0
    assert out["diff"] <= PARALLEL_MATCH_TOL
    assert out["leftover_temp_files"] == []
