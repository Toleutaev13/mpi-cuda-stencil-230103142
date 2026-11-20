import os
import sys
import json
import time
import ctypes
import hashlib
import platform
import threading
import subprocess
import numpy as np
from numba import njit

try:
    import resource
except ImportError:          # Windows
    resource = None
try:
    import psutil            # optional: pip install psutil (context switches on Windows)
except ImportError:
    psutil = None

# Replace with your exact Git handle before executing
GIT_USERNAME = "toleutaev13"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
STRIDES = (1, 4, 8, 16)
REPEATS = 3                  # each config is run 3 times, median is reported
PURE_PY_ITERS = 1_000_000    # for the Q5 GIL experiment


def detect_cache_line_size() -> int:
    """Query L1 data cache coherency line size in bytes from OS sysfs/sysctl."""
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], encoding="utf-8")
        return int(out.strip())
    return 64


# ==============================================================================
# Core affinity helpers (Linux: sched_setaffinity, Windows: SetThreadAffinityMask)
# ==============================================================================
def pick_cores(n: int) -> list:
    """Pick n logical CPUs, one per physical core when topology is known."""
    if hasattr(os, "sched_getaffinity"):
        avail = sorted(os.sched_getaffinity(0))
        chosen, seen = [], set()
        for c in avail:
            p = f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"
            try:
                with open(p, "r", encoding="utf-8") as f:
                    key = f.read().strip()
            except OSError:
                key = str(c)
            if key not in seen:
                seen.add(key)
                chosen.append(c)
            if len(chosen) == n:
                return chosen
        return avail[:n]
    if sys.platform == "win32":
        logical = os.cpu_count() or 1
        # Windows enumerates SMT siblings as adjacent pairs (0,1), (2,3)...
        return [2 * i for i in range(n)] if logical >= 2 * n else list(range(n))
    return []


def pin_current_thread(core: int) -> bool:
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {core})   # pid 0 = calling thread on Linux
            return True
        if sys.platform == "win32":
            k32 = ctypes.windll.kernel32
            k32.GetCurrentThread.restype = ctypes.c_void_p
            k32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            k32.SetThreadAffinityMask.restype = ctypes.c_size_t
            return k32.SetThreadAffinityMask(k32.GetCurrentThread(), 1 << core) != 0
    except Exception:
        pass
    return False


# ==============================================================================
# TASK 1: Hardware Topology & Student Identity Context
# ==============================================================================
def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running."
    )
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")

    line_size = detect_cache_line_size()
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{logical_cores}"

    digest = hashlib.sha256(f"{git_user}|{cpu_sig}".encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)                      # deterministic 64-bit seed

    cores = pick_cores(NUM_THREADS)
    return {
        "git_username": git_user,
        "logical_cores": logical_cores,
        "cache_line_size": line_size,
        "cpu_signature": cpu_sig,
        "instance_seed": instance_seed,
        "affinity_cores": cores,
        "affinity_supported": len(cores) == NUM_THREADS,
        "python": platform.python_version(),
        "os": platform.platform(),
    }


# ==============================================================================
# TASK 2: GIL-free native counter kernel
# The store address depends on a value just loaded from memory
# (buffer[slot] >> 63 is always 0 at runtime, but the compiler cannot prove it),
# so LLVM cannot promote buffer[slot] to a register: every iteration loads and
# stores to memory.
# ==============================================================================
@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, seed):
    inc = seed % np.uint64(7) + np.uint64(1)
    for _ in range(iters):
        idx = slot + np.int64(buffer[slot] >> np.uint64(63))
        buffer[idx] = buffer[idx] + inc


# GIL-holding twin of the same kernel (for Q5): compiled, but nogil=False
worker_strided_gil = njit(nogil=False)(worker_strided_accumulate.py_func)


# ==============================================================================
# TASK 4: Thread-local register privatization baseline
# ==============================================================================
@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, seed):
    inc = seed % np.uint64(7) + np.uint64(1)
    acc = np.uint64(0)
    for _ in range(iters):
        acc += inc
    buffer[slot] = acc            # single shared write at the end


# ==============================================================================
# Helpers
# ==============================================================================
def aligned_buffer(n_elems: int) -> np.ndarray:
    """uint64 array whose first element sits on a 64-byte boundary."""
    raw = np.zeros(n_elems + 8, dtype=np.uint64)
    off = ((-raw.ctypes.data) % 64) // 8
    buf = raw[off:off + n_elems]          # view keeps `raw` alive via buf.base
    assert buf.ctypes.data % 64 == 0
    return buf


def ctx_switches():
    if resource is not None:
        r = resource.getrusage(resource.RUSAGE_SELF)
        return r.ru_nvcsw, r.ru_nivcsw
    if psutil is not None:
        c = psutil.Process().num_ctx_switches()
        return c.voluntary, c.involuntary
    return None


def launch(kernel, buf, stride, iters, seed, cores):
    """Run NUM_THREADS threads, thread t works on buffer[t*stride]. Returns (elapsed_ns, ctx_delta)."""
    barrier = threading.Barrier(NUM_THREADS + 1)
    useed = np.uint64(seed)

    def run(tid):
        if cores:
            pin_current_thread(cores[tid])
        barrier.wait()
        kernel(buf, tid * stride, iters, useed)

    threads = [threading.Thread(target=run, args=(t,)) for t in range(NUM_THREADS)]
    for t in threads:
        t.start()
    before = ctx_switches()
    barrier.wait()
    t0 = time.perf_counter_ns()
    for t in threads:
        t.join()
    t1 = time.perf_counter_ns()
    after = ctx_switches()
    ctx = None
    if before and after:
        ctx = {"voluntary": after[0] - before[0], "involuntary": after[1] - before[1]}
    return t1 - t0, ctx


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


# ==============================================================================
# TASK 3: Multi-threaded stride sweep (+ Task 4 baseline)
# ==============================================================================
def run_stride_benchmark(ctx: dict) -> dict:
    seed, cores = ctx["instance_seed"], ctx["affinity_cores"] if ctx["affinity_supported"] else []
    out = {"strides": {}, "private": None}

    for stride in STRIDES:
        runs, finals, ctxs = [], None, []
        for _ in range(REPEATS):
            buf = aligned_buffer(NUM_THREADS * 16 + 16)
            ns, cs = launch(worker_strided_accumulate, buf, stride, ITERATIONS_PER_THREAD, seed, cores)
            runs.append(ns)
            ctxs.append(cs)
            finals = [int(buf[t * stride]) for t in range(NUM_THREADS)]
        med = median(runs)
        out["strides"][stride] = {
            "runs_ns": runs,
            "median_ns": med,
            "ns_per_update": med / ITERATIONS_PER_THREAD,
            "context_switches_last_run": ctxs[-1],
            "final_values": finals,
        }
        print(f"stride={stride:>2}: median {med/1e9:8.3f} s  "
              f"{med/ITERATIONS_PER_THREAD:7.3f} ns/update  runs={[round(r/1e9,3) for r in runs]}")

    runs, finals = [], None
    for _ in range(REPEATS):
        buf = aligned_buffer(NUM_THREADS * 16 + 16)
        ns, cs = launch(worker_private_accumulate, buf, 8, ITERATIONS_PER_THREAD, seed, cores)
        runs.append(ns)
        finals = [int(buf[t * 8]) for t in range(NUM_THREADS)]
    med = median(runs)
    out["private"] = {"runs_ns": runs, "median_ns": med,
                      "ns_per_update": med / ITERATIONS_PER_THREAD, "final_values": finals}
    print(f"private  : median {med/1e9:8.3f} s  {med/ITERATIONS_PER_THREAD:7.3f} ns/update")
    return out


# ==============================================================================
# Q5 helper: pure-Python bytecode threads (GIL) 1 thread vs 4 threads, stride=1
# ==============================================================================
def run_gil_experiment(seed: int) -> dict:
    py_kernel = worker_strided_accumulate.py_func     # plain interpreted bytecode
    useed = np.uint64(seed)
    res = {}
    for nthreads in (1, NUM_THREADS):
        buf = aligned_buffer(64)
        threads = [threading.Thread(target=py_kernel, args=(buf, t, PURE_PY_ITERS, useed))
                   for t in range(nthreads)]
        t0 = time.perf_counter_ns()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        res[f"threads_{nthreads}_ns"] = time.perf_counter_ns() - t0
    res["iters_per_thread"] = PURE_PY_ITERS
    res["slowdown_4_vs_1"] = res[f"threads_{NUM_THREADS}_ns"] / res["threads_1_ns"]
    return res


# ==============================================================================
# TASK 5: Parity + LLVM IR verification + manifest
# ==============================================================================
def audit_llvm_and_parity(ctx: dict, bench: dict, gil: dict) -> dict:
    inc = int(np.uint64(ctx["instance_seed"]) % np.uint64(7)) + 1
    expected = ITERATIONS_PER_THREAD * inc

    all_vals = [bench["strides"][s]["final_values"] for s in STRIDES] + [bench["private"]["final_values"]]
    parity = all(v == [expected] * NUM_THREADS for v in all_vals)

    irs = worker_strided_accumulate.inspect_llvm()
    full_ir = "\n".join(irs.values())
    # keep only the native kernel body (drop the cpython:: arg-unpacking wrapper,
    # which legitimately calls Py* API, and the cfunc/NRT helpers)
    blocks = ("\n" + full_ir).split("\ndefine ")[1:]
    ir = "\n".join(b for b in blocks
                   if "@_ZN8__main__" in b.split("\n", 1)[0]
                   and "cpython" not in b.split("\n", 1)[0]) or full_ir
    stores_i64 = ir.count("store i64")
    loads_i64 = ir.count("load i64")
    py_fallback = any(tok in ir for tok in ("PyObject_", "PyNumber_", "PyEval_"))
    llvm_ok = stores_i64 > 0 and not py_fallback

    t = {s: bench["strides"][s]["median_ns"] for s in STRIDES}
    verification = parity and llvm_ok
    manifest = {
        "git_username": ctx["git_username"],
        "instance_seed": ctx["instance_seed"],
        "cpu_signature": ctx["cpu_signature"],
        "logical_cores": ctx["logical_cores"],
        "cache_line_size": ctx["cache_line_size"],
        "affinity_cores": ctx["affinity_cores"],
        "affinity_applied": ctx["affinity_supported"],
        "python": ctx["python"],
        "os": ctx["os"],
        "num_threads": NUM_THREADS,
        "iterations_per_thread": ITERATIONS_PER_THREAD,
        "increment": inc,
        "expected_final_value": expected,
        "results": {f"stride_{s}": bench["strides"][s] for s in STRIDES},
        "private_baseline": bench["private"],
        "slowdown_stride1_vs_stride8": t[1] / t[8],
        "slowdown_stride8_vs_stride16": t[8] / t[16],
        "gil_experiment": gil,
        "llvm": {"store_i64_count": stores_i64, "load_i64_count": loads_i64,
                 "python_object_fallback": py_fallback},
        "parity_passed": parity,
        "llvm_passed": llvm_ok,
        "verification_passed": verification,
    }
    with open("cache_lab_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ctx = init_system_context(GIT_USERNAME)
    print(json.dumps({k: ctx[k] for k in ("logical_cores", "cache_line_size", "affinity_cores", "affinity_supported")}))
    if not ctx["affinity_supported"]:
        print("WARNING: core pinning unavailable; results may be noisier.")
    if resource is None and psutil is None:
        print("NOTE: no 'resource' module and no psutil -> context switches not recorded (pip install psutil).")

    # JIT warm-up so compile time is not measured
    w = aligned_buffer(64)
    s = np.uint64(ctx["instance_seed"])
    for k in (worker_strided_accumulate, worker_private_accumulate, worker_strided_gil):
        k(w, 0, 10, s)

    bench = run_stride_benchmark(ctx)
    gil = run_gil_experiment(ctx["instance_seed"])
    print(f"pure-Python GIL threads: 1 thr {gil['threads_1_ns']/1e9:.3f} s, "
          f"{NUM_THREADS} thr {gil['threads_4_ns']/1e9:.3f} s (x{gil['slowdown_4_vs_1']:.2f})")

    m = audit_llvm_and_parity(ctx, bench, gil)
    print(f"stride1/stride8 slowdown: x{m['slowdown_stride1_vs_stride8']:.2f}")
    print(f"parity={m['parity_passed']} llvm={m['llvm_passed']} verification_passed={m['verification_passed']}")


if __name__ == "__main__":
    main()
