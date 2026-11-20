# Cache Line & False Sharing — Analysis

Host: Intel64 Family 6 Model 186 (GenuineIntel), 12 logical cores, Windows 11, Python 3.13 · cache line 64 B · 4 threads pinned to logical CPUs 0, 2, 4, 6
Git handle: `toleutaev13` · INSTANCE_SEED: `5580014552038820860` · manifest: `cache_lab_manifest.json` (`verification_passed: true`)
Method: 60,000,000 updates per thread, 4 threads, 3 runs per layout, median reported. All final counters equal 120,000,000 (increment 2 x 60M) in every layout and in the privatized baseline (parity passed).

## Q1. MESI transitions (stride = 1)

`buffer[0]` and `buffer[1]` are 8 bytes apart, so both live in the same 64-byte line L.

| Step | Event | Core 0 L1 (L) | Core 1 L1 (L) |
|---|---|---|---|
| 0 | L last written by Core 1 | I | M |
| 1 | Core 0 stores `buffer[0]`: miss, issues RFO | I (waiting) | snoop hit in M: line forwarded / written back, then I |
| 2 | RFO completes, store commits | **M** | I |
| 3 | Core 1 stores `buffer[1]`: miss, issues RFO | snoop hit in M: line forwarded, then I | I (waiting) |
| 4 | RFO completes, store commits | I | **M** |
| 5 | Core 0 stores again | repeat from step 1 | |

(If the line were also held by readers it would be S and the RFO would invalidate those copies, S to I; a line held cleanly by one core is E and becomes M silently on a store. With two cores constantly writing, the line is in M in exactly one core at any time and moves back and forth.)

Why the RFO stalls the store buffer: x86 is TSO, so stores must become globally visible in program order. The store at the head of Core 1's store buffer cannot commit to L1 until the core owns L in M or E. While the RFO round trip to the other core is outstanding, that store blocks, younger stores queue behind it, and the store buffer fills up. In this kernel the next iteration's load of the same line also misses, so each iteration of the load, add, store dependency chain waits for a cross-core line transfer.

## Q2. Quantitative slowdown

| Regime | Total time (ns, median of 3) | ns / update (time / 60,000,000) |
|---|---|---|
| stride = 1 | 425,910,300 | 7.099 |
| stride = 4 | 441,232,900 | 7.354 |
| stride = 8 | 261,394,400 | 4.357 |
| stride = 16 | 267,516,400 | 4.459 |
| register-privatized | 465,800 | 0.0078 |

- Slowdown of stride = 1 relative to stride = 8: 425,910,300 / 261,394,400 = **1.63x** (manifest: `slowdown_stride1_vs_stride8`).
- Coherence overhead per update: 7.099 - 4.357 = about 2.74 ns. Aggregate throughput: 240M updates / 0.426 s = about 563 M updates/s at stride 1 versus about 918 M/s at stride 8.
- ns/update is per thread (all 4 threads run concurrently).
- The slowdown is moderate because even without false sharing an update costs 4.36 ns: the loop is a memory dependency chain (load of `buffer[slot]`, load/store of `buffer[idx]` with store-to-load forwarding), so part of the coherence latency overlaps with this chain.
- stride = 4 (two threads per line) was not faster than stride = 1: 7.35 vs 7.10 ns, and the individual runs overlap (384-448 ms vs 421-521 ms). So going from 4-way to 2-way sharing did not help measurably here; my interpretation is that the cost is dominated by the line-transfer latency each thread waits for, not by the number of sharers.
- Privatized baseline: 0.0078 ns/update (about 560x faster than stride = 8). LLVM reduces `acc += inc` to a multiplication, so this is a lower bound that removes memory traffic completely. The gap between it and stride = 8 is the cost of the L1 load/store dependency (store forwarding); the gap between stride = 8 and stride = 1 is the cross-core coherence cost.

## Q3. Adjacent-line prefetcher (stride 8 vs 16)

Measured: stride = 8 gives 261,394,400 ns (4.357 ns/update), stride = 16 gives 267,516,400 ns (4.459 ns/update); ratio 16/8 = 1.02 (manifest `slowdown_stride8_vs_stride16` = 0.977). The difference is within run-to-run noise (stride 8 runs: 259-275 ms, stride 16 runs: 258-280 ms), so on this machine I see no measurable penalty from the 128-byte pairing.

Intel's L2 spatial (adjacent cache line) prefetcher treats memory as 128-byte sectors made of an even and an odd 64-byte line. When one line is fetched on demand, the hardware may also fetch the other half of the pair. With stride = 8 the thread slots are at 0, 64, 128, 192 B: threads 0 and 1 occupy the two halves of one sector, threads 2 and 3 the two halves of the next. Each core's request for its own line can therefore also pull in, or contend for, its neighbour's line, causing secondary invalidations even though no 64-byte line is shared. 64-byte padding still suffers when the prefetcher is enabled and neighbouring slots fall in the same 128-byte-aligned pair, when the buffer base is not 128-byte aligned (my buffer is only 64-byte aligned), or when the prefetch is triggered by load-heavy access patterns. stride = 16 spaces slots 128 B apart so each thread owns a separate sector. That I measured no difference means the effect is small or absent for this kernel and CPU (modern cores may also have the prefetcher disabled or throttled under store-heavy traffic).

## Q4. Register promotion vs. memory stores

An optimizing compiler (LLVM LICM / scalar promotion) can turn `for i: a[slot] += inc` into "load once, add in a register, store once", because nothing marks the access volatile or atomic and a single-threaded view of the program allows it. That would remove all memory traffic, and with it the false sharing. My privatized kernel shows exactly this: it ran in 0.0078 ns/update.

In `worker_strided_accumulate` the store address is `slot + (buffer[slot] >> 63)`. The compiler cannot prove that the top bit of the loaded value is 0, so each iteration's address depends on a value loaded from memory (which the previous iteration stored). The location may change, so the counter cannot be kept in a register and every iteration issues a real load and store.

IR evidence (from `inspect_llvm()`, native kernel body, loop): `<paste 4-6 lines from ir.txt: the load i64, lshr i64 ..., 63, getelementptr, load i64, add, store i64>`. Manifest counts for the kernel: `store_i64_count = 5`, `load_i64_count = 10`, no Python C-API calls in the kernel body (`python_object_fallback: false`), so the kernel is native i64 code with no Python object fallback.

## Q5. GIL vs. `nogil=True`

Pure-Python bytecode threads, same stride = 1 layout, 1,000,000 iterations per thread: 1 thread = 991,935,200 ns (about 992 ns/iteration), 4 threads = 4,069,729,800 ns (about 1017 ns/iteration of per-thread work serialized); 4 threads take 4.10x the time of 1 thread (manifest `gil_experiment`). For comparison, 4 `nogil` Numba threads doing 60M updates each finish in 0.426 s.

Under the GIL only one thread executes bytecode at any instant, so two cores never store to the same cache line simultaneously. There is no concurrent RFO ping-pong and therefore no false sharing: the 4x slowdown is just 4x the work executed serially, plus about 2.5% extra per iteration for GIL hand-off (switch interval 5 ms, condition-variable/futex wake-ups, convoy effects). That is software lock contention, handled by the OS scheduler. With `nogil=True` the threads really run in parallel on separate cores, and the slowdown is hardware cache-line contention (coherence traffic and RFO latency), which needs no lock or context switch.

Note on context switches: `context_switches_last_run` in the manifest (voluntary -3 to -8, involuntary 0) is not reliable. It comes from `psutil` on Windows, where the process counter can decrease between reads (threads exiting), so I do not use these numbers as evidence.
