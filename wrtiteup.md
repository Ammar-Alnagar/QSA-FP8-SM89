# Sparse Paged GQA FP8 Kernels on SM89

**Author:** Ammar Alnagar

**Target used for the reported run:** NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9 (Ada Lovelace), 8 GiB VRAM

**Original run date:** October 21, 2026 (historical 64-request workload)

**Current test workload:** 256 requests, 32 query heads, 8 KV heads, head dimension 128, Top-K 512, BF16 query/gate/output, FP8 E4M3FN K/V cache, page size 16, and 32 pages per request.

This report describes the implementations and the larger test shape now configured in the sources. A focused Nsight Compute capture was collected on the RTX 4060 Laptop GPU for the optimized CuTe kernel at the 256-request shape. The NCU duration is profiler-instrumented and is not a normal benchmark latency. No new baseline, CUDA, or Mojo profiles have been collected at this shape. The older timing figures below are retained only as historical results from the 64-request shape.

## Implementations

- [`sm89_cutedsl.py`](sm89_cutedsl.py): optimized CuTe DSL implementation. One 128-thread CTA handles one request and one KV head; each warp handles one of the four query heads sharing that KV head.
- [`backup.py`](backup.py): CuTe DSL comparison kernel. One CTA handles one request and one query head, and its four warps cooperate on a 128-element dot product using shared memory and block barriers.
- [`sm89_cuda.cu`](sm89_cuda.cu): standalone CUDA C++ implementation with the same request/KV-head CTA mapping as optimized CuTe. It uses inline PTX for FP8-to-FP16 conversion and CUDA warp shuffles.
- [`sm89_mojo.mojo`](sm89_mojo.mojo): MAX GPU implementation with the same request/KV-head CTA mapping. Its loop tile is now 32 tokens, matching the other implementations. The source masks invalid tokens out of the online-softmax denominator and no longer reports hard-coded correctness values.

All four implement the same intended operation for valid inputs: sparse scaled dot-product attention over the selected logical token indices, followed by a sigmoid gate. The two CuTe scripts construct randomized BF16/FP8 tensors and compare their outputs to a PyTorch reference. The CUDA and Mojo paths currently use constant-filled data and do not calculate a reference comparison; both now label validation as not checked.

## Workload and math

The current synthetic setup uses 256 requests × 32 pages/request × 16 tokens/page, with 8 KV heads and 128 values per head. Since 32 pages cover Top-K 512, this removes the prior allocation of 512 pages per request while preserving the selected-token workload. The physical K cache and V cache each contain:

`256 × 32 × 16 × 8 × 128 = 134,217,728` FP8 bytes (128 MiB).

Together they occupy 256 MiB, before query, output, metadata, allocator overhead, and temporary allocations. The selected Top-K is 512 per request. Indices select the first 512 logical tokens and the block table maps each request to its own contiguous physical pages.

The grouped CuTe/CUDA/Mojo launch has 256 × 8 = 2,048 CTAs; the baseline has 256 × 32 = 8,192. These provide substantially more independent blocks than the previous 512/2,048-block setup. CTA count alone does not prove full occupancy: achieved occupancy and utilization require a live GPU profile.

For request `r`, query head `h`, and its KV head `g = floor(h / 4)`, the kernel computes scores from `Q[r,h] · (K[token,g] × k_scale) / sqrt(128)`, normalizes across the selected tokens, accumulates the corresponding scaled values, and multiplies each output element by `sigmoid(gate[r,h])`. The query/key and value reductions use float32 accumulators; output is stored as BF16.

The optimized CuTe and CUDA kernels fold `k_scale / sqrt(D) × log2(e)` into the score scale so that softmax exponentials use base 2. They accumulate unscaled V values and apply `v_scale` once in the final normalization. The baseline multiplies `v_scale` into each loaded V value. These transformations are algebraically equivalent for finite inputs, subject to floating-point rounding.

The nominal operation count used by the scripts is `4 × B × TopK × Hq × D` FLOPs: multiply-add work for QK and probability-weighted V. This is 2,147,483,648 FLOPs for the current 256-request shape and 536,870,912 FLOPs for the historical 64-request shape. TFLOP/s values use this convention and are derived rates, not hardware counters.

## Kernel organization in the source

### Optimized CuTe DSL and CUDA

Both use a grid of `[256, 8, 1]` and 128 threads per CTA. Four warps correspond to the four query heads in a GQA group. Each lane owns four adjacent dimensions (`4 × lane + {0,1,2,3}`), so one warp covers all 128 dimensions of one query head. It computes a four-element partial dot product and reduces it across 32 lanes with five butterfly shuffles. This avoids cross-warp reduction and shared memory for QK.

The token loop resolves a logical token into page and offset, looks up the physical page in the block table, loads K and V, updates the online softmax maximum/normalizer, and updates four per-lane output accumulators. The CUDA implementation explicitly packs four FP8 bytes into a 32-bit load and uses `cvt.rn.f16x2.e4m3x2`; the CuTe implementation indexes FP8 values through CUTLASS tensor wrappers. The source establishes these load/conversion choices; it does not by itself establish a particular cycle count or prove that a given instruction is the performance bottleneck.

### Baseline CuTe DSL

The baseline launch grid is `[256, 32, 1]`, with one CTA per query head. The four warps split the 128 dimensions and use shared memory plus block synchronization to combine their partial dot products. This repeats K/V work for query heads that share a KV head and launches four times as many CTAs as the grouped mapping (8,192 instead of 2,048).

### Mojo / MAX

Mojo also launches `[256, 8, 1]` with 128 threads per CTA and maps a warp to a query head. Its source uses `BLOCK_N = 32`, `unsafe_load[width=4]` for vector elements, and a compile-time five-step shuffle reduction. Invalid indices are skipped in the online-softmax state. The benchmark uses `perf_counter_ns` around 100 enqueues followed by `ctx.synchronize()`, so its reported interval includes host/runtime enqueue cost as well as GPU execution. It is labeled enqueue-plus-sync wall time and is not directly comparable to CUDA-event kernel timing. Mojo still has no reference comparison, so its output is marked not checked.

## Current 256-request benchmark run

The following run was collected on the RTX 4060 Laptop GPU on October 22, 2026. CuTe and CUDA use CUDA events; Mojo reports host wall time around enqueueing and synchronization. The E2E columns are the values printed by each script and do not represent a common timing interval.

| Implementation | Kernel / timed interval | E2E reported | Derived rate | Validation |
|---|---:|---:|---:|---|
| Optimized CuTe DSL | 4.0915 ms | 25.988 ms | 0.525 TFLOP/s | max error 0.000004, PASS |
| Baseline CuTe DSL | 13.8669 ms | 39.289 ms | 0.155 TFLOP/s | max error 0.00000, PASS |
| Raw CUDA C++ | 3.8575 ms | 3.858 ms | 0.557 TFLOP/s | not checked |
| Mojo / MAX | 15.099 ms enqueue+sync | — | 0.142 TFLOP/s* | not checked |

Optimized CuTe is **3.39× faster** than the baseline by the CUDA-event timing in this run (`13.8669 / 4.0915`). CUDA's reported interval is 1.06× faster than optimized CuTe, but its output is not reference-validated. Mojo's displayed rate divides the nominal FLOP count by a host interval that includes enqueue overhead, so it is not a device throughput figure and should not be ranked with the CUDA-event rates.

`*` The Mojo interval is the mean host enqueue-plus-synchronization time across 100 submissions. It is retained as printed for completeness, not as a GPU-only rate.

## Historical 64-request benchmark run

The following values are transcribed from the earlier 64-request run. They do not measure the current 256-request configuration. CuTe and CUDA report CUDA-event kernel timing; their E2E columns are the scripts' separate end-to-end measurements. The old Mojo timing is host wall-clock time around enqueueing and synchronization.

| Implementation | Kernel / timed interval | E2E reported | Derived throughput | Printed validation |
|---|---:|---:|---:|---|
| Optimized CuTe DSL | 0.7933 ms (793.3 µs) | 59.512 ms | 0.677 TFLOP/s | max error 0.000004, PASS |
| Baseline CuTe DSL | 2.5078 ms (2507.8 µs) | 61.272 ms | 0.214 TFLOP/s | max error 0.00000, PASS |
| Raw CUDA C++ | 1.0042 ms (1004.2 µs) | 1.004 ms | 0.535 TFLOP/s | max error 0.000004, PASS |
| Mojo / MAX | 2.5402 ms (2540.2 µs) | 2.5402 ms | 0.211 TFLOP/s | old source printed PASS (not verified)* |

`*` The historical Mojo source printed validation values without calculating them. The current source marks validation as not checked. Its timing includes runtime enqueue overhead, so the table does not support a direct pure-kernel speed ranking for Mojo.

For that old shape, optimized CuTe was about **3.16× faster** than the baseline (`2.5078 / 0.7933`). Raw CUDA was about **2.50× faster** than baseline and about **1.27× slower** than optimized CuTe in that run. These ratios do not establish performance at 256 requests.

The historical `nvidia-smi` snapshot reported driver 595.91.07, CUDA 13.2, GPU memory capacity 8,188 MiB, and 76 MiB in use after those scripts exited. This was a post-run idle snapshot, not a measurement of each process's peak allocation.

## Current Nsight Compute profile

The optimized CuTe kernel was captured on October 22, 2026 with Nsight Compute 2026.1.1, using the `basic` set and filtering on `qsa_sm89_kernel`. The report is [`qsa_sm89_cutedsl_256.ncu-rep`](qsa_sm89_cutedsl_256.ncu-rep). It records grid `(256, 8, 1)` (2,048 CTAs), block size 128, 24 SMs, 38 registers/thread, and 7.11 waves/SM. NCU reports 100% theoretical occupancy and 95.15% achieved occupancy (45.67 active warps/SM). This confirms that the larger test supplies enough work to keep the GPU highly occupied for this kernel.

NCU reports 4.01 ms duration for the profiled launch, 75.2% compute throughput, 69.2% memory throughput, and 27.2% DRAM throughput. Treat the duration as profiler-instrumented; NCU replays the kernel to collect counters, so it is not comparable to CUDA-event timing outside the profiler. The values establish occupancy for optimized CuTe only; they do not establish a performance speedup over the old shape or explain a bottleneck by themselves.

## Historical saved Nsight Compute captures

The three original `.ncu-rep` files in the repository are older captures of the 64-request shape. Reading their recorded metrics with Nsight Compute gives:

| Capture | Grid | Duration | L1/TEX hit rate | Registers/thread | Dynamic shared memory/block |
|---|---:|---:|---:|---:|---:|
| `qsa_sm89_baseline.ncu-rep` | 64 × 32 CTAs | 3.41 ms | 49.79% | 40 | 16 B |
| `qsa_sm89_cutedsl.ncu-rep` | 64 × 8 CTAs | 1.12 ms | 93.90% | 38 | 0 B |
| `qsa_sm89_raw_cuda.ncu-rep` | 64 × 8 CTAs | 1.00 ms | 79.29% | 34 | 0 B |

These reports support the observed differences in CTA mapping, register count, shared-memory allocation, and the recorded L1/TEX hit rates for those specific captures. They do not include a Mojo capture. The reports also show equal grid/block dimensions for optimized CuTe and CUDA, and an L1/TEX hit rate that differs between those captures; therefore the hit rate should not be presented as identical across implementations or used alone to explain runtime. These artifacts are not profiles of the current 256-request shape.

The older values previously presented as Mojo NCU metrics, including estimated hit rates, bandwidth, register counts, barrier counts, and IPC, are not backed by a Mojo Nsight report in this repository and have been removed. Likewise, claims about exact cycle savings, bus burst utilization, compiler scheduling superiority, and causal attribution of the speed difference are not established by the available source and reports.

## Benchmark interpretation and limits

- In the historical run, the CuTe event times are the closest pure-device measurements in the supplied log. The separately reported CuTe E2E times are much larger, indicating substantial host/runtime work in that path; they should not be conflated with kernel execution time.
- CUDA's E2E value in that historical run matches its kernel value at the displayed precision. This does not mean the other scripts measure the same timing interval.
- Mojo's wall-clock interval includes repeated `enqueue_function` submissions and a final synchronization. A CUDA-event or Nsight capture would be needed to report its isolated GPU duration.
- The scripts use synthetic data and a fixed Top-K selection. They demonstrate this workload and implementation, not end-to-end model serving performance or an adaptive QSA selection strategy.
- The displayed latency and derived throughput are measurements from the supplied run and hardware/software state; they are not guaranteed results on another machine or run.

## Reproduction commands

```bash
python3 sm89_cutedsl.py
python3 backup.py
cudarun sm89_cuda.cu
mojo sm89_mojo.mojo
```

Profile the optimized CuTe kernel after the GPU is visible to the process:

```bash
ncu --set basic --target-processes all .venv/bin/python sm89_cutedsl.py --profile
```

Recreate the focused CuTe capture with:

```bash
ncu --set basic --kernel-name-base function --kernel-name regex:qsa_sm89_kernel \
  --target-processes all --export qsa_sm89_cutedsl_256 \
  .venv/bin/python sm89_cutedsl.py --profile
```

Open either current or historical reports in Nsight Compute with `ncu-ui <report>.ncu-rep`. Only the new filtered report measures the 256-request optimized CuTe kernel; the other checked-in reports are historical 64-request captures.
