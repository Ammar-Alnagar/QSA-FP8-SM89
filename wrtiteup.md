# Sparse Paged GQA FP8 Kernels on SM89

**Author:** Ammar Alnagar

**Target used for the reported run:** NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9 (Ada Lovelace), 8 GiB VRAM

**Current test workload:** 256 requests, 32 query heads, 8 KV heads, head dimension 128, Top-K 512, BF16 query/gate/output, FP8 E4M3FN K/V cache, page size 16, and 32 pages per request.

This report describes the implementations and the larger test shape now configured in the sources. Focused Nsight Compute captures were collected on the RTX 4060 Laptop GPU for all five kernels at the 256-request shape. NCU durations are profiler-instrumented and are not normal benchmark latencies. The older timing figures below are retained only as historical results from the 64-request shape.

## Implementations

- [`sm89_cutedsl.py`](sm89_cutedsl.py): optimized CuTe DSL implementation. One 128-thread CTA handles one request and one KV head; each warp handles one of the four query heads sharing that KV head.
- [`backup.py`](backup.py): CuTe DSL comparison kernel. One CTA handles one request and one query head, and its four warps cooperate on a 128-element dot product using shared memory and block barriers.
- [`sm89_triton.py`](sm89_triton.py): Triton implementation, one program per request/query head, with a 32-token loop tile and online softmax.
- [`sm89_cuda.cu`](sm89_cuda.cu): standalone CUDA C++ implementation with the same request/KV-head CTA mapping as optimized CuTe. It uses inline PTX for FP8-to-FP16 conversion and CUDA warp shuffles.
- [`sm89_mojo.mojo`](sm89_mojo.mojo): MAX GPU implementation with the same request/KV-head CTA mapping. Its loop tile is now 32 tokens, matching the other implementations. The source masks invalid tokens out of the online-softmax denominator and no longer reports hard-coded correctness values.

All five implement the same intended operation for valid inputs: sparse scaled dot-product attention over selected logical token indices, followed by a sigmoid gate. The CuTe and Triton scripts use the same randomized BF16/FP8 inputs and compare against the shared PyTorch reference. CUDA and Mojo use constant fixtures and check every output against the closed-form BF16(0.05) result; these checks exercise the output path but do not cover randomized inputs.

## Workload and math

The current synthetic setup uses 256 requests × 32 pages/request × 16 tokens/page, with 8 KV heads and 128 values per head. Since 32 pages cover Top-K 512, this removes the prior allocation of 512 pages per request while preserving the selected-token workload. The physical K cache and V cache each contain:

`256 × 32 × 16 × 8 × 128 = 134,217,728` FP8 bytes (128 MiB).

Together they occupy 256 MiB, before query, output, metadata, allocator overhead, and temporary allocations. The selected Top-K is 512 per request. Indices select the first 512 logical tokens and the block table maps each request to its own contiguous physical pages.

The grouped CuTe/CUDA/Mojo launch has 256 × 8 = 2,048 CTAs. Baseline CuTe and Triton each launch 256 × 32 = 8,192 CTAs. These provide substantially more independent blocks than the previous 512/2,048-block setup.

For request `r`, query head `h`, and its KV head `g = floor(h / 4)`, the kernel computes scores from `Q[r,h] · (K[token,g] × k_scale) / sqrt(128)`, normalizes across the selected tokens, accumulates the corresponding scaled values, and multiplies each output element by `sigmoid(gate[r,h])`. The query/key and value reductions use float32 accumulators; output is stored as BF16.

The optimized CuTe and CUDA kernels fold `k_scale / sqrt(D) × log2(e)` into the score scale so that softmax exponentials use base 2. They accumulate unscaled V values and apply `v_scale` once in the final normalization. The baseline multiplies `v_scale` into each loaded V value. These transformations are algebraically equivalent for finite inputs, subject to floating-point rounding.

The nominal operation count used by the scripts is `4 × B × TopK × Hq × D` FLOPs: multiply-add work for QK and probability-weighted V. This is 2,147,483,648 FLOPs for the current 256-request shape and 536,870,912 FLOPs for the historical 64-request shape. TFLOP/s values use this convention and are derived rates, not hardware counters.

## Kernel organization in the source

### Optimized CuTe DSL and CUDA

Both use a grid of `[256, 8, 1]` and 128 threads per CTA. Four warps correspond to the four query heads in a GQA group. Each lane owns four adjacent dimensions (`4 × lane + {0,1,2,3}`), so one warp covers all 128 dimensions of one query head. It computes a four-element partial dot product and reduces it across 32 lanes with five butterfly shuffles. This avoids cross-warp reduction and shared memory for QK.

The token loop resolves a logical token into page and offset, looks up the physical page in the block table, loads K and V, updates the online softmax maximum/normalizer, and updates four per-lane output accumulators. The CUDA implementation explicitly packs four FP8 bytes into a 32-bit load and uses `cvt.rn.f16x2.e4m3x2`; the CuTe implementation indexes FP8 values through CUTLASS tensor wrappers. The source establishes these load/conversion choices; it does not by itself establish a particular cycle count or prove that a given instruction is the performance bottleneck.

### Baseline CuTe DSL

The baseline launch grid is `[256, 32, 1]`, with one CTA per query head. The four warps split the 128 dimensions and use shared memory plus block synchronization to combine their partial dot products. This repeats K/V work for query heads that share a KV head and launches four times as many CTAs as the grouped mapping (8,192 instead of 2,048).

### Triton

Triton launches one program per `(request, query head)` (8,192 programs). Each program processes Top-K in 32-token blocks, loads that head's query and the corresponding KV head's paged K/V rows, and maintains online-softmax state before writing its output row. It shares the CuTe test input and PyTorch reference, but repeats K/V loads across the four query heads in each GQA group.

### Mojo / MAX

Mojo also launches `[256, 8, 1]` with 128 threads per CTA and maps a warp to a query head. Its source uses `BLOCK_N = 32`, `unsafe_load[width=4]` for vector elements, and a compile-time five-step shuffle reduction. Invalid indices are skipped in the online-softmax state. The benchmark uses `perf_counter_ns` around 100 enqueues followed by `ctx.synchronize()`, so its reported interval includes host/runtime enqueue cost as well as GPU execution. It is labeled enqueue-plus-sync wall time and is not directly comparable to CUDA-event kernel timing. Its constant-input output is checked against BF16(0.05).

## Current 256-request benchmark run

The following run was collected with `python3 run_all.py` on the RTX 4060 Laptop GPU. CuTe, Triton, and CUDA use CUDA events; Mojo reports host wall time around enqueueing and synchronization. The E2E columns are the values printed by each script and do not represent a common timing interval.

| Speed rank | Implementation | Kernel / timed interval | E2E reported | Derived rate | Validation |
|---:|---|---:|---:|---:|---|
| 1 | Triton | 2.5130 ms | — | 0.855 TFLOP/s | max error 0.00000381, PASS |
| 2 | Raw CUDA C++ | 3.0692 ms | 3.069 ms | 0.700 TFLOP/s | max error 0.00000000, PASS† |
| 3 | Optimized CuTe DSL | 4.2694 ms | 26.104 ms | 0.503 TFLOP/s | max error 0.000004, PASS |
| 4 | Baseline CuTe DSL | 26.8541 ms | 41.601 ms | 0.080 TFLOP/s | max error 0.00000, PASS |
| — | Mojo / MAX | 16.840 ms enqueue+sync | — | 0.127 TFLOP/s* | max error 0.00004883, PASS† |

In this run, Triton had the lowest CUDA-event latency. Its test uses the same randomized data and PyTorch reference as CuTe. CUDA and Mojo passed their constant-fixture checks, which are narrower than the randomized reference comparison. Timings vary across runs; the table records one execution of the all-kernels command.

`*` Mojo's reported rate divides the nominal FLOP count by host enqueue-plus-synchronization time, so it is not a GPU-only throughput and should not be ranked with CUDA-event rates. `†` CUDA/Mojo checks compare all output elements to the analytically expected BF16(0.05) output for their constant fixtures.

### Speed order and data movement

For comparable CUDA-event kernel times, the speed order is **Triton → raw CUDA → optimized CuTe → baseline CuTe**. The derived rate column follows the same order: **0.855 → 0.700 → 0.503 → 0.080 TFLOP/s**. Relative to optimized CuTe, this run measured Triton at 1.70× faster, raw CUDA at 1.39× faster, and baseline CuTe at 6.29× slower. These are single-run results; small timing changes can shift close rankings. Mojo is excluded because its enqueue-plus-sync wall interval includes host/runtime work.

The K/V cache allocation is **256 MiB** (128 MiB each for K and V). Based on the launch mapping and Top-K, the grouped implementations (optimized CuTe, CUDA, Mojo) request about **256 MiB of K/V payload per invocation**: each of 2,048 request/KV-head CTAs consumes 128 KiB. Baseline CuTe and Triton request about **1 GiB logically** because 8,192 request/query-head programs each load the same 128 KiB K/V slice, repeating work across the four query heads sharing a KV head. These figures are logical requested payload from the kernel mapping, not measured DRAM bytes; cache reuse can lower physical traffic. Q and output each occupy about 2 MiB, while gate and index/block-table metadata are smaller.

The RTX 4060 Laptop GPU has 24 SMs. All kernels launch enough CTAs for multiple waves, but their mapping and generated work differ. Triton leads this run despite the repeated logical K/V requests: its Nsight capture reports 86.31% SM throughput and 41.35% DRAM throughput, with 72 registers/thread and 57.81% achieved occupancy. This indicates strong utilization for this implementation, but does not isolate a single cause; cache reuse means the 1 GiB logical request estimate does not imply 1 GiB from DRAM. Raw CUDA and optimized CuTe use the grouped mapping and avoid the fourfold logical K/V repetition. CUDA measured faster here; its capture has 34 registers/thread and 94.80% occupancy versus CuTe's 38 registers and 95.15%, but these counters alone do not establish why. The baseline's per-query-head CTA mapping repeats K/V work and adds shared-memory cross-warp reduction and barriers; it is slowest in this run even though its achieved occupancy is high. Mojo has grouped mapping and similar occupancy, but its displayed host timing cannot support a GPU-kernel speed ranking.

## Historical 64-request benchmark run

The following values are transcribed from the earlier 64-request run. They do not measure the current 256-request configuration. CuTe and CUDA report CUDA-event kernel timing; their E2E columns are the scripts' separate end-to-end measurements. The old Mojo timing is host wall-clock time around enqueueing and synchronization.

| Implementation | Kernel / timed interval | E2E reported | Derived throughput | Printed validation |
|---|---:|---:|---:|---|
| Optimized CuTe DSL | 0.7933 ms (793.3 µs) | 59.512 ms | 0.677 TFLOP/s | max error 0.000004, PASS |
| Baseline CuTe DSL | 2.5078 ms (2507.8 µs) | 61.272 ms | 0.214 TFLOP/s | max error 0.00000, PASS |
| Raw CUDA C++ | 1.0042 ms (1004.2 µs) | 1.004 ms | 0.535 TFLOP/s | max error 0.000004, PASS |
| Mojo / MAX | 2.5402 ms (2540.2 µs) | 2.5402 ms | 0.211 TFLOP/s | old source printed PASS (not verified)* |

`*` The historical Mojo source printed validation values without calculating them; validation was not checked for that old result. The current source checks every output element against the closed-form fixture. The historical timing includes runtime enqueue overhead, so the table does not support a direct pure-kernel speed ranking for Mojo.

For that old shape, optimized CuTe was about **3.16× faster** than the baseline (`2.5078 / 0.7933`). Raw CUDA was about **2.50× faster** than baseline and about **1.27× slower** than optimized CuTe in that run. These ratios do not establish performance at 256 requests.

The historical `nvidia-smi` snapshot reported driver 595.91.07, CUDA 13.2, GPU memory capacity 8,188 MiB, and 76 MiB in use after those scripts exited. This was a post-run idle snapshot, not a measurement of each process's peak allocation.

## Current Nsight Compute profiles

All five kernels were captured with Nsight Compute and the `basic` metric set. The matching reports are checked in as `qsa_sm89_{cutedsl,baseline,raw_cuda,mojo,triton}_256.ncu-rep`. The workload and launch shapes were the same as in the timing table above.

| Kernel | Grid CTAs | NCU duration* | Achieved occupancy | Registers/thread | Waves/SM | SM throughput | Memory throughput | DRAM throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Optimized CuTe | 2,048 | 4.013 ms | 95.15% | 38 | 7.11 | 75.19% | 69.20% | 27.23% |
| Baseline CuTe | 8,192 | 13.725 ms | 98.86% | 40 | 28.44 | 70.13% | 70.13% | 15.46% |
| Raw CUDA | 2,048 | 3.548 ms | 94.80% | 34 | 7.11 | 78.72% | 46.99% | 30.97% |
| Mojo / MAX | 2,048 | 13.824 ms | 94.01% | 38 | 7.11 | 89.96% | 20.10% | 8.05% |
| Triton | 8,192 | 2.651 ms | 57.81% | 72 | 48.76 | 86.31% | 86.31% | 41.35% |

The device has 24 SMs and all five kernels launch enough blocks for multiple waves. Achieved occupancy ranges from 57.81% to 98.86%; occupancy alone does not predict latency. Triton has the lowest reported occupancy but the highest CUDA-event rate in this run, while grouped kernels reduce logical K/V requests. Cache behavior, instruction mix, reduction strategy, and launch overhead also affect results.

`*` NCU replays launches while collecting metrics; these instrumented durations are not directly comparable to the CUDA-event timings in the benchmark table. Throughput percentages are NCU's peak-relative counters, not percentages of application-level useful work.

## Historical saved Nsight Compute captures

The three original `.ncu-rep` files in the repository are older captures of the 64-request shape. Reading their recorded metrics with Nsight Compute gives:

| Capture | Grid | Duration | L1/TEX hit rate | Registers/thread | Dynamic shared memory/block |
|---|---:|---:|---:|---:|---:|
| `qsa_sm89_baseline.ncu-rep` | 64 × 32 CTAs | 3.41 ms | 49.79% | 40 | 16 B |
| `qsa_sm89_cutedsl.ncu-rep` | 64 × 8 CTAs | 1.12 ms | 93.90% | 38 | 0 B |
| `qsa_sm89_raw_cuda.ncu-rep` | 64 × 8 CTAs | 1.00 ms | 79.29% | 34 | 0 B |

These historical reports support the observed differences in CTA mapping, register count, shared-memory allocation, and the recorded L1/TEX hit rates for those specific captures. They do not include a Mojo capture. The reports also show equal grid/block dimensions for optimized CuTe and CUDA, and an L1/TEX hit rate that differs between those captures; therefore the hit rate should not be presented as identical across implementations or used alone to explain runtime. These artifacts are not profiles of the current 256-request shape.

The older values previously presented as Mojo NCU metrics, including estimated hit rates, bandwidth, register counts, barrier counts, and IPC, are not backed by a Mojo Nsight report in this repository and have been removed. Likewise, claims about exact cycle savings, bus burst utilization, compiler scheduling superiority, and causal attribution of the speed difference are not established by the available source and reports.

## Benchmark interpretation and limits

- In the historical run, the CuTe event times are the closest pure-device measurements in the supplied log. The separately reported CuTe E2E times are much larger, indicating substantial host/runtime work in that path; they should not be conflated with kernel execution time.
- CUDA's E2E value in that historical run matches its kernel value at the displayed precision. This does not mean the other scripts measure the same timing interval.
- Mojo's benchmark interval includes repeated `enqueue_function` submissions and a final synchronization. The Mojo Nsight report provides hardware counters, but its replay duration is instrumented and is not directly comparable to CUDA-event timings.
- The scripts use synthetic data and a fixed Top-K selection. They demonstrate this workload and implementation, not end-to-end model serving performance or an adaptive QSA selection strategy.
- The displayed latency and derived throughput are measurements from the supplied run and hardware/software state; they are not guaranteed results on another machine or run.

## Reproduction commands

Run all five implementations and print their comparison table:

```bash
python3 run_all.py
```

Run individual implementations if needed:

```bash
python3 sm89_cutedsl.py
python3 backup.py
cudarun sm89_cuda.cu
mojo sm89_mojo.mojo
```

Recreate the focused CuTe capture with:

```bash
ncu --set basic --kernel-name-base function --kernel-name regex:qsa_sm89_kernel \
  --target-processes all --export qsa_sm89_cutedsl_256 \
  .venv/bin/python sm89_cutedsl.py --profile
```

The current filtered captures are `qsa_sm89_baseline_256.ncu-rep`, `qsa_sm89_raw_cuda_256.ncu-rep`, `qsa_sm89_mojo_256.ncu-rep`, and `qsa_sm89_triton_256.ncu-rep`. Open current or historical reports in Nsight Compute with `ncu-ui <report>.ncu-rep`. The three original reports are historical 64-request captures.
