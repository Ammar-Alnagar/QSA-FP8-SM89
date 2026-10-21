# Sparse Paged GQA FP8 Kernels — SM89

This repository contains synthetic sparse Grouped-Query Attention (GQA) benchmarks for NVIDIA Ada Lovelace GPUs (compute capability 8.9). It includes optimized and baseline CuTe DSL kernels, a raw CUDA C++ kernel, and a Mojo / MAX implementation.

## Workload

- 64 requests, 32 query heads, 8 KV heads, and head dimension 128
- Top-K 512, page size 16, FP8 E4M3FN K/V cache
- BF16 query, gate, and output tensors
- Fused sigmoid gate and online softmax

## Latest supplied results

Measurements below are from the RTX 4060 Laptop run supplied on October 21, 2026. CuTe and CUDA kernel durations use CUDA events. Mojo uses host wall-clock timing around enqueue calls and a final synchronization, so it is not a pure GPU duration.

| Implementation | Latency | Derived compute rate | Validation output |
|---|---:|---:|---|
| Baseline CuTe (`backup.py`) | 2.5078 ms | 0.214 TFLOP/s | PASS |
| **Optimized CuTe (`sm89_cutedsl.py`)** | **0.7933 ms** | **0.677 TFLOP/s** | **PASS** |
| Raw CUDA (`sm89_cuda.cu`) | 1.0042 ms | 0.535 TFLOP/s | PASS |
| Mojo / MAX (`sm89_mojo.mojo`) | 2.5402 ms* | 0.211 TFLOP/s* | Prints PASS* |

The optimized CuTe result is about 3.16× faster than the baseline in this run. The saved Nsight Compute reports are older and record 3.41 ms for baseline CuTe, 1.12 ms for optimized CuTe, and 1.00 ms for CUDA; these are kept separate from the latest run. The Mojo source currently prints hard-coded error/pass values rather than calculating a reference comparison.

## Repository files

- `sm89_cutedsl.py` — optimized CuTe DSL kernel, one CTA per request/KV head
- `backup.py` — baseline CuTe kernel, one CTA per request/query head
- `sm89_cuda.cu` — CUDA C++ implementation
- `sm89_mojo.mojo` — Mojo / MAX implementation
- `qsa_sm89_*.ncu-rep` — saved Nsight Compute captures for baseline CuTe, optimized CuTe, and CUDA
- [`wrtiteup.md`](wrtiteup.md) — implementation details, profiling notes, and measurement caveats

## Run

```bash
python3 sm89_cutedsl.py
python3 backup.py
cudarun sm89_cuda.cu
mojo sm89_mojo.mojo
```

The CUDA command above uses the local `cudarun` helper. To compile directly with NVCC, use:

```bash
nvcc -O3 -arch=sm_89 -lineinfo sm89_cuda.cu -o sm89_cuda
./sm89_cuda
```

## Kernel organization

The optimized CuTe and CUDA versions launch one 128-thread CTA per `(request, KV head)`. Its four warps each handle one query head in that GQA group, and each warp reduces a 128-dimensional dot product with five butterfly shuffles. The baseline instead launches one CTA per query head and uses shared-memory cross-warp reduction. The Mojo version uses the grouped launch mapping but has its own host-timed benchmark path.

