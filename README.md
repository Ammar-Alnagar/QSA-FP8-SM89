# Sparse Paged GQA FP8 Kernels — SM89

This repository contains synthetic sparse Grouped-Query Attention (GQA) benchmarks for NVIDIA Ada Lovelace GPUs (compute capability 8.9). It includes optimized and baseline CuTe DSL kernels, a raw CUDA C++ kernel, and a Mojo / MAX implementation.

## Workload

- 256 requests, 32 query heads, 8 KV heads, and head dimension 128
- Top-K 512, page size 16, 32 physical pages per request
- FP8 E4M3FN K/V cache; BF16 query, gate, and output tensors
- Fused sigmoid gate and online softmax

The grouped kernels launch 2,048 CTAs at this size (one per request/KV-head); the baseline launches 8,192 (one per request/query-head). The cache now holds exactly the pages needed for Top-K instead of allocating 512 pages per request. An Nsight Compute profile of optimized CuTe measured 95.15% achieved occupancy at this shape; old 64-request timing figures do not describe this configuration.

The source still contains prior 64-request measurements and saved Nsight Compute captures. See [`wrtiteup.md`](wrtiteup.md) for their provenance and the current profiling status. CuTe performs a reference comparison; CUDA and Mojo report validation as not checked because those paths do not calculate a reference comparison.

## Current run

RTX 4060 Laptop GPU, October 22, 2026. The optimized CuTe kernel measured 4.0915 ms and passed validation; the baseline measured 13.8669 ms and passed. Raw CUDA measured 3.8575 ms but has no reference check. Mojo reported 15.099 ms for enqueue plus synchronization, not isolated GPU time. The focused Nsight capture measured 95.15% achieved occupancy for optimized CuTe. Full timing and profile details are in [`wrtiteup.md`](wrtiteup.md).

## Repository files

- `sm89_cutedsl.py` — optimized CuTe DSL kernel, one CTA per request/KV head
- `backup.py` — baseline CuTe kernel, one CTA per request/query head
- `sm89_cuda.cu` — CUDA C++ implementation
- `sm89_mojo.mojo` — Mojo / MAX implementation
- `qsa_sm89_*.ncu-rep` — saved Nsight Compute captures for baseline CuTe, optimized CuTe, and CUDA
- `qsa_sm89_cutedsl_256.ncu-rep` — focused Nsight Compute profile for optimized CuTe at the current 256-request shape
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

