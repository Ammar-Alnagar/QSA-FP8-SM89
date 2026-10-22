# Sparse Paged GQA FP8 Kernels — SM89

This repository contains synthetic sparse Grouped-Query Attention (GQA) benchmarks for NVIDIA Ada Lovelace GPUs (compute capability 8.9). It includes optimized and baseline CuTe DSL kernels, a Triton implementation, a raw CUDA C++ kernel, and a Mojo / MAX implementation.

## Workload

- 256 requests, 32 query heads, 8 KV heads, and head dimension 128
- Top-K 512, page size 16, 32 physical pages per request
- FP8 E4M3FN K/V cache; BF16 query, gate, and output tensors
- Fused sigmoid gate and online softmax

Grouped kernels launch 2,048 CTAs at this size (one per request/KV-head); baseline CuTe and Triton launch 8,192 (one per request/query-head). The cache holds exactly the pages needed for Top-K. Current Nsight Compute captures report 57.81–98.86% achieved occupancy across the five kernels; old 64-request timing figures do not describe this configuration.

All five implementations report validation against either the shared PyTorch reference or an independent closed-form constant-input reference. The source also contains prior 64-request measurements and saved Nsight Compute captures. See [`wrtiteup.md`](wrtiteup.md) for their provenance and the current profiling status.

## Current run

Run `python3 run_all.py` for the current five-kernel comparison table. In the recorded run, CUDA-event speed order was Triton, raw CUDA, optimized CuTe, then baseline CuTe. CuTe and Triton use the shared randomized PyTorch reference; CUDA and Mojo use a constant-input closed-form check. Mojo timing includes enqueue plus synchronization, so it is not ranked against device-event timings. The write-up includes measured rates, logical K/V payload estimates, and the matching Nsight profiles with caveats in [`wrtiteup.md`](wrtiteup.md).

## Repository files

- `sm89_cutedsl.py` — optimized CuTe DSL kernel, one CTA per request/KV head
- `backup.py` — baseline CuTe kernel, one CTA per request/query head
- `sm89_cuda.cu` — CUDA C++ implementation
- `sm89_mojo.mojo` — Mojo / MAX implementation
- `sm89_triton.py` — Triton implementation with PyTorch reference validation
- `run_all.py` — runs all five implementations and prints a comparison table
- `qsa_sm89_*.ncu-rep` — saved Nsight Compute captures for the original CuTe/CUDA implementations
- `qsa_sm89_*_256.ncu-rep` — focused Nsight Compute profiles for all five kernels at the current 256-request shape
- [`wrtiteup.md`](wrtiteup.md) — implementation details, profiling notes, and measurement caveats

## Run

Run all implementations and print a comparison table:

```bash
python3 run_all.py
```

Or run individual implementations:

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

