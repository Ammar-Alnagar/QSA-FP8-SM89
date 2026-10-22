# Sparse Paged GQA FP8 Kernels — SM89

This repository contains synthetic sparse Grouped-Query Attention (GQA) benchmarks for NVIDIA Ada Lovelace GPUs (compute capability 8.9). It includes optimized and baseline CuTe DSL kernels, a raw CUDA C++ kernel, and a Mojo / MAX implementation.

## Workload

- 256 requests, 32 query heads, 8 KV heads, and head dimension 128
- Top-K 512, page size 16, 32 physical pages per request
- FP8 E4M3FN K/V cache; BF16 query, gate, and output tensors
- Fused sigmoid gate and online softmax

The grouped kernels launch 2,048 CTAs at this size (one per request/KV-head); the baseline launches 8,192 (one per request/query-head). The cache now holds exactly the pages needed for Top-K instead of allocating 512 pages per request. The old 64-request timing figures below describe a different workload and are not results for this configuration. A live profile could not be collected in the current environment because CUDA reports no available GPU.

The source still contains prior 64-request measurements and saved Nsight Compute captures. See [`wrtiteup.md`](wrtiteup.md) for their provenance and the current profiling status. The Mojo source currently prints hard-coded error/pass values rather than calculating a reference comparison.

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

