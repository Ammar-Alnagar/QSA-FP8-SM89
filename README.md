
# QSA Sparse Paged GQA — SM89 CuTe DSL

High-performance sparse FP8 Grouped-Query Attention kernel for NVIDIA Ada Lovelace (SM89).

**Author**: Ammar Alnagar  
**Target Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (sm_89, 8 GB VRAM)  
**Software**: NVIDIA CuTe DSL (`nvidia-cutlass-dsl 4.8.1`), CUDA 13.4, PyTorch 2.x

---

## Overview

This repository contains a complete end-to-end implementation of **Quick Sparse Attention (QSA)** with:

- Grouped-Query Attention (GQA)
- Paged KV cache
- Top-K sparsity
- FP8-e4m3fn Key/Value cache
- Online softmax (Milakov-Gimelshteyn algorithm)
- Fused output gating

The kernel is written entirely in **NVIDIA CuTe DSL** and is optimized specifically for Ada Lovelace (SM89).

### Key Result

| Kernel              | Latency (64 req × 512 top-k) | Speedup |
|---------------------|------------------------------|---------|
| Baseline            | 3.41 ms                      | 1.00×   |
| **Optimized (this)**| **1.12 ms**                  | **3.04×** |

---

## Repository Structure

```text
.
├── sm89_cutedsl.py              # Optimized production kernel (GQA warp tiling)
├── backup.py                    # Baseline kernel (1 CTA per query head)
├── qsa_sm89_cutedsl.ncu-rep     # Nsight Compute report – optimized
├── qsa_sm89_baseline.ncu-rep    # Nsight Compute report – baseline
└── README.md
```

---

## Quick Start

### Requirements

```bash
pip install torch nvidia-cutlass-dsl
```

CUDA 12.4+ / 13.x and an SM89 GPU (RTX 40-series) are required.

### Run the Optimized Kernel

```bash
python3 sm89_cutedsl.py
```

Expected output (on RTX 4060 Laptop):

```text
GPU   : NVIDIA GeForce RTX 4060 Laptop GPU
SM    : sm_89
[reference] running ...
[CuTe DSL] compiling + warmup ...
  max abs err : ~0.00000x
  PASS        : True
  Latency     : ~1.12 ms/iter
  Throughput  : ~0.48 TFLOP/s
```

### Run the Baseline (for comparison)

```bash
python3 backup.py
```

---

## Architecture Highlights

### Optimized Design (`sm89_cutedsl.py`)

- **1 CTA per (request, KV head)** → 4 warps cooperate on the same GQA group
- **Warp-level butterfly shuffle** for the full 128-dim QK reduction (no shared memory)
- **0 `sync_threads()`** barriers in the token loop
- **128-byte coalesced FP8 loads**
- **Scale folding**: `k_scale` into softmax scale, `v_scale` into post-loop normalizer
- Pre-compiled with `cute.compile` to eliminate Python JIT overhead

### Baseline Design (`backup.py`)

- 1 CTA per query head
- Shared-memory cross-warp reduction tree
- 2 barriers per token
- Used only for performance comparison

---

## Performance Summary (Nsight Compute)

| Metric                      | Baseline     | Optimized    | Improvement      |
|-----------------------------|--------------|--------------|------------------|
| Kernel Duration             | 3.41 ms      | **1.12 ms**  | **3.04×**        |
| Compute Throughput          | 0.16 TFLOP/s | **0.48 TFLOP/s** | **3.05×**    |
| DRAM Bandwidth              | 20.5 GB/s    | **62.1 GB/s**| **3.03×**        |
| L1 Cache Hit Rate           | 49.8 %       | **93.9 %**   | +44.1 pp         |
| Block Barriers per CTA      | 1 024        | **0**        | 100 % eliminated |
| Dynamic Shared Memory       | 16 B         | **0 B**      | Full L1 capacity |

---

## Mathematical Core

The kernel implements gated sparse attention:

```
Output = (Σ pᵢ · (Vᵢ × v_scale)) ⊙ σ(Gate)
```

with online softmax:

```
m_new = max(m, s)
α     = 2^(m - m_new)
p     = 2^(s - m_new)
acc   = acc × α + p · V
ℓ     = ℓ   × α + p
```

All scales are folded so that the inner token loop contains only FMAs and `ex2.approx.f32`.

---

## Engineering Notes

Six major issues were solved during development:

1. MLIR SSA region leak (Constexpr annotations must match between `@cute.jit` and `@cute.kernel`)
2. Independent Thread Scheduling deadlock on `__shfl_sync`
3. Silent FP8 type trap (`Float8E4M3` vs `Float8E4M3FN`)
4. 39 ms Python JIT overhead → fixed with `cute.compile`
5. `UNSUP_EARLY_EXIT` → replaced by structured predicates
6. VRAM explosion on 8 GB laptop → direct FP8 allocation

Full technical write-up is available in the accompanying engineering document.

---

## License

This code is provided for research and educational purposes.

