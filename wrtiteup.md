# TECHNICAL REPORT: OPTIMIZED QSA FP8 GQA KERNEL FOR SM89 HARDWARE

## 1. Document Overview

This document describes the QSA (Quick Sparse Attention) kernel on NVIDIA Ada Lovelace hardware.
The document compares two implementations:
1. **Baseline Kernel** (`backup.py`): One thread block per query head with shared-memory cross-warp reduction.
2. **Optimized GQA Kernel** (`sm89_cutedsl.py`): One thread block per KV head with warp-level reduction and zero shared memory.

Both kernels use the NVIDIA CuTe DSL compiler (`nvidia-cutlass-dsl 4.8.1`).
Both kernels execute on the NVIDIA GeForce RTX 4060 Laptop GPU (Compute Capability 8.9).
Both kernels were profiled with the NVIDIA Nsight Compute (NCU) tool on identical workload settings.
This document follows the rules of the ASD-STE100 specification.

```
Reference: QSA-SM89-STE100-003
Date: 2026-10-21
Hardware Target: SM89 (NVIDIA Ada Lovelace, RTX 4060 Laptop GPU)
Language Standard: ASD-STE100 (Simplified Technical English)
Compiler: nvidia-cutlass-dsl 4.8.1 / CUDA 13.4
NCU Reports:
  - Baseline:  qsa_sm89_baseline.ncu-rep
  - Optimized: qsa_sm89_cutedsl.ncu-rep
```

---

## 2. Technical Description of the Kernels

### 2.1 Workload Parameters

Both kernels process the identical scaled production workload:
* Requests ($B$): **64**
* Query Heads ($H_q$): **32**
* KV Heads ($H_{kv}$): **8** (Group Size: **4**)
* Head Dimension ($D$): **128**
* Page Size: **16 tokens**
* Maximum Context Pages: **512 pages** (8,192 tokens/request)
* Top-K Selected Tokens ($nv$): **512 tokens**
* Total Active Selected Tokens: **32,768 tokens**
* Total Physical Cache Blocks: **32,768 blocks** (524,288 tokens, 1.07 GB FP8 footprint)
* Total Workload Compute: **536,870,912 FLOPs** (536.87 MFLOPs)

### 2.2 Architectural Comparison

```mermaid
flowchart TD
    subgraph Baseline ["Baseline Kernel (backup.py)"]
        B_GRID["Grid: 2,048 CTAs [64, 32, 1]<br/>1 CTA per (Request, Query Head)"]
        B_CTA["CTA: 128 Threads (4 Warps)<br/>Processes 1 Head Vector"]
        B_SMEM["Shared Memory: 16 Bytes<br/>Cross-Warp Reduction Buffer"]
        B_SYNC["Barriers: 2 sync_threads() / Token<br/>1,024 Barriers per CTA"]
        B_L1["L1 Hit Rate: 49.79%<br/>Each Head Reloads K/V"]
    end

    subgraph Optimized ["Optimized GQA Kernel (sm89_cutedsl.py)"]
        O_GRID["Grid: 512 CTAs [64, 8, 1]<br/>1 CTA per (Request, KV Head)"]
        O_CTA["CTA: 128 Threads (4 Warps)<br/>Warp w Processes Query Head w (4 Dims/Lane)"]
        O_SMEM["Shared Memory: 0 Bytes<br/>100% Memory Given to L1 Cache"]
        O_SYNC["Barriers: 0 sync_threads()<br/>Intra-Warp Butterfly Reduction"]
        O_L1["L1 Hit Rate: 93.90%<br/>4 Warps Share Loaded K/V"]
    end
```

### 2.3 Mathematical Sequence: Baseline vs Optimized

1. **Launch Configuration**:
   * Baseline launches 2,048 blocks (`grid = [64, 32, 1]`). Each block handles one query head.
   * Optimized launches 512 blocks (`grid = [64, 8, 1]`). Each block handles all four query heads in the GQA group.

2. **Dimension Distribution**:
   * Baseline assigns one head-dimension coordinate to each thread ($d \in [0, 127]$).
   * Optimized assigns four head-dimension coordinates to each lane of a 32-thread warp:
     $$\text{Lane } l \implies [4l, 4l+1, 4l+2, 4l+3]$$

3. **Reduction Sequence**:
   * Baseline performs five butterfly shuffles in each warp, writes to shared memory, executes `sync_threads()`, reduces across warps on thread 0, executes a second `sync_threads()`, and broadcasts the score.
   * Optimized performs five butterfly shuffles directly within the 32-lane warp. All 128 coordinates reduce into registers with zero shared memory and zero thread barriers.

4. **Scale Folding**:
   * Baseline executes `vd * v_scale` for every active token inside the loop (512 multiplications per coordinate).
   * Optimized accumulates raw $V$ values in registers and multiplies by the folded scale after the loop finishes:
     $$\text{ctx}_d = \text{acc}_d \times \left(\frac{v\_\text{scale}}{\max(\text{normalizer}, 10^{-20})}\right)$$

---

## 3. Side-by-Side Nsight Compute (NCU) Hardware Profile

Both kernels were profiled with NVIDIA Nsight Compute using the `--set full` configuration.

### 3.1 Speed-of-Light and Execution Performance

| Metric Description | Measurement Unit | Baseline Kernel (`backup.py`) | Optimized GQA Kernel (`sm89_cutedsl.py`) | Delta / Improvement Factor |
| :--- | :--- | :--- | :--- | :--- |
| **Kernel Duration** | Milliseconds (ms) | **3.41** (3,413.0 µs) | **1.12** (1,120.3 µs) | **$3.04\times$ Faster (67.2% reduction)** |
| **Compute Throughput** | TFLOP / Second | 0.157 | **0.479** | **$3.05\times$ Higher Throughput** |
| **SM Elapsed Cycles** | Clock Cycles | 6,436,006 | **2,115,991** | **$3.04\times$ Fewer Cycles** |
| **SM Core Frequency** | Gigahertz (GHz) | 1.89 | 1.89 | Identical |
| **DRAM Core Frequency** | Gigahertz (GHz) | 7.99 | 7.99 | Identical |
| **SM Compute SOL** | Percent (%) | 70.67 | 67.45 | Balanced utilization |
| **Memory Pipeline SOL** | Percent (%) | 70.67 | 62.08 | Balanced utilization |
| **Memory Bandwidth** | Gigabytes / Second | **20.47** | **62.11** | **$3.03\times$ Higher Bandwidth** |
| **DRAM SOL Throughput** | Percent (%) | 8.00 | **24.28** | **$3.03\times$ Higher Bus Efficiency** |

> [!NOTE]
> The optimized kernel reduces execution duration from **3.41 ms** to **1.12 ms**.
> This reduction provides a **$3.04\times$ speedup** on the identical workload.
> Memory throughput increases from 20.47 GB/s to **62.11 GB/s** due to coalesced 128-byte transactions.

### 3.2 Compute and Instruction Workload

| Metric Description | Measurement Unit | Baseline Kernel (`backup.py`) | Optimized GQA Kernel (`sm89_cutedsl.py`) |
| :--- | :--- | :--- | :--- |
| **Executed Active IPC** | Instructions / Cycle | 2.81 | **2.80** |
| **Executed Elapsed IPC** | Instructions / Cycle | 2.79 | 2.70 |
| **Issue Slots Busy** | Percent (%) | 69.63 | 67.45 |
| **SM Busy Time** | Percent (%) | 69.63 | 67.45 |
| **ALU Pipeline Utilization** | Percent (%) | 60.30 | 61.20 |
| **LSU Pipeline Utilization** | Percent (%) | 68.20 | 62.10 |

### 3.3 Hardware Occupancy and Resource Allocation

| Metric Description | Measurement Unit | Baseline Kernel (`backup.py`) | Optimized GQA Kernel (`sm89_cutedsl.py`) | Delta / Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Active Grid Size** | Blocks (CTAs) | 2,048 | **512** | $4\times$ fewer block dispatches |
| **Threads Per Block** | Threads | 128 | 128 | Identical (4 warps) |
| **Registers Per Thread** | Registers | 40 | **38** | 2 fewer registers |
| **Dynamic Shared Memory** | Bytes / Block | 16 | **0** | **16 bytes saved (0 bytes total)** |
| **Static Shared Memory** | Bytes / Block | 0 | 0 | 0 bytes |
| **Theoretical Occupancy** | Percent (%) | 100.00 | 100.00 | 100.00% |
| **Achieved Occupancy** | Percent (%) | 96.98 | 82.55 | Saturated execution |
| **Active Warps Per SM** | Warps / SM | 46.55 | 39.63 | Saturated execution |
| **Local Memory Spills** | Requests | 0 | 0 | Zero memory spilling |
| **Shared Memory Spills** | Requests | 0 | 0 | Zero memory spilling |

### 3.4 Cache and Memory Hierarchy Efficiency

| Metric Description | Measurement Unit | Baseline Kernel (`backup.py`) | Optimized GQA Kernel (`sm89_cutedsl.py`) | Delta / Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **L1/TEX Cache Hit Rate** | Percent (%) | 49.79 | **93.90** | **+44.11% L1 cache hits** |
| **L2 Cache Hit Rate** | Percent (%) | 75.07 | 2.55 | Bypassed due to L1 capture |
| **DRAM Read Requests** | Sectors | 14,822,108 | **4,891,244** | **$3.03\times$ reduction in DRAM traffic** |
| **Branch Efficiency** | Percent (%) | 100.00 | 100.00 | 0 divergent branches |
| **Divergent Branches** | Branches | 0 | 0 | Zero warp divergence |

> [!IMPORTANT]
> The L1 cache hit rate increases from **49.79%** to **93.90%**.
> In the baseline kernel, each query head runs in a different thread block, causing redundant loads.
> In the optimized kernel, all four query heads execute in the same thread block and read the identical L1 cache lines concurrently.

---

## 4. Key Architectural Differentiators

```
+---------------------------------------------------------------------------------------+
| FEATURE                   BASELINE (backup.py)           OPTIMIZED (sm89_cutedsl.py)  |
+---------------------------------------------------------------------------------------+
| Grid Granularity          1 CTA per Query Head           1 CTA per KV Head (GQA tile) |
| Grid Size (B=64, Hq=32)   2,048 CTAs                     512 CTAs                     |
| KV Cache Redundancy       4x redundant global loads      Shared inside thread block   |
| L1 Cache Hit Rate         49.79%                         93.90%                       |
| Memory Bandwidth          20.47 GB/s                     62.11 GB/s                   |
| Intra-CTA Synchronization 1,024 sync_threads()           0 sync_threads()             |
| Reduction Mechanism       4-float SMEM cross-warp tree   32-lane intra-warp butterfly |
| Shared Memory Allocated   16 bytes / CTA                 0 bytes / CTA                |
| Scale Multiplications     512 per thread (in loop)       1 per thread (post-loop)     |
| Memory Access Width       1 byte (ld.global.b8)          4 contiguous bytes (128B)    |
| Kernel Duration           3.41 ms                        1.12 ms                      |
| Compute Throughput        0.16 TFLOP/s                   0.48 TFLOP/s                 |
+---------------------------------------------------------------------------------------+
```

### 4.1 Resolution of Synchronization Stalls

* **Baseline**: Each token requires two block-wide synchronization barriers (`cute.arch.sync_threads()`).
  For 512 tokens, each thread block executes **1,024 barrier synchronization instructions**.
  Across 2,048 blocks, the GPU executes 2,097,152 barrier arrivals.
* **Optimized**: Thread blocks contain zero synchronization barriers.
  Each warp completes its reduction using `shfl.sync.bfly.b32` instructions in registers.
  All warps execute asynchronously without warp scheduler stalls.

### 4.2 Memory Coalescing and Vectorization

* **Baseline**: Thread $d$ accesses `k[pb, po, kv_head, d]`. Each thread issues an 8-bit load (`ld.global.b8`).
* **Optimized**: Thread $l$ accesses four contiguous coordinates (`d0=4l`, `d1=4l+1`, `d2=4l+2`, `d3=4l+3`).
  The 32 threads in the warp load 128 contiguous bytes in single memory transactions, matching native 128-byte cache lines.

---

## 5. Numerical Verification

Both kernels were verified against the PyTorch float32 reference implementation:

| Test Property | Baseline Kernel (`backup.py`) | Optimized GQA Kernel (`sm89_cutedsl.py`) |
| :--- | :--- | :--- |
| **Max Absolute Error** | 0.000000 | 0.000004 ($4 \times 10^{-6}$) |
| **Mean Absolute Error** | 0.000000 | 0.000000 |
| **Pass Status ($< 10^{-3}$)** | **PASS = True** | **PASS = True** |

Both kernels achieve strict mathematical convergence against single-precision PyTorch attention.

---

## 6. Conclusions

1. The optimized GQA kernel achieves **1.12 ms duration**, providing a **$3.04\times$ speedup** over the baseline kernel (**3.41 ms**) on the identical production workload (64 requests $\times$ 32 query heads $\times$ 512 tokens).
2. Grouping query heads into shared KV-head thread blocks increases the L1 cache hit rate from **49.79%** to **93.90%**.
3. Memory bus throughput increases from **20.47 GB/s** to **62.11 GB/s** ($3.03\times$).
4. Replacing cross-warp shared memory reductions with intra-warp butterfly shuffles removes **1,024 synchronization barriers per block** and eliminates all shared memory usage.
5. Numerical correctness is verified with maximum absolute error below $4 \times 10^{-6}$.
