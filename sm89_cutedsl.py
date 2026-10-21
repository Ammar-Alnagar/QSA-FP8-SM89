"""
QSA Sparse Paged GQA — SM89 port using CuTe DSL (nvidia-cutlass-dsl 4.8.1)
RTX 4060 Laptop GPU  |  compute_capability = 8.9  |  FP8-e4m3fn K/V cache

Optimizations for SM89 (Ada Lovelace):
  1. Launch Granularity: 1 CTA per (row, kv_head) processing all GQA query heads.
  2. Warp-Level Reduction: 1 warp per query head (32 threads x 4 dims = 128 dims),
     eliminating all cross-warp shared memory synchronization barriers.
  3. Scale Folding: v_scale folded into post-normalization; k_scale folded into softmax_scale.
  4. Memory Coalescing: 4 contiguous FP8 elements per thread -> 128-byte coalesced loads.
  5. Native C-API Dispatch: Precompiled with cute.compile to eliminate Python JIT overhead.
  6. Strict MLIR Compliance: No early returns or breaks/continues inside staged loops.

Run:  .venv/bin/python sm89_cutedsl.py
"""

import math
import time

import cutlass
import cutlass.cute as cute
import torch
from cutlass.utils import create_cute_tensor_for_fp8

# ── config ──────────────────────────────────────────────────────────────────
NUM_REQUESTS = 32
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS  # 4
HEAD_DIM = 128
PAGE_SIZE = 16
MAX_PAGES = 256
TOPK = 256
BLOCK_N = 32

WARMUP = 10
BENCH = 100


# ── synthetic tensors ────────────────────────────────────────────────────────
def make_tensors(device="cuda"):
    torch.manual_seed(42)
    nblocks = NUM_REQUESTS * MAX_PAGES
    rows = NUM_REQUESTS

    q = torch.randn(rows, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    kf = torch.randn(nblocks, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device) * 0.1
    k_fp8 = kf.to(torch.float8_e4m3fn)
    del kf
    vf = torch.randn(nblocks, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device) * 0.1
    v_fp8 = vf.to(torch.float8_e4m3fn)
    del vf
    torch.cuda.empty_cache()

    btable = torch.arange(nblocks, dtype=torch.int32, device=device).view(
        NUM_REQUESTS, MAX_PAGES
    )
    t2r = torch.arange(rows, dtype=torch.int32, device=device)

    indices = torch.full((rows, TOPK + 1), -1, dtype=torch.int32, device=device)
    nv = TOPK
    for r in range(rows):
        indices[r, :nv] = torch.arange(nv, dtype=torch.int32, device=device)
        indices[r, TOPK] = nv

    gate = torch.randn(rows, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    return q, k_fp8, v_fp8, 0.1, 0.1, btable, t2r, indices, gate


# ── reference ────────────────────────────────────────────────────────────────
def reference(q, k_fp8, v_fp8, k_scale, v_scale, btable, t2r, indices, gate):
    rows, nqh, hd = q.shape
    nkv = k_fp8.shape[2]
    gs = nqh // nkv
    topk = indices.shape[1] - 1
    scale = hd**-0.5
    kf = k_fp8.to(torch.float32) * k_scale
    vf = v_fp8.to(torch.float32) * v_scale
    out = torch.zeros(rows, nqh, hd, dtype=torch.float32, device=q.device)
    for r in range(rows):
        req = int(t2r[r])
        vc = min(int(indices[r, topk]), topk)
        if vc <= 0 or req < 0:
            continue
        tok = indices[r, :vc]
        tok = tok[tok >= 0]
        if tok.numel() == 0:
            continue
        lp = tok.long() // PAGE_SIZE
        po = tok.long() % PAGE_SIZE
        pb = btable[req, lp]
        bm = (pb >= 0) & (pb < k_fp8.shape[0])
        pb = pb[bm]
        po = po[bm]
        if pb.numel() == 0:
            continue
        for g in range(nkv):
            keys = kf[pb, po, g, :]  # [N, hd]
            vals = vf[pb, po, g, :]
            h0 = g * gs
            qg = q[r, h0 : h0 + gs, :].float()
            sc = qg @ keys.T * scale
            pr = torch.softmax(sc, dim=-1)
            out[r, h0 : h0 + gs, :] = pr @ vals
    out = out * torch.sigmoid(gate.float())
    return out.bfloat16()


# ── CuTe DSL kernel ──────────────────────────────────────────────────────────
# Grid:  [rows, num_kv_heads, 1]
# Block: 128 threads (4 warps).
# Warp w in [0, GROUP_SZ-1] processes query head (kv_head * GROUP_SZ + w).
# Lane l in [0, 31] processes 4 coordinates: d0=l*4, d1=l*4+1, d2=l*4+2, d3=l*4+3.
# Within each warp: 5 butterfly shuffles reduce across the full 128 dimensions.
# Zero shared memory and zero cross-warp barriers needed.


@cute.kernel
def qsa_sm89_kernel(
    q,
    k,
    v,
    idx,
    bt,
    t2r,
    gate,
    out,
    softmax_scale: cutlass.Float32,
    v_scale: cutlass.Float32,
    nblocks: cutlass.Int32,
    nreqs: cutlass.Int32,
    TOPK: cutlass.Constexpr,
    PAGE_SZ: cutlass.Constexpr,
    PG_WIDTH: cutlass.Constexpr,
    BLOCK_N: cutlass.Constexpr,
    GROUP_SZ: cutlass.Constexpr,
):
    row, kv_head, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()

    warp_id = tid // 32
    lane_id = tid % 32
    head = kv_head * GROUP_SZ + warp_id

    req = t2r[row]
    if not (req < 0 or req >= nreqs):
        vc = idx[row, TOPK]
        vc = cute.clamp(vc, 0, TOPK)

        d0 = lane_id * 4
        d1 = d0 + 1
        d2 = d0 + 2
        d3 = d0 + 3

        qd0 = cutlass.Float32(q[row, head, d0])
        qd1 = cutlass.Float32(q[row, head, d1])
        qd2 = cutlass.Float32(q[row, head, d2])
        qd3 = cutlass.Float32(q[row, head, d3])

        acc0 = cutlass.Float32(0.0)
        acc1 = cutlass.Float32(0.0)
        acc2 = cutlass.Float32(0.0)
        acc3 = cutlass.Float32(0.0)
        nm = cutlass.Float32(0.0)
        mx = cutlass.Float32(-1e20)

        ntiles = (vc + BLOCK_N - 1) // BLOCK_N
        for tile in range(ntiles):
            for n in range(BLOCK_N):
                col = tile * BLOCK_N + n
                if col < vc:
                    ltok = idx[row, col]
                    lp = ltok // PAGE_SZ
                    po = ltok % PAGE_SZ
                    pb = bt[req, cute.clamp(lp, 0, PG_WIDTH - 1)]
                    valid = (ltok >= 0) & (lp < PG_WIDTH) & (pb >= 0) & (pb < nblocks)
                    if valid:
                        pb = cute.clamp(pb, 0, nblocks - 1)

                        kd0 = cutlass.Float32(k[pb, po, kv_head, d0])
                        kd1 = cutlass.Float32(k[pb, po, kv_head, d1])
                        kd2 = cutlass.Float32(k[pb, po, kv_head, d2])
                        kd3 = cutlass.Float32(k[pb, po, kv_head, d3])

                        partial = qd0 * kd0 + qd1 * kd1 + qd2 * kd2 + qd3 * kd3

                        # Intra-warp butterfly reduction (5 shuffles sum all 128 dimensions)
                        for i in cutlass.range_constexpr(5):
                            partial += cute.arch.shuffle_sync_bfly(
                                partial, 1 << i, 0xFFFFFFFF
                            )

                        score = partial * softmax_scale

                        new_mx = cute.math.max(mx, score)
                        alpha = cute.math.exp2(mx - new_mx)
                        p = cute.math.exp2(score - new_mx)
                        mx = new_mx

                        vd0 = cutlass.Float32(v[pb, po, kv_head, d0])
                        vd1 = cutlass.Float32(v[pb, po, kv_head, d1])
                        vd2 = cutlass.Float32(v[pb, po, kv_head, d2])
                        vd3 = cutlass.Float32(v[pb, po, kv_head, d3])

                        acc0 = acc0 * alpha + p * vd0
                        acc1 = acc1 * alpha + p * vd1
                        acc2 = acc2 * alpha + p * vd2
                        acc3 = acc3 * alpha + p * vd3
                        nm = nm * alpha + p

        safe_nm = cute.math.max(nm, cutlass.Float32(1e-20))
        inv_nm_vscale = v_scale / safe_nm

        ctx0 = acc0 * inv_nm_vscale
        ctx1 = acc1 * inv_nm_vscale
        ctx2 = acc2 * inv_nm_vscale
        ctx3 = acc3 * inv_nm_vscale

        gd0 = cutlass.Float32(gate[row, head, d0])
        gd1 = cutlass.Float32(gate[row, head, d1])
        gd2 = cutlass.Float32(gate[row, head, d2])
        gd3 = cutlass.Float32(gate[row, head, d3])

        log2e = cutlass.Float32(1.4426950408889634)
        sig0 = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp2(-gd0 * log2e))
        sig1 = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp2(-gd1 * log2e))
        sig2 = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp2(-gd2 * log2e))
        sig3 = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp2(-gd3 * log2e))

        out[row, head, d0] = cutlass.BFloat16(ctx0 * sig0)
        out[row, head, d1] = cutlass.BFloat16(ctx1 * sig1)
        out[row, head, d2] = cutlass.BFloat16(ctx2 * sig2)
        out[row, head, d3] = cutlass.BFloat16(ctx3 * sig3)


@cute.jit
def launch_qsa(
    q,
    k_cute,
    v_cute,
    indices,
    btable,
    t2r,
    gate,
    out,
    ss_log2,
    v_scale,
    nblocks,
    nreqs,
    rows,
    nkvheads,
    topk: cutlass.Constexpr,
    page_sz: cutlass.Constexpr,
    max_pages: cutlass.Constexpr,
    block_n: cutlass.Constexpr,
    group_sz: cutlass.Constexpr,
):
    qsa_sm89_kernel(
        q,
        k_cute,
        v_cute,
        indices,
        btable,
        t2r,
        gate,
        out,
        ss_log2,
        v_scale,
        nblocks,
        nreqs,
        topk,
        page_sz,
        max_pages,
        block_n,
        group_sz,
    ).launch(
        grid=[rows, nkvheads, 1],
        block=[128, 1, 1],
        smem=0,
    )


# ── driver ───────────────────────────────────────────────────────────────────
_COMPILED_LAUNCH = None


def get_compiled_kernel(q, k_cute, v_cute, indices, btable, t2r, gate, out, ss_log2, v_scale, nblocks, nreqs, rows, nkvheads):
    global _COMPILED_LAUNCH
    if _COMPILED_LAUNCH is None:
        args = [
            q, k_cute, v_cute, indices, btable, t2r, gate, out,
            ss_log2, v_scale, nblocks, nreqs, rows, nkvheads,
            TOPK, PAGE_SIZE, MAX_PAGES, BLOCK_N, GROUP_SIZE,
        ]
        _COMPILED_LAUNCH = cute.compile(launch_qsa, *args)
    return _COMPILED_LAUNCH


def run(q, k_fp8, v_fp8, k_scale, v_scale, btable, t2r, indices, gate):
    rows = q.shape[0]
    nqh = q.shape[1]
    hd = q.shape[2]
    nblocks = k_fp8.shape[0]
    nreqs = btable.shape[0]
    max_pg = btable.shape[1]
    topk = indices.shape[1] - 1
    pg_sz = k_fp8.shape[1]
    nkv = k_fp8.shape[2]
    log2e = 1.4426950408889634
    ss_log2 = float((hd**-0.5) * k_scale * log2e)

    out = torch.zeros(rows, nqh, hd, dtype=torch.bfloat16, device=q.device)
    k_fp8_c = k_fp8.contiguous()
    v_fp8_c = v_fp8.contiguous()
    k_cute = create_cute_tensor_for_fp8(
        k_fp8_c.view(dtype=torch.uint8),
        cutlass.Float8E4M3FN,
        leading_dim=k_fp8_c.ndim - 1,
        source_f32_tensor=k_fp8_c.to(torch.float32),
    )
    v_cute = create_cute_tensor_for_fp8(
        v_fp8_c.view(dtype=torch.uint8),
        cutlass.Float8E4M3FN,
        leading_dim=v_fp8_c.ndim - 1,
        source_f32_tensor=v_fp8_c.to(torch.float32),
    )

    compiled_fn = get_compiled_kernel(
        q.contiguous(), k_cute, v_cute, indices.contiguous(), btable.contiguous(),
        t2r.contiguous(), gate.contiguous(), out,
        ss_log2, float(v_scale), nblocks, nreqs, rows, nkv,
    )
    # Compiled kernel accepts the non-Constexpr runtime arguments
    compiled_fn(
        q.contiguous(), k_cute, v_cute, indices.contiguous(), btable.contiguous(),
        t2r.contiguous(), gate.contiguous(), out,
        ss_log2, float(v_scale), nblocks, nreqs, rows, nkv,
    )
    torch.cuda.synchronize()
    return out


def vram():
    return torch.cuda.memory_allocated() / 1e6, torch.cuda.memory_reserved() / 1e6


def main():
    dev = "cuda"
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(
        f"SM    : sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}"
    )

    tensors = make_tensors(dev)
    q, k_fp8, v_fp8, k_scale, v_scale, btable, t2r, indices, gate = tensors

    print("\n[reference] running ...")
    ref = reference(*tensors)
    print(f"  shape={tuple(ref.shape)}, dtype={ref.dtype}")

    print("\n[CuTe DSL] compiling + warmup ...")
    m0_alloc, m0_res = vram()
    out = run(*tensors)
    m1_alloc, m1_res = vram()
    print(f"  VRAM allocated : {m1_alloc:.1f} MB   reserved: {m1_res:.1f} MB")

    # correctness check
    diff = (out.float() - ref.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    passed = max_err < 1e-3
    print(f"\n  max abs err : {max_err:.6f}")
    print(f"  mean abs err: {mean_err:.6f}")
    print(f"  PASS (<1e-3): {passed}")

    # warmup
    for _ in range(WARMUP):
        run(*tensors)
    torch.cuda.synchronize()

    # Pre-bind CuTe tensors for pure kernel benchmark
    rows, nqh, hd = q.shape
    nblocks = k_fp8.shape[0]
    nreqs = btable.shape[0]
    nkv = k_fp8.shape[2]
    topk = indices.shape[1] - 1
    log2e = 1.4426950408889634
    ss_log2 = float((hd**-0.5) * k_scale * log2e)

    k_fp8_c = k_fp8.contiguous()
    v_fp8_c = v_fp8.contiguous()
    k_cute = create_cute_tensor_for_fp8(
        k_fp8_c.view(dtype=torch.uint8),
        cutlass.Float8E4M3FN,
        leading_dim=k_fp8_c.ndim - 1,
        source_f32_tensor=k_fp8_c.to(torch.float32),
    )
    v_cute = create_cute_tensor_for_fp8(
        v_fp8_c.view(dtype=torch.uint8),
        cutlass.Float8E4M3FN,
        leading_dim=v_fp8_c.ndim - 1,
        source_f32_tensor=v_fp8_c.to(torch.float32),
    )
    q_c = q.contiguous()
    idx_c = indices.contiguous()
    bt_c = btable.contiguous()
    t2r_c = t2r.contiguous()
    gate_c = gate.contiguous()
    v_sc_f = float(v_scale)

    compiled_fn = get_compiled_kernel(
        q_c, k_cute, v_cute, idx_c, bt_c, t2r_c, gate_c, out,
        ss_log2, v_sc_f, nblocks, nreqs, rows, nkv,
    )
    runtime_args = [
        q_c, k_cute, v_cute, idx_c, bt_c, t2r_c, gate_c, out,
        ss_log2, v_sc_f, nblocks, nreqs, rows, nkv,
    ]

    # benchmark pure kernel
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(BENCH):
        compiled_fn(*runtime_args)
    t1.record()
    torch.cuda.synchronize()
    kernel_ms = t0.elapsed_time(t1) / BENCH

    # benchmark end-to-end driver (including cute tensor wrapping)
    t0.record()
    for _ in range(min(BENCH, 20)):
        run(*tensors)
    t1.record()
    torch.cuda.synchronize()
    e2e_ms = t0.elapsed_time(t1) / min(BENCH, 20)

    nv = int(indices[:, -1].float().mean().item())
    flops = 4.0 * rows * nv * nqh * hd  # QK dot (2x) + AV matmul (2x)
    tflops = flops / (kernel_ms * 1e-3) / 1e12

    print(f"\n  Kernel Latency: {kernel_ms:.4f} ms/iter ({kernel_ms * 1000:.1f} us)")
    print(f"  E2E Latency   : {e2e_ms:.3f} ms/iter")
    print(f"  Throughput    : {tflops:.3f} TFLOP/s (compute)")
    print(f"\n{'=' * 55}")
    print(f"  SUMMARY  CuTe DSL sparse GQA FP8 (32 reqs, TopK=256)")
    print(f"  Kernel: {kernel_ms * 1000:.1f} us  |  {tflops:.2f} TFLOP/s  |  PASS={passed}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
