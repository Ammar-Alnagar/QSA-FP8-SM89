"""
QSA Sparse Paged GQA — SM89 port using CuTe DSL (nvidia-cutlass-dsl 4.8.1)
Baseline kernel (1 CTA per query head, cross-warp SMEM reduction)
RTX 4060 Laptop GPU  |  compute_capability = 8.9  |  FP8-e4m3fn K/V cache

Run:  .venv/bin/python backup.py
"""

import math
import sys

import cutlass
import cutlass.cute as cute
import torch
from cutlass.utils import create_cute_tensor_for_fp8

# ── config ──────────────────────────────────────────────────────────────────
NUM_REQUESTS = 64
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS  # 4
HEAD_DIM = 128
PAGE_SIZE = 16
MAX_PAGES = 512
TOPK = 512
BLOCK_N = 32

WARMUP = 5 if "--profile" in sys.argv else 10
BENCH = 1 if "--profile" in sys.argv else 50


# ── synthetic tensors ────────────────────────────────────────────────────────
def make_tensors(device="cuda"):
    torch.manual_seed(42)
    nblocks = NUM_REQUESTS * MAX_PAGES
    rows = NUM_REQUESTS

    q = torch.randn(rows, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k_fp8 = (
        torch.randn(
            nblocks, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        * 0.1
    ).to(torch.float8_e4m3fn)
    torch.cuda.empty_cache()

    v_fp8 = (
        torch.randn(
            nblocks, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        * 0.1
    ).to(torch.float8_e4m3fn)
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
            keys = k_fp8[pb, po, g, :].to(torch.float32) * k_scale  # [N, hd]
            vals = v_fp8[pb, po, g, :].to(torch.float32) * v_scale
            h0 = g * gs
            qg = q[r, h0 : h0 + gs, :].float()
            sc = qg @ keys.T * scale
            pr = torch.softmax(sc, dim=-1)
            out[r, h0 : h0 + gs, :] = pr @ vals
    out = out * torch.sigmoid(gate.float())
    return out.bfloat16()


# ── CuTe DSL kernel ──────────────────────────────────────────────────────────
# One CUDA block per (query_row, query_head).
# blockDim = (HEAD_DIM, 1, 1) = 128 threads; each thread owns 1 dim slot.
# QK dot-product uses warp shuffles (4 warps → log2(32)=5 shuffles each,
# then a 4-entry shared-mem cross-warp reduction).


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
    row, head, _ = cute.arch.block_idx()  # grid=[rows, nqheads, 1]
    d, _, _ = cute.arch.thread_idx()  # block=[head_dim, 1, 1]

    # Shared memory: 4 floats for cross-warp dot-product reduction
    smem_alloc = cutlass.memory.SmemAllocator()
    smem = smem_alloc.allocate_tensor(cutlass.Float32, 4)

    req = t2r[row]
    if not (req < 0 or req >= nreqs):
        vc = idx[row, TOPK]  # valid token count (last col)
        vc = cute.clamp(vc, 0, TOPK)

        # Load query dim d for this head
        qd = cutlass.Float32(q[row, head, d])

        # Accumulation registers
        acc = cutlass.Float32(0.0)
        nm = cutlass.Float32(0.0)
        mx = cutlass.Float32(-1e20)

        # Iterate over selected KV tokens
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
                        kv_head = head // GROUP_SZ

                        kd = cutlass.Float32(k[pb, po, kv_head, d])

                        # Partial dot for this thread's dim d
                        partial = qd * kd

                        # Warp-level reduction (32 threads per warp, 4 warps = 128 threads)
                        for i in cutlass.range_constexpr(5):  # log2(32)=5
                            partial += cute.arch.shuffle_sync_bfly(
                                partial, 1 << i, 0xFFFFFFFF
                            )

                        # Lane 0 of each warp writes to smem
                        warp_id = d // 32
                        lane_id = d % 32
                        if lane_id == 0:
                            smem[warp_id] = partial

                        cute.arch.sync_threads()

                        # Thread 0 sums all 4 warps and multiplies by softmax_scale
                        if d == 0:
                            smem[0] = (
                                smem[0] + smem[1] + smem[2] + smem[3]
                            ) * softmax_scale

                        cute.arch.sync_threads()

                        # All threads read the final score
                        score = smem[0]

                        # Online softmax update
                        new_mx = cute.math.max(mx, score)
                        alpha = cute.math.exp2(mx - new_mx)
                        p = cute.math.exp2(score - new_mx)
                        mx = new_mx

                        # Load V dim d and accumulate
                        vd = cutlass.Float32(v[pb, po, kv_head, d]) * v_scale

                        acc = acc * alpha + p * vd
                        nm = nm * alpha + p

        # Normalise and write output
        safe_nm = cute.math.max(nm, cutlass.Float32(1e-20))
        ctx_d = acc / safe_nm

        gd = cutlass.Float32(gate[row, head, d])
        sig_g = cutlass.Float32(1.0) / (
            cutlass.Float32(1.0) + cute.math.exp2(-gd * 1.4426950408889634)
        )
        result = cutlass.BFloat16(ctx_d * sig_g)
        out[row, head, d] = result


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
    nqheads,
    topk: cutlass.Constexpr,
    page_sz: cutlass.Constexpr,
    max_pages: cutlass.Constexpr,
    head_dim: cutlass.Constexpr,
    block_n: cutlass.Constexpr,
    group_sz: cutlass.Constexpr,
):
    smem_bytes = 4 * 4  # 4 floats for cross-warp reduction

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
        grid=[rows, nqheads, 1],
        block=[head_dim, 1, 1],
        smem=smem_bytes,
    )


# ── driver ───────────────────────────────────────────────────────────────────
_COMPILED_LAUNCH = None


def get_compiled_kernel(
    q, k_cute, v_cute, indices, btable, t2r, gate, out, ss_log2, v_scale, nblocks, nreqs, rows, nqh
):
    global _COMPILED_LAUNCH
    if _COMPILED_LAUNCH is None:
        args = [
            q, k_cute, v_cute, indices, btable, t2r, gate, out,
            ss_log2, v_scale, nblocks, nreqs, rows, nqh,
            TOPK, PAGE_SIZE, MAX_PAGES, HEAD_DIM, BLOCK_N, GROUP_SIZE,
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
        ss_log2, float(v_scale), nblocks, nreqs, rows, nqh,
    )
    compiled_fn(
        q.contiguous(), k_cute, v_cute, indices.contiguous(), btable.contiguous(),
        t2r.contiguous(), gate.contiguous(), out,
        ss_log2, float(v_scale), nblocks, nreqs, rows, nqh,
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

    print("\n[CuTe DSL baseline] compiling + warmup ...")
    m0_alloc, m0_res = vram()
    out = run(*tensors)
    m1_alloc, m1_res = vram()
    print(f"  VRAM allocated : {m1_alloc:.1f} MB   reserved: {m1_res:.1f} MB")

    # correctness
    diff = (out.float() - ref.float()).abs()
    print(f"\n  max abs err : {diff.max():.5f}")
    print(f"  mean abs err: {diff.mean():.6f}")
    print(f"  PASS        : {diff.max() < 0.12}")

    # warmup
    for _ in range(WARMUP):
        run(*tensors)
    torch.cuda.synchronize()

    # Pre-bind CuTe tensors for pure kernel benchmark
    rows, nqh, hd = q.shape
    nblocks = k_fp8.shape[0]
    nreqs = btable.shape[0]
    max_pg = btable.shape[1]
    topk = indices.shape[1] - 1
    pg_sz = k_fp8.shape[1]
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
        ss_log2, v_sc_f, nblocks, nreqs, rows, nqh,
    )
    runtime_args = [
        q_c, k_cute, v_cute, idx_c, bt_c, t2r_c, gate_c, out,
        ss_log2, v_sc_f, nblocks, nreqs, rows, nqh,
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

    # benchmark end-to-end driver
    t0.record()
    for _ in range(min(BENCH, 10)):
        run(*tensors)
    t1.record()
    torch.cuda.synchronize()
    e2e_ms = t0.elapsed_time(t1) / min(BENCH, 10)

    nv = int(indices[:, -1].float().mean().item())
    flops = 4.0 * rows * nv * nqh * hd  # QK dot (2x) + AV matmul (2x)
    tflops = flops / (kernel_ms * 1e-3) / 1e12

    print(f"\n  Kernel Latency: {kernel_ms:.4f} ms/iter ({kernel_ms * 1000:.1f} us)")
    print(f"  E2E Latency   : {e2e_ms:.3f} ms/iter")
    print(f"  Throughput    : {tflops:.3f} TFLOP/s (compute)")
    print(f"\n{'=' * 55}")
    print(f"  SUMMARY  CuTe DSL baseline GQA FP8 ({rows} reqs, TopK={nv})")
    print(
        f"  Kernel: {kernel_ms * 1000:.1f} us  |  {tflops:.2f} TFLOP/s  |  PASS={diff.max() < 0.12}"
    )
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
