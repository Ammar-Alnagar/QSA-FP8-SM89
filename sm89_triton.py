"""Sparse paged GQA reference kernel in Triton for SM89."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from sm89_cutedsl import (
    BLOCK_N,
    HEAD_DIM,
    MAX_PAGES,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    NUM_REQUESTS,
    PAGE_SIZE,
    TOPK,
    make_tensors,
    reference,
)

GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS
WARMUP = 10
BENCH = 50


@triton.jit
def _qsa_triton_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    idx_ptr,
    btable_ptr,
    t2r_ptr,
    gate_ptr,
    out_ptr,
    softmax_scale_log2: tl.constexpr,
    v_scale: tl.constexpr,
    num_requests: tl.constexpr,
    topk: tl.constexpr,
    max_pages: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    group_size: tl.constexpr,
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // num_q_heads
    q_head = pid % num_q_heads
    kv_head = q_head // group_size
    req = tl.load(t2r_ptr + row)

    count = tl.load(idx_ptr + row * (topk + 1) + topk)
    count = tl.minimum(tl.maximum(count, 0), topk)
    query_offsets = tl.arange(0, head_dim)
    query_base = (row * num_q_heads + q_head) * head_dim
    query = tl.load(q_ptr + query_base + query_offsets).to(tl.float32)

    running_max = tl.full((), -1.0e20, tl.float32)
    normalizer = tl.full((), 0.0, tl.float32)
    value_acc = tl.full((head_dim,), 0.0, tl.float32)
    token_offsets = tl.arange(0, block_n)

    # A fixed trip count avoids a device-side loop bound while the masks support
    # any valid-token count from zero through Top-K.
    for block_start in range(0, topk, block_n):
        logical_tokens = block_start + token_offsets
        active = logical_tokens < count
        logical = tl.load(
            idx_ptr + row * (topk + 1) + logical_tokens,
            mask=active,
            other=0,
        )
        logical_page = logical // page_size
        token_offset = logical % page_size
        page_valid = active & (logical >= 0) & (logical_page < max_pages)
        physical_page = tl.load(
            btable_ptr + req * max_pages + logical_page,
            mask=page_valid,
            other=0,
        )
        valid = page_valid & (physical_page >= 0) & (physical_page < num_requests * max_pages)

        physical_token = physical_page * page_size + token_offset
        kv_base = (physical_token * num_kv_heads + kv_head) * head_dim
        k_values = tl.load(
            k_ptr + kv_base[:, None] + query_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k_values * query[None, :], axis=1) * softmax_scale_log2
        scores = tl.where(valid, scores, -1.0e20)

        block_max = tl.max(scores, axis=0)
        next_max = tl.maximum(running_max, block_max)
        alpha = tl.exp2(running_max - next_max)
        probabilities = tl.exp2(scores - next_max)
        probabilities = tl.where(valid, probabilities, 0.0)

        v_values = tl.load(
            v_ptr + kv_base[:, None] + query_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=0)
        value_acc = value_acc * alpha + tl.sum(probabilities[:, None] * v_values, axis=0)
        running_max = next_max

    gate = tl.load(gate_ptr + query_base + query_offsets).to(tl.float32)
    output = (value_acc * (v_scale / tl.maximum(normalizer, 1.0e-20))) * tl.sigmoid(gate)
    tl.store(out_ptr + query_base + query_offsets, output)


def launch(q, k, v, indices, btable, t2r, gate, out, k_scale, v_scale):
    softmax_scale_log2 = (HEAD_DIM**-0.5) * k_scale * math.log2(math.e)
    _qsa_triton_kernel[(NUM_REQUESTS * NUM_Q_HEADS,)](
        q,
        k,
        v,
        indices,
        btable,
        t2r,
        gate,
        out,
        softmax_scale_log2,
        v_scale,
        NUM_REQUESTS,
        TOPK,
        MAX_PAGES,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        GROUP_SIZE,
        PAGE_SIZE,
        HEAD_DIM,
        BLOCK_N,
        num_warps=4,
    )
    return out


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Triton benchmark requires a visible CUDA GPU")

    device = "cuda"
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    capability = torch.cuda.get_device_capability(0)
    print(f"SM    : sm_{capability[0]}{capability[1]}\n")

    tensors = make_tensors(device)
    q, k, v, k_scale, v_scale, btable, t2r, indices, gate = tensors
    expected = reference(*tensors)
    output = torch.empty_like(q)

    launch(q, k, v, indices, btable, t2r, gate, output, k_scale, v_scale)
    torch.cuda.synchronize()
    max_err = (output.float() - expected.float()).abs().max().item()
    passed = max_err < 1e-3
    print(f"  shape={tuple(q.shape)}, dtype={q.dtype}")
    print(f"  max abs err : {max_err:.8f}")
    print(f"  PASS (<1e-3): {passed}")

    for _ in range(WARMUP):
        launch(q, k, v, indices, btable, t2r, gate, output, k_scale, v_scale)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(BENCH):
        launch(q, k, v, indices, btable, t2r, gate, output, k_scale, v_scale)
    stop.record()
    torch.cuda.synchronize()
    latency_ms = start.elapsed_time(stop) / BENCH
    flops = 4.0 * NUM_REQUESTS * TOPK * NUM_Q_HEADS * HEAD_DIM
    tflops = flops / (latency_ms * 1e-3) / 1e12

    print(f"  Kernel Latency: {latency_ms:.4f} ms/iter ({latency_ms * 1000:.1f} us)")
    print(f"  Throughput    : {tflops:.3f} TFLOP/s (compute)")
    print(f"  SUMMARY Triton sparse GQA FP8 ({NUM_REQUESTS} reqs, TopK={TOPK})")
    print(f"  Kernel: {latency_ms * 1000:.1f} us | {tflops:.3f} TFLOP/s | PASS={passed}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
