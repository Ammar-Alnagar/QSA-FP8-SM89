"""Eager PyTorch baseline for the sparse paged GQA workload."""

from __future__ import annotations

import math
import time

import torch

from sm89_cutedsl import (
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
WARMUP = 2
BENCH = 5


@torch.inference_mode()
def torch_eager(q, k, v, k_scale, v_scale, btable, t2r, indices, gate):
    """Vectorized eager PyTorch ops; intentionally no custom Triton/CUDA kernel."""
    logical = indices[:, :TOPK].to(torch.long)
    valid = (logical >= 0) & (logical < TOPK)
    logical = logical.clamp(0, TOPK - 1)
    logical_page = logical // PAGE_SIZE
    page_offset = logical % PAGE_SIZE

    request_ids = t2r.to(torch.long).clamp(0, NUM_REQUESTS - 1)
    physical_page = btable[request_ids[:, None], logical_page]
    valid &= (physical_page >= 0) & (physical_page < NUM_REQUESTS * MAX_PAGES)
    physical_page = physical_page.clamp(0, NUM_REQUESTS * MAX_PAGES - 1)

    # Advanced indexing materializes all selected K/V rows, then eager PyTorch
    # converts them to FP32 for batched matmul and softmax.
    keys = k[physical_page, page_offset].permute(0, 2, 3, 1).to(torch.float32)
    values = v[physical_page, page_offset].permute(0, 2, 1, 3).to(torch.float32)
    queries = q.reshape(NUM_REQUESTS, NUM_KV_HEADS, GROUP_SIZE, HEAD_DIM).float()

    scores = torch.matmul(queries, keys) * (k_scale / math.sqrt(HEAD_DIM))
    scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    context = torch.matmul(probabilities, values) * v_scale
    gated = context * torch.sigmoid(
        gate.reshape(NUM_REQUESTS, NUM_KV_HEADS, GROUP_SIZE, HEAD_DIM).float()
    )
    return gated.reshape(NUM_REQUESTS, NUM_Q_HEADS, HEAD_DIM).to(torch.bfloat16)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch eager benchmark requires a visible CUDA GPU")

    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    capability = torch.cuda.get_device_capability(0)
    print(f"SM    : sm_{capability[0]}{capability[1]}")
    print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})\n")

    tensors = make_tensors("cuda")
    q, k, v, k_scale, v_scale, btable, t2r, indices, gate = tensors
    print(f"[PyTorch eager] workload shape={tuple(q.shape)}, dtype={q.dtype}")
    print("  Reference: shared randomized-input PyTorch reference")

    expected = reference(*tensors)
    output = torch_eager(q, k, v, k_scale, v_scale, btable, t2r, indices, gate)
    torch.cuda.synchronize()
    max_err = (output.float() - expected.float()).abs().max().item()
    passed = max_err < 1e-3
    print(f"  max abs err : {max_err:.8f}")
    print(f"  PASS (<1e-3): {passed}")
    del expected, output
    torch.cuda.empty_cache()

    for _ in range(WARMUP):
        output = torch_eager(q, k, v, k_scale, v_scale, btable, t2r, indices, gate)
    torch.cuda.synchronize()
    del output
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(BENCH):
        output = torch_eager(q, k, v, k_scale, v_scale, btable, t2r, indices, gate)
    stop.record()
    torch.cuda.synchronize()
    event_ms = start.elapsed_time(stop) / BENCH
    del output

    wall_start = time.perf_counter()
    for _ in range(BENCH):
        output = torch_eager(q, k, v, k_scale, v_scale, btable, t2r, indices, gate)
    torch.cuda.synchronize()
    e2e_ms = (time.perf_counter() - wall_start) * 1000 / BENCH
    del output

    peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
    flops = 4.0 * NUM_REQUESTS * TOPK * NUM_Q_HEADS * HEAD_DIM
    tflops = flops / (event_ms * 1e-3) / 1e12
    print(f"  Peak allocated VRAM: {peak_mib:.1f} MiB")
    print("  Timing method: PyTorch eager CUDA events")
    print(f"  Kernel Latency: {event_ms:.4f} ms/iter")
    print(f"  E2E Latency   : {e2e_ms:.3f} ms/iter")
    print(f"  Throughput    : {tflops:.3f} TFLOP/s (derived from CUDA-event time)")
    print("  SUMMARY PyTorch eager sparse GQA FP8")
    print(f"  Kernel: {event_ms:.4f} ms | {tflops:.3f} TFLOP/s | PASS={passed}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
