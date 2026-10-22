# =============================================================================
# QSA Sparse Paged GQA — Mojo kernel (SM89)
# Complete production-scale implementation matching CuTe DSL and Raw CUDA
# =============================================================================

from std.math import ceildiv, exp2, max, sqrt
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, Dim
from max.gpu import thread_idx, block_idx
from max.gpu.primitives.warp import shuffle_xor
from std.memory import Pointer

# ---------------------------------------------------------------------------
# Compile-time constants (exact match with CuTe DSL & Raw CUDA workload)
# ---------------------------------------------------------------------------
comptime NUM_REQUESTS = 256
comptime NUM_Q_HEADS  = 32
comptime NUM_KV_HEADS = 8
comptime GROUP_SIZE   = 4
comptime HEAD_DIM     = 128
comptime PAGE_SIZE    = 16
comptime TOPK         = 512
comptime MAX_PAGES    = TOPK // PAGE_SIZE
comptime BLOCK_N      = 32
comptime WARP_SIZE    = 32
comptime NUM_WARPS    = GROUP_SIZE
comptime THREADS      = WARP_SIZE * NUM_WARPS   # 128
comptime LOG2E: Float32 = 1.4426950408889634

# ---------------------------------------------------------------------------
# Initialization Kernel for Metadata Buffers
# ---------------------------------------------------------------------------
def init_data_kernel(
    idx_ptr:   Pointer[Int32, MutAnyOrigin],
    bt_ptr:    Pointer[Int32, MutAnyOrigin],
    t2r_ptr:   Pointer[Int32, MutAnyOrigin],
    nreqs:     Int32,
    topk:      Int32,
    max_pages: Int32,
):
    var r = Int(block_idx.x)
    if r < Int(nreqs):
        t2r_ptr[unsafe_offset=r] = Int32(r)
        idx_ptr[unsafe_offset=r * (Int(topk) + 1) + Int(topk)] = Int32(topk)
        for i in range(Int(topk)):
            idx_ptr[unsafe_offset=r * (Int(topk) + 1) + i] = Int32(i)
        for p in range(Int(max_pages)):
            bt_ptr[unsafe_offset=r * Int(max_pages) + p] = Int32(r * Int(max_pages) + p)

# ---------------------------------------------------------------------------
# Device kernel
# ---------------------------------------------------------------------------
def qsa_sm89_kernel(
    q_ptr:    Pointer[BFloat16, ImmutAnyOrigin],
    gate_ptr: Pointer[BFloat16, ImmutAnyOrigin],
    out_ptr:  Pointer[BFloat16, MutAnyOrigin],
    k_ptr:    Pointer[Float8_e4m3fn, ImmutAnyOrigin],
    v_ptr:    Pointer[Float8_e4m3fn, ImmutAnyOrigin],
    idx_ptr:  Pointer[Int32, ImmutAnyOrigin],
    bt_ptr:   Pointer[Int32, ImmutAnyOrigin],
    t2r_ptr:  Pointer[Int32, ImmutAnyOrigin],
    softmax_scale: Float32,
    v_scale:       Float32,
    nblocks:       Int32,
    nreqs:         Int32,
    topk:          Int32,
    max_pages:     Int32,
    nq_heads:      Int32,
    nkv_heads:     Int32,
):
    # Thread coordinates
    var row     = Int(block_idx.x)
    var kv_head = Int(block_idx.y)
    var tid     = Int(thread_idx.x)

    var warp_id = tid // WARP_SIZE
    var lane    = tid %  WARP_SIZE

    var q_head  = kv_head * GROUP_SIZE + warp_id

    var d0 = lane * 4
    var d1 = d0 + 1
    var d2 = d0 + 2
    var d3 = d0 + 3

    # Request validity
    var req = Int(t2r_ptr[unsafe_offset=row])
    var valid_req = (req >= 0) and (req < Int(nreqs))

    var vc = Int(idx_ptr[unsafe_offset=row * (Int(topk) + 1) + Int(topk)])
    if vc < 0:
        vc = 0
    if vc > Int(topk):
        vc = Int(topk)
    if not valid_req:
        vc = 0

    # Load query (vectorized SIMD[BFloat16, 4])
    var q_offset = (row * Int(nq_heads) + q_head) * HEAD_DIM + d0
    var q_vec = q_ptr.unsafe_load[width=4](q_offset)
    var qd0 = Float32(q_vec[0])
    var qd1 = Float32(q_vec[1])
    var qd2 = Float32(q_vec[2])
    var qd3 = Float32(q_vec[3])

    # Online-softmax state
    var acc0: Float32 = 0.0
    var acc1: Float32 = 0.0
    var acc2: Float32 = 0.0
    var acc3: Float32 = 0.0
    var nm:   Float32 = 0.0
    var mx:   Float32 = -1e20

    var ntiles = ceildiv(vc, BLOCK_N)

    # Token loop
    for tile in range(ntiles):
        for n in range(BLOCK_N):
            var col = tile * BLOCK_N + n
            var active = col < vc

            var kd0: Float32 = 0.0
            var kd1: Float32 = 0.0
            var kd2: Float32 = 0.0
            var kd3: Float32 = 0.0
            var vd0: Float32 = 0.0
            var vd1: Float32 = 0.0
            var vd2: Float32 = 0.0
            var vd3: Float32 = 0.0

            var token_valid = False
            if active:
                var ltok = Int(idx_ptr[unsafe_offset=row * (Int(topk) + 1) + col])
                var lp   = ltok // PAGE_SIZE
                var po   = ltok %  PAGE_SIZE
                var pb   = Int(bt_ptr[unsafe_offset=req * Int(max_pages) + max(0, min(lp, Int(max_pages) - 1))])

                var valid = (ltok >= 0) and (lp < Int(max_pages)) and (pb >= 0) and (pb < Int(nblocks))
                if valid:
                    token_valid = True
                    pb = max(0, min(pb, Int(nblocks) - 1))
                    var base = ((pb * PAGE_SIZE + po) * Int(nkv_heads) + kv_head) * HEAD_DIM + d0
                    var k_vec = k_ptr.unsafe_load[width=4](base)
                    var v_vec = v_ptr.unsafe_load[width=4](base)
                    kd0 = Float32(k_vec[0])
                    kd1 = Float32(k_vec[1])
                    kd2 = Float32(k_vec[2])
                    kd3 = Float32(k_vec[3])
                    vd0 = Float32(v_vec[0])
                    vd1 = Float32(v_vec[1])
                    vd2 = Float32(v_vec[2])
                    vd3 = Float32(v_vec[3])

            # 4-element partial dot
            var partial = qd0 * kd0 + qd1 * kd1 + qd2 * kd2 + qd3 * kd3

            # Intra-warp butterfly reduction (compile-time unrolled)
            comptime for i in range(5):
                partial += shuffle_xor(partial, UInt32(1 << i))

            var score = partial * softmax_scale

            # Online softmax (masked)
            var new_mx = max(mx, score)
            var alpha  = exp2(mx - new_mx)
            var p      = exp2(score - new_mx)

            if not token_valid:
                alpha = 1.0
                p     = 0.0

            if token_valid:
                mx = new_mx

            acc0 = acc0 * alpha + p * vd0
            acc1 = acc1 * alpha + p * vd1
            acc2 = acc2 * alpha + p * vd2
            acc3 = acc3 * alpha + p * vd3
            nm   = nm   * alpha + p

    # Final normalisation + gate + store
    var safe_nm = max(nm, Float32(1e-20))
    var inv     = v_scale / safe_nm

    var ctx0 = acc0 * inv
    var ctx1 = acc1 * inv
    var ctx2 = acc2 * inv
    var ctx3 = acc3 * inv

    var g_vec = gate_ptr.unsafe_load[width=4](q_offset)
    var gd0 = Float32(g_vec[0])
    var gd1 = Float32(g_vec[1])
    var gd2 = Float32(g_vec[2])
    var gd3 = Float32(g_vec[3])

    var sig0 = 1.0 / (1.0 + exp2(-gd0 * LOG2E))
    var sig1 = 1.0 / (1.0 + exp2(-gd1 * LOG2E))
    var sig2 = 1.0 / (1.0 + exp2(-gd2 * LOG2E))
    var sig3 = 1.0 / (1.0 + exp2(-gd3 * LOG2E))

    if valid_req:
        out_ptr.unsafe_store[width=4](
            q_offset,
            SIMD[DType.bfloat16, 4](
                BFloat16(ctx0 * sig0),
                BFloat16(ctx1 * sig1),
                BFloat16(ctx2 * sig2),
                BFloat16(ctx3 * sig3),
            )
        )
    else:
        out_ptr.unsafe_store[width=4](
            q_offset,
            SIMD[DType.bfloat16, 4](BFloat16(0.0), BFloat16(0.0), BFloat16(0.0), BFloat16(0.0))
        )


# ---------------------------------------------------------------------------
# Benchmark & Verification Main
# ---------------------------------------------------------------------------
def main() raises:
    if not has_accelerator():
        print("No compatible GPU found")
        return

    var ctx = DeviceContext()
    print("GPU   : " + ctx.name())
    print("SM    : sm_89\n")

    print("[Mojo / MAX] workload")
    print("  shape=(" + String(NUM_REQUESTS) + ", " + String(NUM_Q_HEADS) + ", " + String(HEAD_DIM) + "), dtype=torch.bfloat16\n")

    var nblocks = NUM_REQUESTS * MAX_PAGES
    var q_size = NUM_REQUESTS * NUM_Q_HEADS * HEAD_DIM
    var kv_size = nblocks * PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM
    var btable_size = NUM_REQUESTS * MAX_PAGES
    var indices_size = NUM_REQUESTS * (TOPK + 1)
    var t2r_size = NUM_REQUESTS

    print("[Mojo / MAX] compiling + warmup ...")
    print("  K/V payload    : " + String(2 * kv_size) + " FP8 elements (" + String(Float64(2 * kv_size) / 1048576.0) + " MiB)\n")

    # Allocate GPU buffers
    var q_buf    = ctx.enqueue_create_buffer[DType.bfloat16](q_size)
    var gate_buf = ctx.enqueue_create_buffer[DType.bfloat16](q_size)
    var out_buf  = ctx.enqueue_create_buffer[DType.bfloat16](q_size)
    var k_buf    = ctx.enqueue_create_buffer[DType.float8_e4m3fn](kv_size)
    var v_buf    = ctx.enqueue_create_buffer[DType.float8_e4m3fn](kv_size)
    var idx_buf  = ctx.enqueue_create_buffer[DType.int32](indices_size)
    var bt_buf   = ctx.enqueue_create_buffer[DType.int32](btable_size)
    var t2r_buf  = ctx.enqueue_create_buffer[DType.int32](t2r_size)

    q_buf.enqueue_fill(1.0)
    gate_buf.enqueue_fill(0.0)
    out_buf.enqueue_fill(0.0)
    # Constant 1.0 keys and values with a zero gate yield the closed-form
    # output BF16(0.05) for every element, independently checked below.
    k_buf.enqueue_fill(1.0)
    v_buf.enqueue_fill(1.0)

    # Initialize metadata on GPU
    ctx.enqueue_function[func=init_data_kernel](
        idx_buf.device_ptr(),
        bt_buf.device_ptr(),
        t2r_buf.device_ptr(),
        Int32(NUM_REQUESTS),
        Int32(TOPK),
        Int32(MAX_PAGES),
        grid_dim=Dim(NUM_REQUESTS),
        block_dim=Dim(1),
    )
    ctx.synchronize()

    var k_scale: Float32 = 0.1
    var v_scale: Float32 = 0.1
    var softmax_scale = (1.0 / sqrt(Float32(HEAD_DIM))) * k_scale * LOG2E

    # Warmup
    for _ in range(10):
        ctx.enqueue_function[func=qsa_sm89_kernel](
            q_buf.device_ptr(),
            gate_buf.device_ptr(),
            out_buf.device_ptr(),
            k_buf.device_ptr(),
            v_buf.device_ptr(),
            idx_buf.device_ptr(),
            bt_buf.device_ptr(),
            t2r_buf.device_ptr(),
            softmax_scale,
            v_scale,
            Int32(nblocks),
            Int32(NUM_REQUESTS),
            Int32(TOPK),
            Int32(MAX_PAGES),
            Int32(NUM_Q_HEADS),
            Int32(NUM_KV_HEADS),
            grid_dim=Dim(NUM_REQUESTS, NUM_KV_HEADS),
            block_dim=Dim(THREADS),
        )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[DType.bfloat16](q_size)
    ctx.enqueue_copy(dst_buf=out_host, src_buf=out_buf)
    ctx.synchronize()

    var max_err: Float32 = 0.0
    for i in range(q_size):
        var err = Float32(out_host[i]) - 0.05
        if err < 0.0:
            err = -err
        if err > max_err:
            max_err = err
    var validation_passed = max_err < 1e-3
    print("  max abs err : " + String(max_err))
    print("  PASS (<1e-3): " + String(validation_passed) + "\n")

    # Benchmark iterations
    var iters = 100
    var t0 = perf_counter_ns()
    for _ in range(iters):
        ctx.enqueue_function[func=qsa_sm89_kernel](
            q_buf.device_ptr(),
            gate_buf.device_ptr(),
            out_buf.device_ptr(),
            k_buf.device_ptr(),
            v_buf.device_ptr(),
            idx_buf.device_ptr(),
            bt_buf.device_ptr(),
            t2r_buf.device_ptr(),
            softmax_scale,
            v_scale,
            Int32(nblocks),
            Int32(NUM_REQUESTS),
            Int32(TOPK),
            Int32(MAX_PAGES),
            Int32(NUM_Q_HEADS),
            Int32(NUM_KV_HEADS),
            grid_dim=Dim(NUM_REQUESTS, NUM_KV_HEADS),
            block_dim=Dim(THREADS),
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()

    var elapsed_ms = Float64(t1 - t0) / 1_000_000.0
    var avg_ms = elapsed_ms / Float64(iters)
    var avg_us = avg_ms * 1000.0
    var flops = 4.0 * Float64(NUM_REQUESTS) * Float64(TOPK) * Float64(NUM_Q_HEADS) * Float64(HEAD_DIM)
    var tflops = flops / (avg_ms * 1e-3) / 1e12

    # Formatted output matching exactly
    var avg_ms_str = String(avg_ms)[byte=0:6]
    var avg_us_str = String(avg_us)[byte=0:6]
    var tflops_str = String(tflops)[byte=0:5]

    print("  Enqueue+sync wall time: " + avg_ms_str + " ms/iter (" + avg_us_str + " us)")
    print("  Throughput    : " + tflops_str + " TFLOP/s (compute)\n")

    print("=======================================================")
    print("  SUMMARY  Mojo / MAX sparse GQA FP8 (256 reqs, TopK=512)")
    print("  Enqueue+sync: " + avg_us_str + " us  |  " + tflops_str + " TFLOP/s  |  validation=" + String(validation_passed))
    print("=======================================================")
