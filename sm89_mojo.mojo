# =============================================================================
# QSA Sparse Paged GQA — Mojo kernel (SM89)
# Fully corrected for current Mojo (max.gpu, var, raises, Pointer)
# =============================================================================

from std.math import ceildiv, exp2, max
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.primitives.warp import shuffle_xor
from std.memory import Pointer

# ---------------------------------------------------------------------------
# Compile-time constants
# ---------------------------------------------------------------------------
comptime HEAD_DIM     = 128
comptime GROUP_SIZE   = 4
comptime PAGE_SIZE    = 16
comptime BLOCK_N      = 16
comptime WARP_SIZE    = 32
comptime NUM_WARPS    = GROUP_SIZE
comptime THREADS      = WARP_SIZE * NUM_WARPS   # 128
comptime LOG2E: Float32 = 1.4426950408889634

# ---------------------------------------------------------------------------
# Device kernel
# ---------------------------------------------------------------------------
def qsa_sm89_kernel(
    q_ptr:    Pointer[BFloat16, ...],
    gate_ptr: Pointer[BFloat16, ...],
    out_ptr:  Pointer[BFloat16, ...],
    k_ptr:    Pointer[Float8_e4m3fn, ...],
    v_ptr:    Pointer[Float8_e4m3fn, ...],
    idx_ptr:  Pointer[Int32, ...],
    bt_ptr:   Pointer[Int32, ...],
    t2r_ptr:  Pointer[Int32, ...],
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

    # Flat index helpers (inlined to avoid capture issues)
    # Q / gate / out : [rows, nq_heads, HEAD_DIM]
    # K / V          : [nblocks, PAGE_SIZE, nkv_heads, HEAD_DIM]
    # indices        : [rows, topk+1]
    # block table    : [nreqs, max_pages]

    # Request validity
    var req = Int(t2r_ptr[row])
    var valid_req = (req >= 0) and (req < Int(nreqs))

    var vc = Int(idx_ptr[row * (Int(topk) + 1) + Int(topk)])
    if vc < 0:
        vc = 0
    if vc > Int(topk):
        vc = Int(topk)
    if not valid_req:
        vc = 0

    # Load query
    var qd0 = Float32(q_ptr[(row * Int(nq_heads) + q_head) * HEAD_DIM + d0])
    var qd1 = Float32(q_ptr[(row * Int(nq_heads) + q_head) * HEAD_DIM + d1])
    var qd2 = Float32(q_ptr[(row * Int(nq_heads) + q_head) * HEAD_DIM + d2])
    var qd3 = Float32(q_ptr[(row * Int(nq_heads) + q_head) * HEAD_DIM + d3])

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

            if active:
                var ltok = Int(idx_ptr[row * (Int(topk) + 1) + col])
                var lp   = ltok // PAGE_SIZE
                var po   = ltok %  PAGE_SIZE
                var pb   = Int(bt_ptr[req * Int(max_pages) + max(0, min(lp, Int(max_pages) - 1))])

                var valid = (ltok >= 0) and (lp < Int(max_pages)) and (pb >= 0) and (pb < Int(nblocks))
                if valid:
                    pb = max(0, min(pb, Int(nblocks) - 1))
                    var base = ((pb * PAGE_SIZE + po) * Int(nkv_heads) + kv_head) * HEAD_DIM
                    kd0 = Float32(k_ptr[base + d0])
                    kd1 = Float32(k_ptr[base + d1])
                    kd2 = Float32(k_ptr[base + d2])
                    kd3 = Float32(k_ptr[base + d3])
                    vd0 = Float32(v_ptr[base + d0])
                    vd1 = Float32(v_ptr[base + d1])
                    vd2 = Float32(v_ptr[base + d2])
                    vd3 = Float32(v_ptr[base + d3])

            # 4-element partial dot
            var partial = qd0 * kd0 + qd1 * kd1 + qd2 * kd2 + qd3 * kd3

            # Intra-warp butterfly reduction
            for i in range(5):
                var delta = 1 << i
                partial += shuffle_xor(partial, UInt32(delta))

            var score = partial * softmax_scale

            # Online softmax (masked)
            var new_mx = max(mx, score)
            var alpha  = exp2(mx - new_mx)
            var p      = exp2(score - new_mx)

            if not active:
                alpha = 1.0
                p     = 0.0

            if active:
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

    var gd0 = Float32(gate_ptr[(row * Int(nq_heads) + q_head) * HEAD_DIM + d0])
    var gd1 = Float32(gate_ptr[(row * Int(nq_heads) + q_head) * HEAD_DIM + d1])
    var gd2 = Float32(gate_ptr[(row * Int(nq_heads) + q_head) * HEAD_DIM + d2])
    var gd3 = Float32(gate_ptr[(row * Int(nq_heads) + q_head) * HEAD_DIM + d3])

    var sig0 = 1.0 / (1.0 + exp2(-gd0 * LOG2E))
    var sig1 = 1.0 / (1.0 + exp2(-gd1 * LOG2E))
    var sig2 = 1.0 / (1.0 + exp2(-gd2 * LOG2E))
    var sig3 = 1.0 / (1.0 + exp2(-gd3 * LOG2E))

    var out_base = (row * Int(nq_heads) + q_head) * HEAD_DIM
    if valid_req:
        out_ptr[out_base + d0] = BFloat16(ctx0 * sig0)
        out_ptr[out_base + d1] = BFloat16(ctx1 * sig1)
        out_ptr[out_base + d2] = BFloat16(ctx2 * sig2)
        out_ptr[out_base + d3] = BFloat16(ctx3 * sig3)
    else:
        out_ptr[out_base + d0] = BFloat16(0.0)
        out_ptr[out_base + d1] = BFloat16(0.0)
        out_ptr[out_base + d2] = BFloat16(0.0)
        out_ptr[out_base + d3] = BFloat16(0.0)


# ---------------------------------------------------------------------------
# Host entry point (minimal – just verifies compilation)
# ---------------------------------------------------------------------------
def main() raises:
    if not has_accelerator():
        print("No compatible GPU found")
        return

    var ctx = DeviceContext()
    print("GPU:", ctx.name())
    print("Mojo QSA Sparse Paged GQA kernel compiled successfully.")
