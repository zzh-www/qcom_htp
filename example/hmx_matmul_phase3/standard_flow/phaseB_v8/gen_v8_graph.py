#!/usr/bin/env python3
"""
V8 shape-adaptive ONNX generator.

Implements the QNN MatMul design recipe (docs/qnn_custom_op_design_recipe.md):
- tile planner: M_TILE=256 fixed (matches QNN), N_TILE adaptive to VTCM budget
- emits M_ROUNDS × N_ROUNDS MatMulV8 instances, each producing a small tile
- M_ROUNDS pack_act instances + N_ROUNDS pack_wt instances
- weight and bias are Python-split into STATIC sub-tensors (compile-time)
- activation is runtime-sliced via onnx::Slice
- reassembly uses onnx::Concat (maps to qti.aisw::Concat built-in)

Usage:
    python gen_v8_graph.py --M 4096 --K 4096 --N 4096 --vtcm-mb 8
    python gen_v8_graph.py --M 512 --K 512 --N 512            # default VTCM 8 MB

Outputs:
    v8_model.onnx                      the adaptive graph
    runtime_inputs_u8/act.raw          runtime activation input
    input_list.txt                     qnn-net-run input list
"""
import argparse
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

DOMAIN = "hmx"


def plan_matmul_graph(M, K, N, vtcm_mb=8, n_tile_override=None, m_tile_override=None):
    """Shape → tile plan. Mirrors QNN's internal MatMul planner per
    observed 32³/128³/256³/512³/4096³ chrometrace behavior.

    Keeps ~40% of VTCM free as margin for compiler-inserted staging.
    `n_tile_override` / `m_tile_override` let callers pin tile sizes for perf
    tuning (e.g. M_TILE=512 at 512³ to skip slice_act).

    Sweep 2026-04-26 (t256 vs t128, V8/native iter2 wall):
        256³ : 38/39  vs 28us   (256 marginally better)
        512³ : 85/94  vs 64us   (256 better, t128 +11%)
        1024³: 363/378 vs 585us (256 ≈ t128, both faster than native)
        2048³: 2411/2492 vs 2786us (256 better, both faster than native)
        4096³: 79963/17531 vs 13240us (**t128 4.5× faster!**)
    Conclusion: tile=128 at K≥4096 to avoid VTCM thrashing
    (V8 t256 4096³ fill/spill = 7×; t128 = 18× but smaller absolute
    spill, total wall drops 4.5×). For K<4096 keep tile=256.
    """
    if m_tile_override is not None:
        M_TILE = ((m_tile_override + 31) // 32) * 32
    else:
        # Adaptive default: smaller tiles at K>=4096 to keep VTCM-cached
        # tile reuse high enough that compiler-inserted @Spill/@Fill doesn't
        # explode (validated at 4096³).
        if K >= 4096 or N >= 4096 or M >= 4096:
            M_TILE = 128
        else:
            M_TILE = 256
        if M < M_TILE:
            M_TILE = ((M + 31) // 32) * 32  # round up to 32

    if n_tile_override is not None:
        N_TILE = n_tile_override
        if N_TILE % 32 != 0:
            N_TILE = ((N_TILE + 31) // 32) * 32
    else:
        # Pin N_TILE to the same band as M_TILE — empirically the dominant
        # axis for VTCM scheduling is symmetric tiling.
        if K >= 4096 or N >= 4096 or M >= 4096:
            N_TILE_target = 128
        else:
            N_TILE_target = 256
        workspace = int(0.6 * vtcm_mb * 1024 * 1024)
        if K + M_TILE > 0:
            n_tile_max = (workspace - M_TILE * K) // (K + M_TILE)
        else:
            n_tile_max = N_TILE_target
        N_TILE = max(32, min(N_TILE_target, (n_tile_max // 32) * 32))

    if N < N_TILE:
        N_TILE = ((N + 31) // 32) * 32

    M_ROUNDS = (M + M_TILE - 1) // M_TILE
    N_ROUNDS = (N + N_TILE - 1) // N_TILE

    return dict(
        M_TILE=M_TILE, N_TILE=N_TILE,
        M_ROUNDS=M_ROUNDS, N_ROUNDS=N_ROUNDS,
        total_instances=M_ROUNDS * N_ROUNDS,
    )


def fp32_to_fp16_u16(f):
    return np.float16(f).view(np.uint16).item()


def build_bias_chunk(N_chunk, K, n_offset):
    """Per-column fp16 bias encoding (quant scale + baseline).
    Matches HmxMatMulV8Op.cpp's silicon formula:
        bias[nt*128 + 2c]   = fp16(512 * scale_c)     (scale)
        bias[nt*128 + 2c+1] = fp16(2.0) = 0x4000      (zp=128 baseline)
    """
    assert N_chunk % 32 == 0
    n_tiles = N_chunk // 32
    b = np.zeros((1, 1, n_tiles, 128), dtype=np.uint16)
    for nt in range(n_tiles):
        for c in range(32):
            n_abs = n_offset + nt * 32 + c
            scale_n = 1.0 / (K * (1.0 + 0.1 * (n_abs % 7)))
            b[0, 0, nt, 2 * c]     = fp32_to_fp16_u16(512.0 * scale_n)
            b[0, 0, nt, 2 * c + 1] = 0x4000
    return b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=512)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--N", type=int, default=512)
    p.add_argument("--vtcm-mb", type=int, default=8)
    p.add_argument("--n-tile", type=int, default=None, help="Override N_TILE (default: auto from VTCM budget)")
    p.add_argument("--m-tile", type=int, default=None,
                   help="Override M_TILE (default: 256). NOTE: M_TILE > 256 currently "
                        "breaks correctness (output ~bias-only) at 512³. Lever for future "
                        "investigation if 512³ slice_act overhead becomes critical.")
    p.add_argument("--seed", type=int, default=0xB17E)
    p.add_argument("-o", "--out", default=None, help="ONNX output path (default v8_model.onnx)")
    args = p.parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))
    M, K, N = args.M, args.K, args.N
    np.random.seed(args.seed)
    assert M % 32 == 0 and K % 32 == 0 and N % 32 == 0, "M/K/N must be /32"

    plan = plan_matmul_graph(M, K, N, args.vtcm_mb,
                             n_tile_override=args.n_tile,
                             m_tile_override=args.m_tile)
    M_TILE, N_TILE = plan["M_TILE"], plan["N_TILE"]
    M_ROUNDS, N_ROUNDS = plan["M_ROUNDS"], plan["N_ROUNDS"]
    K_T = K // 32
    print(f"== V8 shape-adaptive graph ({M}x{K}x{N}, VTCM={args.vtcm_mb} MB) ==")
    print(f"   M_TILE={M_TILE}  N_TILE={N_TILE}  "
          f"rounds={M_ROUNDS}×{N_ROUNDS}={plan['total_instances']}  K_T={K_T}")

    # ---------- static data (weight, bias, scratch) ----------
    wRaw = np.array([((i * 13) % 15) - 7 for i in range(K * N)],
                    dtype=np.int8).reshape(1, 1, K, N)
    wRaw_u8 = wRaw.view(np.uint8)   # u8 bit-pattern preserved; UFIXED_POINT_8 in graph

    # Weight split into N_ROUNDS chunks
    wt_initializers = []
    for nr in range(N_ROUNDS):
        n0 = nr * N_TILE
        n1 = min(n0 + N_TILE, N)
        pad = N_TILE - (n1 - n0)
        chunk = wRaw_u8[:, :, :, n0:n1]
        if pad > 0:
            # zero-pad the N dim to N_TILE (QNN does this for tile alignment)
            chunk = np.pad(chunk, ((0, 0), (0, 0), (0, 0), (0, pad)))
        wt_initializers.append(numpy_helper.from_array(chunk.astype(np.uint8),
                                                       name=f"wt_N{nr}"))

    # Bias split into N_ROUNDS chunks (STATIC, per-N-column fp16)
    bias_initializers = []
    for nr in range(N_ROUNDS):
        n0 = nr * N_TILE
        n_chunk = min(N_TILE, N - n0)
        # pad to N_TILE with zeros (trailing cols in last chunk are inert)
        if n_chunk < N_TILE:
            b = np.zeros((1, 1, N_TILE // 32, 128), dtype=np.uint16)
            b_real = build_bias_chunk(((n_chunk + 31) // 32) * 32, K, n0)
            b[:, :, :b_real.shape[2], :] = b_real
        else:
            b = build_bias_chunk(N_TILE, K, n0)
        bias_initializers.append(numpy_helper.from_array(b, name=f"bias_N{nr}"))

    # Scratch: single 2 KiB zero buffer shared by all MatMulV8 instances
    scratch_arr = np.zeros((1, 1, 1, 2048), dtype=np.uint8)
    scratch_init = numpy_helper.from_array(scratch_arr, name="scratch")

    # ---------- graph tensors ----------
    act_in = helper.make_tensor_value_info("act_raw", TensorProto.UINT8, [1, 1, M, K])
    out_shape = [1, M // 32, N // 32, 1024]
    out_info = helper.make_tensor_value_info("out", TensorProto.UINT8, out_shape)

    nodes = []

    # Slice activation along M axis into M_ROUNDS chunks ----------------------
    act_chunks = []
    for mr in range(M_ROUNDS):
        m0 = mr * M_TILE
        m1 = min(m0 + M_TILE, M)
        # If last chunk is smaller, pad up (not expected for M multiples of M_TILE)
        # We just use onnx::Slice when m1 <= M. Padding handled by pack kernel via shape.
        starts = numpy_helper.from_array(np.array([m0], dtype=np.int64),
                                         name=f"slice_starts_M{mr}")
        ends = numpy_helper.from_array(np.array([m1], dtype=np.int64),
                                       name=f"slice_ends_M{mr}")
        axes = numpy_helper.from_array(np.array([2], dtype=np.int64),
                                       name=f"slice_axes_M{mr}")
        nodes.extend([
            helper.make_node("Slice",
                inputs=["act_raw", f"slice_starts_M{mr}", f"slice_ends_M{mr}", f"slice_axes_M{mr}"],
                outputs=[f"act_M{mr}"],
                name=f"slice_act_M{mr}"),
        ])
        act_chunks.append((f"act_M{mr}", starts, ends, axes))

    # Collect slice param initializers
    slice_initializers = []
    for (_, s, e, a) in act_chunks:
        slice_initializers += [s, e, a]

    # Pack activation per M_ROUND (MT=true → auto multi-HVX) ------------------
    PACK_ACT_OP = os.environ.get("V8_PACK_ACT_OP", "PackActivationU8RowMajor")
    for mr in range(M_ROUNDS):
        nodes.append(helper.make_node(
            PACK_ACT_OP,
            inputs=[f"act_M{mr}"],
            outputs=[f"packed_act_M{mr}"],
            name=f"pack_act_M{mr}",
            domain=DOMAIN,
        ))

    # Pack weight per N_ROUND (MT=false currently — one per nr) ----------------
    for nr in range(N_ROUNDS):
        nodes.append(helper.make_node(
            "PackWeightToHmxTileV3",
            inputs=[f"wt_N{nr}"],
            outputs=[f"packed_wt_N{nr}"],
            name=f"pack_wt_N{nr}",
            domain=DOMAIN,
        ))

    # MatMulV8 per (mr, nr) — produces one tile-layout sub-output -------------
    out_tiles = []
    for mr in range(M_ROUNDS):
        for nr in range(N_ROUNDS):
            out_name = f"out_tile_M{mr}_N{nr}"
            nodes.append(helper.make_node(
                "MatMulV8",
                inputs=[f"packed_act_M{mr}", f"packed_wt_N{nr}",
                        f"bias_N{nr}", "scratch"],
                outputs=[out_name],
                name=f"mmv8_M{mr}_N{nr}",
                domain=DOMAIN,
            ))
            out_tiles.append((mr, nr, out_name))

    # Concat reassembly ----------------------------------------------------
    # Each out_tile_M{mr}_N{nr} has shape [1, M_TILE/32, N_TILE/32, 1024] in VTCM.
    # Large shapes: VTCM can't hold the full concat'd output. Strategy:
    #   1) Concat N-tiles in VTCM (each row_stripe ≤ ~1 MB even at 4096³ → fits)
    #   2) TcmDramCopy each row_stripe to DDR (scatters to disjoint DDR rows)
    #   3) (optional) DDR-side Concat if more than one M_ROUND
    # Per Agent/qnn_matmul_design_principles §3: leave VTCM room for staging.
    row_stripes_ddr = []
    for mr in range(M_ROUNDS):
        # Step 1: concat_N within VTCM (→ [1, M_TILE/32, N/32, 1024])
        if N_ROUNDS == 1:
            stripe_vtcm = f"out_tile_M{mr}_N0"
        else:
            stripe_vtcm = f"row_stripe_vtcm_M{mr}"
            nodes.append(helper.make_node(
                "Concat",
                inputs=[f"out_tile_M{mr}_N{nr}" for nr in range(N_ROUNDS)],
                outputs=[stripe_vtcm],
                name=f"concat_N_M{mr}",
                axis=2,
            ))

        # Step 2: drain to DDR (per-stripe, keeps VTCM free for next M_ROUND)
        stripe_ddr = f"row_stripe_ddr_M{mr}"
        nodes.append(helper.make_node(
            "TcmDramCopy",
            inputs=[stripe_vtcm],
            outputs=[stripe_ddr],
            name=f"tcm2ddr_M{mr}",
            domain=DOMAIN,
        ))
        row_stripes_ddr.append(stripe_ddr)

    # Step 3: DDR-side concat_M assembles final output.
    # At 4096³ this is 16 MB DDR Concat — qti.aisw::Concat handles this fine.
    if M_ROUNDS == 1:
        # Rename last stripe to "out" — need an explicit Identity-style pass-through
        # since ONNX graph output name is fixed. Use TcmDramCopy but on DDR tensor:
        # actually simplest is to just re-assign output name in the last tcm2ddr.
        # Rewire: change the last tcm2ddr node's output to "out".
        for n in nodes:
            if n.name == f"tcm2ddr_M0":
                n.output[0] = "out"
        # pop the fake stripe name we added
    else:
        nodes.append(helper.make_node(
            "Concat",
            inputs=row_stripes_ddr,
            outputs=["out"],
            name="concat_M",
            axis=1,
        ))

    # ---------- assemble ONNX graph ----------
    all_initializers = (
        slice_initializers
        + wt_initializers
        + bias_initializers
        + [scratch_init]
    )
    graph = helper.make_graph(
        nodes,
        name="v8_model",
        inputs=[act_in],
        outputs=[out_info],
        initializer=all_initializers,
    )
    model = helper.make_model(
        graph,
        producer_name=f"v8_adaptive_{M}x{K}x{N}",
        opset_imports=[
            helper.make_opsetid("", 13),
            helper.make_opsetid(DOMAIN, 1),
        ],
    )
    model.ir_version = 8
    # skip onnx.checker — custom domain nodes trip the checker

    out_path = args.out or os.path.join(HERE, "v8_model.onnx")
    onnx.save(model, out_path)
    print(f"   -> {out_path}")
    print(f"   graph nodes: {len(nodes)}  "
          f"(slices={len(act_chunks)}, pack_act={M_ROUNDS}, pack_wt={N_ROUNDS}, "
          f"matmul={M_ROUNDS*N_ROUNDS}, concat_N={M_ROUNDS if N_ROUNDS>1 else 0}, "
          f"concat_M={1 if M_ROUNDS>1 else 0}, tcm2ddr=1)")

    # ---------- runtime activation input ----------
    u8_dir = os.path.join(HERE, "runtime_inputs_u8")
    os.makedirs(u8_dir, exist_ok=True)
    aRaw = np.array([(i * 37) & 0xFF for i in range(M * K)],
                    dtype=np.uint8).reshape(1, 1, M, K)
    aRaw.tofile(os.path.join(u8_dir, "act.raw"))
    with open(os.path.join(HERE, "input_list.txt"), "w") as f:
        f.write("act_raw:=runtime_inputs_u8/act.raw\n")
    print(f"   -> runtime_inputs_u8/act.raw ({M*K} bytes)")
    print(f"   -> input_list.txt")


if __name__ == "__main__":
    main()
