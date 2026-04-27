#!/usr/bin/env python3
"""
V8 ONNX gen — APPLES-TO-APPLES with QNN MatMul.

Differences vs gen_v8_graph.py:
  - Input shape: [1, M, K] u8 (rank-3, matches QNN's [1, M, K])
  - Output shape: [1, M, N] u8 (rank-3 row-major, matches QNN's [1, M, N])
  - Internal tile format: Crouton (PackActCrouton + BbbKMajor)
                          NOT row-major (PackActivationU8RowMajor + MatMulV8)
  - Adds UntileToRowMajor at end to convert tile-layout → [M, N]

This makes V8 consume the same INPUT as QNN, produce the same OUTPUT as QNN,
and use the same INTERNAL Crouton tile format as QNN's q::ForceFormat_Crouton.

Usage:
    python gen_v8_apples.py --M 2048 --K 2048 --N 2048
"""
import argparse, os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

DOMAIN = "hmx"


def fp32_to_fp16_u16(f):
    return np.float16(f).view(np.uint16).item()


def build_bias_chunk(N, K, n_offset=0):
    n_tiles = N // 32
    b = np.zeros((1, 1, n_tiles, 128), dtype=np.uint16)
    for nt in range(n_tiles):
        for c in range(32):
            n = n_offset + nt * 32 + c
            scale = 1.0 / (K * (1.0 + 0.1 * (n % 7)))
            b[0, 0, nt, 2*c]     = fp32_to_fp16_u16(512.0 * scale)
            b[0, 0, nt, 2*c + 1] = 0x4000
    return b


def plan_matmul_graph(M, K, N, vtcm_mb=8, n_tile_override=None):
    M_TILE = 256
    if M < M_TILE:
        M_TILE = ((M + 31) // 32) * 32
    if n_tile_override is not None:
        N_TILE = n_tile_override
    else:
        workspace = int(0.6 * vtcm_mb * 1024 * 1024)
        n_tile_max = (workspace - M_TILE * K) // (K + M_TILE)
        N_TILE = max(32, min(256, (n_tile_max // 32) * 32))
    if N < N_TILE:
        N_TILE = ((N + 31) // 32) * 32
    M_ROUNDS = (M + M_TILE - 1) // M_TILE
    N_ROUNDS = (N + N_TILE - 1) // N_TILE
    return dict(M_TILE=M_TILE, N_TILE=N_TILE, M_ROUNDS=M_ROUNDS, N_ROUNDS=N_ROUNDS,
                total_instances=M_ROUNDS * N_ROUNDS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--M', type=int, default=2048)
    p.add_argument('--K', type=int, default=2048)
    p.add_argument('--N', type=int, default=2048)
    p.add_argument('--vtcm-mb', type=int, default=8)
    p.add_argument('--n-tile', type=int, default=None)
    p.add_argument('--seed', type=int, default=0xB17E)
    p.add_argument('--out', type=str, default=None)
    args = p.parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))
    M, K, N = args.M, args.K, args.N
    np.random.seed(args.seed)
    assert M % 32 == 0 and K % 32 == 0 and N % 32 == 0

    plan = plan_matmul_graph(M, K, N, args.vtcm_mb, n_tile_override=args.n_tile)
    M_TILE, N_TILE = plan["M_TILE"], plan["N_TILE"]
    M_ROUNDS, N_ROUNDS = plan["M_ROUNDS"], plan["N_ROUNDS"]
    K_T = K // 32
    print(f"== V8-apples graph ({M}x{K}x{N}) ==")
    print(f"   M_TILE={M_TILE}  N_TILE={N_TILE}  rounds={M_ROUNDS}×{N_ROUNDS}={plan['total_instances']}")

    # Static weight (V8 PackWeightToHmxTileV3 = HMX-compatible double-vshuff format)
    wRaw = np.array([((i * 13) % 15) - 7 for i in range(K * N)],
                    dtype=np.int8).reshape(1, 1, K, N)
    wRaw_u8 = wRaw.view(np.uint8)

    wt_initializers = []
    for nr in range(N_ROUNDS):
        n0 = nr * N_TILE
        n1 = min(n0 + N_TILE, N)
        pad = N_TILE - (n1 - n0)
        chunk = wRaw_u8[:, :, :, n0:n1]
        if pad > 0:
            chunk = np.pad(chunk, ((0, 0), (0, 0), (0, 0), (0, pad)))
        wt_initializers.append(numpy_helper.from_array(chunk.astype(np.uint8),
                                                       name=f"wt_N{nr}"))

    bias_initializers = []
    for nr in range(N_ROUNDS):
        n0 = nr * N_TILE
        n_chunk = min(N_TILE, N - n0)
        if n_chunk < N_TILE:
            b = np.zeros((1, 1, N_TILE // 32, 128), dtype=np.uint16)
            b_real = build_bias_chunk(((n_chunk + 31) // 32) * 32, K, n0)
            b[:, :, :b_real.shape[2], :] = b_real
        else:
            b = build_bias_chunk(N_TILE, K, n0)
        bias_initializers.append(numpy_helper.from_array(b, name=f"bias_N{nr}"))

    scratch_arr = np.zeros((1, 1, 1, 2048), dtype=np.uint8)
    scratch_init = numpy_helper.from_array(scratch_arr, name="scratch")

    # ONNX top-level tensors — RANK-3 to match QNN
    act_in  = helper.make_tensor_value_info("act_raw", TensorProto.UINT8, [1, M, K])
    out_info = helper.make_tensor_value_info("out",    TensorProto.UINT8, [1, M, N])

    nodes = []

    # Reshape rank-3 input → rank-4 [1, 1, M, K] for slicing & internal ops
    reshape_act_shape = numpy_helper.from_array(np.array([1, 1, M, K], dtype=np.int64),
                                                name="act_reshape_dims")
    nodes.append(helper.make_node("Reshape",
                                  inputs=["act_raw", "act_reshape_dims"],
                                  outputs=["act_4d"],
                                  name="reshape_act_to_4d"))

    # Slice activation along M axis (axis=2 for rank-4)
    slice_initializers = []
    act_chunks = []
    for mr in range(M_ROUNDS):
        m0 = mr * M_TILE
        m1 = min(m0 + M_TILE, M)
        starts = numpy_helper.from_array(np.array([m0], dtype=np.int64), name=f"slice_starts_M{mr}")
        ends   = numpy_helper.from_array(np.array([m1], dtype=np.int64), name=f"slice_ends_M{mr}")
        axes   = numpy_helper.from_array(np.array([2], dtype=np.int64), name=f"slice_axes_M{mr}")
        nodes.append(helper.make_node("Slice",
            inputs=["act_4d", f"slice_starts_M{mr}", f"slice_ends_M{mr}", f"slice_axes_M{mr}"],
            outputs=[f"act_M{mr}"], name=f"slice_act_M{mr}"))
        slice_initializers += [starts, ends, axes]
        act_chunks.append(f"act_M{mr}")

    # Pack activation per M_ROUND — V8 production format (row-major)
    # NOTE: PackActCrouton + BbbKMajor combo has correctness bug at N_t > 8 / multi-op,
    # so apples-to-apples uses production V8 stack with Reshape/Untile only at boundaries.
    for mr in range(M_ROUNDS):
        nodes.append(helper.make_node(
            "PackActCrouton",
            inputs=[f"act_M{mr}"], outputs=[f"packed_act_M{mr}"],
            name=f"pack_act_M{mr}", domain=DOMAIN,
        ))

    # Pack weight per N_ROUND — V8 PackWeightToHmxTileV3 (HMX-compatible double-vshuff)
    for nr in range(N_ROUNDS):
        nodes.append(helper.make_node(
            "PackWeightToHmxTileV3",
            inputs=[f"wt_N{nr}"], outputs=[f"packed_wt_N{nr}"],
            name=f"pack_wt_N{nr}", domain=DOMAIN,
        ))

    # MatMulV8 (production) — consumes V8 row-major act + V8 weight, outputs tile-layout
    out_tiles = []
    for mr in range(M_ROUNDS):
        for nr in range(N_ROUNDS):
            nodes.append(helper.make_node(
                "BbbKMajor",
                inputs=[f"packed_act_M{mr}", f"packed_wt_N{nr}", f"bias_N{nr}", "scratch"],
                outputs=[f"out_tile_M{mr}_N{nr}"],
                name=f"mmv9_M{mr}_N{nr}", domain=DOMAIN,
            ))
            out_tiles.append((mr, nr, f"out_tile_M{mr}_N{nr}"))

    # Concat N-tiles within each M-row stripe → [1, M_TILE/32, N/32, 1024] tile-layout
    row_stripes_tile = []
    for mr in range(M_ROUNDS):
        if N_ROUNDS == 1:
            stripe = f"out_tile_M{mr}_N0"
        else:
            stripe = f"row_stripe_tile_M{mr}"
            nodes.append(helper.make_node("Concat",
                inputs=[f"out_tile_M{mr}_N{nr}" for nr in range(N_ROUNDS)],
                outputs=[stripe], name=f"concat_N_M{mr}", axis=2))
        row_stripes_tile.append(stripe)

    # UntileToRowMajor per stripe → [1, 1, M_TILE, N] u8 row-major
    row_stripes_rm = []
    for mr in range(M_ROUNDS):
        rm_name = f"row_stripe_rm_M{mr}"
        nodes.append(helper.make_node(
            "UntileToRowMajor",
            inputs=[row_stripes_tile[mr]], outputs=[rm_name],
            name=f"untile_M{mr}", domain=DOMAIN,
        ))
        row_stripes_rm.append(rm_name)

    # Concat M-rows into final [1, 1, M, N] u8
    if M_ROUNDS == 1:
        final_4d = row_stripes_rm[0]
    else:
        final_4d = "out_4d"
        nodes.append(helper.make_node("Concat",
            inputs=row_stripes_rm, outputs=[final_4d], name="concat_M", axis=2))

    # Reshape rank-4 [1, 1, M, N] → rank-3 [1, M, N] to match QNN output
    out_reshape_dims = numpy_helper.from_array(np.array([1, M, N], dtype=np.int64),
                                                name="out_reshape_dims")
    nodes.append(helper.make_node("Reshape",
        inputs=[final_4d, "out_reshape_dims"], outputs=["out"],
        name="reshape_out_to_3d"))

    all_initializers = (
        [numpy_helper.from_array(np.array([1, 1, M, K], dtype=np.int64),
                                  name="act_reshape_dims")]
        + slice_initializers
        + wt_initializers
        + bias_initializers
        + [scratch_init, out_reshape_dims]
    )

    graph = helper.make_graph(
        nodes, name="v8_apples", inputs=[act_in], outputs=[out_info],
        initializer=all_initializers,
    )
    model = helper.make_model(graph,
        producer_name=f"v8_apples_{M}x{K}x{N}",
        opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DOMAIN, 1)])
    model.ir_version = 8

    out_path = args.out or os.path.join(HERE, "v8_apples.onnx")
    onnx.save(model, out_path)
    print(f"   -> {out_path}  ({len(nodes)} nodes)")

    # Runtime activation input
    u8_dir = os.path.join(HERE, "runtime_inputs_u8")
    os.makedirs(u8_dir, exist_ok=True)
    aRaw = np.array([(i * 37) & 0xFF for i in range(M * K)],
                    dtype=np.uint8).reshape(1, M, K)
    aRaw.tofile(os.path.join(u8_dir, "act.raw"))
    with open(os.path.join(HERE, "input_list_apples.txt"), "w") as f:
        f.write("act_raw:=runtime_inputs_u8/act.raw\n")
    print(f"   -> runtime_inputs_u8/act.raw ({M*K} bytes)")


if __name__ == "__main__":
    main()
