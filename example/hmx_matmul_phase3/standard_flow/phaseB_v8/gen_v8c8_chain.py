#!/usr/bin/env python3
"""
gen_v8c8_test.py — native-aligned 3-input C8 framework test.

ONNX graph (3 inputs to BbbKMajor):
  act_raw   [1, M/32, 32, K]      u8     →  Crouton_8 via QNN auto-insertion
  wt_packed [1, K/32, N/32, 1024] u8     →  pre-packed native ConvLayer layout,
                                              STATIC, weights_to_vtcm DMA inserted
  bias      [N]                   int32  →  raw quantized bias, STATIC,
                                              q::ConvLayer.opt.bias_to_vtcm DMA
                                              expected (slot-2 fold dispatcher
                                              does the weight-aware fold itself —
                                              we no longer pre-bake it here)

Native equivalent (qairt-dlc-info on a u8×u8 256³ MatMul):
  W (data type: sFxp_8;  tensor dimension: [256,256]; tensor type: STATIC)
  B (data type: sFxp_32; tensor dimension: [256];     tensor type: STATIC)

Then UntileToRowMajor + Reshape produce row-major rank-3 [1, M, N] u8 output.
"""
import argparse, os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

DOMAIN = "hmx"

# ACT_ZP used by QNN's q::ConvLayer.opt.weights_to_vtcm@Fi.fi. dispatcher
# when it folds our int32 bias. EMPIRICALLY VERIFIED at 64³ (probe of bias
# VTCM bytes returned exactly -128 × Σwt + bias_q). QNN's default for u8
# inputs is 128 regardless of quant_overrides act_offset value.
ACT_ZP = 128


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=256)
    p.add_argument("--K", type=int, default=256)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--seed", type=int, default=0xB17E)
    p.add_argument("-o", "--out", default="v8c8_chain.onnx")
    p.add_argument("--chain", type=int, default=8,
                   help="number of BbbKMajor ops")
    p.add_argument("--mode", choices=["chain", "independent"], default="chain",
                   help="chain: out[i] = bbb(out[i-1], wt). "
                        "independent: each bbb takes a fresh model input act_i, "
                        "no data dep — apples-to-apples with native independent.")
    p.add_argument("--wt_layout", choices=["nmaj", "kmaj"], default="nmaj",
                   help="nmaj: [1, N_t, K_t, 1024] (V8C8 / V73 non-deep). "
                        "kmaj: [1, K_t, N_t, 1024] (V73DEEP — native q::ConvLayer_s1.opt).")
    args = p.parse_args()
    assert args.M == args.K == args.N, "chain only works for square matmul"

    M, K, N = args.M, args.K, args.N
    np.random.seed(args.seed)
    assert M % 32 == 0 and K % 32 == 0 and N % 32 == 0

    HERE = os.path.dirname(os.path.abspath(__file__))
    K_t, N_t = K // 32, N // 32

    # ── Weight: [1, N_t, K_t, 1024] N-tile-outer / K-tile-inner native
    #    layout (Step 2 — matches native q::ConvLayer.opt.weights_to_vtcm
    #    end-state). Within each tile, byte layout is the same 4-row-group:
    #    dst = (r//4)*128 + c*4 + (r%4). QNN's weights_to_vtcm@FB.fB. is
    #    verbatim byte-copy DMA, so swapping the outer two dims here is
    #    invisible to QNN — only the SkelOp kernel addressing changes.
    # Asymmetric wRaw so K-major vs N-major layouts produce DIFFERENT byte
    # arrangements (the previous (i*13)%15 was symmetric → tests indistinguishable).
    # Asymmetric:  wRaw[k, n] = ((k*31 + n*13) % 15) - 7
    k_idx, n_idx = np.meshgrid(np.arange(K), np.arange(N), indexing='ij')
    wRaw_KN = (((k_idx * 31 + n_idx * 13) % 15) - 7).astype(np.int8)
    if args.wt_layout == "nmaj":
        wt_packed = np.zeros((1, N_t, K_t, 1024), dtype=np.int8)
        for nt in range(N_t):
            for kt in range(K_t):
                for r in range(32):
                    for c in range(32):
                        dst = (r // 4) * 128 + c * 4 + (r % 4)
                        wt_packed[0, nt, kt, dst] = wRaw_KN[kt * 32 + r, nt * 32 + c]
    else:  # kmaj — for V73DEEP: outer dim is K_t, inner dim is N_t
        wt_packed = np.zeros((1, K_t, N_t, 1024), dtype=np.int8)
        for kt in range(K_t):
            for nt in range(N_t):
                for r in range(32):
                    for c in range(32):
                        dst = (r // 4) * 128 + c * 4 + (r % 4)
                        wt_packed[0, kt, nt, dst] = wRaw_KN[kt * 32 + r, nt * 32 + c]
    # Reshape declared dims to [1, 1, K, N] u8q to match native ConvLayer_s1.opt's
    # in[1] wt = [1, 1, 256, 256]. Bytes stay in our pre-pack layout (same 65 K B
    # contents); QNN's weights_to_vtcm DMA copies verbatim either way, only the
    # shape label changes.
    wt_init = numpy_helper.from_array(
        wt_packed.view(np.uint8).reshape(1, 1, K, N), name="wt_packed")

    # ── Bias: NATIVE bias_to_vtcm layout, host-pre-folded.
    #
    # Earlier hypothesis "QNN auto-folds our int32" was a misread. QNN
    # delivers our int32 bytes VERBATIM. So we do the fold (= what
    # native q::ConvLayer.opt.bias_to_vtcm does at its host prepare time)
    # right here, in the gen-script. End-state in VTCM is byte-1:1 with
    # native bias_to_vtcm output.
    #
    # Per N-tile (256 B):
    #   bytes 0..127   : 32 × (fp16 scale, fp16 baseline)   per-channel
    #   bytes 128..255 : 32 × int32 effective_bias[c]
    #     effective[c] = -ACT_ZP × Σ_k W[k,c] + bias_q[c]
    bias_q_int32   = np.arange(1, N + 1, dtype=np.int32)
    sum_w          = wRaw_KN.astype(np.int32).sum(axis=0)
    effective_int32 = (-ACT_ZP) * sum_w + bias_q_int32              # (N,)

    # quant_overrides: scale=1.0, out_zp=0 → runtime scale=1, baseline=0.
    runtime_scale_per_chan = np.ones(N, dtype=np.float32)
    out_zp = 0

    n_tiles = N // 32
    bias_fold_bytes = np.zeros((n_tiles, 256), dtype=np.uint8)
    for nt in range(n_tiles):
        for c in range(32):
            n_abs = nt * 32 + c
            scale_u16    = np.float16(512.0 * runtime_scale_per_chan[n_abs]) \
                              .view(np.uint16).item()
            baseline_u16 = (out_zp & 0x1FF) << 7
            bias_fold_bytes[nt, 4*c    : 4*c + 2] = \
                np.array([scale_u16],    np.uint16).view(np.uint8)
            bias_fold_bytes[nt, 4*c + 2: 4*c + 4] = \
                np.array([baseline_u16], np.uint16).view(np.uint8)
            bias_fold_bytes[nt, 128 + 4*c : 128 + 4*c + 4] = \
                np.array([int(effective_int32[n_abs])], np.int32).view(np.uint8)

    # Bias dims [1, N_t, 1, 64] i32 to match native ConvLayer_s1.opt's
    # in[2] bias = [1, 8, 1, 64] (N_t=8 tiles × 64 int32 per tile). Same total
    # bytes (2N int32 = N_t × 64), just factorised the same way as native.
    # bias_fold_bytes is (n_tiles, 256) bytes; view as int32 → (n_tiles, 64),
    # then reshape to (1, n_tiles, 1, 64).
    bias_fold_i32 = bias_fold_bytes.view(np.int32).reshape(1, n_tiles, 1, 64).copy()
    bias_init = numpy_helper.from_array(bias_fold_i32, name="bias")

    # ── VTCM scratch — 2 KB static zero, declared TCM_Only in op signature.
    # QNN auto-DMAs to VTCM (via the same path as wt/bias). At runtime the op
    # uses this region as backing store for act_tbl_all/out_tbl_all so the
    # kernel reads pointer tables from VTCM (matches native ConvLayer_s1.opt
    # 0xfc02_xxxx layout) instead of stack/DDR.
    vtcm_scratch_init = numpy_helper.from_array(
        np.zeros((1, 1, 1, 2048), dtype=np.uint8), name="vtcm_scratch")

    # ── ONNX graph build ─────────────────────────────────────────────
    chain = max(1, int(args.chain))
    in_reshape_dims = numpy_helper.from_array(
        np.array([1, M // 32, 32, K], dtype=np.int64), name="in_reshape_dims")
    out_reshape_dims = numpy_helper.from_array(
        np.array([1, 1, M, N], dtype=np.int64), name="out_reshape_dims")

    initializers = [wt_init, bias_init, vtcm_scratch_init,
                    in_reshape_dims, out_reshape_dims]
    inputs_info = []
    outputs_info = []
    nodes = []
    chain_value_infos = []

    if args.mode == "chain":
        # User-facing single input act_raw, chain of bbb out → in.
        inputs_info.append(helper.make_tensor_value_info(
            "act_raw", TensorProto.UINT8, [1, 1, M, K]))
        outputs_info.append(helper.make_tensor_value_info(
            "out", TensorProto.UINT8, [1, 1, M, N]))
        nodes.append(helper.make_node(
            "Reshape", ["act_raw", "in_reshape_dims"], ["act_4d"],
            name="reshape_in_to_4d"))
        prev = "act_4d"
        for i in range(chain):
            out_name = f"mm_c8_{i}" if i < chain - 1 else "mm_c8"
            nodes.append(helper.make_node(
                "BbbKMajor",
                inputs=["bias", "wt_packed", prev, "vtcm_scratch"],
                outputs=[out_name],
                name=f"bbb_chain{i}",
                domain=DOMAIN,
            ))
            chain_value_infos.append(helper.make_tensor_value_info(
                out_name, TensorProto.UINT8, [1, M // 32, 32, N]))
            prev = out_name
        nodes.append(helper.make_node(
            "Reshape", ["mm_c8", "out_reshape_dims"], ["out"],
            name="reshape_out_to_4d"))
    else:
        # Independent mode: 8 separate model inputs act_raw_i, 8 BbbKMajor,
        # 8 outputs out_i. No data dep between bbb. shared wt/bias static.
        for i in range(chain):
            in_name  = f"act_raw_{i}" if i > 0 else "act_raw"
            out_name = f"out_{i}"
            inputs_info.append(helper.make_tensor_value_info(
                in_name, TensorProto.UINT8, [1, 1, M, K]))
            outputs_info.append(helper.make_tensor_value_info(
                out_name, TensorProto.UINT8, [1, 1, M, N]))
            # per-instance reshape in
            in_reshape_name = f"in_reshape_dims_{i}" if i > 0 else "in_reshape_dims"
            if i > 0:
                initializers.append(numpy_helper.from_array(
                    np.array([1, M // 32, 32, K], dtype=np.int64),
                    name=in_reshape_name))
            act4d_name = f"act_4d_{i}"
            mm_c8_name = f"mm_c8_{i}"
            nodes.append(helper.make_node(
                "Reshape", [in_name, in_reshape_name], [act4d_name],
                name=f"reshape_in_{i}"))
            nodes.append(helper.make_node(
                "BbbKMajor",
                inputs=["bias", "wt_packed", act4d_name, "vtcm_scratch"],
                outputs=[mm_c8_name],
                name=f"bbb_indep{i}",
                domain=DOMAIN,
            ))
            chain_value_infos.append(helper.make_tensor_value_info(
                mm_c8_name, TensorProto.UINT8, [1, M // 32, 32, N]))
            out_reshape_name = f"out_reshape_dims_{i}" if i > 0 else "out_reshape_dims"
            if i > 0:
                initializers.append(numpy_helper.from_array(
                    np.array([1, 1, M, N], dtype=np.int64),
                    name=out_reshape_name))
            nodes.append(helper.make_node(
                "Reshape", [mm_c8_name, out_reshape_name], [out_name],
                name=f"reshape_out_{i}"))

    graph = helper.make_graph(
        nodes, name=f"v8c8_{args.mode}",
        inputs=inputs_info, outputs=outputs_info,
        initializer=initializers,
        value_info=chain_value_infos,
    )
    model = helper.make_model(graph,
        producer_name=f"v8c8_{M}x{K}x{N}",
        opset_imports=[
            helper.make_opsetid("", 13),
            helper.make_opsetid(DOMAIN, 1),
        ])
    model.ir_version = 8

    out_path = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    onnx.save(model, out_path)
    # Write per-mode quant_overrides.json next to the ONNX so the runner
    # can use it (independent mode needs entries for act_raw_i / out_i).
    import json as _json
    u8 = {"bitwidth": 8, "dtype": "int", "is_symmetric": "False",
          "scale": 1.0, "offset": 0, "min": 0.0, "max": 255.0}
    bias_enc = {"bitwidth": 32, "dtype": "int", "is_symmetric": "True",
                "scale": 1.0, "offset": 0, "min": -2147483648.0, "max": 2147483647.0}
    if args.mode == "chain":
        act_encs = {"act_raw": [u8], "out": [u8]}
    else:
        act_encs = {"act_raw": [u8]}
        for i in range(chain):
            if i > 0:
                act_encs[f"act_raw_{i}"] = [u8]
            act_encs[f"out_{i}"] = [u8]
    overrides = {"activation_encodings": act_encs,
                 "param_encodings": {"bias": [bias_enc]}}
    with open(os.path.join(os.path.dirname(out_path), "quant_overrides.json"), "w") as f:
        _json.dump(overrides, f, indent=2)
    print(f"  -> {out_path}")
    print(f"  shape: M={M} K={K} N={N}  ACT_ZP={ACT_ZP}")
    print(f"  graph: BbbKMajor(act,wt,bias) → UntileToRowMajor → Reshape  "
          f"(output [1, {M}, {N}] u8)")
    print(f"  wt_packed:    {wt_packed.size} B  shape {list(wt_packed.shape)}  ({args.wt_layout})")
    print(f"  bias:         {bias_fold_i32.nbytes} B  shape [2N={2*N}] int32  "
          f"(NATIVE 256-B/N-tile fold layout, ACT_ZP={ACT_ZP})")
    print(f"    sample: bias_q[0..3]={bias_q_int32[:4].tolist()}, "
          f"effective[0..3]={effective_int32[:4].tolist()}")

    np.save(out_path + ".wRaw_KN.npy", wRaw_KN)
    np.save(out_path + ".bias_q_int32.npy", bias_q_int32)
    np.save(out_path + ".effective_int32.npy", effective_int32)

    # Generate runtime act for device.
    u8_dir = os.path.join(HERE, "runtime_inputs_u8")
    os.makedirs(u8_dir, exist_ok=True)

    def make_a(idx):
        seed = (idx + 1) * 374761393
        return np.array([((i * 37 + seed) & 0xFF) for i in range(M * K)],
                        dtype=np.uint8).reshape(1, 1, M, K)

    if args.mode == "chain":
        aRaw = make_a(0)
        aRaw.tofile(os.path.join(u8_dir, "act_v8c8.raw"))
        # Reference: chained matmul with saturation per step
        cur = aRaw.reshape(M, K).astype(np.int32)
        wt_2d = wRaw_KN.astype(np.int32)
        for _ in range(chain):
            acc = (cur - ACT_ZP) @ wt_2d + bias_q_int32
            cur = np.clip(acc, 0, 255)
        out_ref = cur.astype(np.uint8)
        np.save(out_path + ".out_ref_u8.npy", out_ref)
        print(f"  chain={chain}; ref out[0..3, 0]={out_ref[:4, 0].tolist()}")
    else:
        # independent: 1 fresh act_raw per BbbKMajor; reference per output
        for i in range(chain):
            a_i = make_a(i)
            fname = f"act_v8c8_{i}.raw" if i > 0 else "act_v8c8.raw"
            a_i.tofile(os.path.join(u8_dir, fname))
        # Single reference (output[0]); others have their own bytes,
        # bit-exact verification per output is in run script.
        cur = make_a(0).reshape(M, K).astype(np.int32)
        wt_2d = wRaw_KN.astype(np.int32)
        acc = (cur - ACT_ZP) @ wt_2d + bias_q_int32
        out_ref = np.clip(acc, 0, 255).astype(np.uint8)
        np.save(out_path + ".out_ref_u8.npy", out_ref)
        print(f"  independent mode: {chain} fresh inputs, {chain} outputs; ref out[0..3,0]={out_ref[:4, 0].tolist()}")


if __name__ == "__main__":
    main()
