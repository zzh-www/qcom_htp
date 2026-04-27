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
    p.add_argument("-o", "--out", default="v8c8_test.onnx")
    args = p.parse_args()

    M, K, N = args.M, args.K, args.N
    np.random.seed(args.seed)
    assert M % 32 == 0 and K % 32 == 0 and N % 32 == 0

    HERE = os.path.dirname(os.path.abspath(__file__))
    K_t, N_t = K // 32, N // 32

    # ── Weight: [1, K_t, N_t, 1024] in native ConvLayer K-tile / N-tile /
    #    4-row-group / 1024-byte tile layout. ────────────────────────────
    wRaw_KN = np.array([((i * 13) % 15) - 7 for i in range(K * N)],
                       dtype=np.int8).reshape(K, N)
    wt_packed = np.zeros((1, K_t, N_t, 1024), dtype=np.int8)
    for kt in range(K_t):
        for nt in range(N_t):
            for r in range(32):
                for c in range(32):
                    dst = (r // 4) * 128 + c * 4 + (r % 4)
                    wt_packed[0, kt, nt, dst] = wRaw_KN[kt * 32 + r, nt * 32 + c]
    wt_init = numpy_helper.from_array(wt_packed.view(np.uint8), name="wt_packed")

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

    bias_fold_i32 = bias_fold_bytes.reshape(-1).view(np.int32).copy()  # (2N,) int32
    bias_init = numpy_helper.from_array(bias_fold_i32, name="bias")

    # ── ONNX graph (Step 3 — Crouton_8 output, no UntileToRowMajor) ──
    # BbbKMajor output is now [1, M/32, 32, N] (logical Crouton_8).
    # Reshape collapses to user-facing [1, M, N]. QNN compiler should
    # auto-insert q::ForceFormat_Flat between BbbKMajor and Reshape so
    # the user tensor lands as flat row-major in DDR via the framework
    # Output op (replaces our hand-rolled UntileToRowMajor).
    act_in = helper.make_tensor_value_info(
        "act_raw", TensorProto.UINT8, [1, M // 32, 32, K])
    out_info = helper.make_tensor_value_info("out", TensorProto.UINT8, [1, M, N])

    bbb_node = helper.make_node(
        "BbbKMajor",
        inputs=["act_raw", "wt_packed", "bias"],
        outputs=["mm_c8"],
        name="bbb_M0_N0",
        domain=DOMAIN,
    )
    out_reshape_dims = numpy_helper.from_array(
        np.array([1, M, N], dtype=np.int64), name="out_reshape_dims")
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["mm_c8", "out_reshape_dims"],
        outputs=["out"],
        name="reshape_out_to_3d",
    )

    graph = helper.make_graph(
        [bbb_node, reshape_node], name="v8c8_test",
        inputs=[act_in], outputs=[out_info],
        initializer=[wt_init, bias_init, out_reshape_dims],
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
    print(f"  -> {out_path}")
    print(f"  shape: M={M} K={K} N={N}  ACT_ZP={ACT_ZP}")
    print(f"  graph: BbbKMajor(act,wt,bias) → UntileToRowMajor → Reshape  "
          f"(output [1, {M}, {N}] u8)")
    print(f"  wt_packed:    {wt_packed.size} B  shape [1,{K_t},{N_t},1024]")
    print(f"  bias:         {bias_fold_i32.nbytes} B  shape [2N={2*N}] int32  "
          f"(NATIVE 256-B/N-tile fold layout, ACT_ZP={ACT_ZP})")
    print(f"    sample: bias_q[0..3]={bias_q_int32[:4].tolist()}, "
          f"effective[0..3]={effective_int32[:4].tolist()}")

    np.save(out_path + ".wRaw_KN.npy", wRaw_KN)
    np.save(out_path + ".bias_q_int32.npy", bias_q_int32)
    np.save(out_path + ".effective_int32.npy", effective_int32)

    # Generate runtime act for device
    u8_dir = os.path.join(HERE, "runtime_inputs_u8")
    os.makedirs(u8_dir, exist_ok=True)
    aRaw_flat = np.array([(i * 37) & 0xFF for i in range(M * K)], dtype=np.uint8)
    aRaw = aRaw_flat.reshape(1, 1, M, K)
    aRaw.tofile(os.path.join(u8_dir, "act_v8c8.raw"))

    # Reference matmul matching native ConvLayer math (centered u8 act):
    #   acc[m,n] = effective_int32[n] + Σ_k act_u8[m,k] × wRaw_i8[k,n]
    #            = bias_q[n] + Σ_k (act_u8[m,k] - ACT_ZP) × wRaw_i8[k,n]
    #   out_u8 = saturate_u8(top9(baseline) + floor(acc × scale_fp16 / 512))
    # For our quant config (scale=1, out_zp=0): out = saturate_u8(acc).
    act_2d = aRaw_flat.reshape(M, K).astype(np.int32)
    wt_2d  = wRaw_KN.astype(np.int32)
    acc    = (act_2d - ACT_ZP) @ wt_2d         # centered act × wt
    acc   += bias_q_int32                       # + bias
    out_ref = np.clip(acc, 0, 255).astype(np.uint8)
    np.save(out_path + ".out_ref_u8.npy", out_ref)
    print(f"  ref out: [{M},{N}] u8 (saturated). ref[0..3, 0] = {out_ref[:4, 0].tolist()}")


if __name__ == "__main__":
    main()
