#!/usr/bin/env python3
"""
V9 test: same matmul semantics as gen_v8_onnx.py at small shapes, but
using PackActCrouton + MatMulV9 (Route 1 stack). The OUTPUT must match V8's
output byte-for-byte, since both use the same HMX inner formula.

Usage:
    python gen_v9_test.py --M 32 --K 128 --N 128
"""
import argparse, os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

DOMAIN = "hmx"

def fp32_to_fp16_u16(f):
    return np.float16(f).view(np.uint16).item()

def build_bias_chunk(N, K):
    n_tiles = N // 32
    b = np.zeros((1, 1, n_tiles, 128), dtype=np.uint16)
    for nt in range(n_tiles):
        for c in range(32):
            n = nt * 32 + c
            scale = 1.0 / (K * (1.0 + 0.1 * (n % 7)))
            b[0, 0, nt, 2*c]     = fp32_to_fp16_u16(512.0 * scale)
            b[0, 0, nt, 2*c + 1] = 0x4000
    return b

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--M', type=int, default=32)
    p.add_argument('--K', type=int, default=128)
    p.add_argument('--N', type=int, default=128)
    p.add_argument('--seed', type=int, default=0xB17E)
    args = p.parse_args()

    M, K, N = args.M, args.K, args.N
    assert M % 4 == 0 and K % 128 == 0 and N % 4 == 0
    HERE = os.path.dirname(os.path.abspath(__file__))

    np.random.seed(args.seed)

    # Same static weight pattern as V8.
    wRaw = np.array([((i * 13) % 15) - 7 for i in range(K * N)],
                    dtype=np.int8).reshape(1, 1, K, N).view(np.uint8)
    bias = build_bias_chunk(N, K)
    scratch = np.zeros((1, 1, 1, 2048), dtype=np.uint8)

    aRaw = np.array([(i * 37) & 0xFF for i in range(M * K)],
                    dtype=np.uint8).reshape(1, 1, M, K)

    M_t, K_t, N_t = M // 32, K // 32, N // 32

    # Tensors
    act_in   = helper.make_tensor_value_info("act_raw",   TensorProto.UINT8,  [1, 1, M, K])
    bias_in  = helper.make_tensor_value_info("bias_fp16", TensorProto.UINT16, [1, 1, N_t, 128])
    scr_in   = helper.make_tensor_value_info("scratch",   TensorProto.UINT8,  [1, 1, 1, 2048])
    out_t    = helper.make_tensor_value_info("out",       TensorProto.UINT8,  [1, M_t, N_t, 1024])

    wt_init  = numpy_helper.from_array(wRaw, name="wt_raw")

    # Intermediates: act in Crouton K-major (PackActCrouton), wt in V8 N-major P2 (PackWeightToHmxTileV3)
    packed_act = helper.make_tensor_value_info("packed_act", TensorProto.UINT8, [1, K_t, M_t, 1024])
    packed_wt  = helper.make_tensor_value_info("packed_wt",  TensorProto.UINT8, [1, N_t, K_t, 1024])
    out_tile   = helper.make_tensor_value_info("out_tile",   TensorProto.UINT8, [1, M_t, N_t, 1024])

    # PackActCrouton on activation [1,1,M,K] → [1, K/32, M/4, 128]
    n_pa = helper.make_node("PackActCrouton", ["act_raw"], ["packed_act"],
                            name="pack_act", domain=DOMAIN)
    # PackActCrouton on weight [1,1,K,N] → [1, N/32, K/4, 128]
    # — wait: input rank-4 needs M, K dims; for weight we want K=Mx, N=Kx.
    # Skipping: current PackActCrouton kernel reads dims[-2], dims[-1] as M, K.
    # For weight [1,1,K,N], K becomes "M" arg and N becomes "K" arg of pack.
    # Output shape via shape inference: [1, N/32, K/4, 128]. Match our packed_wt.
    n_pw = helper.make_node("PackActCrouton", ["wt_raw"], ["packed_wt_pre"],
                            name="pack_wt", domain=DOMAIN)
    # Hmm — PackActCrouton outputs [1, K_arg/32, M_arg/4, 128].
    # For weight input [1,1,K,N]: M_arg=K, K_arg=N → output [1, N/32, K/4, 128].
    # MatMulV9 wants packed_wt as [1, K_t, N_grp=N/4, 128] where K_t = K/32.
    # So we need [1, K/32, N/4, 128] but pack gives [1, N/32, K/4, 128]. WRONG.
    #
    # Fix: transpose the weight before pack. Since weight is static, transpose
    # in Python at graph-build time (not a runtime op). Reshape input dims
    # [1,1,K,N] → [1,1,N,K] effectively swaps which is "M" and which is "K".
    # Then pack output [1, K/32, N/4, 128] = packed_wt format MatMulV9 wants.
    pass  # see node redefinitions below

    # Use V8's PackWeightToHmxTileV3 (produces HMX-compatible double-vshuff
     # P2 layout). V9 dlsym path expects this format for hmx_convbbb1x1_stride1.
    wt_init = numpy_helper.from_array(wRaw, name="wt_raw")
    n_pw = helper.make_node("PackWeightToHmxTileV3", ["wt_raw"], ["packed_wt"],
                            name="pack_wt", domain=DOMAIN)
    # packed_wt = [1, N/32, K/32, 1024] (V8 weight format)

    # Update the static initializer ref (we're using wt_init not wT_init now)
    wT_init = wt_init

    # MatMulV9 (BbbKMajor)
    n_mm = helper.make_node(
        "BbbKMajor",
        ["packed_act", "packed_wt", "bias_fp16", "scratch"],
        ["out_tile"],
        name="mmv9", domain=DOMAIN,
    )
    # TcmDramCopy → out
    n_cp = helper.make_node("TcmDramCopy", ["out_tile"], ["out"],
                            name="tcm2ddr", domain=DOMAIN)

    graph = helper.make_graph(
        [n_pa, n_pw, n_mm, n_cp],
        "v9_test",
        [act_in, bias_in, scr_in],
        [out_t],
        [wT_init],
        value_info=[packed_act, packed_wt, out_tile],
    )
    model = helper.make_model(
        graph,
        producer_name="v9_test",
        opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DOMAIN, 1)],
    )
    model.ir_version = 8
    onnx.save(model, os.path.join(HERE, "v9_model.onnx"))
    print(f"  -> v9_model.onnx (M={M}, K={K}, N={N})")

    # Runtime inputs
    in_dir = os.path.join(HERE, "runtime_inputs_v9")
    os.makedirs(in_dir, exist_ok=True)
    aRaw.tofile(os.path.join(in_dir, "act.raw"))
    bias.tofile(os.path.join(in_dir, "bias.raw"))
    scratch.tofile(os.path.join(in_dir, "scratch.raw"))
    print(f"  -> runtime_inputs_v9/{{act,bias,scratch}}.raw")

    with open(os.path.join(HERE, "input_list_v9.txt"), "w") as f:
        f.write("act_raw:=runtime_inputs_v9/act.raw bias_fp16:=runtime_inputs_v9/bias.raw scratch:=runtime_inputs_v9/scratch.raw\n")

if __name__ == "__main__":
    main()
