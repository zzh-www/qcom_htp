#!/usr/bin/env python3
"""
Generate ONNX + reference output for PackActCrouton at a given (M, K).

Output layout per kernel doc: [1, K/32, M/4, 128] u8 contiguous.
Block (g, h) at offset (g*M_grp + h)*128 holds 4 spatial × 32 depth bytes:
    block[r*32 + c] = act[h*4+r][g*32+c]   for r in 0..3, c in 0..31

Usage:
    python gen_pack_act_crouton_test.py --M 32 --K 128
    python gen_pack_act_crouton_test.py --M 64 --K 256
"""
import argparse, os
import numpy as np
import onnx
from onnx import helper, TensorProto

def crouton_pack_b_reference(act, M, K):
    """Output flat [1, K/32, M/32, 1024]; bytes are identical to old
    [1, K/32, M/4, 128] layout — only the rank-4 view differs."""
    out = np.zeros(M * K, dtype=np.uint8)
    M_grp = M // 4
    for h in range(M_grp):
        for g in range(K//32):
            blk_off = (g * M_grp + h) * 128
            for r in range(4):
                for c in range(32):
                    out[blk_off + r*32 + c] = act[h*4+r, g*32+c]
    return out.reshape(1, K//32, M//32, 1024)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--M', type=int, default=32)
    p.add_argument('--K', type=int, default=128)
    p.add_argument('--seed', type=int, default=0xCAFE)
    args = p.parse_args()

    M, K = args.M, args.K
    assert M % 4 == 0, f"M must be %4 (got {M})"
    assert K % 128 == 0, f"K must be %128 (got {K})"

    HERE = os.path.dirname(os.path.abspath(__file__))
    DOMAIN = "hmx"

    np.random.seed(args.seed)
    act = np.random.randint(0, 256, size=(M, K), dtype=np.uint8)

    # ONNX
    act_in  = helper.make_tensor_value_info("act_raw",  TensorProto.UINT8, [1, 1, M, K])
    out_t   = helper.make_tensor_value_info("crouton_out", TensorProto.UINT8, [1, K//32, M//32, 1024])
    n = helper.make_node("PackActCrouton", ["act_raw"], ["crouton_out"],
                         name="pack", domain=DOMAIN)
    g = helper.make_graph([n], "pack_act_crouton_test", [act_in], [out_t])
    m = helper.make_model(g, producer_name="pack_act_crouton_test",
                          opset_imports=[helper.make_opsetid("", 13),
                                         helper.make_opsetid(DOMAIN, 1)])
    m.ir_version = 8
    onnx.save(m, os.path.join(HERE, "pack_act_test.onnx"))
    print(f"  -> pack_act_test.onnx (M={M}, K={K})")

    # Inputs / reference
    in_dir = os.path.join(HERE, "runtime_inputs_pack_act")
    os.makedirs(in_dir, exist_ok=True)
    act.reshape(1,1,M,K).tofile(os.path.join(in_dir, "act.raw"))

    ref = crouton_pack_b_reference(act, M, K)
    ref.tofile(os.path.join(HERE, "pack_act_ref.raw"))
    print(f"  -> runtime_inputs_pack_act/act.raw ({act.nbytes} B)")
    print(f"  -> pack_act_ref.raw ({ref.nbytes} B)")

    with open(os.path.join(HERE, "input_list_pack_act.txt"), "w") as f:
        f.write("act_raw:=runtime_inputs_pack_act/act.raw\n")

if __name__ == "__main__":
    main()
