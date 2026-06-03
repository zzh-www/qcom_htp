#!/usr/bin/env bash
# Attribute U's 1.1e-2 device error: T-floor (solve's 8-bit) vs U's own matmul quant.
# Build sub_U = [vc, betac, T] → U, inject T as fp32 truth (no encoding), quantize the rest as usual,
# run on device, compare to fp64 U. U_self (this) vs U_full (1.1e-2 from gdn_stage_error) ⇒ split.
set -uo pipefail
cd "$(dirname "$0")/quant_v2_L0"
ROOT_DIR="$(cd ../../.. && pwd)"
DEVICE="${DEVICE:-oneplus}"; ARCH="${ARCH:-v75}"; SOC_ID="${SOC_ID:-57}"
source "$ROOT_DIR/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT_DIR/.venv/bin/python"
T=/Add_32_output_0; U=/Transpose_17_output_0

# 1) extract sub_U and write its override from full-graph ORT ranges (T skipped → float)
"$PY" - <<PY
import onnx, onnx.utils, onnxruntime as ort, numpy as np, json, os, sys
sys.path.insert(0,"$ROOT_DIR/scripts"); os.environ["GDN_NO_VSCALE"]="1"
from gdn_onnx_kernel import INPUT_NAMES, const_inputs, per_head_vscale, _golden_chunk_args
T="$T"; U="$U"
m=onnx.load("gdn_q.onnx"); have={o.name for o in m.graph.output}
for n in m.graph.node:
    for o in n.output:
        if o and o not in have: vi=onnx.ValueInfoProto(); vi.name=o; m.graph.output.append(vi); have.add(o)
s=ort.InferenceSession(m.SerializeToString(),providers=["CPUExecutionProvider"])
a=_golden_chunk_args("$ROOT_DIR/tests/gdn/golden/p00_L00.npz",0)
vs,ivs=per_head_vscale(a[2],a[5])
feed={n:t.float().numpy().astype(np.float32) for n,t in zip(INPUT_NAMES,a)}
feed.update({k:v.astype(np.float32) for k,v in const_inputs().items()})
feed["vscale"]=vs.numpy().astype(np.float32); feed["inv_vscale"]=ivs.numpy().astype(np.float32)
on=[o.name for o in s.get_outputs()]; vals=dict(zip(on,s.run(on,feed)))
onnx.utils.extract_model("gdn_q.onnx","subU.onnx",["vc","betac","vscale",T],[U])
sub=onnx.load("subU.onnx")
i8={n.input[1] for n in sub.graph.node if n.op_type=="MatMul" and len(n.input)>=2}
tensors=set(["vc","betac","vscale"]) | {o for n in sub.graph.node for o in n.output if o}
encs=[]
for tn in tensors:
    if tn==T or tn not in vals and tn not in feed: continue          # T → float (skip); skip unknowns
    v=np.asarray(vals.get(tn, feed.get(tn))); amax=float(np.abs(v).max()) or 1e-6
    if tn in i8: encs.append({"name":tn,"output_dtype":"int8","y_scale":amax/127.0})
    else: encs.append({"name":tn,"output_dtype":"int16","y_scale":amax/32767.0})
json.dump({"version":"2.0.0","encodings":encs}, open("subU_ovr.json","w"))
print("subU inputs:", [i.name for i in sub.graph.input], "| i8:", sorted(i8), "| T floated:", T)
PY

# 2) convert + ctxgen (T input is fp32/float, no encoding)
KEEP=$("$PY" -c "import onnx;print(' '.join(i.name for i in onnx.load('subU.onnx').graph.input))")
LAY=(); for n in $KEEP; do LAY+=(--source_model_input_layout "$n" NONTRIVIAL --desired_input_layout "$n" NONTRIVIAL); done
LAY+=(--source_model_output_layout "$U" NONTRIVIAL --desired_output_layout "$U" NONTRIVIAL)
qairt-converter -i subU.onnx --target_backend HTP \
    -d vc 1,32,64,128 -d betac 1,32,64 -d vscale 1,32,1,1 -d "$T" 1,32,64,64 \
    "${LAY[@]}" --quantization_overrides subU_ovr.json -o subU.dlc > _u.log 2>&1 || { echo CONVERT-FAIL; tail -6 _u.log; exit 1; }
rm -rf uctx; qnn-context-binary-generator --dlc_path subU.dlc --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" --config_file _cfg.json --binary_file u_ctx --output_dir uctx > _ux.log 2>&1 || { echo "CTXGEN-FAIL: $(grep -iE 'error|incorrect|expected' _ux.log|head -1)"; exit 1; }

# 3) device: feed vc, betac (truth) + T = fp64 reference (/tmp/ref_02_T.raw)
DEV="$(ssh "$DEVICE" 'echo $HOME/qnn_run')"; W="$DEV/gdn_stage"; ssh "$DEVICE" "mkdir -p $W"
ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./htp.json\"}}' > $W/cfg.json"
ssh "$DEVICE" "cat > $W/htp.json" < _htp.json
ssh "$DEVICE" "cat > $W/vc.raw" < vc.raw; ssh "$DEVICE" "cat > $W/betac.raw" < betac.raw
ssh "$DEVICE" "cat > $W/vscale.raw" < vscale.raw
ssh "$DEVICE" "cat > $W/T.raw" < /tmp/ref_02_T.raw
echo "vc:=vc.raw betac:=betac.raw vscale:=vscale.raw $T:=T.raw" > u_list.txt
ssh "$DEVICE" "cat > $W/u_list.txt" < u_list.txt
ssh "$DEVICE" "cat > $W/u_ctx.bin" < uctx/u_ctx.bin
ssh "$DEVICE" "cd $W && rm -rf out && LD_LIBRARY_PATH=..:/vendor/lib64 ADSP_LIBRARY_PATH=.. ../qnn-net-run --backend ../libQnnHtp.so --retrieve_context u_ctx.bin --config_file cfg.json --input_list u_list.txt --output_dir out --perf_profile burst" > _ur.log 2>&1 || true
rm -rf usout; ssh "$DEVICE" "cd $W && tar cf - out" | tar xf - -C . 2>/dev/null && mv out usout
"$PY" - <<'PY'
import numpy as np, glob
ref=np.fromfile("/tmp/ref_03_U.raw","<f4"); f=glob.glob("usout/**/*.raw",recursive=True)
if not f: print("NO-OUTPUT"); raise SystemExit
raw=np.fromfile(f[0],np.uint8); dev=(raw.view('<f2') if raw.size==ref.size*2 else raw.view('<f4')).astype(np.float32)
e=np.linalg.norm(dev.ravel()-ref)/(np.linalg.norm(ref)+1e-12)
print(f"\n  U_self (T injected fp32) device relerr = {e:.4e}")
print(f"  U_full (T from solve)              = 1.12e-02   [from gdn_stage_error]")
print(f"  => T-floor contribution ≈ sqrt(1.12e-2^2 - U_self^2);  U-own matmul ≈ U_self")
PY