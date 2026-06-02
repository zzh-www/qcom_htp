#!/usr/bin/env bash
# Device validation for GdnSolveDiag (M6 Op1): A->GdnSolveDiag->T, HVX diagonal solves only,
# multithreaded=true (central tiler over heads).  Reports block-diagonal T relerr vs np.linalg.inv +
# GdnSolveDiag op cyc (optrace aggregate + PROBE_CYCLES per-stage work-volume).
#
# Env: H (heads, default 16), CB (chunk 128|256, default 256), EXTRA_DEFS (debug build flags).
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../../.. && pwd)"; OPDIR="$(cd .. && pwd)"
PKG="GdnSolveDiagPackage"; PROV="${PKG}InterfaceProvider"; ARCH=v75; DEVICE="${DEVICE:-oneplus}"
source "$ROOT/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT/.venv/bin/python"
H="${H:-16}"; CB="${CB:-256}"; EXTRA_DEFS="${EXTRA_DEFS:-}"

EXTRA_DEFS="${EXTRA_DEFS} -DGDN_BR_C=${CB} -DGDN_DIAG_HD_DDR" bash "$OPDIR/build.sh" >_build.log 2>&1 || { echo BUILDFAIL; tail -20 _build.log; exit 1; }
X86="$OPDIR/build/x86_64-linux-clang/lib${PKG}.so"; HTP="$OPDIR/build/hexagon-$ARCH/lib${PKG}_htp.so"
CPU="$OPDIR/build/aarch64/lib${PKG}_cpu.so"; CPL="$OPDIR/converter/build/libConverterOpPackage.so"
cat > _htp.json <<EOF
{"devices":[{"dsp_arch":"$ARCH","soc_id":57,"pd_session":"unsigned","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
cat > _cfg.json <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$(pwd)/_htp.json"}}
EOF

"$PY" "$ROOT/scripts/gdn_solve_diag_probe.py" . "$H" "$CB" || { echo PROBEFAIL; exit 1; }

qairt-converter -i solve_diag.onnx --target_backend HTP \
   --source_model_input_layout A NONTRIVIAL --desired_input_layout A NONTRIVIAL \
   --source_model_output_layout T NONTRIVIAL --desired_output_layout T NONTRIVIAL \
   --op_package_config "$OPDIR/${PKG}.xml" --converter_op_package_lib "$CPL" \
   --quantization_overrides ovr_solve_diag.json -o solve_diag.dlc >_c.log 2>&1 || { echo CVTFAIL; tail -8 _c.log; exit 1; }
rm -rf ctx_s
qnn-context-binary-generator --dlc_path solve_diag.dlc --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
   --op_packages "$X86:$PROV" --config_file _cfg.json --profiling_level detailed --profiling_option optrace \
   --binary_file solve_diag_ctx --output_dir ctx_s >_x.log 2>&1 || { echo CTXFAIL; tail -8 _x.log; exit 1; }
for s in *schematic.bin ctx_s/*schematic.bin; do [ -f "$s" ] && mv -f "$s" ctx_s/ 2>/dev/null || true; done

W="$(ssh "$DEVICE" 'echo $HOME/qnn_run')/solve_diag"
ssh "$DEVICE" "mkdir -p $W"
ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./htp.json\"}}' > $W/cfg.json"
ssh "$DEVICE" "cat > $W/htp.json" < _htp.json
ssh "$DEVICE" "cat > $W/solve_diag_ctx.bin" < ctx_s/solve_diag_ctx.bin
ssh "$DEVICE" "cat > $W/lib${PKG}_htp.so" < "$HTP"; ssh "$DEVICE" "cat > $W/lib${PKG}_cpu.so" < "$CPU"
ssh "$DEVICE" "cat > $W/A.raw" < A.raw
ssh "$DEVICE" "printf 'A:=A.raw\n' > $W/list.txt"
ssh "$DEVICE" "cd $W && rm -rf out && LD_LIBRARY_PATH=..:.:/vendor/lib64 ADSP_LIBRARY_PATH='..;.;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' ../qnn-net-run \
   --backend ../libQnnHtp.so --retrieve_context solve_diag_ctx.bin --config_file cfg.json \
   --op_packages ./lib${PKG}_cpu.so:$PROV:CPU,./lib${PKG}_htp.so:$PROV:HTP \
   --input_list list.txt --output_dir out --profiling_level detailed --profiling_option optrace --perf_profile burst" >_r.log 2>&1 || true
grep -q 'Finished Executing Graphs' _r.log || { echo "RUNFAIL"; tail -12 _r.log; exit 1; }
rm -rf out_s; mkdir -p out_s; ssh "$DEVICE" "cd $W && tar cf - out" | tar xf - -C out_s --strip-components=1 2>/dev/null

wall=$(qnn-profile-viewer --input_log out_s/qnn-profiling-data_0.log 2>/dev/null | grep -i 'QNN accelerator (execute) time' | grep -io '[0-9]* us' | head -1)
T=$(ls out_s/Result_0/T.raw 2>/dev/null || ls out_s/*/T.raw 2>/dev/null | head -1)
echo "  >>> H=$H CB=$CB WALL=$wall  T=$T  EXTRA_DEFS='${EXTRA_DEFS}'"

"$PY" - "$H" "$T" "$CB" <<'PY'
import sys, numpy as np
H=int(sys.argv[1]); Tf=sys.argv[2]; C=int(sys.argv[3]); BL=64; NB=C//BL
t=np.fromfile(Tf,dtype=np.float32)
if t.size < H*C*C:
    print("  OUTPUT TRUNCATED size",t.size,"expected",H*C*C); sys.exit()
t=t[:H*C*C].reshape(H,C,C)
r=np.fromfile('T_diag_ref.raw',dtype=np.float32)[:H*C*C].reshape(H,C,C)
def rel(a,b):
    d=np.linalg.norm(a-b); n=np.linalg.norm(b); return d/(n+1e-12)
hs = list(range(1,H)) if H>1 else [0]
# diagonal-block relerr (the blocks Op1 writes)
dg=[]
for h in hs:
    vals=[rel(t[h,i*BL:(i+1)*BL,i*BL:(i+1)*BL], r[h,i*BL:(i+1)*BL,i*BL:(i+1)*BL]) for i in range(NB)]
    dg.append(np.mean(vals))
print(f"  diag-block T relerr vs np.linalg.inv (heads {hs[0]}..{hs[-1]}): mean {np.mean(dg):.3e} max {np.max(dg):.3e}")
for n,h in enumerate(hs[:4]):
    print(f"    head {h}: diag relerr {dg[n]:.3e}")
print(f"  PASS gate ~1.5e-3 (int16 diag faithful): {'PASS' if np.mean(dg)<3e-3 else 'CHECK'}")
PY

# per-stage PROBE_CYCLES (uint32 in T head 0): p0=diag p4=quantA p5=requant p6=tiles, /(h1-h0) per head
if echo "$EXTRA_DEFS" | grep -q PROBE_CYCLES; then
"$PY" - "$T" "$CB" <<'PY'
import sys,numpy as np
Tf=sys.argv[1]; C=int(sys.argv[2])
raw=np.fromfile(Tf,dtype=np.float32).view(np.uint32)
p=raw[:8]
nh=max(int(p[3]),1)
print(f"  PROBE Op1 per-head (÷{nh} heads/thread, nslices={p[1]}, heads={p[2]}):")
print(f"    diag={p[0]/nh:,.0f}  quantA(diag-i8)={p[4]/nh:,.0f}  requant(T-out)={p[5]/nh:,.0f}  offdiagA-tiles={p[6]/nh:,.0f}")
print(f"    Op1 stage-sum/head={(int(p[0])+int(p[4])+int(p[5])+int(p[6]))/nh:,.0f}")
PY
fi

# aggregate op cyc
"$PY" "$ROOT/scripts/decode_qnn_optrace.py" out_s --profile-log out_s/qnn-profiling-data_0.log --schematic "$(ls ctx_s/*schematic.bin|head -1)" >/dev/null 2>&1
"$PY" - out_s "$H" "$CB" <<'PY'
import json,sys,numpy as np
d=sys.argv[1]; H=int(sys.argv[2]); C=int(sys.argv[3])
try:
    ct=json.load(open(f"{d}/optrace/chrometrace.json"))
except Exception as e:
    print("  (no chrometrace:",e,")"); sys.exit()
ev=ct['traceEvents'] if isinstance(ct,dict) else ct
durs=sorted(e.get('dur',0) for e in ev if 'GdnSolveDiag' in str(e.get('name','')) and e.get('dur',0)>200)
print(f"  GdnSolveDiag op_dur events (cyc): {durs}")
if durs:
    print(f"    n_tile_ops={len(durs)} max_dur={durs[-1]:,} -> per-head(max tile,8h)={durs[-1]/8:,.0f}  sum_dur={sum(durs):,}")
PY
