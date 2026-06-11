#!/usr/bin/env bash
# M6: C=256 block-recursive inverse as a GRAPH of SEPARATE ops (HVX GdnSolve diagonal solve ||
# HMX MatMul merges).  Builds 3 graphs at the same H: split (full), diag (HVX-only), merge (HMX-only).
# Runs each on ssh oneplus with optrace; reports per-tid (HVX 512-515 vs HMX 256) busy spans + overlap %,
# Accelerator (execute) cycles total + per-head, DRAM bytes (VTCM residency), and split-T relerr.
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../../.. && pwd)"; OPDIR="$(cd .. && pwd)"
PKG="GdnSolvePackage"; PROV="${PKG}InterfaceProvider"; ARCH=v75; DEVICE="${DEVICE:-oneplus}"
source "$ROOT/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT/.venv/bin/python"
H="${H:-8}"
bash "$OPDIR/build.sh" >/dev/null 2>&1
X86="$OPDIR/build/x86_64-linux-clang/lib${PKG}.so"; HTP="$OPDIR/build/hexagon-$ARCH/lib${PKG}_htp.so"
CPU="$OPDIR/build/aarch64/lib${PKG}_cpu.so"; CPL="$OPDIR/converter/build/libConverterOpPackage.so"
[ -f A_ref.raw ] || cp A.raw A_ref.raw
cat > _htp.json <<EOF
{"devices":[{"dsp_arch":"$ARCH","soc_id":57,"pd_session":"unsigned","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
cat > _cfg.json <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$(pwd)/_htp.json"}}
EOF

OUT=sbr
"$PY" "$ROOT/scripts/gdn_split_br_probe.py" "$OUT" "$H" A_ref.raw || exit 1

# all tensor names that can be graph inputs across the 3 graphs (for layout flags + transfer)
INPUTS_ALL="Adiag Tdiag_i8 Aoff_10 Aoff_20 Aoff_21 Aoff_30 Aoff_31 Aoff_32"

run() {  # <graphname> <onnx-basename> <inputs-list-line>
  local nm=$1 base=$2 inputs=$3 d=ctx_$1
  local layoutflags=""
  for t in $INPUTS_ALL Tout; do
    layoutflags="$layoutflags --source_model_input_layout $t NONTRIVIAL --desired_input_layout $t NONTRIVIAL"
    layoutflags="$layoutflags --source_model_output_layout $t NONTRIVIAL --desired_output_layout $t NONTRIVIAL"
  done
  qairt-converter -i "$OUT/$base.onnx" --target_backend HTP $layoutflags \
     --op_package_config "$OPDIR/${PKG}.xml" --converter_op_package_lib "$CPL" \
     --quantization_overrides "$OUT/$base.ovr.json" -o "$OUT/$nm.dlc" >"_c_$nm.log" 2>&1 \
     || { echo "$nm CVTFAIL"; tail -8 "_c_$nm.log"; return 1; }
  rm -rf "$d"
  qnn-context-binary-generator --dlc_path "$OUT/$nm.dlc" --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
     --op_packages "$X86:$PROV" --config_file _cfg.json --profiling_level detailed --profiling_option optrace \
     --binary_file "${nm}_ctx" --output_dir "$d" >"_x_$nm.log" 2>&1 \
     || { echo "$nm CTXFAIL"; tail -10 "_x_$nm.log"; return 1; }
  for s in *schematic.bin "$d"/*schematic.bin; do [ -f "$s" ] && mv -f "$s" "$d"/ 2>/dev/null || true; done
  local W; W="$(ssh "$DEVICE" 'echo $HOME/qnn_run')/sbr"
  ssh "$DEVICE" "mkdir -p $W"
  ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./htp.json\"}}' > $W/cfg.json"
  ssh "$DEVICE" "cat > $W/htp.json" < _htp.json
  ssh "$DEVICE" "cat > $W/${nm}_ctx.bin" < "$d/${nm}_ctx.bin"
  ssh "$DEVICE" "cat > $W/lib${PKG}_htp.so" < "$HTP"; ssh "$DEVICE" "cat > $W/lib${PKG}_cpu.so" < "$CPU"
  for f in $INPUTS_ALL; do [ -f "$OUT/$f.raw" ] && ssh "$DEVICE" "cat > $W/$f.raw" < "$OUT/$f.raw"; done
  ssh "$DEVICE" "printf '%s\n' '$inputs' > $W/list.txt"
  ssh "$DEVICE" "cd $W && rm -rf out && LD_LIBRARY_PATH=..:.:/vendor/lib64 ADSP_LIBRARY_PATH='..;.;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' ../qnn-net-run \
     --backend ../libQnnHtp.so --retrieve_context ${nm}_ctx.bin --config_file cfg.json \
     --op_packages ./lib${PKG}_cpu.so:$PROV:CPU,./lib${PKG}_htp.so:$PROV:HTP \
     --input_list list.txt --output_dir out --profiling_level detailed --profiling_option optrace --perf_profile burst" >"_r_$nm.log" 2>&1 || true
  grep -q 'Finished Executing Graphs' "_r_$nm.log" || { echo "$nm RUNFAIL"; tail -10 "_r_$nm.log"; return 1; }
  rm -rf "out_$nm"; mkdir -p "out_$nm"; ssh "$DEVICE" "cd $W && tar cf - out" | tar xf - -C "out_$nm" --strip-components=1 2>/dev/null
  local cyc; cyc=$(qnn-profile-viewer --input_log "out_$nm/qnn-profiling-data_0.log" 2>/dev/null | grep -iE 'Accelerator \(execute\) time \(cycles\)' | grep -oE '[0-9]+ cycles' | head -1 | grep -oE '[0-9]+')
  local us; us=$(qnn-profile-viewer --input_log "out_$nm/qnn-profiling-data_0.log" 2>/dev/null | grep -i 'QNN accelerator (execute) time' | grep -io '[0-9]* us' | head -1)
  echo "$nm cyc=$cyc us=$us H=$H" >> /tmp/sbr_cyc.txt
  echo "  [$nm] Accelerator(execute) cyc=$cyc  $us"
  # decode optrace -> chrometrace.json (net-run only emits the profiling log; viewer builds the trace)
  cp -f "$d"/*schematic.bin "out_$nm"/ 2>/dev/null || true
  "$PY" "$ROOT/scripts/decode_qnn_optrace.py" "out_$nm" --profile-log "out_$nm/qnn-profiling-data_0.log" \
       --schematic "$(ls "$d"/*schematic.bin 2>/dev/null | head -1)" >/dev/null 2>&1 || true
  # analyze optrace: per-tid spans + overlap
  "$PY" "$ROOT/scripts/gdn_split_overlap_analyze.py" "out_$nm/optrace/chrometrace.json" "$nm" "$H" || true
}

: > /tmp/sbr_cyc.txt
echo "=== M6 split block-recursive C=256  H=$H ==="
run merge merge "Tdiag_i8:=Tdiag_i8.raw Aoff_10:=Aoff_10.raw Aoff_20:=Aoff_20.raw Aoff_21:=Aoff_21.raw Aoff_30:=Aoff_30.raw Aoff_31:=Aoff_31.raw Aoff_32:=Aoff_32.raw" || true
run diag  diag  "Adiag:=Adiag.raw" || true
run split split "Adiag:=Adiag.raw Aoff_10:=Aoff_10.raw Aoff_20:=Aoff_20.raw Aoff_21:=Aoff_21.raw Aoff_30:=Aoff_30.raw Aoff_31:=Aoff_31.raw Aoff_32:=Aoff_32.raw" || true

# accuracy: split Tout vs np.linalg.inv
T=$(ls out_split/Result_0/Tout.raw 2>/dev/null || ls out_split/*/Tout.raw 2>/dev/null | head -1)
if [ -n "$T" ]; then
  "$PY" - "$T" "$OUT/Tref.raw" "$H" <<'PY'
import numpy as np, sys
# net-run writes graph outputs dequantized to float32.  Tout = 10 lower-tri blocks stacked block-major
# on axis1: [1, ntri*H, 64, 64].  Reconstruct the assembled [H,256,256] lower-tri T and compare to golden.
BL=64; NB=4; C=256; H=int(sys.argv[3])
tri=[(i,j) for i in range(NB) for j in range(i+1)]
T=np.fromfile(sys.argv[1],dtype=np.float32).reshape(len(tri), H, BL, BL)   # block-major
r=np.fromfile(sys.argv[2],dtype=np.float32).reshape(H,C,C)
Tasm=np.zeros((H,C,C))
for b,(i,j) in enumerate(tri):
    Tasm[:, i*BL:(i+1)*BL, j*BL:(j+1)*BL]=T[b]
# zero the strictly-upper of ref for fair lower-tri compare
rl=np.tril(r)
print(f"  [accuracy] whole-T (lower-tri) relerr vs np.linalg.inv = {np.linalg.norm(Tasm-rl)/np.linalg.norm(rl):.3e}  (H={H})")
dg=np.mean([np.linalg.norm(Tasm[:,i*BL:(i+1)*BL,i*BL:(i+1)*BL]-r[:,i*BL:(i+1)*BL,i*BL:(i+1)*BL])
            /(np.linalg.norm(r[:,i*BL:(i+1)*BL,i*BL:(i+1)*BL])+1e-12) for i in range(NB)])
print(f"  [accuracy] diagonal-block relerr mean = {dg:.3e}")
PY
fi
echo "=== summary ==="
"$PY" - <<'PY'
d={}
for l in open("/tmp/sbr_cyc.txt"):
    p=dict(x.split("=") for x in l.split() if "=" in x)
    if "cyc" in p and p["cyc"].isdigit(): d[l.split()[0]]={"cyc":int(p["cyc"]),"H":int(p.get("H",1))}
if d:
    H=next(iter(d.values()))["H"]
    for k in ("diag","merge","split"):
        if k in d: print(f"  {k:6} total={d[k]['cyc']:>12,} cyc   per-head={d[k]['cyc']//max(H,1):>10,} cyc")
    if {"diag","merge","split"}<=set(d):
        S,M,X=d["diag"]["cyc"],d["merge"]["cyc"],d["split"]["cyc"]
        print(f"  max(diag,merge)={max(S,M):,}  diag+merge={S+M:,}  split={X:,}")
        print(f"  overlap (by total cyc) = (diag+merge-split)/max(diag,merge) = {(S+M-X)/max(S,M,1):.2f}")
        print(f"  split per-head={X//max(H,1):,}  vs baseline 70,201 -> ratio {X/max(H,1)/70201:.2f}x")
PY
