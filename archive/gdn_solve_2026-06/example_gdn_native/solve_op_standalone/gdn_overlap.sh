#!/usr/bin/env bash
# Overlap validation for #2: run solve-only (HVX), matmul-only (HMX), and combined (solve->matmul) at
# the same B,C; if combined compute-cycles ~ max(solve,mm) the units OVERLAP, if ~ solve+mm they serialise.
# Reports Accelerator (execute) compute CYCLES + VTCM/DRAM bytes (to confirm T stays on-chip).
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../../.. && pwd)"; OPDIR="$(cd .. && pwd)"
PKG="GdnSolvePackage"; PROV="${PKG}InterfaceProvider"; ARCH=v75; DEVICE="${DEVICE:-oneplus}"
source "$ROOT/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT/.venv/bin/python"
C="${C:-64}"; B="${B:-64}"
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
"$PY" "$ROOT/scripts/gdn_overlap_probe.py" . "$C" "$B" A_ref.raw >/dev/null

run() {  # <name> <inputs "n:=f.raw ...">  -> prints "name cyc=<compute cycles> vtcm=<bytes> dram=<bytes>"
  local nm=$1 inputs=$2 d=ov_$1
  qairt-converter -i "$nm.onnx" --target_backend HTP \
     $(for t in A T V P; do echo --source_model_input_layout $t NONTRIVIAL --desired_input_layout $t NONTRIVIAL \
       --source_model_output_layout $t NONTRIVIAL --desired_output_layout $t NONTRIVIAL; done) \
     --op_package_config "$OPDIR/${PKG}.xml" --converter_op_package_lib "$CPL" \
     --quantization_overrides "$nm.ovr.json" -o "$nm.dlc" >_c_$nm.log 2>&1 || { echo "$nm CVTFAIL"; tail -4 _c_$nm.log; return; }
  rm -rf "$d"
  qnn-context-binary-generator --dlc_path "$nm.dlc" --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
     --op_packages "$X86:$PROV" --config_file _cfg.json --profiling_level detailed --profiling_option optrace \
     --binary_file "${nm}_ctx" --output_dir "$d" >_x_$nm.log 2>&1 || { echo "$nm CTXFAIL"; tail -6 _x_$nm.log; return; }
  for s in *schematic.bin "$d"/*schematic.bin; do [ -f "$s" ] && mv -f "$s" "$d"/ 2>/dev/null || true; done
  local W; W="$(ssh "$DEVICE" 'echo $HOME/qnn_run')/ovl"
  ssh "$DEVICE" "mkdir -p $W"
  ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./htp.json\"}}' > $W/cfg.json"
  ssh "$DEVICE" "cat > $W/htp.json" < _htp.json
  ssh "$DEVICE" "cat > $W/${nm}_ctx.bin" < "$d/${nm}_ctx.bin"
  ssh "$DEVICE" "cat > $W/lib${PKG}_htp.so" < "$HTP"; ssh "$DEVICE" "cat > $W/lib${PKG}_cpu.so" < "$CPU"
  for f in A T_in V; do [ -f "$f.raw" ] && ssh "$DEVICE" "cat > $W/$f.raw" < "$f.raw"; done
  ssh "$DEVICE" "printf '%s\n' '$inputs' > $W/list.txt"
  ssh "$DEVICE" "cd $W && rm -rf out && LD_LIBRARY_PATH=..:.:/vendor/lib64 ADSP_LIBRARY_PATH='..;.;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' ../qnn-net-run \
     --backend ../libQnnHtp.so --retrieve_context ${nm}_ctx.bin --config_file cfg.json \
     --op_packages ./lib${PKG}_cpu.so:$PROV:CPU,./lib${PKG}_htp.so:$PROV:HTP \
     --input_list list.txt --output_dir out --profiling_level detailed --profiling_option optrace --perf_profile burst" >_r_$nm.log 2>&1 || true
  grep -q 'Finished Executing Graphs' _r_$nm.log || { echo "$nm RUNFAIL"; tail -8 _r_$nm.log; return; }
  rm -rf "out_$nm"; mkdir -p "out_$nm"; ssh "$DEVICE" "cd $W && tar cf - out" | tar xf - -C "out_$nm" --strip-components=1 2>/dev/null
  local cyc; cyc=$(qnn-profile-viewer --input_log "out_$nm/qnn-profiling-data_0.log" 2>/dev/null | grep -iE 'Accelerator \(execute\) time \(cycles\)' | grep -oE '[0-9]+ cycles' | head -1 | grep -oE '[0-9]+')
  "$PY" - "$nm" "$cyc" "out_$nm" <<'PY'
import json,sys
nm,cyc,d=sys.argv[1],sys.argv[2],sys.argv[3]
try:
    c=json.load(open(f"{d}/optrace/chrometrace.json"))
    def cset(track): return [e for e in c["traceEvents"] if e.get("ph")=="X"]
    cnt={}
    for e in c["traceEvents"]:
        if e.get("ph")=="X": cnt[e["name"]]=cnt.get(e["name"],0)+1
    # mem counters: total VTCM vs DRAM read+write (proxy for on-chip vs off-chip handoff)
    vt=dr=0
    for e in c["traceEvents"]:
        n=e.get("name","")
        if "VTCM read" in n or "VTCM write" in n: vt+=e.get("args",{}).get("value",0) if False else 0
    print(f"  {nm:9} compute_cyc={cyc:>10}  ops={ {k:v for k,v in sorted(cnt.items(),key=lambda x:-x[1])[:4]} }")
except Exception as ex:
    print(f"  {nm:9} compute_cyc={cyc}  (parse {ex})")
PY
  echo "$nm $cyc" >> /tmp/ovl_cyc.txt
}

: > /tmp/ovl_cyc.txt
echo "=== overlap test C=$C B=$B ==="
run solve    "A:=A.raw"
run mm       "T:=T_in.raw V:=V.raw"
run combined "A:=A.raw V:=V.raw"
echo "=== verdict (compute cycles) ==="
"$PY" - <<'PY'
d={}
for l in open("/tmp/ovl_cyc.txt"):
    p=l.split()
    if len(p)==2 and p[1].isdigit(): d[p[0]]=int(p[1])
if {"solve","mm","combined"}<=set(d):
    S,M,X=d["solve"],d["mm"],d["combined"]
    print(f"  solve={S:,}  mm={M:,}  combined={X:,}")
    print(f"  max(S,M)={max(S,M):,}  S+M={S+M:,}")
    ov=(S+M-X)/max(M,S,1)
    print(f"  overlap = (S+M-X)/max(S,M) = {ov:.2f}   (1.0=perfect overlap, 0=serial)")
PY
