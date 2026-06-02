#!/usr/bin/env bash
# FAST batched CK sweep for the M8 per-chunk split.  The op .so's are IDENTICAL across CK (CK only
# changes the ONNX graph topology), so BUILD ONCE + PUSH LIBS ONCE, then loop CK doing only
# {gen onnx -> convert -> ctxgen -> push ctx.bin -> net-run -> tar back -> decode + ASCII timeline}.
# This turns the old per-CK-rebuild sweep (tens of min) into ~build-once + 4x(ctxgen+run) (~minutes).
#
# Env: H (heads, default 16), CB (chunk C, default 256), CKS (chunk sizes, default "16 4 2 1"), EXTRA_DEFS.
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../../.. && pwd)"; DIAGDIR="$(cd ../../solve_diag_op && pwd)"; MERGEDIR="$(cd .. && pwd)"
ARCH=v75; DEVICE="${DEVICE:-oneplus}"
DPKG="GdnSolveDiagPackage"; DPROV="${DPKG}InterfaceProvider"
MPKG="GdnMergeHmxPackage";  MPROV="${MPKG}InterfaceProvider"
source "$ROOT/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT/.venv/bin/python"
H="${H:-16}"; CB="${CB:-256}"; CKS="${CKS:-16 4 2 1}"; EXTRA_DEFS="${EXTRA_DEFS:-}"

echo "=== BUILD ONCE (both ops, C=$CB) ==="
EXTRA_DEFS="${EXTRA_DEFS} -DGDN_BR_C=${CB}" bash "$DIAGDIR/build.sh"  >_build_diag.log  2>&1 || { echo BUILDFAIL_DIAG;  tail -20 _build_diag.log;  exit 1; }
EXTRA_DEFS="${EXTRA_DEFS} -DGDN_BR_C=${CB}" bash "$MERGEDIR/build.sh" >_build_merge.log 2>&1 || { echo BUILDFAIL_MERGE; tail -20 _build_merge.log; exit 1; }
DX86="$DIAGDIR/build/x86_64-linux-clang/lib${DPKG}.so"; DHTP="$DIAGDIR/build/hexagon-$ARCH/lib${DPKG}_htp.so"; DCPU="$DIAGDIR/build/aarch64/lib${DPKG}_cpu.so"
MX86="$MERGEDIR/build/x86_64-linux-clang/lib${MPKG}.so"; MHTP="$MERGEDIR/build/hexagon-$ARCH/lib${MPKG}_htp.so"; MCPU="$MERGEDIR/build/aarch64/lib${MPKG}_cpu.so"
CCPL_DIR="$MERGEDIR/converter/build_combined"; mkdir -p "$CCPL_DIR"; CCPL="$CCPL_DIR/libConverterCombined.so"
clang++ -std=c++17 -O2 -shared -fPIC -I "$QNN_SDK_ROOT/include/QNN" -o "$CCPL" \
   "$DIAGDIR/converter/ConverterOpPackage.cpp" "$MERGEDIR/converter/ConverterOpPackage.cpp" || { echo CCPLFAIL; exit 1; }

cat > _htp.json <<EOF
{"devices":[{"dsp_arch":"$ARCH","soc_id":57,"pd_session":"unsigned","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
W="$(ssh "$DEVICE" 'echo $HOME/qnn_run')/split"
echo "=== PUSH LIBS + INPUT ONCE (device $W) ==="
"$PY" "$ROOT/scripts/gdn_split_probe.py" . "$H" "$CB" >/dev/null || { echo PROBEFAIL; exit 1; }   # A.raw + T_full_ref.raw (CK-independent)
ssh "$DEVICE" "mkdir -p $W"
ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./htp.json\"}}' > $W/cfg.json"
ssh "$DEVICE" "cat > $W/htp.json" < _htp.json
ssh "$DEVICE" "cat > $W/lib${DPKG}_htp.so" < "$DHTP"; ssh "$DEVICE" "cat > $W/lib${DPKG}_cpu.so" < "$DCPU"
ssh "$DEVICE" "cat > $W/lib${MPKG}_htp.so" < "$MHTP"; ssh "$DEVICE" "cat > $W/lib${MPKG}_cpu.so" < "$MCPU"
ssh "$DEVICE" "cat > $W/A.raw" < A.raw
ssh "$DEVICE" "printf 'A:=A.raw\n' > $W/list.txt"

OPP="$DX86:$DPROV,$MX86:$MPROV"
RUNOPP="./lib${DPKG}_cpu.so:$DPROV:CPU,./lib${DPKG}_htp.so:$DPROV:HTP,./lib${MPKG}_cpu.so:$MPROV:CPU,./lib${MPKG}_htp.so:$MPROV:HTP"
cat > _cfg.json <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$(pwd)/_htp.json"}}
EOF

for CK in $CKS; do
  echo; echo "########## CK=$CK (n_chains=$(( (H + CK - 1) / CK )) ) ##########"
  GDN_CK=$CK "$PY" "$ROOT/scripts/gdn_split_probe.py" . "$H" "$CB" >/dev/null || { echo "CK$CK PROBEFAIL"; continue; }
  qairt-converter -i split.onnx --target_backend HTP \
     --source_model_input_layout A NONTRIVIAL --desired_input_layout A NONTRIVIAL \
     --source_model_output_layout T NONTRIVIAL --desired_output_layout T NONTRIVIAL \
     --op_package_config "$DIAGDIR/${DPKG}.xml" "$MERGEDIR/${MPKG}.xml" --converter_op_package_lib "$CCPL,$CCPL" \
     --quantization_overrides ovr_split.json -o split.dlc >_c.log 2>&1 || { echo "CK$CK CVTFAIL"; tail -8 _c.log; continue; }
  rm -rf ctx_$CK
  qnn-context-binary-generator --dlc_path split.dlc --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
     --op_packages "$OPP" --config_file _cfg.json --profiling_level detailed --profiling_option optrace \
     --binary_file split_ctx --output_dir ctx_$CK >_x.log 2>&1 || { echo "CK$CK CTXFAIL"; tail -8 _x.log; continue; }
  for s in *schematic.bin ctx_$CK/*schematic.bin; do [ -f "$s" ] && mv -f "$s" ctx_$CK/ 2>/dev/null || true; done
  ssh "$DEVICE" "cat > $W/split_ctx.bin" < ctx_$CK/split_ctx.bin
  ssh "$DEVICE" "cd $W && rm -rf out && LD_LIBRARY_PATH=..:.:/vendor/lib64 ADSP_LIBRARY_PATH='..;.;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' ../qnn-net-run \
     --backend ../libQnnHtp.so --retrieve_context split_ctx.bin --config_file cfg.json --op_packages $RUNOPP \
     --input_list list.txt --output_dir out --profiling_level detailed --profiling_option optrace --perf_profile burst" >_r.log 2>&1 || true
  grep -q 'Finished Executing Graphs' _r.log || { echo "CK$CK RUNFAIL"; tail -15 _r.log; continue; }
  rm -rf out_$CK; mkdir -p out_$CK; ssh "$DEVICE" "cd $W && tar cf - out" | tar xf - -C out_$CK --strip-components=1 2>/dev/null
  "$PY" "$ROOT/scripts/decode_qnn_optrace.py" out_$CK --profile-log out_$CK/qnn-profiling-data_0.log --schematic "$(ls ctx_$CK/*schematic.bin|head -1)" >/dev/null 2>&1
  T=$(ls out_$CK/Result_0/T.raw 2>/dev/null || ls out_$CK/*/T.raw 2>/dev/null | head -1)
  "$PY" "$ROOT/scripts/gdn_timeline.py" "out_$CK/optrace/chrometrace.json" "$H" "$T" "$CB"
done
