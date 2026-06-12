#!/usr/bin/env bash
# Standalone GdnSolve kernel benchmark: A->GdnSolve->T (64 heads, NO matmuls), so the whole-graph
# wall == the op's wall — a clean low-noise signal for HVX kernel A/B (the full GDN graph's wall is
# dominated by matmul dispatch and drowns a ~20% solve change in run-to-run noise).
# Reuses solve.dlc/ovr.json/A.raw/T_ref.raw already in this dir. Rebuilds the op each run.
#   prints: wall us (QNN accelerator execute) + T relerr vs T_ref.
set -uo pipefail
cd "$(dirname "$0")"
ROOT_DIR="$(cd ../../../.. && pwd)"
OPDIR="$(cd .. && pwd)"
PKG="GdnSolvePackage"; PROV="${PKG}InterfaceProvider"
ARCH="${ARCH:-v75}"; DEVICE="${DEVICE:-oneplus}"
source "$ROOT_DIR/scripts/env.sh" >/dev/null 2>&1
source "$ROOT_DIR/scripts/dssh.sh"
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT_DIR/.venv/bin/python"

echo "[1/4] build op"
bash "$OPDIR/build.sh" >_build.log 2>&1 || { echo "  BUILD FAIL"; tail -15 _build.log; exit 1; }
X86_PKG="$OPDIR/build/x86_64-linux-clang/lib${PKG}.so"
HTP_PKG="$OPDIR/build/hexagon-${ARCH}/lib${PKG}_htp.so"
CPU_PKG="$OPDIR/build/aarch64/lib${PKG}_cpu.so"

echo "[2/4] ctxgen + optrace (fresh schematic matching solve.dlc)"
rm -rf ctx; mkdir -p ctx
qnn-context-binary-generator --dlc_path solve.dlc --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --op_packages "$X86_PKG:$PROV" --config_file _cfg.json \
    --profiling_level detailed --profiling_option optrace \
    --binary_file solve_ctx --output_dir ctx >_ctx.log 2>&1 || { echo "  CTXGEN FAIL"; tail -15 _ctx.log; exit 1; }
for s in *schematic.bin ctx/*schematic.bin; do [ -f "$s" ] && mv -f "$s" ctx/ 2>/dev/null || true; done

echo "[3/4] device run + optrace ($DEVICE)"
DEV="$(ssh "$DEVICE" 'echo $HOME/qnn_run')"; W="$DEV/gdn_solve_bench"
ssh "$DEVICE" "mkdir -p $W"
ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./htp.json\"}}' > $W/cfg.json"
ssh "$DEVICE" "cat > $W/htp.json" < _htp.json
ssh "$DEVICE" "cat > $W/solve_ctx.bin" < ctx/solve_ctx.bin
ssh "$DEVICE" "cat > $W/lib${PKG}_htp.so" < "$HTP_PKG"
ssh "$DEVICE" "cat > $W/lib${PKG}_cpu.so" < "$CPU_PKG"
ssh "$DEVICE" "cat > $W/A.raw" < A.raw
ssh "$DEVICE" "printf 'A:=A.raw\n' > $W/list.txt"
ssh "$DEVICE" "cd $W && rm -rf out && LD_LIBRARY_PATH=..:.:/vendor/lib64 ADSP_LIBRARY_PATH='..;.;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' ../qnn-net-run \
    --backend ../libQnnHtp.so --retrieve_context solve_ctx.bin --config_file cfg.json \
    --op_packages ./lib${PKG}_cpu.so:$PROV:CPU,./lib${PKG}_htp.so:$PROV:HTP \
    --input_list list.txt --output_dir out --profiling_level detailed --profiling_option optrace --perf_profile burst" >_run.log 2>&1 || true
grep -q 'Finished Executing Graphs' _run.log || { echo "  RUN FAIL"; tail -20 _run.log; exit 1; }
rm -rf out; mkdir -p out; ssh "$DEVICE" "cd $W && tar cf - out" | tar xf - -C out --strip-components=1

echo "[4/4] wall + correctness"
qnn-profile-viewer --input_log out/qnn-profiling-data_0.log 2>/dev/null | grep -iE 'accelerator \(execute\) time' | grep -iv cycles | sed 's/^ */    /' | sort -u
T=$(ls out/Result_0/T.raw 2>/dev/null || ls out/*/T.raw 2>/dev/null | head -1)
"$PY" - "$T" T_ref.raw <<'PY'
import numpy as np,sys
t=np.fromfile(sys.argv[1],dtype=np.float32).astype(np.float64)
r=np.fromfile(sys.argv[2],dtype=np.float32).astype(np.float64)
n=min(t.size,r.size); t,r=t[:n],r[:n]
rel=np.linalg.norm(t-r)/(np.linalg.norm(r)+1e-12)
print(f"    T relerr vs T_ref: {rel:.3e}  ({'OK' if rel<5e-3 else 'CHECK'})  [fp32, n={n}]")
PY