#!/usr/bin/env bash
# P3.1 device proof: SHIP/ARES HVXMix consumer PMU真值 pass (mirror of pure-HMX P2.3).
# For each impl: build CLEAN + PMU variants, ACAC-interleave (clean,PMU,clean,PMU,...) over reps,
# prove (a) the PMU真值 prints, (b) the PMU build's clean wall stats[0] matches the clean build's
# wall within thermal noise (=> the separate post-solve PMU pass did NOT pollute the timing window).
# metric = stats[0] graph-wall (32-head TOTAL, VTCM-only), reps2-N median.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/example/gdn_native/baremetal"
source "$ROOT/scripts/dssh.sh"; dssh_open oneplus >/dev/null
W=$(dssh 'echo $HOME/gdnbm_p31')
dssh "mkdir -p $W"
REPS="${REPS:-8}"
SHIP="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL -DGDN_BR_SBOOST -DGDN_BR_DIAG_I16 -DGDN_BR_REQ_FUSE -DGDN_BR_FBOOST"
ARES_BASE="$SHIP -DGDN_BR_W16 -DGDN_BR_W16_N8 -DGDN_BR_W16_N8_FUSEDEP -DGDN_BR_W16_N8_ACTCACHE"
SCALES="32 256 32768 32768 2.770166930875267e-05 6.103701895199438e-05"

build_to() { local defs="$1" tag="$2"
  EXTRA_DEFS="$defs" bash build.sh >/tmp/p31_build_$tag.log 2>&1 || { echo "BUILD FAIL $tag"; tail -15 /tmp/p31_build_$tag.log; exit 1; }
  cp build/libgdnbm_skel.so /tmp/p31_${tag}.so; cp build/gdnbm /tmp/p31_${tag}.bin; }

deploy() { local tag="$1"
  dssh "cat > $W/lib_${tag}.so" < /tmp/p31_${tag}.so
  dssh "cat > $W/gdnbm_${tag}" < /tmp/p31_${tag}.bin; dssh "chmod +x $W/gdnbm_${tag}"; }

run_one() { local tag="$1" Araw="$2"
  dssh "cd $W && cp lib_${tag}.so libgdnbm_skel.so && GDNBM_REPS=$REPS \
    LD_LIBRARY_PATH=$W:/vendor/lib64:/system/lib64 \
    ADSP_LIBRARY_PATH='$W;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' \
    ./gdnbm_${tag} 4 $Araw T_${tag}.raw $SCALES 2>&1"; }

# median of reps2-N walls from a run log
med_wall() { python3 - <<PY
import sys,re
walls=[]
for ln in '''$1'''.splitlines():
    m=re.search(r'rep=(\d+)/\d+ +wall=(\d+)',ln)
    if m and int(m.group(1))>=2: walls.append(int(m.group(2)))
walls.sort()
print(walls[len(walls)//2] if walls else 0)
PY
}

impl_proof() {  # $1=label $2=clean_defs $3=pmu_defs $4=Araw
  local label="$1" cdefs="$2" pdefs="$3" Araw="$4"
  echo "################## $label ##################"
  build_to "$cdefs" "${label}_clean"; build_to "$pdefs" "${label}_pmu"
  deploy "${label}_clean"; deploy "${label}_pmu"
  # ACAC interleave: clean,PMU,clean,PMU  (one GDNBM_REPS run each = one hot sub-window)
  declare -a CW PW
  for win in 1 2 3; do
    oc=$(run_one "${label}_clean" "$Araw"); op=$(run_one "${label}_pmu" "$Araw")
    cm=$(med_wall "$oc"); pm=$(med_wall "$op")
    CW+=($cm); PW+=($pm)
    if [ "$win" = "1" ]; then
      echo "--- $label PMU真值 line (from PMU build, win1) ---"
      echo "$op" | grep -iE 'PMU_UTIL|raw PMU stats' | head -3
      echo "--- $label clean wall=$cm  PMU-build wall=$pm ---"
    else
      echo "win$win: clean=$cm  PMU=$pm"
    fi
  done
  python3 - "$label" "${CW[@]}" "--" "${PW[@]}" <<PY
import sys
a=sys.argv; lbl=a[1]; i=a.index('--')
C=[int(x) for x in a[2:i] if int(x)>0]; P=[int(x) for x in a[i+1:] if int(x)>0]
def med(v): v=sorted(v); return v[len(v)//2] if v else 0
cm,pm=med(C),med(P)
d=(pm-cm)/cm*100 if cm else 0
print(f"=== {lbl} 3-window: clean_med={cm} PMU_med={pm}  dWall={d:+.2f}% (must be in thermal noise => no pollution) ===")
print(f"    clean wins={C}  PMU wins={P}")
PY
}

# ---- SHIP: plain natural A ----
dssh "cat > $W/A_u16_h32.raw" < A_u16_h32.raw
impl_proof SHIP "$SHIP" "$SHIP -DGDN_BR_PMU_UTIL" "A_u16_h32.raw"

# ---- ARES: needs extended A (natural + cv-block). Build DUMP, assemble, then ACAC. ----
echo "################## ARES (assemble extended A) ##################"
build_to "$ARES_BASE -DGDN_BR_W16_N8_ACVRES_DUMP" "ARES_dump"; deploy "ARES_dump"
run_one "ARES_dump" "A_u16_h32.raw" | grep -iE 'solve rc' | head -1
dssh "cat $W/T_ARES_dump.raw" > /tmp/p31_T_dump.raw
NATA="$ROOT/example/gdn_native/baremetal/A_u16_h32.raw" python3 - <<'PY'
import numpy as np, os
H=32; CC=256*256
dump=np.fromfile("/tmp/p31_T_dump.raw", dtype="<u2")
nat =np.fromfile(os.environ["NATA"], dtype="<u2")
cv=np.zeros((H,6*4096),dtype="<u2")
for h in range(H): cv[h]=dump[h*CC : h*CC + 6*4096]
np.concatenate([nat, cv.ravel()]).tofile("/tmp/p31_A_ares.raw")
print("  assembled /tmp/p31_A_ares.raw", os.path.getsize("/tmp/p31_A_ares.raw"), "bytes")
PY
dssh "cat > $W/A_ares.raw" < /tmp/p31_A_ares.raw
impl_proof ARES "$ARES_BASE -DGDN_BR_W16_N8_ARES" "$ARES_BASE -DGDN_BR_W16_N8_ARES -DGDN_BR_PMU_UTIL" "A_ares.raw"

echo "DONE"
