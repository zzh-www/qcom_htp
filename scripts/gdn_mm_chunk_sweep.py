#!/usr/bin/env python3
"""Isolated BL^3 matmul cycle vs chunk size — GDNSolveHVX (vrmpy) leg.

Measures the steady-state cycle of one BL x BL x BL int8 matmul done with HVX vrmpy
(4 MACs/lane), the kernel GDNSolveHVX uses for the divide-and-conquer triangular-inverse
off-diagonal merges. Size-generic form of gdn_matmul_i8_vrmpy (which is hardcoded to BL=64).
Run in hexagon-sim (HVX is deterministic -> sim == device; no SSR, fast sweep).

  python3 scripts/gdn_mm_chunk_sweep.py --bl 32 64 128 256
Reports cyc/matmul per BL.  (HMX leg measured separately.)
"""
from __future__ import annotations
import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HARNESS = r"""
#include <stdint.h>
#include <stdio.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_cycles.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

#define BL   @@BL@@
#define NV   (BL/32)            /* output HVX vectors per row */
#define REPS @@REPS@@

/* size-generic GDNSolveHVX matmul: C[BL][BL] i32 = A[BL][BL] @ B[BL][BL], int8 via vrmpy.
   B pre-packed as btp[g][v] (g=K/4 groups, v=N/32 col vectors). cycle is data-independent. */
static void vrmpy_mm(const int8_t *A, const int8_t *btp, int32_t *C) {
  for (int i = 0; i < BL; ++i) {
    const int32_t *Aw = (const int32_t *)(A + (size_t)i * BL);
    HVX_Vector acc[NV];
    for (int v = 0; v < NV; ++v) acc[v] = Q6_V_vzero();
    for (int g = 0; g < BL / 4; ++g) {
      HVX_Vector vA = Q6_V_vsplat_R(Aw[g]);
      const HVX_Vector *bt = (const HVX_Vector *)(btp + (size_t)g * NV * 128);
      for (int v = 0; v < NV; ++v) acc[v] = Q6_Vw_vrmpyacc_VwVbVb(acc[v], vA, bt[v]);
    }
    for (int v = 0; v < NV; ++v) ((HVX_Vector *)(C + (size_t)i * BL))[v] = acc[v];
  }
}

int main(void) {
  unsigned int vb = h2_info(INFO_VTCM_BASE), vs = h2_info(INFO_VTCM_SIZE);
  if (vb == 0 || vs < 1024u) { h2_thread_stop(1); return 1; }
  uint8_t *base = (uint8_t *)(uintptr_t)(vb + 0x10000u);
  int8_t  *A   = (int8_t  *)(base + 0x00000u);
  int8_t  *btp = (int8_t  *)(base + 0x40000u);
  int32_t *C   = (int32_t *)(base + 0x80000u);
  for (int i = 0; i < BL * BL; ++i) { A[i] = (int8_t)((i % 11) - 5); }
  for (int i = 0; i < (BL/4) * NV * 128; ++i) { btp[i] = (int8_t)((i % 7) - 3); }
  vrmpy_mm(A, btp, C);                                   /* warm */
  unsigned long long t0 = h2_get_pcycles();
  for (int r = 0; r < REPS; ++r) vrmpy_mm(A, btp, C);
  unsigned long long t1 = h2_get_pcycles();
  printf("[CYC]%llu\n", (unsigned long long)((t1 - t0) / REPS));
  printf("[CHK]%d\n", (int)C[0]);
  h2_thread_stop(0);
  return 0;
}
"""


def tool(name: str) -> Path:
    f = shutil.which(name)
    if f:
        return Path(f)
    c = sorted((ROOT / "tools" / "hexagon-sdk" / "tools" / "HEXAGON_Tools").glob("*/Tools/bin/" + name))
    if c:
        return c[0]
    raise FileNotFoundError(name)


def run_bl(bl: int, reps: int) -> int | None:
    clang, sim = tool("hexagon-clang"), tool("hexagon-sim")
    h2_install = ROOT / "tools" / "h2-install"
    h2_root = h2_install.resolve().parent if h2_install.is_symlink() else h2_install.parent
    booter = h2_install / "bin" / "booter"
    work = Path(tempfile.mkdtemp(prefix="mm_sweep_"))
    (work / "h.c").write_text(HARNESS.replace("@@BL@@", str(bl)).replace("@@REPS@@", str(reps)))
    cc = [str(clang), "-O2", "-mv75", "-mhvx", "-mhvx-length=128B", "-mhmx", "-DARCHV=75",
          "-I", str(h2_install / "include"), "-I", str(h2_root / "kernel" / "include"),
          "-moslib=h2", "-Wl,-L," + str(h2_install / "lib"),
          "-Wl,--section-start=.start=0x02000000", "-o", str(work / "h"), str(work / "h.c")]
    r = subprocess.run(cc, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"BL={bl} COMPILE FAIL:\n" + r.stdout[-1500:])
        return None
    sc = [str(sim), "--mv75", "--mhmx", "1", "--simulated_returnval", "--",
          str(booter), "--ext_power", "1", "--use_ext", "1", "--fence_hi", "0xfe000000", str(work / "h")]
    sr = subprocess.run(sc, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    shutil.rmtree(work, ignore_errors=True)
    m = re.search(r"\[CYC\](\d+)", sr.stdout)
    return int(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bl", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()
    print(f"GDNSolveHVX vrmpy int8 matmul, isolated cyc/matmul (hexagon-sim, REPS={a.reps}):")
    print(f"{'BL':>5} {'cyc/matmul':>12} {'cyc/MAC':>10}")
    for bl in a.bl:
        c = run_bl(bl, a.reps)
        if c is None:
            print(f"{bl:>5} {'FAIL':>12}")
            continue
        macs = bl ** 3
        print(f"{bl:>5} {c:>12} {c / macs:>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
