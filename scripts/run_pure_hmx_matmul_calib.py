#!/usr/bin/env python3
"""Calibrate the REAL w16a16 64^3 matmul on device using the C packers (pure_hmx_solve/w16a16_pack.h).

Embeds a known small A (int16 signed, fed u16 zp=32768) and W (int16), PACKS ON DEVICE with the
byte-exact C packers, runs the byte-proven kernel, DEPACKS ON DEVICE with the C depack, and FARF-logs
the real output values. The driver prints the numpy A_signed @ W so we can read the exact drain
(output_code = round(A@W * gain) + out_zp) -> pins gain, giving a real "int16 matmul + scale" primitive.

  uv run python scripts/run_pure_hmx_matmul_calib.py
"""
from __future__ import annotations
import subprocess, sys, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from run_handwritten_artifact_body_device import EXAMPLE, RUN_MAIN, RUN_MAIN_SKEL, SDK, ssh_cat_to, ssh_text, tool  # noqa: E402
from run_handwritten_artifact_body_sim import c_array  # noqa: E402

M = K = N = 64
PACK_H = ROOT / "example/gdn_native/pure_hmx_solve"
KERN_INC = EXAMPLE / "include"


def main():
    rng = np.random.default_rng(1)
    # q16-magnitude inputs (~+-4096 = value +-0.125): A@W ~ 64*4096^2/.. /32767 lands in mid int16 range.
    A_signed = rng.integers(-4096, 4097, size=(M, K)).astype(np.int64)
    W = rng.integers(-4096, 4097, size=(K, N)).astype(np.int64)
    A_u16 = (A_signed + 32768).astype("<u2")
    W_i16 = W.astype("<i2")
    acc = (A_signed @ W)                                             # exact int32 accumulator
    print("acc=A@W       [0..7] =", acc.flatten()[:8].tolist())
    for g, lbl in [(1/32767, "/32767"), (1/32768, "/32768"), (2/32767, "*2/32767")]:
        print(f"  round(acc*{lbl})+32768 [0..7] =",
              [int(np.clip(round(acc.flatten()[i]*g)+32768,0,65535)) for i in range(8)])

    m_t = k_t = n_t = 2
    mt_groups = m_t * 8
    src = "\n".join([
        "#include <stdint.h>", "#include <string.h>",
        '#include "HAP_compute_res.h"', '#include "HAP_farf.h"', '#include "HAP_power.h"',
        '#include "handwritten_hmx_w16a16_kernel.h"',
        '#include "w16a16_pack.h"',
        '#define TAG "[HM_CALIB]"',
        c_array("k_A", A_u16.tobytes()), c_array("k_W", W_i16.tobytes()),
        "static int g_pc;",
        "int main(void){",
        "  HAP_power_request_t r; memset(&r,0,sizeof(r)); r.type=HAP_power_set_apptype; r.apptype=HAP_POWER_COMPUTE_CLIENT_CLASS;",
        "  HAP_power_set((void*)&g_pc,&r);",
        "  memset(&r,0,sizeof(r)); r.type=HAP_power_set_HMX; r.hmx.power_up=1; HAP_power_set((void*)&g_pc,&r);",
        "  unsigned vsz=8u*1024u*1024u; HAP_compute_res_query_VTCM(0,&vsz,0,0,0);",
        "  compute_res_attr_t at; HAP_compute_res_attr_init(&at);",
        "  HAP_compute_res_attr_set_vtcm_param(&at,vsz,1); HAP_compute_res_attr_set_hmx_param(&at,1);",
        "  unsigned cid=HAP_compute_res_acquire(&at,100000); if(!cid){FARF(ALWAYS,TAG\" vtcm_fail\");return 2;}",
        "  uint8_t*base=(uint8_t*)HAP_compute_res_attr_get_vtcm_ptr(&at);",
        "  if(HAP_compute_res_hmx_lock(cid)){FARF(ALWAYS,TAG\" hmxlock_fail\");return 3;}",
        "  uint16_t*act=(uint16_t*)(base+0x00000u); int8_t*wt=(int8_t*)(base+0x10000u);",
        "  int32_t*bias=(int32_t*)(base+0x20000u); uint16_t*out=(uint16_t*)(base+0x30000u);",
        "  uint16_t*pub=(uint16_t*)(base+0x80000u);",
        "  int32_t*atab=(int32_t*)(base+0xC0000u),*otab=(int32_t*)(base+0xC0400u);",
        "  w16a16_pack_act_crouton16(k_A, act, 64,64);",
        "  w16a16_pack_wt_kmajor((const int16_t*)k_W, (uint8_t*)wt, 64,64);",
        "  w16a16_pack_bias((const int16_t*)k_W, bias, 64,64);",
        "  for(int i=0;i<0x10000/2;i++) out[i]=0;",
        "  for(int r4=0;r4<16;r4++)for(int t=0;t<2;t++){",
        "    atab[r4*2+t]=(int32_t)(uintptr_t)((uint8_t*)act + (((r4&7)*2+t)*512));",
        "    otab[r4*2+t]=(int32_t)(uintptr_t)((uint8_t*)out + (((r4&7)*2+t)*2048)); }",
        "  uint32_t mask[16] __attribute__((aligned(16)))={0x0u,0x700u,0x0u,0x77cu,0x0u,0x0u,0x3ffu,0x0u,0x0u,0x0u,0x0u,0x0u,0x80u,0x0u,0x0u,0x0u};",
        "  uint32_t extra[2] __attribute__((aligned(16)))={1u,1536u}; mask[14]=(uint32_t)(uintptr_t)extra;",
        f"  HmW16A16ActDesc ad={{atab,{k_t}u,{k_t*64}u}};",
        f"  HmW16A16OutDesc od={{otab,{n_t}u,{M}u,{M}u,1,{n_t*32}u}};",
        "  hm_w16a16_v73_kernel(&od,&ad,(const uint8_t*)wt,(const uint8_t*)bias,(const HmW16A16MaskDesc*)mask,extra);",
        "  w16a16_depack_crouton16(out, pub, 64,64);",
        '  FARF(ALWAYS,TAG" out[0..15]= %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",'
        + "".join(f"(int)pub[{i}]," for i in range(16))[:-1] + ");",
        "  HAP_compute_res_hmx_unlock(cid); HAP_compute_res_release(cid); return 0;}",
    ])
    work = Path("/tmp/pure_hmx_calib"); work.mkdir(exist_ok=True)
    (work / "calib.c").write_text(src)
    clang = tool("hexagon-clang")
    cmd = [str(clang), "-mv75", "-O2", "-G0", "-fPIC", "-shared", "-mhvx", "-mhvx-length=128B", "-mhmx",
           "-I", str(SDK/"incs"), "-I", str(SDK/"incs"/"stddef"),
           "-I", str(SDK/"rtos"/"qurt"/"computev75"/"include"),
           "-I", str(SDK/"rtos"/"qurt"/"computev75"/"include"/"qurt"),
           "-I", str(KERN_INC), "-I", str(PACK_H), str(work/"calib.c"), "-o", str(work/"libcalib.so")]
    r = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.returncode:
        print("COMPILE FAIL\n", r.stdout); return 1
    dev = "oneplus"; rd = "pure_hmx_calib"
    remote = ssh_text(dev, f"rm -rf {rd} && mkdir -p {rd} && cd {rd} && pwd").strip()
    ssh_cat_to(dev, f"{remote}/run_main_on_hexagon", RUN_MAIN.read_bytes())
    ssh_cat_to(dev, f"{remote}/librun_main_on_hexagon_skel.so", RUN_MAIN_SKEL.read_bytes())
    ssh_cat_to(dev, f"{remote}/libcalib.so", (work/"libcalib.so").read_bytes())
    ssh_text(dev, f"chmod +x {remote}/run_main_on_hexagon", timeout=20)
    ssh_text(dev, "logcat -c >/dev/null 2>&1 || true", timeout=20)
    ssh_text(dev, f"echo '0x1f' > {remote}/run_main_on_hexagon.farf", timeout=20)
    lp = f"{remote}:/vendor/lib64:/system/vendor/lib64:/odm/lib64"
    out = ssh_text(dev, f"cd {remote} && DSP_LIBRARY_PATH={remote} LD_LIBRARY_PATH={lp} ADSP_LIBRARY_PATH={remote} ./run_main_on_hexagon 3 libcalib.so 2>&1", timeout=90)
    log = ssh_text(dev, "logcat -d -v brief 2>/dev/null | grep '\\[HM_CALIB\\]' || true", timeout=30)
    print(out);
    mm = re.search(r"out\[0\.\.15\]=\s*(.+)$", out + "\n" + log, re.M)
    if mm:
        dev_vals = [int(x) for x in mm.group(1).split()][:16]
        a = acc.flatten()
        print("device out [0..7] =", dev_vals[:8])
        for g, lbl in [(1/32767, "/32767"), (1/32768, "/32768"), (2/32767, "*2/32767"), (1/65534, "/65534")]:
            pred = [int(np.clip(round(a[i]*g)+32768, 0, 65535)) for i in range(8)]
            match = sum(1 for i in range(8) if pred[i] == dev_vals[i])
            print(f"  acc*{lbl:>9}+zp -> {pred}  match {match}/8")
    else:
        print("NO OUTPUT parsed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
