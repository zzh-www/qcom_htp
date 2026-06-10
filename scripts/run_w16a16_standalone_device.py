#!/usr/bin/env python3
"""QNN-free standalone W16A16 HMX MatMul on REAL device (CDSP), to decide whether
the :2x2-store col-half1 corruption seen in hexagon-sim is a simulator artifact
or a genuine output-layout mismatch.

Mirrors scripts/run_w16a16_standalone_kernel.py (single output crouton surface,
per-split stride-4 sub-table, op-faithful descriptors) but runs the byte-verified
kernel body on real silicon via run_main_on_hexagon, deblocks several candidate
layouts ON DEVICE, and FARF-logs each variant's diff vs the QNN native Y.raw.

If any variant reaches diffs=0 on device (while sim matched none), the sim's
mxmem:2x2 store emulation is the culprit and the standalone is correct on HW.

  uv run python scripts/run_w16a16_standalone_device.py --artifact /tmp/hw_w16a16_check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_handwritten_artifact_body_device import (  # noqa: E402
    EXAMPLE, RUN_MAIN, RUN_MAIN_SKEL, SDK, ssh_cat_to, ssh_text, tool,
)
from run_handwritten_artifact_body_sim import c_array  # noqa: E402

BLOCK = 2048
KSPLIT = 4


def c_u32(name, vals):
    return f"static const uint32_t {name}[{len(vals)}] = {{{', '.join(str(v & 0xffffffff) + 'u' for v in vals)}}};"


def read_u32(path):
    d = path.read_bytes()
    return [int.from_bytes(d[i:i+4], "little") for i in range(0, len(d), 4)]


# One deblock variant body (the proven w8a16 row4 inverse) + 4 alternates, each
# logged on device. col-half1 handling differs per variant.
def deblock_variants(m, n):
    n2 = n * 2
    common_head = lambda name: [
        f"static void {name}(uint8_t *dst, const uint8_t *src) {{",
        "  for (uint32_t rp4 = 0; rp4 < 8u; ++rp4)",
        f"   for (uint32_t nt = 0; nt < {n//32}u; ++nt) {{",
        f"    const uint8_t *block = src + ((rp4*{n//32}u+nt)*2048u);",
        f"    for (uint32_t m32 = 0; m32 < {m//32}u; ++m32)",
        "     for (uint32_t rp = 0; rp < 2u; ++rp) {",
        "      uint32_t r0 = m32*32u+rp4*4u+rp*2u, r1 = r0+1u;",
        f"      uint8_t *d0 = dst + r0*{n2}u + nt*64u;",
        f"      uint8_t *d1 = dst + r1*{n2}u + nt*64u;",
        "      const uint8_t *sp = block + (m32*2u+rp)*128u;",
    ]
    tail = ["     }", "    }", "}"]
    row4 = common_head("deblock_row4") + [
        "      for (uint32_t c=0;c<32u;++c){const uint8_t*w=sp+c*4u;"
        "d0[c*2]=w[0];d0[c*2+1]=w[1];d1[c*2]=w[2];d1[c*2+1]=w[3];}",
    ] + tail
    col16 = common_head("deblock_col16") + [
        "      for (uint32_t c=0;c<32u;++c){uint32_t dc=c^16u;const uint8_t*w=sp+c*4u;"
        "d0[dc*2]=w[0];d0[dc*2+1]=w[1];d1[dc*2]=w[2];d1[dc*2+1]=w[3];}",
    ] + tail
    # halves: cols0-15 at sp[0..63], cols16-31 at block+0x40 region (separate slab)
    halves = [
        "static void deblock_halves(uint8_t *dst, const uint8_t *src) {",
        "  for (uint32_t rp4 = 0; rp4 < 8u; ++rp4)",
        f"   for (uint32_t nt = 0; nt < {n//32}u; ++nt) {{",
        f"    const uint8_t *block = src + ((rp4*{n//32}u+nt)*2048u);",
        f"    for (uint32_t m32 = 0; m32 < {m//32}u; ++m32)",
        "     for (uint32_t rp = 0; rp < 2u; ++rp) {",
        "      uint32_t r0 = m32*32u+rp4*4u+rp*2u, r1 = r0+1u;",
        f"      uint8_t *d0 = dst + r0*{n2}u + nt*64u, *d1 = dst + r1*{n2}u + nt*64u;",
        "      const uint8_t *lo = block + (m32*2u+rp)*64u;",       # half0 packed 64B/pair
        f"      const uint8_t *hi = block + 0x400u + (m32*2u+rp)*64u;",  # half1 slab
        "      for (uint32_t c=0;c<16u;++c){const uint8_t*w=lo+c*4u;"
        "d0[c*2]=w[0];d0[c*2+1]=w[1];d1[c*2]=w[2];d1[c*2+1]=w[3];}",
        "      for (uint32_t c=0;c<16u;++c){const uint8_t*w=hi+c*4u;"
        "d0[(16u+c)*2]=w[0];d0[(16u+c)*2+1]=w[1];d1[(16u+c)*2]=w[2];d1[(16u+c)*2+1]=w[3];}",
        "     }",
        "    }",
        "}",
    ]
    return row4 + col16 + halves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, type=Path)
    ap.add_argument("--device", default="oneplus")
    ap.add_argument("--remote-dir", default="w16a16_standalone_device")
    ap.add_argument("--native-raw", type=Path)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    artifact = args.artifact.resolve()
    prepared = artifact / "prepared_state"
    oracle = json.loads((ROOT / "example/handwritten_hmx_matmul/oracles.json").read_text())["families"]["w16a16"]
    m, k, n = (int(v) for v in oracle["shape_mkn"])
    native_raw_path = args.native_raw.resolve() if args.native_raw else ROOT / oracle["raw_output"]["path"]

    activation = (prepared / "activation.raw").read_bytes()
    packed_weight = (prepared / "packed_weight.raw").read_bytes()
    folded_bias = (prepared / "folded_bias.raw").read_bytes()
    mask_control = (prepared / "mask_control.raw").read_bytes()
    act_offsets = read_u32(prepared / "activation_table.raw")
    native_raw = native_raw_path.read_bytes()

    m_t, k_t, n_t = m//32, k//32, n//32
    mt_groups = m_t*8
    act_entries = mt_groups*k_t
    out_entries = mt_groups*n_t
    out_offsets = [(((rg & 7)*n_t)+nt)*BLOCK for rg in range(mt_groups) for nt in range(n_t)]
    out_bytes = m*n*2
    split_wbytes = k_t*KSPLIT*BLOCK
    split_bbytes = KSPLIT*512

    src = "\n".join([
        "#include <stdint.h>", "#include <string.h>",
        '#include "HAP_compute_res.h"', '#include "HAP_farf.h"', '#include "HAP_power.h"',
        '#include "handwritten_hmx_w16a16_kernel.h"',
        '#define TAG "[HM_DEVICE]"',
        c_array("k_act", activation), c_array("k_wt", packed_weight),
        c_array("k_bias", folded_bias), c_array("k_mask", mask_control),
        c_array("k_native", native_raw),
        c_u32("k_actoff", act_offsets[:act_entries]),
        c_u32("k_outoff", out_offsets[:out_entries]),
        "static int g_power_ctx;",
        "static void cb(uint8_t*d,const uint8_t*s,uint32_t n){for(uint32_t i=0;i<n;++i)d[i]=s[i];}",
        "static uint32_t diffcount(const uint8_t*a,const uint8_t*b,uint32_t n){uint32_t d=0;for(uint32_t i=0;i<n;++i)if(a[i]!=b[i])d++;return d;}",
        *deblock_variants(m, n),
        "static int power_on(void){HAP_power_request_t req;memset(&req,0,sizeof(req));"
        "req.type=HAP_power_set_apptype;req.apptype=HAP_POWER_COMPUTE_CLIENT_CLASS;"
        "if(HAP_power_set((void*)&g_power_ctx,&req))return -1;"
        "memset(&req,0,sizeof(req));req.type=HAP_power_set_HVX;req.hvx.power_up=1;"
        "if(HAP_power_set((void*)&g_power_ctx,&req))return -3;"
        "memset(&req,0,sizeof(req));req.type=HAP_power_set_HMX;req.hmx.power_up=1;"
        "if(HAP_power_set((void*)&g_power_ctx,&req))return -4;return 0;}",
        "int main(int argc,char**argv){(void)argc;(void)argv;",
        "  if(power_on()){FARF(ALWAYS,TAG\" w16dev power_fail\");return 1;}",
        "  unsigned int vsz=8u*1024u*1024u;HAP_compute_res_query_VTCM(0,&vsz,0,0,0);",
        "  compute_res_attr_t at;HAP_compute_res_attr_init(&at);",
        "  HAP_compute_res_attr_set_vtcm_param(&at,vsz,1);HAP_compute_res_attr_set_hmx_param(&at,1);",
        "  unsigned int cid=HAP_compute_res_acquire(&at,100000);",
        "  if(!cid){FARF(ALWAYS,TAG\" w16dev vtcm_fail\");return 2;}",
        "  uint8_t*base=(uint8_t*)HAP_compute_res_attr_get_vtcm_ptr(&at);",
        "  if(HAP_compute_res_hmx_lock(cid)){FARF(ALWAYS,TAG\" w16dev hmxlock_fail\");HAP_compute_res_release(cid);return 3;}",
        "  uint8_t*act=base+0x000000u,*wt=base+0x040000u,*bias=base+0x080000u;",
        "  uint8_t*out=base+0x0a0000u,*pub=base+0x100000u,*pub2=base+0x140000u;",
        "  int32_t*atab=(int32_t*)(base+0x180000u),*otab=(int32_t*)(base+0x190000u),*stab=(int32_t*)(base+0x1a0000u);",
        "  cb(act,k_act,sizeof(k_act));cb(wt,k_wt,sizeof(k_wt));cb(bias,k_bias,sizeof(k_bias));",
        f"  for(uint32_t i=0;i<{out_bytes}u;++i)out[i]=0;",
        f"  for(uint32_t i=0;i<{act_entries}u;++i)atab[i]=(int32_t)(uintptr_t)(act+k_actoff[i]);",
        f"  for(uint32_t i=0;i<{out_entries}u;++i)otab[i]=(int32_t)(uintptr_t)(out+k_outoff[i]);",
        "  uint32_t mask[16] __attribute__((aligned(16)));cb((uint8_t*)mask,k_mask,64u);",
        "  uint32_t extra[2] __attribute__((aligned(16)))={1u,1536u};mask[14]=(uint32_t)(uintptr_t)extra;",
        "  HmW16A16ActDesc ad={atab," + f"{k_t}u,{k_t*64}u}};",
        f"  for(uint32_t s=0;s<{n_t//KSPLIT}u;++s){{",
        f"    for(uint32_t rg=0;rg<{mt_groups}u;++rg)for(uint32_t nt=0;nt<{KSPLIT}u;++nt)"
        f"stab[rg*{KSPLIT}u+nt]=otab[rg*{n_t}u+s*{KSPLIT}u+nt];",
        f"    HmW16A16OutDesc od={{stab,{KSPLIT}u,{m}u,{m}u,1,{KSPLIT*32}u}};",
        f"    hm_w16a16_v73_kernel(&od,&ad,wt+s*{split_wbytes}u,bias+s*{split_bbytes}u,(const HmW16A16MaskDesc*)mask,extra);",
        "  }",
        f"  uint32_t rawhash=0;for(uint32_t i=0;i<{out_bytes}u;++i)rawhash=rawhash*31u+out[i];",
        "  deblock_row4(pub,out);  uint32_t d_row4=diffcount(pub,k_native," + f"{out_bytes}u);",
        "  deblock_col16(pub2,out); uint32_t d_col16=diffcount(pub2,k_native," + f"{out_bytes}u);",
        "  deblock_halves(pub,out); uint32_t d_halves=diffcount(pub,k_native," + f"{out_bytes}u);",
        f'  FARF(ALWAYS,TAG" w16dev rawhash=0x%08x row4=%u col16=%u halves=%u total=%u",rawhash,d_row4,d_col16,d_halves,{out_bytes}u);',
        "  HAP_compute_res_hmx_unlock(cid);HAP_compute_res_release(cid);return 0;}",
    ])

    work = Path("/tmp/w16a16_standalone_device"); work.mkdir(exist_ok=True)
    source = work / "w16dev.c"; binary = work / "libw16dev.so"
    source.write_text(src)
    clang = tool("hexagon-clang")
    import subprocess
    cmd = [str(clang), "-mv75", "-O2", "-G0", "-fPIC", "-shared", "-mhvx", "-mhvx-length=128B", "-mhmx",
           "-I", str(SDK/"incs"), "-I", str(SDK/"incs"/"stddef"),
           "-I", str(SDK/"rtos"/"qurt"/"computev75"/"include"),
           "-I", str(SDK/"rtos"/"qurt"/"computev75"/"include"/"qurt"),
           "-I", str(EXAMPLE/"include"), str(source), "-o", str(binary)]
    r = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.returncode:
        print("COMPILE FAIL\n", r.stdout); return 1

    dev = args.device; rd = args.remote_dir
    remote = ssh_text(dev, f"rm -rf {rd} && mkdir -p {rd} && cd {rd} && pwd").strip()
    ssh_cat_to(dev, f"{remote}/run_main_on_hexagon", RUN_MAIN.read_bytes())
    ssh_cat_to(dev, f"{remote}/librun_main_on_hexagon_skel.so", RUN_MAIN_SKEL.read_bytes())
    ssh_cat_to(dev, f"{remote}/{binary.name}", binary.read_bytes())
    ssh_text(dev, f"chmod +x {remote}/run_main_on_hexagon", timeout=20)
    ssh_text(dev, "logcat -c >/dev/null 2>&1 || true", timeout=20)
    ssh_text(dev, f"echo '0x1f' > {remote}/run_main_on_hexagon.farf", timeout=20)
    lp = f"{remote}:/vendor/lib64:/system/vendor/lib64:/odm/lib64"
    out = ssh_text(dev, f"cd {remote} && DSP_LIBRARY_PATH={remote} LD_LIBRARY_PATH={lp} ADSP_LIBRARY_PATH={remote} ./run_main_on_hexagon 3 {binary.name} 2>&1", timeout=90)
    print(out)
    log = ssh_text(dev, "logcat -d -v brief 2>/dev/null | grep '\\[HM_DEVICE\\]' || true", timeout=30)
    print(log)

    import re
    mm = re.search(r"w16dev rawhash=(0x[0-9a-fA-F]+) row4=(\d+) col16=(\d+) halves=(\d+) total=(\d+)", out + "\n" + log)
    if not mm:
        print("w16a16 standalone DEVICE: NO RESULT (device/log parse failed)")
        return 1
    row4 = int(mm.group(2)); total = int(mm.group(5))
    byte_exact = (row4 == 0)
    payload = {
        "schema": "w16a16_standalone_device.v1", "qnn_used": False, "device": dev,
        "rawhash": mm.group(1), "row4_diff_bytes": row4, "total_bytes": total,
        "byte_exact": byte_exact,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"w16a16 standalone DEVICE (QNN-free): {'BYTE-EXACT' if byte_exact else 'MISMATCH'} "
          f"row4_diff={row4}/{total}")
    return 0 if byte_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
