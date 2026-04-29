---
name: P1.4 — On-device kernel patch BREAKTHROUGH (2026-04-29)
description: Patched libQnnHtpV75Skel.so on-device to dump native's runtime descriptor values for hmx_v73_convbbb1x1deep_stride1. **Native uses n_tiles_pow2=32 (not 64), strides=32 (not 0), mask args 0x700 (not 0x70b), mask[+0x38]=extra_param ptr**. Setting these in our V9 op cuts packets 747→475 (36%) but covers only HALF M (50% bit-exact). Remaining gap is table layout + VTCM placement.
type: project
---

# P1.4 — On-device kernel patch BREAKTHROUGH (2026-04-29)

## 方法

Patched `tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so` 156 bytes
at file offset 0x2ebe40 (= `hmx_v73_convbbb1x1deep_stride1` entry).

Patch payload (compiled from `dump_stub.c` via `hexagon-clang -mv75 -O2`)
replaces kernel body with:
1. Read first output ptr from `od[0]` (= out_tile_ptr_table base, then [0])
2. Write `r0..r5` args to that ptr + 0x00..0x18
3. Write 64 bytes from `r4` (mask) to that ptr + 0x20..0x60
4. Write 24 bytes from `r0` (od) to that ptr + 0x60..0x78
5. Write 12 bytes from `r1` (ad) to that ptr + 0x78..0x84
6. Write 8 bytes from `r5` (extra) to that ptr + 0x84..0x8c
7. Markers `0xDEADBEEF` (head), `0xCAFEBABE` (mid), `0x5A5A5A5A` (tail)
8. `jumpr r31` (skip actual matmul)

Pushed to device's `qnn_run/libQnnHtpV75Skel.so` (replacing user-copy, not
`/vendor/lib64`). Backup at `qnn_run/libQnnHtpV75Skel.so.ORIG_BACKUP_p14`.

Ran native chain8 256³. Output `Y.raw` (262144 bytes fp32-dequantized) reads
back as our dumped descriptor data. **All 3 markers verify correctly**
(0xDEADBEEF, 0xCAFEBABE, 0x5A5A5A5A) → patch executed cleanly.

## Native ConvLayer_s1.opt 的真实 descriptor (from MATMUL_7 last call in chain)

```
r0 (od ptr):    0x02098ce8
r1 (ad ptr):    0x02098cd0
r2 (wt ptr):    0xfc020800   <-- VTCM!
r3 (bias ptr):  0xfc020000   <-- VTCM!
r4 (mask ptr):  0x02098d08
r5 (extra ptr): 0x02101000

mask[+0x04] = 0x000_0700        out_rt_mask
mask[+0x0c] = 0x000_071f        act_rt_base
mask[+0x18] = 0x000_07ff        alt_rt
mask[+0x30] = 0x000_0020        deep flag
mask[+0x38] = 0x02101000        EXTRA_PARAM POINTER  <-- !!!

od.out_table_stride_dwords  = 8         (= N_t)  ✓ match
od.out_y_stride_words       = 32        (≠ ours 0)  <-- !!!
od.n_tiles_pow2             = 32        (≠ ours 64) <-- !!!
od.m_total_minus_step       = 8         ✓ match
od.k_total_bytes            = 256       ✓ match

ad.n_act_pairs              = 8         ✓ match
ad.act_table_y_stride_words = 32        (≠ ours 0)  <-- !!!

extra_param[0] = 1                       ✓ match
extra_param[1] = 0                       ✓ match
```

## 5 个关键差异 vs 我们之前 V73DEEP

| 字段 | Native | 我们之前 |
|---|---|---|
| `mask[+0x0c]` (act_rt_base) | **0x71f** | 0x77c |
| `mask[+0x38]` | **= extra_param ptr** | 0 |
| `od.out_y_stride_words` | **32** (M_t * 4) | 0 |
| `od.n_tiles_pow2` | **32** | 64 |
| `ad.act_table_y_stride_words` | **32** (M_t * 4) | 0 |

`mask[+0x0c]` = 0x71f comes from `set_hmx_params_conv1x1(arg1=0x700, ..., arg5=0x20)`
（we used `arg1=0x70b` 给 0x77c）。

`n_tiles_pow2=32` 而非 64 = **kernel inner loop 减半**！这就是 packet 减半的根源。

## EMPIRICAL 应用 native descriptors → 36% packet 减少

新增 knobs `V73D_OUT_Y_STRIDE`, `V73D_AD_ACT_Y_STRIDE`, `V73D_MASK_38_EXTRA_PTR`：

```bash
DEFS="-DV9_USE_NATIVE_KERNEL -DV9_NATIVE_SINGLE_CALL -DV9_NATIVE_V73DEEP -DV9_C8_ALIGNMENT_TEST"
DEFS="$DEFS -DV73D_N_TILES_POW2=32 -DV73D_OUT_Y_STRIDE=32 -DV73D_AD_ACT_Y_STRIDE=32"
DEFS="$DEFS -DV73DEEP_ARG1=0x700 -DV73D_MASK_38_EXTRA_PTR=1"
EXTRA_DEFS="$DEFS" bash build.sh && bash build_x86.sh
```

| 配置 | hot pkts | hot dur | bit-exact | vs native |
|---|---:|---:|---:|---:|
| V73DEEP main path (旧 baseline) | 747 | 3042 | 100% | 2.16× |
| **+ native descs (n_tiles_pow2=32)** | **475** | **1730** | **50%** | **1.37× / 1.54× cyc** |
| Native ConvLayer_s1.opt | 346 | 1120 | ref | 1.0× |

artefact: `phase1_validation/v73deep_native_full_match/`

## 50% bit-exact 解释 — 还剩 50% 的 gap

`n_tiles_pow2=32` 只让 kernel 内 loop1=4 iters (vs 8)。 per outer 输出量减半。
4 outer × 4 loop1 × 2 (M-fanout) = 32 outputs，**只覆盖 50% 的 64 outputs**。

Native 在同样 `n_tiles_pow2=32` setup 下能覆盖全部 64 outputs，意味着
**native 的 act_tbl_all / out_tbl_all 布局 + stride 安排让 kernel 4 个 loop1 iter
能扫到全部 M+N 组合**。我们的 table 布局 [M-band, M-pair, N-tile] 跟 native
不同，with stride=32 时 kernel reads from wrong locations。

## 还需要确认的 native table layout

Native's act_tbl_all / out_tbl_all 必须有 stride 32 dwords 的 entry pattern
让 kernel `r19 += 32 dwords` 跨 loop1 iter 访问到正确的下一组 (M, N) 组合。

要破解 layout：
1. 修补 patch 让它额外 dump 32 dwords from r19 (act_tbl_all 起始 32 entries)
   和 32 dwords from r18 (out_tbl_all 起始)
2. 跑 native chain8 with patched skel
3. 解析 dumped pointers 推导 layout

## VTCM placement — 次要影响

Native r2 (wt) = `0xfc020800`, r3 (bias) = `0xfc020000` — VTCM addresses.
我们 wt + bias in DDR via qhpi_tensor_raw_data。

VTCM 比 DDR cyc/pkt 低 (3.24 vs 4.07 ratio观察)，但不影响 packet count。
要让 wt/bias 在 VTCM 需要 op signature 改成 TCM_Only — 详见
`qnn_primitive_alignment_phase01_2026-04-26.md`。

## V2 patch 扩展 — table layout RE 部分

V2 patch (`/tmp/p14_patch/dump_stub_v2.c`, 204 bytes) 在 v1 基础上额外 dump
32 dwords from out_tbl_all + 32 dwords from act_tbl_all to offset 0x90..0x18f
plus tail marker `0xFEEDFACE` at 0x190.

### Native out_tbl_all 揭示 (前 28 entries 干净)

```
[ 0..27]: 0xfc010000 + i * 0x800   <-- VTCM, 2KB stride per tile
[28..31]: 垃圾                      <-- 不知是 patch 限制 or layout 特殊
```

**关键事实**:
- 输出 tile 在 **VTCM** (0xfc010000..0xfc01ffff = 64KB, exactly = 256³ output size)
- Tile stride = **0x800 (2KB)** per entry, NOT 1024 — 暗示 tile = 2x M-fanout 64×32 layout
- 28 valid entries - 不能确认是真实数还是我们 readback 限制

### act_tbl_all dump 失败

`ad[0] = 0x02098c10`, but reading 32 dwords from there gave random-looking values
(0x340161bd, 0x22de0600, ...) — not pointers. 推测 readback 时机问题：act_tbl
内容可能在 chain 后续 op 中被覆盖, or 我们解码 layout 在 0x110+ offset 处偏离.

(Markers 在 offset 0x190 处也读不出 0xFEEDFACE，证明 tile 内 row-major 32×32 假设
在 row 8+ 不成立，可能 Crouton tile 实际 layout 复杂——但 row 0..7 是确认正确的.)

## artefact

- `Agent/qnn_re/p1_4_kernel_patch_2026-04-29.md` (本文件)
- `/tmp/p14_patch/dump_stub.c` `dump_stub.bin` `libQnnHtpV75Skel_PATCHED.so` (v1)
- `/tmp/p14_patch/dump_stub_v2.c` `dump_stub_v2.bin` `skel_v2.so` (v2 with table dump)
- `/tmp/p14_patch/native_dump/Y.raw` `native_dump_v2.raw` (raw native outputs)
- `phase1_validation/v73deep_native_full_match/` (with native descs applied — 475 pkts / 50% bit-exact)
- skel restored to original on device after each dump

## 下一步 (P1.5: full layout RE + VTCM placement)

要 close 剩余 1.37× gap, 需要：
1. 解 native act_tbl_all 完整 layout (32 entries × 4 bytes per loop1 iter,
   stride 32 dwords)
2. Match VTCM placement for wt + bias (现在我们用 DDR via qhpi_tensor_raw_data)
3. 重排 act_tbl_all 让 stride=32 + n_tiles_pow2=32 covers full 64 outputs

P1.4 已经把 gap 从 2.16× 减到 1.37× (= 36% 减包数). 剩余 gap 推测 ~80% 在
table layout, ~20% 在 VTCM placement (cyc/pkt ratio).
