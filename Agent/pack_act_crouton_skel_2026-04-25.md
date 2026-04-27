---
name: PackActCrouton via convert_to_crouton_b — production-ready primitive
description: First Route 1 wrapper op; wraps libQnnHtpV75Skel.so::convert_to_crouton_b for activation Crouton pack at any (M, K) where M%4==0, K%128==0
type: project
---

# PackActCrouton — first production Route 1 primitive (2026-04-25)

## Bit-exact validated shapes

| M     | K     | Output bytes | n_diff | Status |
|-------|-------|-------------:|-------:|--------|
| 32    | 128   |     4096     | 0      | PASS   |
| 64    | 128   |     8192     | 0      | PASS   |
| 256   | 128   |    32768     | 0      | PASS   |
| 512   | 128   |    65536     | 0      | PASS   |
| 1024  | 128   |   131072     | 0      | PASS   |
| 32    | 256   |     8192     | 0      | PASS   |
| 128   | 256   |    32768     | 0      | PASS   |
| 32    | 512   |    16384     | 0      | PASS   |
| 256   | 512   |   131072     | 0      | PASS   |
| 32    | 1024  |    32768     | 0      | PASS   |
| 512   | 1024  |   524288     | 0      | PASS   |
| 256   | 2048  |   524288     | 0      | PASS   |
| 32    | 4096  |   131072     | 0      | PASS   |

## Critical descriptor parameters (debugged 2026-04-25)

For `convert_to_crouton_b`:
- `block_offset_table` = K/32 entries, **monotonic depth-lane order** —
  table[g] = output_base + g * M_grp * 128
- `outer_step` = 0
- `row_stride` = K (input row stride bytes)
- `channel_groups` = **4 always** (not K/32!) — represents "blocks per inner
  vshuff iter", NOT total K-lanes; the function's middle loop walks the
  full table by advancing r26 16 bytes (4 entries) per inner iter.
- `height_tiles` = M_grp_in_call (≤ 16 due to chunking — see below)
- `depth` = K (in bytes)
- `aux` = **16** (r1) — controls r25 stride exponent. ct0(16)=4 → per-h-iter
  r25 += 128.

## Hard constraint discovered

**aux=16 only works for height_tiles ≤ 16** (i.e., M_grp ≤ 16, M ≤ 64 per
call). For h ≥ 16, the function advances `r20 += 4*(h>>ct0(aux))*row_stride`
which walks the table base beyond our small table, causing OOB writes.

**Workaround**: split the slice's height range into chunks of ≤ 16 and call
`convert_to_crouton_b` once per chunk, with the table pre-offset by
`chunk_h0 * 128` so each sub-call writes to the correct flat-output bytes.

```c
enum { CHUNK_H = 16 };
for (chunk_h0 = h_start; chunk_h0 < h_end; chunk_h0 += CHUNK_H) {
    chunk_h_count = min(CHUNK_H, h_end - chunk_h0);
    for (g = 0; g < K_grp; g++)
        table[g] = out + g * M_grp * 128 + chunk_h0 * 128;
    p.height_tiles = chunk_h_count;
    convert_to_crouton_b(&p, 16, a + chunk_h0 * 4 * K);
}
```

## Output layout

`[1, K/32, M/4, 128]` u8 contiguous. Block `(g, h)` lives at offset
`(g * M/4 + h) * 128` and contains the 4-spatial × 32-depth slice:
- byte `r * 32 + c` of block `(g, h)` = `act[h*4+r][g*32+c]` for r in 0..3, c in 0..31.

This matches QNN's Crouton-byte layout consumed by
`hmx_convbbb1x1_stride1` weight DMA pattern. Each `K_tile` of 32 K-rows ×
32 N-cols = 8 contiguous h-groups × 128 B = 1024 B (one HMX tile).

## Reuse for weights

The same op with input `[1, 1, K, N]` (treating K as the "M" dim and
N as the "K" dim) produces the layout HMX expects for weights:
`[1, N/32, K/4, 128]`. Each N_lane has K/32 contiguous 1024-B K-tiles.
**No separate PackWtCrouton needed** — just use PackActCrouton in graph.

## Files

- Kernel: `example/hmx_matmul_phase3/kernel/pack_act_crouton_skel.c`
- Op name: `HmxMatMulPhase3Package::PackActCrouton`
- Test driver: `example/hmx_matmul_phase3/standard_flow/phaseB_v8/test_pack_act_crouton.sh`
- Reference + ONNX gen: `gen_pack_act_crouton_test.py`

## What's next

- Phase 3c: MatMulSkelHmx — wrap `hmx_convbbb1x1_stride1`. Needs HMX
  context acquire (`nn_os_vtcm_hmx_acquire`) + 3-descriptor setup.
- Skip explicit PackWtCrouton (reuse PackActCrouton with [K,N] input).
- Phase 4: gen_v10_graph.py emits new topology using PackActCrouton +
  MatMulSkelHmx + Concat + output finalize.
