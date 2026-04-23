# HVX 4-row × 32-col byte-transpose RE (v75)

Target: replace scalar `pack_weight_32x32` in
`example/hmx_matmul_w4a8/kernel/hmx_int4xint8_matmul.c:84-99`.
Per kg (of 8), input is 128 bytes laid out as 4 × 32 rows
`[r0(32)|r1(32)|r2(32)|r3(32)]`; output is 128 bytes laid out as 32 × 4
where `out[4c+i] = ri[c]`.

This is a **self-permute on one HVX vector** with the index mapping
`out[o] = in[π(o)]` where π is *rotate-right-by-2 on a 7-bit index*:

```
o  bits = o6 o5 o4 o3 o2 o1 o0   (0..127)
π(o)     = o1 o0 o6 o5 o4 o3 o2   (row=o1o0, col=o6..o2)
```

## Winning candidate: two back-to-back `Q6_Vb_vshuff_Vb`

### Sequence (per 128-byte kg chunk)

```c
HVX_Vector v0 = *(const HVX_Vector *)(w_32x32 + 128 * kg);  /* aligned load */
HVX_Vector s1 = Q6_Vb_vshuff_Vb(v0);
HVX_Vector s2 = Q6_Vb_vshuff_Vb(s1);
*(HVX_Vector *)(tile + 128 * kg) = s2;                       /* aligned store */
```

Full loop: 8 iterations × (1 vload + 2 vshuff + 1 vstore) = **32 HVX
instructions for the full 32×32 = 1024-byte weight tile**, vs the scalar
version's ~256 byte-loads + 256 u32-stores + shifts/ors.

### Correctness proof

From `80-N2040-54_AB_Hexagon_V73_HVX_PRM.pdf` p.203
(identical text in V75 PRM — `Vd.b=vshuff(Vu.b)` is a v60 baseline
instruction, unchanged on v75):

```
Vd.b = vshuff(Vu.b):
  for (i = 0; i < 64; i++) {          /* VELEM(16) = VBITS/16 = 1024/16 = 64 */
      Vd.uh[i].b[0] = Vu.ub[i];       /* VBITS/16 = 64 */
      Vd.uh[i].b[1] = Vu.ub[i + 64];
  }
```

Header declaration at `hvx_hexagon_protos.h:3635-3641`:

```
Assembly Syntax:       Vd32.b=vshuff(Vu32.b)
C Intrinsic Prototype: HVX_Vector Q6_Vb_vshuff_Vb(HVX_Vector Vu)
Instruction Type:      CVI_VP       /* permute resource only */
Execution Slots:       SLOT0123
```

Equivalent expression: `out.ub[o] = (o&1) ? in.ub[(o>>1)+64] : in.ub[o>>1]`.
As a 7-bit index map `i = π₁(o)`:

```
i6 i5 i4 i3 i2 i1 i0  =  o0 o6 o5 o4 o3 o2 o1    (rotate-right-by-1 by permuting bit labels)
```

Applying π₁ twice gives i = rotate-right(o, 2). Step-by-step trace
(with `v = [r0|r1|r2|r3]`, each ri is 32 bytes):

**Stage 1 — `s1 = Q6_Vb_vshuff_Vb(v)`:**
```
s1.ub[2k]   = v.ub[k]        for k = 0..63
s1.ub[2k+1] = v.ub[k+64]
```
So `s1` = `[r0[0], r2[0], r0[1], r2[1], ..., r0[31], r2[31],
           r1[0], r3[0], r1[1], r3[1], ..., r1[31], r3[31]]`
(low 64B = (r0,r2) byte-pairs, high 64B = (r1,r3) byte-pairs).

**Stage 2 — `s2 = Q6_Vb_vshuff_Vb(s1)`:**
Again `s2.ub[2k] = s1.ub[k]`, `s2.ub[2k+1] = s1.ub[k+64]`. For k = 0..31:
- `s2.ub[4k+0] = s1.ub[2k]   = r0[k]`
- `s2.ub[4k+1] = s1.ub[2k+64]= r1[k]`   (since k+64 ≥ 64 → high half)
- `s2.ub[4k+2] = s1.ub[2k+1] = r2[k]`
- `s2.ub[4k+3] = s1.ub[2k+65]= r3[k]`

So `s2[4c..4c+3] = [r0[c], r1[c], r2[c], r3[c]]` for c=0..31. ✓ Matches
spec exactly.

Independent verification via spot-check of π(o) = rotate_right(o, 2):
- `o=1 (000 0001)` → `π=32 (010 0000)` → `v[32] = r1[0]` ✓
- `o=2 (000 0010)` → `π=64 (100 0000)` → `v[64] = r2[0]` ✓
- `o=3 (000 0011)` → `π=96 (110 0000)` → `v[96] = r3[0]` ✓
- `o=7 (000 0111)` → `π=97 (110 0001)` → `v[97] = r3[1]` ✓

### Cost / cycle estimate

- 2 × `CVI_VP` per 128B = 2 permute-slot ops. On v75, permute is
  single-cycle throughput per slot; both vshuffs issue on
  SLOT0123 and can dual-issue with `vmem` loads/stores. In a tight
  loop of 8 kg's, expected **~2 HVX cycles per 128B chunk** (permute
  and memory issue in parallel), **~16 cycles** for the whole 1024B
  tile, vs **~256** scalar cycles for the current `pack_weight_32x32`.
- Dependency chain: v0 → s1 → s2 is 2 stages. Pipeline latency of
  vshuff on v75 is ~2 cycles each (std permute latency). Across 8
  independent kg's the compiler can software-pipeline and hide latency.

### Constraints / limitations

- **Alignment**: `w_32x32` and `tile` both need 128-byte alignment for
  `HVX_Vector`-typed loads/stores. Current code uses `int8_t*`
  arguments — check the call sites that allocate them. If either is not
  guaranteed aligned, use `Q6_V_vldu_A` / `Q6_V_vstu_AV` (unaligned
  load/store). For this pack path, the tile buffer is in VTCM (aligned)
  and `w_32x32` is the output of `gather_w_col` into a stack/scratch
  buffer — align that allocation to 128 bytes.
- **Row stride = 32**: the 4 input rows MUST be adjacent 32-byte
  stretches in memory so one 128-byte load covers them. If the caller
  ever passes a `row_stride != 32` variant, this approach needs a
  gather/reshuffle step; the scalar version would still work.
- **No inter-kg coupling**: 8 kg's processed independently; no need to
  group loads across kg's.
- **No setup constants**: no LUT, no shuffle control register, no
  predicate — zero setup cost.

## Runner-up: 2-input `Q6_W_vshuff_VVR(V, V, Rt)`

A single `Vdd = vshuff(V, V, Rt=0x41)` (bits 0 and 6 set) would compose
two perfect-shuffle stages in one instruction, and the low half of
`Vdd` would equal `s2` above. Header: line 3671-3677, type `CVI_VP_VS`
(permute + shift resources, still SLOT0123). Why it loses:

1. **Produces a `HVX_VectorPair` (256B) when we only need 128B** — uses
   two vector registers instead of one, increasing register pressure
   in the outer loop.
2. **Uses both permute AND shift resources** (`CVI_VP_VS`) — can stall
   dual-issue with another permute/shift op in the same packet,
   whereas two `CVI_VP` ops can each pair with a memory op.
3. Requires passing `V` as both Vu and Vv — the compiler typically
   handles this fine but it's an extra register-alias hint.

Per-kg cost comparable (1 VVR op vs 2 single-input ops), but the
register-pair output makes the outer pipelined loop noticeably messier.
Benchmark the two if you're chasing the last few cycles, but pick the
two-stage `Q6_Vb_vshuff_Vb` for cleaner codegen.

## Rejected candidates

1. **Single-stage `vshuff_VVR`** (one call for full 4-way zip) —
   impossible: `vshuff_VVR` is a 2-input primitive producing a 2-vector
   pair; it implements a *hierarchical* perfect-shuffle on 256 bytes
   total, not a single-vector 4-way transpose. Even with `Rt=-1` on
   `(V,V)` you get a deinterleave across 256-byte index space, not
   within 128B.
2. **`Q6_Wb_vshuffoe_VbVb`** — produces `Vdd` where
   `Vdd.lo = vshuffe(Vu.b,Vv.b)` and `Vdd.hi = vshuffo(Vu.b,Vv.b)`
   (PRM p.89). This is a 2-input even/odd byte split, not a 4-way zip.
   Would need `V = [r0|r2]` and `V' = [r1|r3]` pre-arranged; producing
   those halves from the contiguous `[r0|r1|r2|r3]` vector already costs
   more than the winning 2-vshuff solution.
3. **`Q6_Vb_vlut32_VbVbR`** — the byte LUT intrinsic takes a 32-byte
   LUT and an index vector; `Rt` selects one of 8 sub-LUTs. To cover a
   128-entry permutation we'd need 4 calls (`Rt=0,1,2,3`) OR-accumulated
   via `Q6_Vb_vlut32or_VbVbVbR`, plus splatting 128 index bytes and
   loading the data. Per PRM timing, each vlut32 is CVI_VP_VS and
   latency ~3 cycles. Total ~8-12 HVX ops per kg vs 2. Strictly worse.
4. **`Q6_Vh_vlut4_VuhPh`** — 16-entry halfword LUT driven by a 64-bit
   pair register. Wrong granularity for byte transpose.
5. **`Q6_V_vdelta_VV` / `Q6_V_vrdelta_VV`** — Benes/inverse-Benes
   butterfly networks. Prior attempt (`Agent/int4_matmul_optimization_log.md`)
   confirmed *vdelta cannot express this permutation at all* — the
   rotate-right-by-2 mapping has a 4-cycle within its permutation
   group (e.g. 1→32→64→... ) and the log2(128)=7-stage butterfly only
   realises permutations whose cycle lengths are powers of 2 with
   butterfly-compatible structure. Empirically 100% mismatches observed.
   `vrdelta` has the same network constraint (reversed direction), so
   it fails for the same reason.
6. **`Q6_Vb_vdeal_Vb`** — inverse of `Q6_Vb_vshuff_Vb`: deinterleaves
   even/odd bytes into the two halves. This is rotate-LEFT-by-1 on the
   7-bit index — applying twice gives rotate-left-by-2, NOT our
   rotate-right-by-2. Equivalent to applying `vshuff` five times; never
   better. (Worth noting: the inverse transpose 32×4 → 4×32 would use
   two `Q6_Vb_vdeal_Vb`.)
7. **`Q6_Wb_vpack*` / `Q6_Vh_vshuffe_VhVh` / `vunpack` / `vzxt`** —
   pack/unpack semantics change element width (byte→halfword). They
   don't implement byte-level permutations and are wrong tools here.
8. **Scalar fallback via HVX bulk loads + scalar permute** — no win:
   you still need 32 scalar ops per kg to do the byte-level reshuffle,
   plus unaligned vload/vstore boilerplate. The current scalar code is
   essentially this approach already.

## Alignment / vectorize-over-multiple-kg considerations

- 128 bytes = 1 HVX vector = exactly 1 kg's data. You *cannot* process
  multiple kg's per `vshuff` call because kg's are independent row
  groups that don't mix bytes — there's no shared byte movement. The
  wins come from *software-pipelining the loop across 8 kg's* (easily
  done by the compiler with a `#pragma unroll` or hand unroll).
- **Alignment audit needed**: check the allocator for `w_32x32`. In
  `hmx_int4xint8_matmul.c` the call chain is `gather_w_col` writes
  into a buffer that's currently `int8_t[1024]` on the stack. Add
  `alignas(128)` (or `__attribute__((aligned(128)))`) to avoid the
  unaligned-vload penalty. Tile buffer is already 128-aligned by the
  VTCM layout (`tile + 128*kg` with VTCM base aligned to 2 KiB).
- **Instruction issue width on v75**: `CVI_VP` issues on any of
  SLOT0123. The typical pair "`vmem` + `vshuff`" fits in one packet
  (mem on slot 0/1, permute on 2/3). Expected steady-state: ~1 packet
  per kg for load, ~1 packet per kg for store, ~2 packets for the two
  vshuffs (dependency-serialized within one kg but fully
  parallelizable across kg's). With unrolling, ≥2 kg's per 4-packet
  loop body is reachable.

## Implementation note

Nothing to set up: no LUT, no shuffle control, no predicate register.
Just include `<hvx_hexagon_protos.h>` (already included by the
existing kernel) and replace the nested loop. Keep the scalar fallback
under `#ifndef __HVX__` for host-side sim / unit-test builds.

## References (quoted lines)

`tools/hexagon-sdk/tools/HEXAGON_Tools/19.0.07/Tools/target/hexagon/include/hvx_hexagon_protos.h:3634-3641`:
```
Assembly Syntax:       Vd32.b=vshuff(Vu32.b)
C Intrinsic Prototype: HVX_Vector Q6_Vb_vshuff_Vb(HVX_Vector Vu)
Instruction Type:      CVI_VP
Execution Slots:       SLOT0123
#define Q6_Vb_vshuff_Vb(Vu) __BUILTIN_VECTOR_WRAP(__builtin_HEXAGON_V6_vshuffb)(Vu)
```

`80-N2040-54_AB_Hexagon_V73_HVX_PRM.pdf` p.203 behavior:
```
Vd.b=vshuff(Vu.b)  for (i = 0; i < VELEM(16); i++) {
                       Vd.uh[i].b[0] = Vu.ub[i];
                       Vd.uh[i].b[1] = Vu.ub[i + VBITS/16];
                   }
```
(V75 PRM 80-N2040-57 carries the identical definition — instruction is
v60 baseline, unchanged on v75.)

`tools/hexagon-sdk/tools/HEXAGON_Tools/19.0.07/Tools/target/hexagon/include/hvx_hexagon_protos.h:3670-3677`
(for reference — the runner-up VVR form):
```
Assembly Syntax:       Vdd32=vshuff(Vu32,Vv32,Rt8)
C Intrinsic Prototype: HVX_VectorPair Q6_W_vshuff_VVR(HVX_Vector Vu, HVX_Vector Vv, Word32 Rt)
Instruction Type:      CVI_VP_VS
Execution Slots:       SLOT0123
```
