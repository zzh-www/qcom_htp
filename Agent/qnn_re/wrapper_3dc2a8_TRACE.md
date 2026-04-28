---
name: wrapper_3dc2a8 static trace
description: Static disassembly trace of the u8 ConvLayer.opt wrapper at libQnnHtpV75Skel.so:0x3dc2a8 — loop trip / per-iter advances / kernel call ABI
type: project
---

# Wrapper @ 0x3dc2a8 — Static Trace (2026-04-28)

Source: `Agent/qnn_re/wrapper_3dc2a8.S` (0x3dc2a8 → 0x3dc4b0)

## Function ABI (entry)

`u8_wrapper(r0=outDesc, r1=actTensorObj, r2=?, r3=arg3, r4=arg4_pathFlags, r5=...)`

The wrapper saves callee-saved registers, allocates a 0xd8-byte frame, computes
two size-products, calls the descriptor builder, then runs a kernel-call loop.

## Pre-call setup (0x3dc2a8 .. 0x3dc314)

Computes two "tile-count" products from sub-descriptors:

```
sd0 = memw(r0 + 0x8)                     ; activations sub-descriptor
sd1 = memw(r1 + 0x8)                     ; weights sub-descriptor

r26 = (sd0[0x18] >> 3) * (sd0[0x1c] >> 3) * (sd0[0x20] >> 5)   ; "act tile count"
r27 = (sd1[0x18] >> 3) * (sd1[0x1c] >> 3) * (sd1[0x20] >> 5)   ; "wt tile count"
```

>> 3 / >> 3 / >> 5 ≡ divide by 8 / 8 / 32 — typical HMX-tile axis sizes.

Then:

```
call descriptor_builder_3d7920(
  r0 = r29 + 0x18,   ; output area pointer (writes desc at +0x10..+0x84)
  r1 = arg0,         ; act tensor obj
  r2 = arg1,         ; wt tensor obj
  r3 = r17_entry,    ; (mystery — NOT explicitly set in caller, must be entry-time r17)
  r4 = arg3,
  r5 = arg4
)
```

Note: descriptor builder calls a vtable method via `callr r2` at 0x3d7b1c
(virtual dispatch on `memw(arg1 + 0)` vtable slot +0x24). This is the
"obstacle" the original NEXT_STEPS already flagged.

## Post-call setup (0x3dc31c .. 0x3dc394)

```
r19 = r29 + 0x28                         ; will be passed to kernel as r1
r20 = r29 + 0x40                         ; will be passed to kernel as r0
r21 = r29 + 0x60                         ; mask desc — passed as r4 to kernel

r2  = memw(arg1 + 0x8)                   ; sd1 again (callee-saves preserved r23=arg1)
r23 = memw(sd1 + 0x4)                    ; ★ LOOP TRIP COUNT ★
if (r23 == 0): exit

r25:r24 = vaslw(r27:r26, #2)
  → r24 = r26 << 2 = (act tile count) * 4    ; per-iter advance for memw(r29+0x40)
  → r25 = r27 << 2 = (wt tile count) * 4

r26 = 0                                   ; loop counter (reused)

memw(r29+0x10) = (bit 4 of arg4 == 0)    ; flag1
memw(r29+0xc)  = (r25 > 0)                ; flag2 — gate for 1st dcfetch loop
memw(r29+0x8)  = (r24 > 0)                ; flag3 — gate for 2nd dcfetch loop
```

## Loop body (0x3dc394 .. 0x3dc4ac)

```
loop_top:
  ; Pre-touch cache lines for the activations & weights this iter will use
  if (flag2): for k in 0..ceil(r25/64): dcfetch(memw(r29+0x28) + offset)
  if (flag3): for k in 0..ceil(r24/64): dcfetch(memw(r29+0x40) + offset)

  ; Dispatch to one of 5 kernels based on tensor type combinations
  ; Our (u8 act × i8 wt × u8 out, 1×1) path lands at 0x3dc440:

  call hmx_v73_convbbb1x1_stride1
    r0 = r20 = &memw(r29+0x40)        ; ★ pointer to act-args struct ★
    r1 = r19 = &memw(r29+0x28)        ; ★ pointer to wt-args struct ★
    r2 = r17                           ; (entry-time r17, opaque)
    r3 = r18                           ; from memw(r29+0x24), opaque
    r4 = r21 = r29+0x60                ; mask_desc base (kernel reads +0x0/+0x8/+0x30)
    r5 = r16                           ; from initial save

  ; Per-iter advance
  r26 += 1
  memw(r29+0x40) += r24                 ; advance act-args first word by act_tiles * 4
  r2 = memw(r29+0x28)
  r2 = addasl(r2, r27, #2)              ; advance wt-args first word by wt_tiles * 4
  memw(r29+0x28) = r2

  if (r26 != r23): jump loop_top
```

## Kernel call ABI (re-derived from kernel disasm at 0x2eadc0)

The kernel reads from r0..r5:

| Arg | Wrapper origin | Kernel use |
|-----|----------------|------------|
| r0 = r29+0x40 | "act struct" addr | `r0[0]`=act ptr array base, `r0[4]`=n_act_pairs, `r0[8]`=accumulator stride, `r0[0xc]`,`r0[0x10]`,`r0[0x14]` = more fields |
| r1 = r29+0x28 | "wt struct" addr | `r1[0]`=wt ptr array base, `r1[4]`=?, `r1[8]`=m_total |
| r2 = r17 | entry-time | (unused early; passed thru) |
| r3 = r18 | stack reload | (entry-saved arg) |
| r4 = r29+0x60 | mask_desc | `r4[0..7]`=mask args 1+2; `r4[0x30]` bit-5 selects deep dispatch |
| r5 = r16 | (initial save) | passed thru |

**The wrapper increments only the FIRST word of the struct at r29+0x40 and r29+0x28**
(i.e. memw(r0+0) and memw(r1+0)). The other fields at +4, +8, +0xc, +0x10,
+0x14 are written ONCE by the descriptor builder and stay fixed across calls.

## What the wrapper varies per call

Only the array-base pointer at offset 0 of each struct. Per call:

```
new_act_ptr_array_base = old + r24 = old + 4 * (sd0_tile_count)
new_wt_ptr_array_base  = old + r25 = old + 4 * (sd1_tile_count)
```

So conceptually each call processes `sd0_tile_count` act tiles and
`sd1_tile_count` wt tiles, and the wrapper just slides through `r23` chunks.

## Total work check

Total act tiles across all calls = `r23 * sd0_tile_count`.
Total wt  tiles across all calls = `r23 * sd1_tile_count`.

For 256³ matmul (M=N=K=256):
- Activation [B=1, M=256, K=256] in HMX tiles = 32×8 = 256 tiles (with M_t=8, K_t=32 maybe?)
- Weight     [K=256, N=256]            = 8×8 = 64 tiles (K_t=N_t=8)

Pending: which tensor fields land in sd[0x18]/sd[0x1c]/sd[0x20] and sd1[0x4].

## 已 RE 出 / 未 RE 出 总结

### Solid (formula-level certain)

- ✓ Loop runs `r23 = memw(memw(arg1+0x8) + 0x4)` times.
- ✓ Per iter advances first-word-of-struct at memw(r29+0x40) by `(act_tile_count) * 4`.
- ✓ Per iter advances first-word-of-struct at memw(r29+0x28) by `(wt_tile_count) * 4`.
- ✓ Tile-count formula: `(sd[0x18]>>3) * (sd[0x1c]>>3) * (sd[0x20]>>5)`.
- ✓ Kernel called: `hmx_v73_convbbb1x1_stride1` (which dispatches to deep variant if mask[0x30] bit-5 set).

### Unknown (blocking blind impl)

1. **Tensor descriptor field semantics** — what real-world axes do sd[0x18]/[0x1c]/[0x20]/[0x4] correspond to for our [1,M,1,K] / [K,N] tensors?
2. **Descriptor builder vtable callback** — what does `callr r2` at 0x3d7b1c read? (Reads vtable[+0x24] of weight tensor obj.) Without this we can't reproduce what it writes to memw(r16+0x10..0x84).
3. **Initial value of memw(r29+0x40) and memw(r29+0x28)** — the descriptor builder writes p3 (a 0/1 flag) to memw(r29+0x28) at the END (3d7dfc). The actual array-base-pointer must be written elsewhere (probably by the inner virtual-dispatch path that we haven't traced).

## Why static RE alone won't close the gap

The blocking unknowns above ALL depend on tensor object internals that are
allocated by `libHtpPrepare.so` (prepare-time, x86 host) and shipped to the
DSP in the ctx-binary. Static RE can show the disasm but not the runtime
field layout, because that depends on the C++ type that `arg1` happens to be
(a `ConcreteTensor<...>` template instance with shape baked into the type).

To make blind progress here we'd need either:
- (a) a memory dump of arg0/arg1 at the wrapper entry during a real
      `q::ConvLayer.opt` execution at 256³, OR
- (b) the prepare-side C++ source that constructs these objects, OR
- (c) static RE of `libHtpPrepare.so` to find the matching constructor.

## Descriptor builder writes (full picture)

Builder is called with `r0 = r29_outer + 0x18` (so r16 in builder = r29_outer + 0x18).
That maps builder's `r16+offset` to wrapper-frame offsets like this:

| Builder `r16+X` | Wrapper-frame offset | Kernel-arg meaning |
|--------|----------------------|--------------------|
| +0x10 | r29+0x28 = r1+0      | wt ptr array base |
| +0x14 | r29+0x2c = r1+4      | (kernel reads as wt count?) |
| +0x18 | r29+0x30 = r1+8      | m_total ish |
| +0x1c | r29+0x34 = r1+0xc    | |
| +0x20 | r29+0x38 = r1+0x10   | |
| +0x24 | r29+0x3c = r1+0x14   | |
| +0x28 | r29+0x40 = r0+0      | act ptr array base |
| +0x2c | r29+0x44 = r0+4      | n_act_pairs |
| +0x30 | r29+0x48 = r0+8      | accumulator stride |
| ...   | ...                  | ... |
| +0x60..+0xa0 | r29+0x78..+0xb8 | mask desc area |

So the builder sets up BOTH "act-args struct" and "wt-args struct" plus mask
desc in one big contiguous area at r29+0x28..r29+0xb8. The wrapper then loops
and per call only ticks the array-base words at +0x28 and +0x40.

## Critical RE roadblock

Builder uses `callr r2` (3d7b1c, 3d7b40) — virtual dispatch through
`memw(arg1+0)+0x24` (vtable slot 0x24 of the weight tensor object). This
callback returns data the builder feeds into **memw(r16+0x10) and onward**
(the kernel's r0/r1 struct contents). Without resolving that vtable target
we can't reproduce the field initialization values blind.

The vtable slot is concrete-tensor specific; to find what method it points
to we'd need to:
- enumerate `*ConcreteTensor*` vtables in libQnnHtpV75Skel.so, OR
- (better) trace the prepare-side constructor in libHtpPrepare.so (x86)
  that builds the tensor object — this gives the field-by-field init.

## Most actionable next step

Static RE has hit the vtable wall. Three concrete options ranked by
effort vs payoff:

### Option 1 — Runtime descriptor dump (lowest effort, highest signal)

Patch our V9 op to dump `mask_desc[0..0x40]`, `od[0..0x80]`, `ad[0..0x40]`
to logcat at runtime. Then write a tiny QNN graph that USES native
ConvLayer.opt and have it ALSO dump (via a sibling custom op that hooks
the kernel symbol). Diff the descriptor bytes — that's the gap.

If we can't easily intercept the native call, simpler form: just dump our
own and visually compare to the static-RE-derived descriptor builder
output formula, then probe to fill gaps.

### Option 2 — RE libHtpPrepare.so x86 binary

The DLC compiler in `tools/qnn-sdk/lib/x86_64-linux-clang/libHtpPrepare.so`
contains the C++ that constructs `ConcreteTensor<...>` objects with the
field semantics we need. Static RE on x86 is faster (better tooling) and
gives ground truth for sd[0x18,0x1c,0x20,0x4] semantics.

### Option 3 — Pivot to NEXT_STEPS Phase 2 (kernel-prologue patching)

Skip wrapper RE entirely. Phase 2 in `NEXT_STEPS_v73deep_gap.md` says we
can patch the kernel's prologue at runtime to skip the ~30-pkt setup
across multiple calls, then reissue with appropriate descriptors. This
addresses the "per-call overhead" theory directly.

**Recommendation**: Option 1 first (~30 min), then if it shows the
descriptor field gap is what's expected, Option 3 to close the gap.
Option 2 is fallback if Option 1 doesn't pinpoint the field.
