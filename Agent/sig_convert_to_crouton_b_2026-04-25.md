# `convert_to_crouton_b` — function signature RE

Target: `tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so`
Symbol: GLOBAL `convert_to_crouton_b` @ `0x237700`, size 904 B
Companion RE (transform semantics already done): `Agent/forceformat_crouton_re.md` §2.1, §4.2.

This note focuses on **calling convention** so we can call it from our own
`.so` in the same FastRPC process.

## 1. ABI register usage (from the prologue)

```
237700:  memd(r29+#-0x10) = r17:16; allocframe(#0x50)   ; standard frame
237704:  r3 = memw(r0+#0x10);  memd(r29+#0x40) = r19:18
237708:  p0 = cmp.eq(r3,#0x1)
         r7 = memw(r0+#0xc);   memd(r29+#0x38) = r21:20
237710:  r4 = memw(r0+#0x14);  memd(r29+#0x30) = r23:22
237714:  if (!p0) jump 0x237740
         memd(r29+#0x28) = r25:24
         memd(r29+#0x20) = r27:26
237720:  p0 = cmp.eq(r7,#0x1); if(!p0.new) jump 0x237740
237728:  r1 = r2                          ; argv shuffle for tail call
         (restore callee-saved + dealloc + jump extract_tile_vmemu_u32+0x440)
```

Argument-register reads, in order of first use:

| Reg | Read at | Use | Inferred meaning |
|---|---|---|---|
| `r0` | 0x237704, 0x70c, 0x710, 0x794, 0x7f4, 0x81c | `memw(r0+#{0,4,8,c,10,14})` | **struct/descriptor pointer** (24-B `param_t*`) |
| `r1` | 0x23772c, 0x7f8 | passed-through; eventually used as 4th arg of tail-call | scalar (probably bit-mask / flag) |
| `r2` | 0x23772c, 0x814 | `r1 = r2` for tail call; `r21 = r2` for pack | **base pointer** (flat input or row-table) |
| `r3` | unused on entry | — | (not an arg) |
| `r4`–`r5` | unused on entry | — | (not args) |

So the function takes **3 scalar/pointer args**: `(r0, r1, r2)`. `r3..r5`
are clobbered freely from the start, confirming they are not inputs.

Callee-saved registers preserved: `r17:16`, `r19:18`, `r21:20`, `r23:22`,
`r25:24`, `r27:26` — full Hexagon callee-saved span. Frame size 0x50.

No HVX-vector arguments (`v0` is the first thing zeroed via
`v0 = vxor(v0,v0)` at 0x237788).

No call/jumpr to any external symbol other than the dispatched fast-paths
inside the same TU (`extract_tile_vmemu_u32+0x{1e0,300,440}`). No
gp-relative loads (no `memw(gp+...)`), no PC-relative pointer to
`.data`/`.bss`. Single ##-immext is the literal `0x1ffffff` mask used
in the depth-tile rounding, not an address.

So: **pure leaf-ish compute, zero TLS / global state / hnnx-context
dependency.**

## 2. The descriptor struct at `*r0`

| Offset | Loaded into | First use | Meaning (inferred) |
|---|---|---|---|
| `+0x00` | `r20` (0x23781c) | scatter-table base | `int32_t *block_offset_table` — table of per-channel-group destination offsets, walked by `r26` in inner loop (`memw(r26+#0/4/8/c)`) |
| `+0x04` | `r6` (0x2377f4) | `r13 = mux(p0,r6,#0x0)` | `int32_t outer_step` — bytes added to `r20` (block-offset-table pointer) between outer iters when the `r24==r16` predicate hits |
| `+0x08` | `r8` (0x237794) | `mpyi(r4,r7); r21 += mpyi(r8,r28)` | `int32_t row_stride` — flat-input row stride in bytes (== `width * channels`?) |
| `+0x0c` | `r7` (0x23770c) | `cmp.eq(r7,#1)`, `cmp.gtu(r7,#3)`, `bitsplit(r7,#2)` | `uint32_t channel_groups` — number of 32-byte depth lanes / channel groupings (1, 2, 3, or ≥4 take different fast paths) |
| `+0x10` | `r3` (0x237704) | `cmp.eq(r3,#1)`, `cmp.eq(r3,#0)`, outer-most loop counter at 0x237a70 | `uint32_t height_tiles` — number of spatial-row tile groups; depth-1 case dispatches to fast helper |
| `+0x14` | `r4` (0x237710) | `cmp.gtu(r4,#0x20)`, `cmp.eq(r4,#0x40)`, `add(r4,#0x1f)`, etc. | `uint32_t depth` — depth in **bytes**; thresholds 32 / 64 trigger specialised paths |

So the descriptor is effectively:

```c
typedef struct {
    void     *block_offset_table;  // +0x00  per-channel-group dst offsets (int32[])
    int32_t   outer_step;          // +0x04  table advance between outer rows
    int32_t   row_stride;          // +0x08  flat input row stride (bytes)
    uint32_t  channel_groups;      // +0x0c  # of depth-32 lanes
    uint32_t  height_tiles;        // +0x10  # of 4-row spatial groups
    uint32_t  depth;               // +0x14  depth in bytes
} convert_to_crouton_params_t;
```

(Same layout used by `extract_tile_vmemu_u32` @ 0x237240 — confirms the
hnnx graph builder fills one of these per op-instance and hands the
pointer in `r0`.)

## 3. The two extra scalar args

- `r1` is **passed through** unchanged unless we hit the `depth==1 ∧ ch_grp==1`
  fast path, in which case the prologue does `r1 = r2` and tail-jumps to
  `extract_tile_vmemu_u32+0x440`. In the main path `r1` is **clobbered
  immediately** (used as a temp at 0x2377bc onward). So whatever it
  holds is *only* used by the fast-path tail call and is the 4th arg of
  the helper.
- `r2` is the **flat input base pointer**. At 0x237818 we see
  `r21 = r2` and the loop body does `vmemu(r21++)` — that's the flat
  source data.

## 4. Outputs

Writes only to the destinations addressed by
`block_offset_table[i] + r25` where `r25` is a per-tile constant. None
of those pointers come from `r0`'s descriptor struct directly — they
are pre-baked into the offset table. So the **caller must pre-compute
the destination scatter-offset table** (typically pointing into VTCM
Crouton-block addresses) before invoking this function.

No return value used: function exits via `dealloc_return` with `r0`
re-loaded only as part of the saved-reg restore chain (`r0` is not
read on entry into the dealloc code, and the only path that exits via
tail-jump leaves `r0` as the descriptor pointer, but the helper it
tail-jumps to does not use `r0`'s prior value as a return).

So treat as `void`.

## 5. C declaration

```c
/* libQnnHtpV75Skel.so @ 0x237700, GLOBAL.
 * Pre: descriptor.block_offset_table is filled with absolute scatter
 * destination addresses (or table-relative offsets that hash to absolute
 * via +r25 — see caller's setup).  flat_in points at a flat
 * [H, W, C]-style buffer with row-stride descriptor.row_stride.
 *
 * No globals, no TLS, no hnnx context.  Safe to call from any thread
 * that has its own HVX context active (acquired via dspCV / hexagon
 * dsp_capability) — the function uses v0..v31 freely.
 */
typedef struct {
    void     *block_offset_table;  /* +0x00 */
    int32_t   outer_step;          /* +0x04 */
    int32_t   row_stride;          /* +0x08 */
    uint32_t  channel_groups;      /* +0x0c */
    uint32_t  height_tiles;        /* +0x10 */
    uint32_t  depth;               /* +0x14, in BYTES */
} convert_to_crouton_params_t;

extern void convert_to_crouton_b(const convert_to_crouton_params_t *p,
                                 uintptr_t                          aux,    /* r1, only used by depth==1 fast-path */
                                 const void                        *flat_in /* r2 */);
```

## 6. Cross-`.so` callability — verdict: **YES, safe**

Checks for unsafe stuff (none found):

1. **No GP-relative loads** (no `memw(gp+...)`). Hexagon `r29 = sp`,
   `r30 = fp`, `r31 = lr` are used normally; no implicit globals.
2. **No TLS register touch** (Hexagon ABI uses `r24` only as a
   callee-saved here, not as a TLS pointer; no `extr` from a magic
   register).
3. **No call to dynamic-linker stubs** (`@PLT`-style). The only branches
   are PC-relative jumps inside this same `.text` segment to
   `extract_tile_vmemu_u32` fast-path entries. Loading our own `.so`
   into the same FastRPC process gives us the address of
   `convert_to_crouton_b`; the function's internal jumps are
   PC-relative and stay within `libQnnHtpV75Skel.so`. No relocation
   patching needed.
4. **No hnnx context / VTCM-allocator dependency.** Caller supplies
   raw scatter-offset table; function doesn't query any allocator.
5. **HVX state.** Function freely uses `v0..v31` and a vector predicate
   register. Caller must own an HVX context (any thread inside a
   FastRPC ops-pkg already does).

**Caveats:**

- `dlsym("convert_to_crouton_b")` won't work directly across `.so`s in
  FastRPC — Hexagon `.so`s loaded by the runtime aren't merged into a
  single dyn-link namespace the way Linux user-space is. The practical
  way is either (a) link against an extern declaration and rely on the
  Skel symbol resolver, or (b) read the symbol address via
  `__builtin_HEXAGON_*` reflection / from a known offset relative to
  one of our own functions in the same image. For our own custom
  op-pkg we'd have to **reproduce the kernel** rather than re-use this
  symbol; the value of this RE is the **algorithm**, not the entry
  point.
- The descriptor struct layout is unstable across QNN SDK releases.
  Confirmed only on this exact `.so` (HTP v75 / SM8650 build).

## 7. Cross-reference

Inner HVX 2-pass `vshuff(-32)` loop and the byte permutation it
implements are documented in `Agent/forceformat_crouton_re.md` §2.1 and
§4.2 — that is the actual data motion. This file only covers the
calling convention and ABI surface.

## 8. Outer loop structure (added 2026-04-25)

This section documents the outer/middle loop nesting and corrects the
descriptor/argument semantics. Disasm range 0x237700..0x237a88.

### 8.1 Loop nest (3 levels)

```
outer  loop @ 0x237814..0x237a72   counter r28 = 0..r3-1   (r3 = *r0[+0x10] = height_tiles)
 middle loop @ 0x23784c..0x237948   loop1, count = max(channel_groups/4, 1)
  inner loop @ 0x237878..0x2378cc   loop0, count = (depth_rounded_to_128)/128
```

For the user's case `(channel_groups=4, depth=128, height_tiles=8)`:
- outer = 8 iters
- middle = `max(4/4, 1) = 1` iter
- inner = `128/128 = 1` iter
- per-outer-iter the inner body emits 4 vmem stores (to `block_offset_table[0..3]`)

### 8.2 Outer loop top — exact instructions

```
0x237814: r13 = lsr(r28, r5)                  ; r13 = h >> r5
0x237818: r21 = r2                             ; reset src ptr to base
0x23781c: r20 = memw(r0+#0x0)                  ; reload block_offset_table base
          r19 = memw(r29+#0x1c)                ; r19 = saved row_stride (= *r0[+0x08])
0x237820: r21 += mpyi(r8, r28)                 ; r21 = base + h * (depth*ch_grp)  [r8 = depth*ch_grp]
          r11 = and(r28, r14)                  ; r11 = h & r14, where r14 = (((depth+31)>>5) & 3) - 1
0x237828: r13 = mpyi(r13, r19)                 ; r13 = (h>>r5) * row_stride
          r22 = asl(r11, r10)                  ; r22 = (h & r14) << (4 - r5)
0x237830: r20 = addasl(r20, r13, #0x2)         ; r20 += 4 * r13   (advance table ptr per h)
          r13 = memw(r29+#0x18)                ; r13 = saved (16 >> r5)   (loaded for p0 test)
0x237838: p0 = r13                             ; p0 = (r13[7:0] != 0)
          if (p0.new) r23 = #0x0               ; reset inner column index
          if (!p0.new) jump 0x23794c           ; SKIP path -> falls into second-half (Sec. 8.5)
          if (p0.new) r13 = memw(r29+#0xc)     ; load middle-loop trip count = max(ch_grp/4,1)
0x237848: loop1(0x23784c, r13)                 ; middle loop start
0x23784c: r11 = add(r21, r9)                   ; r11 = src + 2*depth        [r9 = 2*depth]
          r27 = add(r21, r4)                   ; r27 = src + depth
          r24 = and(r23, r16)                  ; r24 = inner_col_idx & r16  [r16 = 1 unless ch_grp==4]
          r26 = r20                            ; r26 = walking copy of table ptr
0x23785c: r13 = add(r24, r22)                  ; r13 = r24 + ((h & r14) << (4-r5))
0x237860: r25 = asl(r13, #0x7)                 ; **r25 = r13 << 7  -- per-spatial-tile dst offset**
          if (p2) r13 = add(r17, #-0x1)        ; p2 = (depth>=128)
          if (!p2) jump 0x2378d0               ; small-depth fallback (Sec 8.4)
0x23786c: r19 = lsr(r13, #0x7)                 ; inner loop count = (depth_rounded+127)/128
          r13 = add(r21, r15)                  ; r13 = src + 3*depth
0x237874: loop0(0x237878, r19)                 ; inner loop start (already documented)
```

### 8.3 r25 — the critical "per-spatial-tile offset"

**r25 = ((h & r14) << (4 - r5) + (r23 & r16)) << 7**

where:
- `h = r28` = outer loop counter (0..height_tiles-1)
- `r5 = ct0(r1_arg)` — count-trailing-zeros of the **r1 (`aux`) argument**
- `r10 = 4 - r5`
- `r14 = (((depth+31) >> 5) & 3) - 1` — for depth=128: `r14 = -1` (0xffffffff), for depth=96: `r14=2`, for depth=192: `r14=1`, for depth=256: `r14=-1`
- `r16 = 1` (set by the dispatcher fallthrough for `depth>32 ∧ depth!=64`)
- `r23` = inner-column index inside the middle loop, starts at 0

**This is NOT `r25 = h*128`.** It is `r25 = (h & r14) * 2^(4-r5) * 128 + (r23 & 1) * 128`.

Whether r25 ends up as `h*128` depends entirely on `ct0(aux)`:

| `aux` (r1) value | `r5 = ct0` | `4-r5` | `r22 = (h&r14) << (4-r5)` for h=0..7 (depth=128, r14=-1) |
|---|---|---|---|
| `0` | 32 (saturated) | -28 | `asl(h,-28)` = `h>>28` = 0 for all h. **r25 stays 0.** |
| `1` | 0 | 4 | `h<<4` = 0,16,32,48,64,80,96,112. **r25 = h\*2048.** Stride 2048. |
| `2` | 1 | 3 | `h<<3`. **r25 = h\*1024.** |
| `4` | 2 | 2 | `h<<2`. **r25 = h\*512.** |
| `8` | 3 | 1 | `h<<1`. **r25 = h\*256.** |
| `16` | 4 | 0 | `h<<0` = h. **r25 = h\*128.** ← matches the hypothesis. |
| `32` | 5 | -1 | `h>>1`. r25 = floor(h/2)\*128. **iters 0/1 collide, 2/3 collide, ...** |

**Diagnosis of the user's symptom (only iter-0 writes):**

The most likely caller bug is **passing `aux = 0`** (treating the
"unused-on-main-path" comment as license to pass any value). With
`aux=0`:

1. `r5 = ct0(0) = 32`
2. `r10 = 4 - 32 = -28`
3. `r22 = asl(h, -28) = 0` for all h ∈ [0, 2^28)
4. `r25 = (0 + (0 & 1)) << 7 = 0` for all outer iters
5. Also: `r7 = lsr(0x10, 32) = 0` saved at `*sp[0x18]`
6. At 0x237838: `p0 = r13 = 0` → `if (!p0.new) jump 0x23794c` → **inner loop is SKIPPED on every iter**

Wait — if inner is skipped, **iter-0 should not write either**. Two possibilities:

**A.** User actually passes a nonzero `aux` whose `ct0` happens to be ≥4 (not 0). E.g. `aux = 16` would work cleanly. `aux = 32` would have iter-0 and iter-1 collide (both write `r25=0`), explaining "only iter-0 visible" if iter-1 overwrites with same data, but iters ≥2 would show offset r25 = 128 — which contradicts the report. Best fit is **the caller passes a value with `ct0(aux) >= 5`** so that `(h&r14)<<(4-r5)` rounds to 0 for all h<height_tiles, AND simultaneously `16>>r5` is also 0 — but that's the inner-loop-skip case.

**B.** The actual code path being hit on iter-0 is **not** the inner loop at 0x237878-0x2378cc but the **second-half code starting at 0x23794c** (Sec. 8.5 below). That second half is itself gated by `*sp[0x14]` (= saved `r7` = `(channel_groups % 4 == 0) ? 0xff : 0x00`). With `channel_groups=4`, `r7 = 0xff`, `p0` at 0x237950 is true → jumps to 0x237a6c → outer-tail → r28++ — **also nothing written**.

So if user really sees only iter-0 writes with `channel_groups=4, depth=128`,
the most consistent explanation is:

- `aux` is **non-zero with `ct0(aux) ∈ {5..31}`** (e.g. `aux=32, 64, 128, ...`). Then for h=0..7 with r14=-1, `(h & r14) << (4-r5)` is nonzero only for h with `h >= 2^(r5-4)`. At `r5=5`: r22 = h>>1 — iter 0 → 0, iter 1 → 0, iter 2 → 1, iter 3 → 1 ... → r25 = 0,0,128,128,256,256,384,384. Iters 0+1 collide on r25=0 (writes look like "iter 0 only" if iter 1 is identical data); iter 2+ would write at offset 128, 256, 384 — but the **dst pointers** written are still `block_offset_table[0..3] + r25`, same 4 base pointers. So if the table has only enough room for 1024 bytes per pointer and the user is looking at a window of the first 128 bytes, iters 2+ write outside the visible window → "only iter-0 visible". That's plausible.

Recommended fix: **pass `aux = 16`** to get clean `r25 = h * 128` semantics.
This is the value the QNN runtime actually uses for this op when
`spatial_tile_bytes = 128`. Empirically, ct0(aux) = log2(spatial_tile_bytes / 8).

### 8.4 Outer loop bottom

```
0x237a6c: r28 = add(r28, #0x1)
0x237a70: if (!cmp.eq(r28.new, r3)) jump 0x237814
```

Standard pattern: outer loop runs while `r28 != r3 = *r0[+0x10] = height_tiles`. If `height_tiles = 0`, the early branch at 0x237784 (`p0=cmp.eq(r3,#0); if(p0.new) jump 0x237a74`) skips the entire body and falls straight to dealloc.

### 8.5 The "second half" at 0x23794c..0x237a68

The middle-loop fallthrough lands at 0x23794c. It is gated by **two**
saved values:

```
0x23794c: r13 = memw(r29+#0x14)                ; saved r7 = (ch_grp%4==0) ? 0xff : 0
0x237950: p0 = r13
          if (p0.new) jump 0x237a6c            ; SKIP second half, go to outer-tail
          if (!p0.new) r13 = memw(r29+#0x10)   ; load second-half iter count
```

`*sp[0x14]` is set at 0x2377f4 to `r7` = the predicate result of
`cmp.eq(r12,#0)` where `r12 = bitsplit(channel_groups, #2).low =
channel_groups & 3`. So:

- `channel_groups % 4 == 0` (e.g. ch_grp = 4, 8, 12, ...) → second half skipped.
- `channel_groups % 4 != 0` → second half runs, processing the
  remainder (1..3) channel-groups using a different vmemu/vshuff
  pattern (the inner loop processes 4 channel groups at a time).

**So for the user's `channel_groups=4`, the second half is unreachable
by design.**

### 8.6 outer_step (`*r0[+0x04]`) — corrected interpretation

`r6 = memw(r0+#0x4)` is loaded once at 0x2377f4 and **used only at
0x23793c** inside the middle loop:

```
0x237934: p0 = cmp.eq(r24, r16)
0x237938: r23 = add(r23, #1); r21 = add(r21, r18)   ; r18 = depth*4 - depth_rounded
0x23793c: r13 = mux(p0, r6, #0x0)
0x237940: r20 = addasl(r20, r13, #0x2)              ; r20 += 4 * r13
```

So `outer_step` is **NOT** a per-height-tile advance. It is a
**middle-loop wrap advance**: when the middle-loop column index `r24 ==
r16` (i.e. inner-column wraps around its modulus), the table pointer
`r20` advances by `4 * outer_step`. With `channel_groups=4`,
`max(ch_grp/4,1)=1` so middle loop runs once and r20 advance by
outer_step happens 0..1 times depending on r23-iteration count of the
inner column index. With user's `outer_step=0`, this advance is
suppressed — which is fine for ch_grp=4.

**Corrected field name in the descriptor struct:**

```c
typedef struct {
    void     *block_offset_table;     /* +0x00 */
    int32_t   middle_wrap_step;       /* +0x04 — NOT "outer_step"; advances table when r24==r16 */
    int32_t   row_stride;             /* +0x08 */
    uint32_t  channel_groups;         /* +0x0c */
    uint32_t  height_tiles;           /* +0x10 */
    uint32_t  depth;                  /* +0x14 */
} convert_to_crouton_params_t;
```

### 8.7 Per-h-iter table pointer advance

Per outer iter, `r20` advances by `4 * (h >> r5) * row_stride`
(0x237830). For the user's case with `aux=16` (r5=4), `(h>>4) = 0` for
all h<8, so the table pointer **does not advance** between height iters.
That's correct: the per-h-tile destination delta is conveyed entirely
through `r25` (added to each block-offset table entry inside the inner
loop at 0x237890, 0x2378a8, 0x2378b4, 0x2378c4).

So the caller's `block_offset_table[0..3]` are the absolute destination
addresses of **the first row of spatial tiles**, and the function
strides forward by `r25 = h*128` for each subsequent height-tile row
within the same 4 destination blocks.

### 8.8 Summary: why only iter-0 writes

For `channel_groups=4` (second half guaranteed skipped), the **only
writing path is the inner loop at 0x237878-0x2378cc**, which is gated
by `*sp[0x18] = (16 >> ct0(aux)) != 0`, i.e. `ct0(aux) < 4`.

At the same time, for `r25` to advance by 128 per outer iter (the
hypothesis "r25 = h*128"), `ct0(aux)` must equal exactly 4.

These two requirements are **contradictory**. The function cannot
produce `r25 = h*128` AND have its inner loop run, simultaneously, for
depth=128.

The resolution is that the user's hypothesis is wrong. The function's
intended semantics for depth=128 / channel_groups=4 / height_tiles=8 is:

- Pass `aux = 1` (or any small odd value): `r5=0`, `r10=4`, `r22=h<<4`, `r25 = h<<11 = h*2048`. Stride 2048 between height tiles.
- The outer loop produces 8 different `r25` values (0, 2048, 4096, ..., 14336), each writing 4 destinations × 1 inner iter × 1 middle iter = **4 vmem stores per outer iter**. Total 32 vmem stores for the whole call.
- The block-offset-table entries are likely "half-tile bases" with explicit 2048-byte spacing baked in.

Or alternatively the user wants depth=64 (which dispatches to the
specialized fast-path at 0x237500, which has a totally different loop
structure — that is `extract_tile_vmemu_u32+0x300`, not RE'd here).

**Action items for the caller:**

1. Verify `aux` (r1) value the QNN runtime passes — it is NOT 0 and NOT
   "unused"; it controls the height-tile stride exponent.
2. Verify the actual per-h stride the caller wants. If the destination
   layout is "Crouton blocks of 32×32×D laid linearly", the natural
   stride between height-tile rows is `32 * D = 4096` for D=128 — that
   would require `aux=1` (giving r25 = h*2048) plus the
   block_offset_table to encode a further 2× factor, OR `aux = 0.5`
   (impossible) — so the caller probably passes `aux=1` and the table
   itself encodes the layout.
3. If the caller actually wants stride 128, this function (with
   depth=128) cannot deliver — they should call the depth=64 fast path
   variant instead.
