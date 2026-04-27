# `hmx_convbbb1x1_stride1` — function signature RE

Target: `tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so`
Symbol: GLOBAL `hmx_convbbb1x1_stride1` @ `0x2ea740`, size 492 B.
Companion RE (HMX inner-loop semantics already done): `Agent/qnn_hmx_pipelining.md`.

This note focuses on **calling convention** so we can call it from our own
`.so` in the same FastRPC process. This is QNN's hot u8·i8 1×1-conv /
MatMul kernel — the body of `q::ConvLayer_s1.opt` for the bbb (byte
input, byte weight, byte output) datatype combo on v75.

Sibling kernels with the same ABI shape: `hmx_convbbb_stride1`,
`hmx_convbbb1xN_stride2`, `hmx_convbbb_dilate_stride1`,
`hmx_convbbb_stride1_aligned`, plus the f16 / hbh / hnh families.
The **descriptor layout** below is shared across the family (verified by
spot-disassembling `hmx_convbbb1xN_stride2` @ `0x2ec2c0`).

## 1. ABI register usage

Hexagon ABI: scalars in `r0..r5`, then stack. `r29 = sp`, `r31 = lr`.
Vector args would be in `v0..v15` paired — there are **none here**;
this kernel uses HMX `mxmem`-style loads from VTCM only.

### 1.1 Prologue arg-register reads

```
2ea740:  r28 = #0x7e0                              ; mask 0x7e0 (alignment check)
2ea744:  r7:6  = memd(r4+#0x8)                     ; r4[+0x08..0x0f]  — Rt-mask pair
2ea748:  r11:10 = memd(r4+#0x0)                    ; r4[+0x00..0x07]  — output Rt + check val
2ea74c:  p0 = bitsclr(r6, r28)                     ; check r4[+0x08] aligned
         p0 = bitsclr(r10, r28) (same packet)      ; check r4[+0x00] aligned
2ea754:  r9  = memw(r4+#0x18)                      ; r4[+0x18]        — alt Rt for last-K MAC
2ea758:  r15 = memw(r1+#0x8)                       ; r1[+0x08]        — act-list stride
2ea760:  if (!p0) jump hmx_convbbb1x1_stride1_unaligned
;  --- past the unaligned-fallback dispatch ---
2ea7a4:  r17 = memw(r0+#0x4); r16 = memw(r0+#0x8)  ; r0[+0x04], r0[+0x08]
2ea7b0:  r13 = memw(r0+#0x14)                      ; r0[+0x14]        — K-iters (in bytes)
2ea7c0:  r20 = memw(r0+#0xc)                       ; r0[+0x0c]        — N-tile shift count
2ea7d0:  r12 = memw(r0+#0x10)                      ; r0[+0x10]        — M-tile span
2ea7d4:  r0  = memw(r0+#0x0)                       ; r0[+0x00]        — out scatter-table base
2ea788:  r4  = memw(r1+#0x4); r1 = memw(r1+#0x0)   ; r1[+0x04] (count), r1[+0x00] (act-tbl)
```

So the function takes **5 register inputs** `(r0, r1, r2, r3, r4)`:

| Reg | Type | Meaning |
|---|---|---|
| `r0` | `out_desc_t *` | "output / geometry" descriptor (6 fields) |
| `r1` | `act_desc_t *` | "activation tile-pointer table" descriptor (3 fields) |
| `r2` | `void *` | weight tile base ptr (VTCM, 1 KB-aligned, declared `int8` Crouton) |
| `r3` | `void *` | bias base ptr (VTCM, fed to `bias = mxmem2(r3)`) |
| `r4` | `mask_desc_t *` | Rt-mask + alignment-check static descriptor (3 fields used) |

`r5` is **not** read (1×1 variant). The `1xN_stride2` sibling does read
`r5` (probably as a horizontal-stride — irrelevant for the 1×1 case).

Callee-saved preserved: `r17:16, r19:18, r21:20, r23:22, r25:24, r27:26`.
Frame: `r29 += -0x30`, no `allocframe` (just `memd ... = r17:16` on entry,
manual `r29 += 0x30` + `memd` reloads + `jumpr r31` on exit). Pure
leaf-style return — no `dealloc_return` even.

## 2. The three descriptor structs

### 2.1 `*r0` — out_desc_t (output + geometry, 6 × u32)

| Off | Access | Use pattern | Inferred meaning |
|---|---|---|---|
| `+0x00` | `r0 = memw(r0+0x0)` | Then `r14 = r0+4`, eventually `m0 = r17 << 2` post-inc walks `r0++m0` to fetch per-M-row output destinations | `int32_t * out_tile_ptr_table` — array of VTCM tile-output addresses, one per M-row |
| `+0x04` | `r17 = memw(r0+0x4)` | `m0 = r17 << 2`; sets the post-inc stride for `memw(r0++m0)` | `uint32_t out_table_stride_dwords` — **dwords** between adjacent M-row entries in the table (typically 1) |
| `+0x08` | `r16 = memw(r0+0x8)` | `r17:16 = vaslw(r17:16, #2)` then per-outer-iter `r18 += r16` | `uint32_t out_y_stride_words` (× 4) — Y-row advance for output destination, **shifted left by 2** internally |
| `+0x0c` | `r20 = memw(r0+0xc)` | `r20 = lsr(r20+r21-1, r18)`; loop1 trip count | `uint32_t n_tiles_pow2` — N-dim count (rounded up via `r21 = 1<<r18`, where `r18 = ct0(r28&r7) - 5`); inner-tile loop1 trip = `(n_tiles + (1<<r18) - 1) >> r18` |
| `+0x10` | `r12 = memw(r0+0x10)` | Each outer iter: `r17 = r12; r17 -= r22 (= 2<<r19)` until `r17 ≤ 0` | `int32_t m_total_minus_step` — M-iteration counter; loop1 ends when `r17 ≤ 0` |
| `+0x14` | `r13 = memw(r0+0x14)` | `r13 -= 1; per-outer-iter r13 -= 0x20; loop while r13 ≥ 0` | `uint32_t k_total_bytes` — K-dimension, in **bytes**; outer loop covers `ceil(K_bytes / 32)` macro-steps |

```c
typedef struct {
    int32_t *out_tile_ptr_table; /* +0x00 */
    uint32_t out_table_stride_dwords; /* +0x04 */
    uint32_t out_y_stride_words;      /* +0x08, internally <<= 2 */
    uint32_t n_tiles_pow2;            /* +0x0c */
    int32_t  m_total_minus_step;      /* +0x10 */
    uint32_t k_total_bytes;           /* +0x14 */
} hmx_conv_out_desc_t;
```

### 2.2 `*r1` — act_desc_t (activation, 3 × u32)

| Off | Access | Use pattern | Meaning |
|---|---|---|---|
| `+0x00` | `r1 = memw(r1+0x0)` | Inner loop: `r6 = memw(r1++#0x8); r23 = memw(r1+#0x4)` — pair-of-act-ptrs walker | `int32_t * act_ptr_pairs` — flat list of `{act_ptr_lo, act_ptr_hi}` pairs (8 B per pair), feeding the pair-MAC inner loop |
| `+0x04` | `r4 = memw(r1+0x4)` | `cmp.gt(r4,#1)` → main vs. tail dispatch; `r5 = r4>>1` (loop0 trip count); `tstbit(r4, #0)` (odd-tail flag) | `uint32_t n_act_pairs` — number of activation pairs to consume per K-iter (== N-tile-pair count) |
| `+0x08` | `r15 = memw(r1+0x8)` (×4) | Per-outer-iter `r19 += r15` | `uint32_t act_table_y_stride_words` (×4) — bytes to advance `act_ptr_pairs` head per K-step |

```c
typedef struct {
    int32_t *act_ptr_pairs;             /* +0x00 — list of {ptr_lo, ptr_hi} */
    uint32_t n_act_pairs;               /* +0x04 */
    uint32_t act_table_y_stride_words;  /* +0x08, internally <<= 2 */
} hmx_conv_act_desc_t;
```

### 2.3 `*r4` — mask_desc_t (Rt masks + alignment check, 4 × u32 used)

| Off | Access | Meaning |
|---|---|---|
| `+0x00` | `r10 = memw(r4+0x0)` | `int32_t out_check` — alignment-check value: `bitsclr(r10, 0x7e0)` must be true; **also** the value passed as `mxmem(r10, r11):after:cm:sat.ub` for the output mxmem mask. **Wait** — re-reading: `r10` from `memd(r4+0)` is overwritten later by `r10 = memw(r0++m0)` in the inner loop, so the initial `r10` here is **only** the alignment probe. The real per-tile output ptr is fetched from `out_tile_ptr_table`. |
| `+0x04` | `r11 = memw(r4+0x4)` | `uint32_t out_rt_mask` — the Rt mask used for `mxmem(r10, r11):after:cm:sat.ub = acc` output store. Survives the whole function. |
| `+0x08` | `r6 = memw(r4+0x8)` | `int32_t act_check` — alignment-probe value (`bitsclr(r6, 0x7e0)`). Same role as `out_check`. |
| `+0x0c` | `r7 = memw(r4+0xc)` | `uint32_t act_rt_base` — base activation Rt mask. Used as `r24 = or(r7, #0x1c)` (`\| 0x1c` is the `:cm` qualifier bits) and as `extractu(r7, #0xb, #0)` to derive `r18 = ct0(...) - 5` (the per-tile shift). |
| `+0x10` | not read by 1×1 | Used by sibling `1xN_stride2` (filter-x stride). |
| `+0x18` | `r9 = memw(r4+0x18)` | `uint32_t alt_rt` — alternate Rt mask used on the **last** MAC of each K-tile pair: `if (cmp.eq(r26, #2)) r25:24 = combine(r9, r7)`. For 1×1 this is the "last-K" mask (`0x3ff`-equivalent variant). Also used in odd-tail at `r8 += add(r9, #1)`. |

```c
typedef struct {
    int32_t  out_check;       /* +0x00 — must satisfy bitsclr(_, 0x7e0) */
    uint32_t out_rt_mask;     /* +0x04 — Rt for sat.ub store         */
    int32_t  act_check;       /* +0x08 — alignment probe              */
    uint32_t act_rt_base;     /* +0x0c — base Rt for activation MAC   */
    uint32_t filter_x_stride; /* +0x10 — unused by 1×1 (used by 1xN)  */
    uint32_t _pad14;          /* +0x14 — not observed                 */
    uint32_t alt_rt;          /* +0x18 — Rt for last-K MAC + odd-tail */
} hmx_conv_mask_desc_t;
```

The classic **`Rt_wt = 0x3FF` performance unlock** documented in
`qnn_hmx_pipelining.md` corresponds to setting either `out_rt_mask`
or `alt_rt` to `0x3FF`. Both come from the static `mask_desc_t`.

### 2.4 Why three descriptors?

QNN's graph-finalize lowers a single `q::ConvLayer_s1.opt` op into the
following per-instance pre-baked tables (one set per `mt_slice`):

- **`out_desc_t`** — geometry (M, N, K) plus the **scatter table** of
  output-tile pointers in VTCM. The scatter table is what lets QNN
  write directly into Crouton-tile-layout output without an extra
  permute pass — the same trick V8 finally adopted (see
  `Agent/qnn_vs_v8_root_cause_2026-04-24.md`).
- **`act_desc_t`** — gather table of activation tile pointer pairs.
  Pre-baked because in 1×1 conv with stride 1, the activation is
  consumed in fixed (flattened) order; the table also serves spatial
  shuffles for the larger-filter siblings.
- **`mask_desc_t`** — kernel-static knobs (Rt masks, alignment check
  values). Computed once at graph-finalize from the per-shape tile
  geometry; immutable across inferences.

The split lets QNN reuse the `mask_desc_t` across many ConvLayer
instances of the same dtype/shape class while parametrising
`out_desc_t` / `act_desc_t` per-tile-of-the-output.

## 3. Inner-loop core (already documented in `qnn_hmx_pipelining.md`)

Reproduced for completeness:

```
{ p0 = cmp.eq(r26, #2); r26 -= 2
  r6  = memw(r1++#8)              ; act_ptr_pair.lo
  r23 = memw(r1+#4) }             ; act_ptr_pair.hi
{ r8 += 0x400                     ; weight ptr += 1024 (1 wt-tile)
  if (p0) r25:24 = combine(r9, r7) ; last-K? swap to alt_rt + base_rt
  activation.ub = mxmem(r6, r24):cm
  weight.b      = mxmem(r8, r25) }
{ r8 += add(r25, #1)              ; weight ptr += r25+1 = 0x400
  activation.ub = mxmem(r23, r24):cm
  weight.b      = mxmem(r8, r25) }   :endloop0
;  --- after loop0 (one M-tile-row done) ---
{ r10 = memw(r0++m0)              ; fetch next out_tile dest
  mxmem(r10, r11):after:cm:sat.ub = acc }   :endloop1
```

So per (M-tile, N-tile) pair: 1 bias load (`mxmem2(r3)`), 2 K-pair
MACs per K-step, 1 sat.ub store at the end. Output goes to
`out_tile_ptr_table[i] + 0` — pure scatter, no extra permute.

## 4. HMX state assumptions — caller responsibilities

This kernel does NOT call `nn_os_vtcm_hmx_acquire`,
`nn_os_vtcm_hmx_cached_acquire`, or any other hnnx symbol. Verified:

- No `call` instructions in the function body.
- The single non-PC-relative jump is to `hmx_convbbb1x1_stride1_unaligned`
  (sibling fallback, same address space).
- No `memw(gp+...)` / GP-relative loads.
- The only non-arg state read is HMX accumulator state — managed via
  `mxclracc` at entry (line 0x2ea770) and `mxmem ... :after:...sat.ub
  = acc` at exit-of-loop1 (which drains and clears).

So the **caller** must own a live HMX context. In the QNN graph,
`q::ConvLayer_s1.opt` runs inside an HMX-acquired window; the kernel
inherits the context. For us calling from a custom op-pkg, that means
either (a) we already hold HMX (e.g. our op declares
`acquireHmx = true` and runs in the HMX execution slot), or (b) we
explicitly call `nn_os_vtcm_hmx_acquire` ourselves before invoking.

`mxclracc` is part of this kernel's own prologue — it does not assume
the accumulators come in clean. So **two consecutive calls back-to-back
work without external state cleanup**.

## 5. Output layout — confirmed Crouton tile-layout, contiguous 1 KB per tile

The output store is `mxmem(r10, r11):after:cm:sat.ub = acc`, with `r10`
sourced from `out_desc.out_tile_ptr_table[i]`. This is the **same**
write pattern V8 adopted in `project_v8_tile_layout_2026-04-24` —
1024 B per tile written contiguously to VTCM. The scatter table (one
ptr per M-row × N-tile combination, walked via `r0++m0` post-increment)
is what implements the "no permute" output: each entry points directly
to the Crouton-layout slot in the output tensor's VTCM staging buffer.

## 6. C function declaration

```c
/* libQnnHtpV75Skel.so @ 0x2ea740, GLOBAL.
 *
 * Kernel: u8·i8 → u8 1×1-conv / MatMul over Crouton-byte tiles, on
 * HTP v75 HMX. Body of QNN's q::ConvLayer_s1.opt for {bbb} dtype.
 *
 * Pre-conditions (CALLER must establish):
 *  - Caller holds an HMX context (nn_os_vtcm_hmx_acquire). Kernel
 *    issues mxclracc at entry but does not acquire/release the HMX.
 *  - Activation Crouton-byte tiles live in VTCM, addresses gathered in
 *    act_desc->act_ptr_pairs as packed {lo, hi} 8-B records.
 *  - Weight Crouton-int8 tiles live in VTCM at weight_base, contiguous
 *    1 KB per K-tile, advancing by 0x400 per K-step in the inner loop.
 *  - Bias buffer at bias_base in VTCM, in mxmem2-loadable layout
 *    (32-channel × 2 × halfword pair, 256 B per N-tile bias-pair).
 *  - Output VTCM destinations pre-allocated, addresses listed in
 *    out_desc->out_tile_ptr_table, one entry per (M-row × N-tile).
 *  - mask_desc->{out_rt_mask, act_rt_base, alt_rt} are populated with
 *    the v75-correct Rt values (0x3FF for the K-edge, etc. — see
 *    Agent/qnn_hmx_pipelining.md §2 on Rt_wt=0x3FF perf unlock).
 *  - All four addresses (r10, r6, r23, r8) must be 32-B-aligned;
 *    misalignment dispatches to hmx_convbbb1x1_stride1_unaligned at
 *    function entry (probe via bitsclr(_, 0x7e0)).
 *
 * Side effects: writes sat.ub-saturated u8 outputs into the VTCM
 * destinations addressed by out_desc->out_tile_ptr_table[i]. Returns
 * void (r0 is not used as a return). Clobbers HMX accumulators (clears
 * via mxclracc at entry; final sat.ub store also implicit-clears).
 *
 * Re-entrant w.r.t. other HMX kernels in the same thread (caller
 * serialises HMX use). NOT thread-safe vs. another thread doing HMX
 * — HMX is a per-HW-thread resource on v75.
 */
typedef struct {
    int32_t *out_tile_ptr_table;     /* +0x00 */
    uint32_t out_table_stride_dwords;/* +0x04 */
    uint32_t out_y_stride_words;     /* +0x08 */
    uint32_t n_tiles_pow2;           /* +0x0c */
    int32_t  m_total_minus_step;     /* +0x10 */
    uint32_t k_total_bytes;          /* +0x14 */
} hmx_conv_out_desc_t;

typedef struct {
    int32_t *act_ptr_pairs;             /* +0x00 */
    uint32_t n_act_pairs;               /* +0x04 */
    uint32_t act_table_y_stride_words;  /* +0x08 */
} hmx_conv_act_desc_t;

typedef struct {
    int32_t  out_check;       /* +0x00 */
    uint32_t out_rt_mask;     /* +0x04 */
    int32_t  act_check;       /* +0x08 */
    uint32_t act_rt_base;     /* +0x0c */
    uint32_t filter_x_stride; /* +0x10 (unused by 1x1) */
    uint32_t _pad14;          /* +0x14 */
    uint32_t alt_rt;          /* +0x18 */
} hmx_conv_mask_desc_t;

extern void hmx_convbbb1x1_stride1(
    const hmx_conv_out_desc_t  *out_desc,    /* r0 */
    const hmx_conv_act_desc_t  *act_desc,    /* r1 */
    const void                 *weight_base, /* r2 — VTCM, 1 KB-aligned */
    const void                 *bias_base,   /* r3 — VTCM */
    const hmx_conv_mask_desc_t *mask_desc    /* r4 — kernel-static */
    /* r5 unused for the 1x1 variant                                  */
);
```

## 7. Cross-`.so` callability — verdict: **YES, but with significant caveats**

### 7.1 No code-level barriers

Same checks as `convert_to_crouton_b`:

1. No GP-relative loads (no `memw(gp+...)`).
2. No PLT / dynamic-import trampolines.
3. No TLS access.
4. No call to any external symbol — pure leaf except for the
   `_unaligned` sibling tail-dispatch (which is in the same `.so`, so
   PC-relative resolves correctly when invoking by absolute address).
5. `mxclracc` at entry → kernel handles its own HMX accumulator reset.
6. No `dealloc_return` / no `allocframe` — pure manual frame
   management, no allocation-tracking interaction with hnnx runtime.

### 7.2 Where the danger lives — descriptor synthesis

The hard part is **building the three descriptor structs correctly**.
QNN does this in `q::ConvLayer_s1.opt::execute` (graph-finalize-time
side, plus per-inference scatter-table materialisation). Specifically:

- `out_desc.out_tile_ptr_table` — VTCM addresses, computed from the
  output-tensor's VTCM allocation. The allocation comes from
  `nn_os_vtcm_hmx_acquire`'s `VtcmReq` mechanism. **You can't fabricate
  this without knowing where the runtime put the output tensor** —
  in a custom op, you get pointers to your VTCM allocations, so this
  is doable but tedious.
- `act_desc.act_ptr_pairs` — same story for the activation side.
- `mask_desc` — pure-static derivation from (M_tile, N_tile, K_tile)
  geometry. The `Rt_wt=0x3FF` value is a known constant for v75 1×1
  conv. We have full RE of the masks (see `qnn_hmx_pipelining.md` and
  the table in §2.3 above).
- `bias_base` layout — `mxmem2` reads 256 B per call (32 chan × 2 ×
  fp16). The layout is the V8-decoded "two-orthogonal-channels" pair
  (see `project_v8_hmx_semantics_open_2026-04-24.md`); this is fully
  understood.

### 7.3 dlsym viability — **NO across `.so` boundary in skel**

Three issues stack up:

1. **No dynamic symbol resolution.** Hexagon `.so`s under FastRPC are
   loaded by `dspCV` / hexagon-runtime, but the loader does NOT merge
   them into a single dlopen-style namespace. `dlsym("hmx_convbbb1x1_stride1")`
   from our op-pkg `.so` will return NULL — the symbol exists in
   `libQnnHtpV75Skel.so` but is not exported to other `.so`s.
2. **No imports declared.** Our op-pkg's ELF has no `R_HEX_*` reloc
   pointing at this symbol; even if dlsym worked, we'd need to add the
   import at link time. The skel's `.dynsym` has it as GLOBAL, but
   that only matters if a loader follows DT_NEEDED back to the skel —
   our op-pkg doesn't link `libQnnHtpV75Skel.so`, and adding a
   `DT_NEEDED` to it would be unsupported and likely rejected by the
   QNN packager.
3. **Address-by-offset is fragile.** Even if we hardcoded
   `0x2ea740 + skel_load_base`, the skel is part of QNN SDK and the
   offset will change with any QNN release. (Confirmed only for
   QNN SDK in `tools/qnn-sdk/` at this commit.)

### 7.4 Practical paths to actually using this kernel

1. **Reimplement, don't reuse.** This is what V8 already does
   (`example/hmx_matmul_phase3/standard_flow/phaseB_v8/mmv8.c`). With
   the descriptor-layout RE in this doc plus the pipelining unlock in
   `qnn_hmx_pipelining.md`, the kernel is fully recoverable as plain
   inline asm of ~30 packets. V8 already matches V6 perf at 512³
   (368 K cycles) which is on par with QNN's HMX inner-loop budget
   for that shape.
2. **Self-load the skel.** Theoretically possible via
   `dlopen("libQnnHtpV75Skel.so", RTLD_NOW)` from our op-pkg if the
   FastRPC loader exposes it. Untested. Even if it works, ABI drift
   across SDK versions makes this an unmaintainable hack.
3. **Hijack via instruction-trace/binary patching.** Not viable in
   production.

### 7.5 What this RE *is* good for

It nails down the **descriptor layout** so V8 can model the same
3-descriptor-table approach that lets QNN do `mt=4` HVX slicing while
the HMX kernel stays shape-invariant. Specifically:

- The `out_tile_ptr_table` scatter pattern is the key to V8's
  tile-layout output match (already adopted).
- The `act_ptr_pairs` pre-baked indirection is what enables QNN to do
  the **HVX `pack_act` in parallel with HMX** while sharing one weight
  tile across N M-rows. V8 currently doesn't pre-bake the act-ptr
  list — it computes addresses inline in the inner loop. Pre-baking
  could shave a few packets per K-step.
- The `mask_desc` confirms the v75 Rt values we need (`Rt_wt = 0x3FF`,
  `Rt_act_base | 0x1c` for `:cm`).

## 8. Open questions / experiments that would resolve them

| Question | Experiment |
|---|---|
| Is the `_unaligned` sibling functionally identical (just slow path) or does it implement different semantics? | Disasm 0x2eb180–0x2eb358 (472 B); compare loop body. Expectation: same MAC structure, different `mxmemu`-style instructions for unaligned VTCM access. |
| Does the `out_check` field (r4+0x00) actually carry an **address**, or is it a **mask value** like 0x7e0 itself? | The `bitsclr(r10, 0x7e0)` test passes only if `r10 & 0x7e0 == 0`. If `r10` is an arbitrary check value rather than an address, the test is "this magic number must have low 11 bits zero." Probably it IS the alignment-pattern of the FIRST output-tile-table entry, sampled to fast-path-check that the whole table is aligned. Confirm by dumping the value at runtime — patch a print into a custom op-pkg that mirrors QNN's call site. |
| What's at `r4[+0x10]` and `r4[+0x14]`? | Already confirmed: `r4[+0x10]` is the filter-x stride (used by `1xN_stride2`); `r4[+0x14]` not observed in any sibling — likely padding or tile-Y stride. |
| Is the kernel safe to invoke from a non-HMX-acquired thread (does the FastRPC default-thread state include HMX)? | Probe: invoke from a known no-HMX-acquired context and check `tlbmiss_x` / `precise_exception` traps. We have the harness in `example/hmx_matmul_qnn/` to do this. |

## 9. Cross-reference

- `Agent/qnn_hmx_pipelining.md` — inner MAC-loop body, Rt_wt=0x3FF
  perf unlock, `:cm`/`:above`/`mxswapacc` semantics. Same kernel.
- `Agent/qnn_vs_v8_root_cause_2026-04-24.md` — disasm comparison V8 vs
  this kernel; output-layout match.
- `Agent/forceformat_crouton_re.md` §4.2 — Crouton input layout this
  kernel consumes (32-channel-byte interleaved, 1024 B per tile).
- `Agent/sig_convert_to_crouton_b_2026-04-25.md` — companion ABI RE.
- `Agent/qnn_matmul_as_composition_2026-04-25.md` — graph-level view
  of how QNN lowers `MatMul` into `q::ConvLayer_s1.opt` instances,
  each of which dispatches one call to this kernel per slice.
