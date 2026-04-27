# QNN HTP Skel symbol mapping for matmul primitives

**File**: `tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so` (10 765 564 B,
6103 dynsym entries)
**Date**: 2026-04-25
**Method**: `llvm-readelf --dyn-syms` + `llvm-cxxfilt` + `--string-dump=.rodata`.
No disassembly — symbol-name + rodata-string analysis only.

## Critical caveat — read first

Most HTP primitives invoked by the QNN graph (`q::ConvLayer.opt.weights_to_vtcm`,
`q::ConvLayer.opt.bias_to_vtcm`, `q::ConvLayer.opt.activations_to_vtcm`, `q::flat_to_vtcm`,
`q::*InputSlicePad`, `q::*OutputSlice`, `@Spill`, `@Fill`, `q::ForceFormat_Crouton`) are
**registered op-table strings** that live in `.rodata`. The C++ functions that implement
them are template instantiations of `hnnx::TypicalOp<...>`/`hnnx::SimpleOp<...>` and are
**not exported** in `.dynsym` — they're stripped to internal-linkage thunks. Only the
underlying *data-movement helper kernels* are exported by name.

Concretely:

| Op-table string in rodata | C++ class behind it | Exported helper kernels it calls |
|---|---|---|
| `@Spill` | `hnnx::SpillOp` (`N4hnnx7SpillOpE`) | `vmemcpy_2d_*_asm`, `runtime_graph2::dma_*memcpy*`, `hnnx::DMA_Manager::do_memcpy_slowpath` |
| `@Fill` | `hnnx::FillOp` (`N4hnnx6FillOpE`) | same family as @Spill |
| `@DmaCheckpointSet/Wait` | `hnnx::DmaCheckpoint{Set,Wait}` | `runtime_graph2::dma_wait_for`, `hnnx::DMA_Manager::wait_for/wait_desc/wait_all` |
| `q::ConvLayer.opt.weights_to_vtcm` / `bias_to_vtcm` / `activations_to_vtcm` / `q::flat_to_vtcm` | template `TypicalOp` thunks (internal) | `vmemcpy_2d_asm`, `vmemcpy_2d_general_asm`, `vmemcpy_2d_short_general`, `hnnx::vmemcpy_2d_gather_asm`, `hnnx::DMA_Manager::do_memcpy2d_slowpath` |
| `q::ForceFormat_Crouton` | template thunks | `convert_to_crouton_b/h/...`, `align_crouton_b/h` (already known) |
| `q::*InputSlicePad` / `q::*OutputSlice` | template thunks | `extract_tile_vmemu_u8/u16/u32`, `scatter_tile_u8/u16/u32`, `vmemcpy_2d_*` |

Bottom line: **no single `weights_to_vtcm` ELF symbol exists** the way
`hmx_convbbb1x1_stride1` exists. The op is a thin C++ wrapper that picks a
`vmemcpy_2d_*_asm` helper based on tensor layout (the `@F5B.f5B.` etc suffix in the
op-table string encodes the per-rank src/dst type tag).

## Per-target findings

### 1. weights_to_vtcm (q::ConvLayer.opt.weights_to_vtcm)

**Op-table string**: `q::ConvLayer.opt.weights_to_vtcm` at rodata offset `0x72c1b`.
Variants registered: `@F5B.f5B`, `@F5H.f5H`, `@F5e.f5e`, `@F5g.f5g`, `@Xwb.xwb`, `@Xpb.xpb`
(at `0x72fd7`+).

**Best-guess implementation kernel(s)** — confidence MEDIUM:
- `vmemcpy_2d_asm` @ `0x3066c0` size 580 — generic 2D vmem→vmem copy, aligned
- `vmemcpy_2d_general_asm` @ `0x306910` size 180 — unaligned variant
- `vmemcpy_2d_short_general` @ `0x306a00` size 500
- `vmemcpy_3d_short_general` @ `0x306c00` (size 0 — likely tail-call alias)
- `hnnx::vmemcpy_2d_gather_asm` @ `0x289640` size 956 (`_ZN4hnnx21vmemcpy_2d_gather_asmEllPvPKvj`) — gather variant, used when crouton tile-stride mismatches src
- `hnnx::DMA_Manager::do_memcpy2d_slowpath` @ `0x2ac300` size 1024 — slowpath when alignment/strides don't fit fastpath

The op selects one based on the layout-suffix variant in the op-table. **No symbol
literally named `weights_to_vtcm`** is exported.

Adjacent symbols (also called from this path):
- `Graph::dlbc_weight_setup()` @ `0x271e20` size 444 — graph-construction-time setup
  (decoder-side weight-prep registration), **not** the per-tile DMA op
- `weights_signed_trivial_and_shuffled_transpose_128x32b_flat` @ `0x306300` size 752 —
  weight-permutation kernel that runs **once at graphPrepare** to lay weights out for
  later ConvLayer consumption (this is `q::ConvLayer.opt.convert_weights_to_signed_trivially.shuffled.transpose`
  per the rodata string at `0x606e5`/`0x6072d`)
- `weights_signed_trivial_and_shuffled_transpose_8x4x32_crt` @ `0x306600` size 168

### 2. bias_to_vtcm (q::ConvLayer.opt.bias_to_vtcm)

**Op-table string**: `q::ConvLayer.opt.bias_to_vtcm` at rodata `0x72cbc`.

**Best-guess implementation** — confidence MEDIUM:
Same `vmemcpy_2d_*_asm` family as weights_to_vtcm. Bias is small enough that a single
2D copy variant suffices.

**Honest gap**: zero symbols containing the substring `"bias"` are exported (verified
across both raw and demangled dynsym). Bias DMA goes through the same generic helpers
as everything else. The bias-specific *math* (`q::ConvLayer.opt.convert_bias` /
`adjust_bias` / `bias_scale_shuff`) shows up as `.rodata` op-table entries but no
exported `_bias_*` C symbols — those are also internal template thunks.

### 3. @Spill and @Fill

**Op-table strings**: `@Spill` at `0x33147`, `@Fill` at `0x3314e`.
**C++ classes**: `hnnx::SpillOp` (`N4hnnx7SpillOpE` mangled type-info string at `0x33253`)
and `hnnx::FillOp` (`N4hnnx6FillOpE` at `0x33298`). Located in
`super_groups_opts.cc` (path leak in adjacent rodata at `0x332a7`).

**No exported symbols for SpillOp/FillOp constructors, vtables, or execute methods**
— class typeinfo names are present but the implementations are local-linkage. Confidence
HIGH that they exist as compiled code; LOW on which exact ELF offset.

**Implementation kernels they call** — confidence HIGH:
- `runtime_graph2::dma_far_memcpy(gHN::gH, u64, void const*, u32, u32)` @ `0x280940` size 256
- `runtime_graph2::dma_far_memcpy(gHN::gH, u64, u64, u32, u32)` @ `0x280840` size 256 (DDR↔DDR overload)
- `runtime_graph2::dma_memcpy2d(gHN::gH, void*, void const*, u32, u32, u32, u32, u32)` @ `0x280b40` size 32 (thunk)
- `runtime_graph2::dma_memcpy2d_overlapped(...)` @ `0x280c40` size 168
- `runtime_graph2::dma_memcpy(...)` @ `0x280740` size 252
- `hnnx::DMA_Manager::do_memcpy_slowpath` @ `0x2ac140` size 400
- `hnnx::DMA_Manager::do_memcpy2d_slowpath` @ `0x2ac300` size 1024

Async wait/checkpoint pairs:
- `hnnx::DMA_Manager::wait_for(dma_id_t)` @ `0x2abfa0` size 200
- `hnnx::DMA_Manager::wait_desc(PortableDMA*)` @ `0x2ac080` size 168
- `hnnx::DMA_Manager::wait_all()` @ `0x2ac2e0` size 28
- `hnnx::OverlappedDMABase::wait_inner(PortableDMA2D*, u32, PortableDMA2D*)` @ `0x2ac700` size 168
- `runtime_graph2::dma_wait_for(gHN::gH, dma_id_t)` @ `0x280d20` size 8 (thunk)

Setup-time symbols (graph-construction, **not** per-tile execution):
- `Graph::dlbc_spill_fill_setup()` @ `0x271d00` size 288 — confirms spillfill is a
  graph-level resource registered by `setup_vtcm`, then reused by the per-tile
  Spill/Fill ops. (DLBC = Dynamic Load Balance / DDR Caching, the generic VTCM-DDR
  bouncing scheme.)
- `Graph::set_shared_spillfill(...)` @ `0x2726e0` size 24
- `fa::RuntimeAllocator::set_shared_spillfill(...)` @ `0x2b2000` size 292
- `fa::RuntimeAllocator::is_shared_spillfill(u32, u32) const` @ `0x2b0200` size 44
- `hexagon_nn_set_shared_spillfill` @ `0x23fc80` size 52 (host-side handle setter)

**Honest gap**: I cannot point at the per-tile execute-function for SpillOp/FillOp.
It is *inside* this binary but stripped to local linkage. If you need its address
you'd have to scan vtables (e.g. relocations referencing the `N4hnnx7SpillOpE` typeinfo
string at `0x33253`).

### 4. q::*InputSlicePad

**Op-table string**: `q::*InputSlicePad` at rodata `0x9d78b` (sibling: `q::*InputSlice`
at `0x9d77c`). Layout-tag variants: `@FB.s4*3.t.s4*3.` for u8 with pad, `@Ff.s4*3.fi.s4*3.`
for fp32, etc.

**Best-guess implementation kernel(s)** — confidence MEDIUM:
- `extract_tile_vmemu_u8` @ `0x236cc0` size 688
- `extract_tile_vmemu_u16` @ `0x236f80` size 620
- `extract_tile_vmemu_u32` @ `0x237200` size 56
- `vmemset_32_2d_asm` / `vmemset_32_2d_general_asm` / `vmemset_2d_h` / `vmemset_h` —
  for the *Pad* portion (zero-fill out-of-bounds region)
- `qhpi_op_slice` @ `0x2ae880` size 16 (thunk into the QHPI slice op)
- `slice4to3` @ `0x289ac0` size 268 (rank-reduction helper)

The "Pad" form augments the slice with vmemset on the padded region. No exported symbol
combines both — they're sequenced inside the template thunk.

**ERROR string** `"InputSlice with requested padding."` at `0x9d741` — the op rejects
some pad combinations at validate-time.

### 5. q::*OutputSlice

**Op-table string**: `q::*OutputSlice` at rodata `0xce2aa`. Variants like
`@fB.s4*3.`, `@FB.s4*3.`, `@F5B.s5*2.s4.`, etc.

**Best-guess implementation kernel(s)** — confidence MEDIUM:
- `scatter_tile_u8` @ `0x303d40` size 600
- `scatter_tile_u16` @ `0x303960` size 664
- `scatter_tile_u32` @ `0x303c00` size 296
- `vmemcpy_2d_asm` family (when no scatter pattern needed, i.e. tile-aligned)

Mirror image of InputSlicePad — read-side scatter instead of gather.

### Bonus: q::ForceFormat_Crouton

Already known, confirming for completeness — confidence HIGH:
- Op-table string: `q::ForceFormat_Crouton` at `0xb3075`
- Implementations exported as plain C: `convert_to_crouton_b`, `convert_to_crouton_h`,
  `convert_to_crouton_h_dEQ1`, `convert_to_crouton_h_dEQ4to16`, `convert_to_crouton_h_dEQ64m`,
  `convert_to_crouton_h_wrapper`, `convert_from_crouton_b/h/w`, `convert_from_crouton_b_align`,
  `convert_from_crouton_b_wrapper`, `align_crouton_b`, `align_crouton_h`,
  `convert_cast_from_crouton_h_17to32`.

## Confidence summary

| Target | Op-table string found | Exported impl symbol | Confidence |
|---|---|---|---|
| weights_to_vtcm | YES (`q::ConvLayer.opt.weights_to_vtcm`) | NO — uses generic `vmemcpy_2d_*` | MEDIUM |
| bias_to_vtcm | YES (`q::ConvLayer.opt.bias_to_vtcm`) | NO — uses generic `vmemcpy_2d_*` | MEDIUM |
| activations_to_vtcm (bonus) | YES (`q::ConvLayer.opt.activations_to_vtcm`) | NO — uses generic `vmemcpy_2d_*` | MEDIUM |
| @Spill | YES (`@Spill` + `N4hnnx7SpillOpE` typeinfo) | NO direct — uses `runtime_graph2::dma_*memcpy*` + `DMA_Manager` | HIGH (existence) / LOW (offset) |
| @Fill | YES (`@Fill` + `N4hnnx6FillOpE` typeinfo) | NO direct — same DMA family as @Spill | HIGH (existence) / LOW (offset) |
| q::\*InputSlicePad | YES (`q::*InputSlicePad`) | NO direct — uses `extract_tile_vmemu_*` + `vmemset_*` | MEDIUM |
| q::\*OutputSlice | YES (`q::*OutputSlice`) | NO direct — uses `scatter_tile_u*` | MEDIUM |
| ForceFormat_Crouton (known) | YES | YES (`convert_to_crouton_b/h/...`) | HIGH |
| ConvLayer_s1.opt (known) | YES | YES (`hmx_convbbb1x1_stride1`) | HIGH |
| Concat (known) | YES | YES (`qhpi_op_concat` + `concat_*_crouton_*`) | HIGH |

## Implications for V8

1. **You will not find a magic "weights_to_vtcm" kernel to link against.** The QNN
   weights_to_vtcm op is a 2D memcpy wrapped in a TypicalOp template that picks the
   right `vmemcpy_2d_*_asm` variant by layout tag. V8's `pack_wt` already does the
   equivalent in HVX — there's no asm primitive in the skel that's faster than what
   you have.

2. **Spill/Fill is just async DMA.** It's `runtime_graph2::dma_far_memcpy` (descriptor
   enqueue → DMA engine) + `DmaCheckpointWait`. If V8's tile-out path is bounded by
   memcpy throughput, the same DMA path is available — `vmemcpy_2d_asm` (0x3066c0,
   580 B) is the aligned fastpath. Note V8 already uses an explicit HVX vmem loop and
   measured 12K cyc for 256 KB (vs libc memcpy at 1.6M cyc), so the DMA fastpath is
   probably comparable, not faster.

3. **InputSlicePad uses `extract_tile_vmemu_u8`** (0x236cc0, 688 B). This is the slice
   primitive that takes a logical `[N,H,W,C]` flat tensor and produces a tile-shaped
   buffer with optional zero-pad. If you need to replicate this in a custom op, that's
   the kernel to model your implementation after (or to call into via skel-internal,
   not exported but the symbol *is* in dynsym so you could `dlsym` it from inside the
   skel — risky but possible).

4. **OutputSlice uses `scatter_tile_u8`** (0x303d40, 600 B). Mirror of point 3.

5. **No `bias`-named exported symbol exists.** The fact that V8 folds bias into the
   matmul kernel signature is consistent with QNN doing the same — bias is just
   `vmemcpy_2d` of a small tensor at op start, not a dedicated kernel.
