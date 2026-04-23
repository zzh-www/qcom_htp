# ForceFormat_Crouton_f2c — HTP v75 HVX pack RE

Target binary: `tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so`
(SM8650, HTP v75). Goal: find the HVX hot loop QNN uses to pack flat
`[N,H,W,C]` tensors into HMX-ready Crouton blocks, for both byte and
halfword variants, and compare to our scalar `pack_weight_32x32`.

## 1. Symbol hunt — outcome

### 1.1 What I searched for

- `ForceFormat`, `Crouton`, `f2c`, `flat_to_crouton`, `ToCrouton` in
  strings, dynsym, readelf relocations, raw rodata.
- Relocation pointers into the rodata region holding the op-name
  strings (`.rodata:0x834e00..0x835800`).

### 1.2 What I found

The strings exist — 20+ variants of
`ForceFormat_Crouton_f2c@XY.XY`:

- `ForceFormat_Crouton_f2c@CB.FB`  (byte, flat-to-Crouton\_8)
- `ForceFormat_Crouton_f2c@CH.FH`  (halfword, flat-to-Crouton\_16)
- `ForceFormat_Crouton_f2c@Ce.Fe`, `@Cg.Fg`  (32-bit elt / f16 etc.)

They live at `0x8352de` onward in `.rodata`. **No** `R_HEX_RELATIVE`
relocation targets these specific strings directly, which rules out
a function-pointer-vs-name registration table (cf. the
`q::ForceFormat_Flat` *namespace* strings at `0x834ee1`, which *do*
have relocs at `.data.rel.ro:0xa03d40` — class-descriptor metadata,
not the per-instance op entry point).

So the op strings are generated/consumed elsewhere (likely built
programmatically at graph-build time by the hnnx op-registration
templates — `ForceFormat_Crouton_f2c` is a C++ `OpDef` template
instance, and the string is formatted lazily for diagnostics).

### 1.3 What I found instead — the real kernels

The byte/halfword pack work is done by GLOBAL symbols that DO exist
in dynsym:

| Symbol | Addr | Size | Notes |
|---|---|---|---|
| `convert_to_crouton_b` | `0x00237700` | 904 B | byte (CB.FB) |
| `convert_to_crouton_h` | `0x00237c80` | 744 B | halfword (CH.FH) generic |
| `convert_to_crouton_h_dEQ1` | `0x00237aa0` | 192 B | depth == 1 |
| `convert_to_crouton_h_dEQ4to16` | `0x00305d80` | 240 B | depth in [4..16] |
| `convert_to_crouton_h_dEQ64m` | `0x00305e80` | 304 B | depth >= 64 |
| `convert_to_crouton_h_wrapper` | `0x00237f80` | 32 B | dispatch |
| `extract_tile_vmemu_u32` | `0x00237200` | ≈1024 B | tile-grain copy helper |

`convert_to_crouton_b` also branches into three `extract_tile_vmemu_u32`
sub-entries (at `+0x40` / `+0x1e0` / `+0x300` / `+0x440`) for the
`depth==1`, `depth<=32`, `depth==64`, and generic tail paths — the
branch predicate is `cmp.gtu(r4,#0x20)` / `cmp.eq(r4,#0x40)` etc. at
`0x237740..0x237778`. So "pack" is depth-partitioned; the HVX
permutation engine is only engaged for `depth > 64`.

## 2. Hot-loop disassembly

### 2.1 `convert_to_crouton_b` — main HVX loop (`0x237874..0x2378cc`)

Four interleaved flat rows → one Crouton block column group:

```
; r7 = #-0x20   (vshuff control = stride-32, set at 0x237808)
; r21,r11,r27,r13 = 4 row pointers stepping by 128 bytes/iter
; r19 is the block-base, computed from a per-iter table memw(r26+#k)

237874:  loop0(0x237878, r19)            ; r19 = depth_tiles/128

237878:  v3 = vmemu(r21++#0x1)          ; flat row 0 (unaligned)
23787c:  v4 = vmemu(r11++#0x1)          ; flat row 1
237880:  v5 = vmemu(r27++#0x1)          ; flat row 2
237884:  v6 = vmemu(r13++#0x1)          ; flat row 3

237888:  v15:14 = vshuff(v4, v3, r7)    ; pass-1 lo pair,  stride 32
           r19 = memw(r26+#0x0)         ;                   |
          r19 += r25                    ; add block offset
237890:  v17:16 = vshuff(v6, v5, r7)    ; pass-1 hi pair

237898:  v19:18 = vshuff(v16, v14, r7)  ; pass-2 merges lo+hi
          vmem(r19+#0) = v18.new        ; store block slice 0
2378a0:  v21:20 = vshuff(v17, v15, r7)
          r19 = memw(r26+#0x4)
         r19 += r25
2378ac:  vmem(r19+#0) = v19             ; store slice 1

2378b0:  r19 = memw(r26+#0x8)
         r19 += r25
2378b8:  vmem(r19+#0) = v20             ; store slice 2

2378bc:  r26 = add(r26,#0x10)
         r19 = memw(r26+#0xc)
         r19 += r25
2378c8:  vmem(r19+#0) = v21             ; store slice 3
         :endloop0
```

Outer `loop1` starting at `0x237848` walks 8 row groups of 4.
`r26` indexes a table of per-channel-group destination offsets; the
four `memw(r26+#k)` loads = scatter addresses (Crouton blocks are
**not** contiguous in depth). `r25 = block-within-tile scale` pre-
computed at `0x237860 (r25 = asl(v_row_idx,#7))`.

### 2.2 `convert_to_crouton_h` — halfword loop (`0x237d88..0x237e00`)

```
; r3 = #-0x2   (vshuff at halfword granularity, set at 0x237d18)
; 4 row pointers r25/r26/r28/... stepping by r4 = 2*width

237da8:  loop0(0x237db0, r13)           ; r13 = depth_tiles/8 (halfword)

237db0:  r25 = r26
         r26 = add(r26, r4)             ; r4 = 2*W (halfword stride)
         r15 = memw(r24+#0x4)           ; scatter-dst offset (hi pair)
         r27 = memw(r24+#0x0)           ; scatter-dst offset (lo pair)
         r28 = add(r26, r4)
         r27 += r17                     ; add block base
         v6 = vmemu(r26+#0)             ; row 1
         r15 += r17
         r26 = add(r28, r4)
         v5 = vmemu(r25+#0)             ; row 0
         r26 = add(r26, r4)
         v8 = vmemu(r26+#0)             ; row 3
         v7 = vmemu(r28+#0)             ; row 2

         r24 = add(r24, r10)            ; r10 = 4*r5 = step table

237de8:  v3:2   = vshuff(v6, v5, r3)   ; pass-1 lo pair at halfword stride
237dec:  v31:30 = vshuff(v8, v7, r3)   ; pass-1 hi pair
         vmem(r27+#0) = v2
237df4:  vmem(r27+#1) = v30
237df8:  vmem(r15+#0) = v3
237e00:  vmem(r15+#1) = v31
         :endloop0
```

Notice the halfword version uses **only one level** of `vshuff`
(single pass at stride-2), unlike the byte version (two passes at
stride-32). That's because HMX halfword Crouton has depth-32
granularity but the HVX vector already holds 64 halfwords, so one
deinterleave of two halfword-streams is enough. For bytes, an HVX
vector holds 128 bytes, so two passes are needed to untangle the
4-way interleave across two vectors.

## 3. HVX instruction classification

Only three HVX "permute" primitives appear in the hot path:

| Intrinsic (C name) | Opcode mnemonic | Role in pack |
|---|---|---|
| `V6_vshuffvdd` | `Vdd.h = vshuff(Vu,Vv,Rt)` | 2-into-1 interleave; byte-granularity when `Rt=-32`, halfword when `Rt=-2`. **Core of the transform.** |
| `V6_vror` | `Vd = vror(Vu,Rt)` | Byte rotate — used only in `vmemcpy_2d_asm` for unaligned source |
| `V6_vdelta` | `Vd = vdelta(Vu,Vv)` | General permute via control vector — used only in `extract_tile_vmemu_u32` depth-1 variant |

**Notably absent**: `V6_vshuffb` (self-shuffle), `V6_vshufoeb`,
`V6_vdealb`, `V6_vlutvvb`, `V6_vlutvwhi`, `V6_vshuffh`. QNN's f2c
does NOT use a LUT permute; it relies on the built-in paired-deinterleave
semantics of `vshuffvdd`.

Loads are all `vmemu` (unaligned) — flat input has no guarantee of
128-byte alignment per row. Stores are aligned `vmem(Rt+#0)` because
the destination Crouton block is placed at a 2 KB boundary by the
TCM/VTCM allocator.

## 4. Inferred input→output byte permutation

### 4.1 HVX `Vdd.b = vshuff(Vu,Vv,Rt=-32)` semantics

For `Rt=-32`, the transform is a byte-stride-32 deinterleave at
double-vector width: given two 128-byte input vectors `Vu` and `Vv`,
the 256 output bytes are split into 8 chunks of 32:

```
in  = [u0..u31 | u32..u63 | u64..u95 | u96..u127 |
       v0..v31 | v32..v63 | v64..v95 | v96..v127]
Vd0 = [u0..u31 | v0..v31  | u64..u95 | v64..v95 ]
Vd1 = [u32..u63| v32..v63 | u96..u127| v96..v127]
```

### 4.2 The whole pack, end-to-end

The byte-variant pass packs 4 flat rows × 128 cols into 4 Crouton
"slices" of 128 bytes each. Call the input rows A B C D (each 128 B).
After one pair of `vshuff(v4,v3,-32)` and `vshuff(v6,v5,-32)`:

```
v14 = [A0..A31 | B0..B31 | A64..A95 | B64..B95]
v15 = [A32..A63| B32..B63| A96..A127| B96..B127]
v16 = [C0..C31 | D0..D31 | C64..C95 | D64..D95]
v17 = [C32..C63| D32..D63| C96..C127| D96..D127]
```

Then the second pair of `vshuff(v16,v14,-32)` / `vshuff(v17,v15,-32)`
interleaves columns-0..31 of A,B,C,D into v18, columns-32..63 into
v19, etc.:

```
v18 = [A0..A31 | B0..B31 | C0..C31 | D0..D31]    ; depth-slice 0
v19 = [A32..A63| B32..B63| C32..C63| D32..D63]   ; depth-slice 1
v20 = [A64..A95| B64..B95| C64..C95| D64..D95]   ; depth-slice 2
v21 = [A96..A127|B96..B127|C96..C127|D96..D127]  ; depth-slice 3
```

Each v18..v21 is one **Crouton row** of 128 bytes = the "32-deep × 4-row"
layout HMX mxmem expects: 4 consecutive spatial rows for one depth
quadrant, with depth as the fastest-varying inner index (depth-bytes
are still in memory order within each 32-B group).

ASCII diagram (byte path; one "4-spatial × 128-depth" slab):

```
FLAT input ([row 0..3] × [d0..d127]):
  row 0:  A[0] A[1] A[2] ... A[127]
  row 1:  B[0] B[1] B[2] ... B[127]
  row 2:  C[0] C[1] C[2] ... C[127]
  row 3:  D[0] D[1] D[2] ... D[127]

CROUTON output (4 blocks × 128 bytes, each block is one spatial-group
at a different depth-slice; blocks are scatter-stored to different
block addresses via memw(r26+#k) table):

  block[0]: A[ 0.. 31] | B[ 0.. 31] | C[ 0.. 31] | D[ 0.. 31]   <- v18
  block[1]: A[32.. 63] | B[32.. 63] | C[32.. 63] | D[32.. 63]   <- v19
  block[2]: A[64.. 95] | B[64.. 95] | C[64.. 95] | D[64.. 95]   <- v20
  block[3]: A[96..127] | B[96..127] | C[96..127] | D[96..127]   <- v21
```

This is exactly a **4-row × 128-col → 128 × 4 byte transpose,
tiled into four 32-B depth-lanes**. The four output blocks are
*separate* Crouton blocks in memory (not adjacent) — they represent
four different depth groups `d0..31`, `d32..63`, `d64..95`,
`d96..127` of the same spatial tile.

The outer `loop1` repeats for 8 spatial row-groups of 4, so a 32-row
tile is covered in 8 iterations × 4 vmemu loads + 4 vshuffs + 4
stores = **32 HVX instructions per 4-row band** (amortising the
scalar register bookkeeping).

### 4.3 Halfword variant (`CH.FH`)

Identical topology, but input rows are 128 bytes = 64 halfwords, and
one single `vshuff(v,v,-2)` pass suffices — the first half of the
64-halfword vector already carries the "column-pair 0..31" slice, so
no second pass is needed; each output Crouton block takes 2 vmems
(`vmem(r27+#0)=v2`, `vmem(r27+#1)=v30`) since a halfword Crouton
block is 4096 B and each HVX vector holds 2048 half-bytes... wait —
let me be precise. The halfword tile is 8×8×32 × 2 B = 4 KB. The
inner loop stores 4 HVX vectors = 512 B per block; the loop trip
count `r13 = depth_tiles/8` accumulates up to the block size.

## 5. Comparison to our scalar `pack_weight_32x32`

Scalar pack
(`example/hmx_matmul_qnn/kernel/hmx_int4_matmul.c:137-153`):

```c
static void pack_weight_32x32(int8_t *tile, const int8_t *w_32x32) {
    for (int kg = 0; kg < 8; kg++) {                       // 8 K-groups
        uint32_t *dst = (uint32_t *)(tile + 128 * kg);
        const uint8_t *r0..r3 = &w[(kg*4+{0..3}) * 32];
        for (int col = 0; col < 32; col++) {
            dst[col] = r0[col] | r1[col]<<8 | r2[col]<<16 | r3[col]<<24;
        }
    }
}
```

Output layout per K-group (128 B): 32 cells, each u32 =
`[r0,r1,r2,r3]` for one column → this is a **4-row × 32-col byte
transpose** writing stride-4 (the 4 rows become the 4 bytes of each
u32), contiguous within the K-group block.

QNN's `convert_to_crouton_b`:

- Input: 4-row × 128-col (wider — packs 4 HVX vectors of data at
  once).
- Output: 4 separate 128-byte Crouton blocks, each holding one
  depth-quadrant of those 4 rows, with depth-bytes interleaved at
  stride 32 within the vector.

**Are they the same transform?** Same structure, different tile
dimensions and different output memory layout:

| Property | Our scalar pack | QNN `convert_to_crouton_b` |
|---|---|---|
| Rows per iter | 4 | 4 |
| Cols per iter | 32 | 128 |
| Output per iter | 1 × 128-B block (K-group) | 4 × 128-B blocks (scatter) |
| Interleave in output | byte-stride-4 (u32 packs 4 rows) | byte-stride-32 (groups 4 rows × 32-d lane) |
| Depth alignment | 32 depth (matches HMX int4/int8 tile) | 32 depth per lane, 128 depth per iter |
| Destination addresses | contiguous | table-indexed (scatter) |

The **relationship**: both operations map 4 input rows into a layout
where 4 bytes across the 4 rows at the same column are bundled
together. But:

- Our pack: 4 bytes packed into one u32 lane (stride 4 bytes).
- QNN pack: 4 × 32-byte spans concatenated (stride 32 bytes) — this
  is the format HMX's mxmem loader expects because HMX processes
  depth-32 as its innermost dim for int8 operand.

So QNN's output is the layout HMX **actually consumes natively**.
Our scalar pack produces a layout that matches our **custom HMX
decomposition** (4-column Rt-weight pattern with 4 byte-lanes per
u32), which we then hand to mxmem via the ConvLayer 4-channel
broadcast path. They are NOT interchangeable.

If we wanted to reuse QNN's byte pack, we would need to target
HMX depth-32 consumption (i.e. the standard ConvLayer/MatMul
mxmem binding, not our custom bit-shuffle layout). The
`Agent/hvx_4way_byte_transpose_re.md` design — two
`Q6_Vb_vshuff_Vb` self-shuffles — is our own, narrower pack
(32-row × 32-col → 32×4-byte u32) and is **not** byte-equivalent to
QNN's output.

## 6. Key citations

Headers (all under `tools/qnn-sdk/include/QNN/HTP/core/`):

- `QnnHtpTensor.h` — `Qnn_HTP_Format_t` enum with values referenced
  by the op-name strings (not directly grep-visible; format suffixes
  `CB.FB` etc. map to the `TensorFormat_Crouton8` / `_Crouton16` /
  `_Flat8` / `_Flat16` Qnn enums).
- The `ForceFormat_*` ops are **internal** hnnx ops used by the
  graph optimizer, not part of the public Qnn_OpDef_t surface. No
  corresponding public header declares them.

HVX intrinsic reference (not shipped in QNN SDK):

- `tools/hexagon-sdk/incs/qhmath_hvx/qhmath_hvx_vector.h` (none
  reference Crouton).
- Hexagon V75 HVX PRM (80-N2040-54 AB), §4.2 (`vshuff`, `vdeal`),
  §8.2.20 (double-vector shuffle semantics for `Rt` stride encoding).

Disassembler commands used:

```bash
hexagon-llvm-objdump -d --mattr=+hmxv75,+hvxv75,+hvx-length128b \
  tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so \
  > /tmp/hex_disasm.txt
awk 'NR>=149275 && NR<=149503' /tmp/hex_disasm.txt   # convert_to_crouton_b
awk 'NR>=149624 && NR<=149812' /tmp/hex_disasm.txt   # convert_to_crouton_h
```

## 7. Takeaways for our kernels

1. QNN's `convert_to_crouton_b` is **4 rows × 128 cols at once**, using
   two passes of `V6_vshuffvdd(Vu,Vv,-32)`. Output goes to 4 scattered
   Crouton blocks via a pre-computed offset table.

2. The halfword path (`convert_to_crouton_h`) uses one pass of
   `V6_vshuffvdd(Vu,Vv,-2)`, 4 rows × 64 halfwords at once.

3. There is **no LUT/vdelta permute** — the transform is expressed
   entirely via the native interleave semantics of `vshuff`. This
   confirms our RE in `Agent/hvx_4way_byte_transpose_re.md` that
   `vshuff` is the right primitive for HMX-shaped packs.

4. QNN's pack output is the **native HMX depth-32 lane layout**
   (stride-32 byte groups, 4 spatial rows stacked). Our scalar
   `pack_weight_32x32` output is a **different** layout (stride-4
   byte groups, suited to our custom int4×int8 decomposition). They
   cannot be swapped without also changing the HMX binding.

5. If we want an HVX-accelerated version of **our** 32×32 pack, use
   the two-step self-shuffle design in `hvx_4way_byte_transpose_re.md`
   — it outputs the stride-4 format our kernel consumes, and is
   orthogonal to QNN's `convert_to_crouton_b`.
