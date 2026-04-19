# HMX u8×i8 native integer matmul — decoding notes

Running log of reverse-engineering the HMX integer path until the int16
matmul works end-to-end, bit-exact with a plain-C reference.

## Goal

32×32×32 per-tensor symmetric quantized int16 matmul on HMX.

## Facts established so far

1. `activation.ub = mxmem(P, 2047)` + `weight.b = mxmem(Q, 2047)` in one
   VLIW packet issues a u8×i8 MAC into the integer accumulator.  `.ub`
   is uint8 activation, `.b` is int8 weight.
2. `mxclracc` clears the integer accumulator.  (`mxclracc.hf` is the f16
   version, different accumulator.)
3. `bias = mxmem(P)` is **required** before convert-out; without it the
   output is all zeros.  P points to 128 uint16.
4. Filling those 128 uint16s with `0x4000` (= f16 2.0) gives the
   "identity" convert scale.  Smaller f16 values scale down linearly;
   there is a fixed internal /2.
5. The "identity" scale is actually **257/256** — i.e. the output is
   `acc + (acc >> 8)`.  Confirmed exactly by:
   - A=1, W=1, K=32 → 0x0020 (=32)
   - A=128, W=1 → 0x1010 (=4112 = 4096 + 4096/256)
   - A=255, W=127 → wraps 1_036_320 × 257/256 mod 2^16 = 0xDFF0
6. Output convert `mxmem(Rs,Rt):after.uh=acc:2x1` writes **1024 uint16**
   to VTCM, NOT 2048.  First 1024 uint16s hold all output; bytes beyond
   are untouched (sentinel survives).
7. The u8 activation tile is **2048 bytes**, not 1024.  Filling only the
   first 1024 leaves stale data in the upper half that HMX reads.
8. The i8 weight tile is only **1024 bytes**.

## The tile-shape puzzle

Naive assumption was 32-row × 32-col output.  Probes show otherwise:

- **Probe B**: A[i][k] = i (varies by row), W = 1.
  Expected 32 distinct row-sums (32·i for i in 0..31).
  Got **8 distinct values**, each appearing 128 times.
  Values match `out_b = 8 × (A[4b] + A[4b+1] + A[4b+2] + A[4b+3])` scaled
  by 257/256.  So 4 "input rows" collapse into 1 "output row", and
  only 8 output rows are produced.
- **Probe C**: A = 1, W[k][j] = j.  Same 8-value pattern on the col
  dimension: 4 cols pack into pairs in the output.

**Conclusion**: HMX u8×i8 matmul tile is NOT 32×32 output.  It's
something like 8 output rows × 32 cols with a K dimension that spans
multiple "logical 32-byte rows" of my naively-filled activation.
A single HMX call produces ≤ 256 logical outputs, not 1024.  To build
a 32×32 logical matmul we need multiple calls with tile-slicing.

## Complete layout (decoded via single-hot-byte probes)

**Activation tile** (2048 bytes, 2 KiB aligned):

```
A_byte(phys_row, K, stream) = A_tile[ 128 * phys_row + 4 * K + byte_offset ]
    phys_row   ∈ 0..15
    K          ∈ 0..31
    stream     ∈ {0, 1}        # 0 = even-phys-col, 1 = odd-phys-col
    byte_offset = 1 if stream == 0 else 3
```

Bytes at positions `4*K + 0` and `4*K + 2` are IGNORED.  This matches
HMX's underlying 16-bit slot layout where `.ub` u8 only reads the
"upper byte" of each 16-bit field.

**Weight tile** (1024 bytes):

```
W_byte(K, col) = W_tile[ 128 * (K >> 2) + 4 * col + (K & 3) ]
    K        ∈ 0..31
    col      ∈ 0..31
```

Organized as `8 K-groups × 32 cols × 4 K-in-group`.  Total K span = 32.

**Output tile** (1024 uint16, 2048 bytes on `:after.uh=acc:2x1`):

```
Output at  phys_row * 64 + 2 * col + stream
```

**Logical 32×32×32 matmul mapping**: a single HMX u8×i8 call
computes a **32-row × 32-col × K=32 matmul** where the 32 logical
rows are split across the 2 streams:

```
logical row ir  →  phys_row = ir % 16,  stream = ir / 16
logical col jc  →  W col    = jc,       phys output col = 2*jc + stream
```

One HMX call = **full 32×32 logical tile** if we pack both streams.

Verified with targeted hot-byte tests:
- `A[1]×W[0]`  → (r0, c0) = 1   ✓ row 0, K=0 ev × K=0 col=0 → (ir=0,jc=0)
- `A[3]×W[0]`  → (r0, c1) = 1   ✓ row 0, K=0 odd × K=0 col=0 → (ir=16,jc=0)
- `A[5]×W[1]`  → (r0, c0) = 1   ✓ row 0, K=1 ev × K=1 col=0 → (ir=0,jc=0)
- `A[1]×W[4]`  → (r0, c2) = 1   ✓ row 0, K=0 ev × K=0 col=1 → (ir=0,jc=1)

## Gotchas

- "Identity" bias (128 × f16 2.0) has a baked-in **×257/256** scale
  (`raw = acc + (acc >> 8)`).  Model this in the reference.
- Output saturates wrap-around as int16 within each uint16 slot.  For
  K=32 u8(0..255)×i8(-128..127), acc range is ~[-1.04M, +1.04M], which
  DOES wrap — need to be careful about inputs.

## Plan to finish

1. ✅ Write `pack_act()` and `pack_wt()` using the decoded layout.
2. ✅ Write `unpack_out()` (32×32 logical output).
3. ✅ Updated `int16_matmul_hmx.c`.
4. ✅ **Bit-exact match with top-byte-only reference (all scenarios).**
5. Extend to full 4-matmul int16 decomposition — in progress.

## Correction: no 257/256 scale when packing is correct

Earlier we thought the "identity" bias applied a ×257/256 scale.  That
observation came from polluted activation tiles (ignored byte slots held
stale data that was being folded into the MAC).  With proper packing —
only touching `4K+1` and `4K+3` bytes, zeroing the ignored slots — the
output is exactly `acc mod 2^16`, no extra scale.

So the correct reference is:

```c
wrapped = (int16_t)(uint16_t)(base & 0xFFFF);
```

## 4-matmul int16 decomposition

```
a_q    = a_u - 32768,        a_u = 256*A_h + A_l    (A_h, A_l uint8)
w_q    = 256*W_h + W_l,                             (W_h int8, W_l uint8)
a_q·w_q  = 65536·A_h·W_h + 256·(A_h·W_l + A_l·W_h) + A_l·W_l - 32768·w_q
```

### The output-wrap problem

HMX u8×i8 with K=32 has max |acc| ≈ 2^20, but the `:after.uh=acc:2x1`
convert stores only **acc mod 2^16** as int16.  The 2^16-wrap aligns
perfectly with the 65536 coefficient on M1 (because `2^16 × 2^16 ≡ 0
mod 2^32`), so M1 via HMX is safe under mod-2^32 combination.

For M2, M3, M4 (coefficients 256, 256, 1) the mod-2^16 wrap leaks error
of up to several thousand LSB into the final sum.  Verified on S2:
HMX-M1 + CPU-M2/M3/M4 gave `−575` instead of ref `−191`, diff `−384`,
which matches `3 × 2^32 / 2^25` (3 = M1 wrap count, 25 = rq.shift).

### ✅ Shipped: K-sliced HMX bit-exact int16 matmul

The K=32 reduction is broken into **32 single-K HMX MACs per partial
product**.  Each K=1 partial has max |u8·i8| = 32640 which fits int16
losslessly, so HMX's uint16-saturating output preserves the full value.
Summed across K in int32, each partial product is exact.

Per 32×32×32 tile: **128 HMX u8×i8 calls + bit-correction**:
- M1 = Σ_k A_h[k]·W_h[k]      32 HMX calls (u8 × i8 native)
- M3 = Σ_k A_l[k]·W_h[k]      32 HMX calls (u8 × i8 native)
- M2 = Σ_k A_h[k]·W_l[k]      32 HMX calls (u8 × (i8 reinterp of u8))
- M4 = Σ_k A_l[k]·W_l[k]      32 HMX calls (u8 × (i8 reinterp of u8))
- M2, M4 corrections:          scalar `256 · Σ A · topbit(W_l)`
- Combine:                     `65536·M1 + 256·(M2+M3) + M4 − 32768·ColSum_W`
- Requant:                     int64 × `rq.mul`, round-shift by `rq.shift`

**Results** — all scenarios bit-exact with int64 oracle:

| S  | inputs                     | rq (mul, shift) | status      |
|----|----------------------------|-----------------|-------------|
| S1 | A=256,     W=256           | (1, 16)         | PASS 1024/1024 |
| S2 | A=−10000,  W=20000         | (1, 25)         | PASS 1024/1024 |
| S3 | A,W random int16 full-range| (1, 20)         | PASS 1024/1024 |

### Future optimizations

The K-slicing approach used 128 HMX calls per tile.  Superseded by the
dual-scale readback kernel below — see "Final kernel (2026-04)" section.

Historical notes:

1. **Dual-scale readback** — implemented.  Uses `:after:retain.uh` to keep
   acc across two converts at different biases.  Winning scale pair:
   0x4000 (f16 2.0 → scale 1.0) for low byte + 0x2000 (f16 2^-7 → scale
   2^-8) for high bytes.  Drops tile cost to **12 HMX packets** (4 MAC
   + 8 convert).
2. **Bias slots 1–3** — confirmed **ignored** by probe.  The bias buffer
   is 128 u16, one per output column, not 4 per column as initially
   hypothesized.  (Probe: splat-all-512 vs set-slot0-only produced
   identical convert output.)
3. **`:2x2` convert** — not probed once dual-scale won; early run at
   this variant saw duplication of the `:2x1` tile with no extra
   precision.

## Final kernel (2026-04): dual-scale readback, 12 HMX packets per tile

The 2^16-wrap problem on M2/M3/M4 is solved by reading the full int32
accumulator via two `.uh` converts on the *same* accumulator, enabled
by the `:after:retain.uh=acc:2x1` intrinsic (SDK header
`hmx_hexagon_protos.h`, `Q6_mxmem_AR_after_retain_uh_2x1`).

### Convert-scale calibration (from `probe_dual_scale.c`)

With a single full-K u8×i8 MAC and a clean (ignored-byte-zeroed)
activation tile, the output convert computes approximately:

```
out_u16(j) = floor(acc[j] * (bias_f16[j] / 2)) mod 2^16
```

for **normal** f16 bias values.  Measured behavior:

| bias u16 | f16 value | effective scale | output at acc=160000 | status  |
|----------|-----------|-----------------|----------------------|---------|
| 0x4000   | 2.0       | 1.0             | 0x7100  (= acc mod 2^16) | clean |
| 0x3C00   | 1.0       | 0.5             | 0x3880  (= 80000 mod 2^16) | clean |
| 0x2000   | 2^-7      | 2^-8            | 0x0271  (= 160000>>8)  | clean |
| 0x0800   | 2^-13     | 2^-14           | 0x0009  (= 160000>>14) | clean |
| 0x0400   | 2^-14     | 2^-15           | 0x0004  (= 160000>>15) | clean |
| 0x0200   | 2^-15 (denormal) | 2^-16  | 0x0003  (= ceil(2.44)) | 1.5× artifact |

**Denormal f16 biases have a ~1.5× scale multiplier** (observed
effective scale is `3 · 2^-17` instead of the nominal `2^-16`).
Normal-range f16 biases (mantissa/exp in the normal range) behave
cleanly per the `scale/2` rule.

### Chosen scale pair and reconstruction

For K=32 u8·i8 partials, acc magnitude is ≤ 1,044,480 ≈ 2^20.  To read
this with normal-f16 scales alone:

- **Convert 1** with bias 0x4000 (scale 1.0), stored via
  `:after:retain.uh=acc:2x1` → `OUT_LO[j] = acc[j] mod 2^16`.
- **Convert 2** with bias 0x2000 (scale 2^-8), stored via
  `:after.uh=acc:2x1` → `OUT_HI[j] = floor(acc[j] / 256) mod 2^16`,
  interpreted as **int16** gives bits `[8..23]` of acc with correct
  sign extension.

Reconstruction (C):

```c
int32_t acc_j = ((int32_t)(int16_t)OUT_HI[j] << 8)
              | ((int32_t)OUT_LO[j] & 0xFF);
```

Verified bit-exact on the full probe set (positive small/large, negative
small/large, max-magnitude corner `A=255, W=-128`).

### Packet sequence per partial product

```
pack_activation_full(act_tile, A_partial_32x32)   ; CPU-side pack, full K=32
pack_weight_full    (wt_tile,  W_partial_32x32)

bias = mxmem(BIAS_LO)                              ; scale 1.0
mxclracc
{ activation.ub = mxmem(act_tile,2047)
  weight.b      = mxmem(wt_tile,2047) }            ; full-K MAC

mxmem(OUT_LO,0):after:retain.uh = acc:2x1          ; low-byte read, acc retained

bias = mxmem(BIAS_HI)                              ; scale 2^-8
mxmem(OUT_HI,0):after.uh        = acc:2x1          ; high-bytes read (signed)
```

**3 HMX packets per partial** (MAC + 2 converts; bias-load packets are
ordinary vector mem ops and usually issue in the same VLIW packet as
neighbors).  × 4 partials = **12 HMX packets per 32×32×32 tile.**

The four partials (M1 = Σ A_h·W_h, M2 = Σ A_h·W_l, M3 = Σ A_l·W_h,
M4 = Σ A_l·W_l) and the combine step `65536·M1 + 256·(M2+M3) + M4 −
32768·ColSum_W` are unchanged from the K-sliced kernel.  M2/M4
u8·u8 handled via int8-reinterp + `+256·Σ A·topbit(W_l)` correction.

Result: S1, S2, S3 all bit-exact 1024/1024 vs the int64 oracle.
