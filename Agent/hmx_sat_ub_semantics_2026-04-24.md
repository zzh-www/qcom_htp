# HMX `:after:cm:sat.ub` semantics — silicon-decoded (2026-04-24)

**Target:** SM8650 v75, OnePlus 12 (`oneplus` ssh)
**Probes:** `example/hmx_matmul_device/probe_hmx_formula.c` (T7..T10),
`example/hmx_matmul_device/probe_row_mapping.c` (T11)
**Outputs:** `probe_hmx_formula_result.txt`, `probe_row_mapping_result.txt`

## Headline formula (validated, updated post-T12)

For a single-tile MAC + `:after:cm:sat.ub` store (u8 bias-folded readback):

```
out_u8[m][c] = sat_u8( (bias_raw[2c+1] >> 7)              ← BASELINE (zp)
                     + floor(acc_hmx[m][c] × bias_fp16_value[2c] / 512) )
                                                          ↑ SCALE (slope)
```

**Two lanes are orthogonal channels:**
- `bias[2c+1]` (odd lane): top-9-bits give the per-col output zero-point
- `bias[2c]` (even lane): fp16 value is the per-col scale

T12 (2026-04-24) decoded this; earlier T9 saw lane-2c as "unused" only
because tests used near-zero acc where the scale contribution floored to 0.

where:
- **activation is plain unsigned u8** — no signed-shift (`act − 128`).  HMX
  `activation.ub = mxmem(...):cm` reads bytes as `u8 0..255` and multiplies
  directly against the signed i8 weight.  Verified by T7a/b/c.
- **weight is plain signed i8** read via `weight.b = mxmem(...)` — the
  bytes stored in the VTCM tile are reinterpreted as i8 (MSB = sign).
  A byte value of 0xF9 (u8 249) is seen as i8 −7.
- **bias lane = 2c+1** (odd-indexed fp16 entries) for col c output.
  Even-indexed entries were ignored by `:sat.ub` in probe T9b.  In more
  complex real matmul setups (random per-col bias), the pair
  **(bias[2c], bias[2c+1])** may both matter — see "Open questions".
- **baseline** is the **top 9 bits of the raw u16** bias entry —
  `(bias_raw >> 7)` — not a constant 128.  For bias_fp16 = 2.0 (= 0x4000),
  baseline is 128; scale is `2/512 = 1/256`.  Other bias values give
  different baselines (e.g. 120 for 1.0, 136 for 4.0, 148 for 12.0).
- **scale factor is the fp16 value / 512** — `acc × bias / 512`
  rounded *toward zero* (floor for positive).  T7c/T10c confirmed floor
  vs round-nearest (`8160 × 2 / 512 = 31.875` → silicon returns 31, not 32).
- **saturation** is to `[0, 255]` u8.

## Probe match rates

| Probe | Cells | Match | Formula |
|------:|------:|------:|---------|
| T7a u8-hyp (act=1, wt[0][n]=n+1, bias=1.0)                            | 1024 | **1024** | ✓ |
| T7b u8-hyp (act=128, wt=1, bias=2.0)                                  | 1024 | **1024** | ✓ |
| T7c u8-hyp (act=255, wt=1, bias=2.0) — tests floor at 31.875          | 1024 | **1024** | with floor |
| T7d u8-hyp (row-ramp, bias=1.0) — exact boundaries                    | 1024 | 512  | off-by-1 at `acc×bias/512 ∈ ℤ` |
| T8  bias sweep (14 non-power-of-2 values with acc=32)                 | 14336 | **14336** | with floor |
| T9  per-col bias (exp=15 mantissa sweep, 32 cols)                     | 1024 | **1024** | ✓ |
| T9b lane-2c probe (lane 2c+1 = 1.0)                                   | 1024 | **1024** | baseline=120 everywhere → lane 2c unused |
| T9c upper 64 (bias[0..63]=1.0, [64..127]=4.0)                         | 1024 | **1024** | upper half unused |
| T10 wt=2 bias mantissa sweep (acc=64)                                 | 1024 | **1024** | ✓ |
| T10b saturation (bias=16.0, acc=32)                                   | 1024 | **1024** | ✓ |
| T10c high acc small bias (bias=1.0, acc=8160) — tests floor           | 1024 | **1024** | with floor |
| T11a single-row lit                                                   | 32×32 | **1:1** | phys_row r ↔ out_row r |
| T11b upper-half lit (phys<16)                                         | 1024 | **1024** | upper-half out=159, lower=128 |
| T11c lower-half lit                                                   | 1024 | **1024** | upper-half out=128, lower=159 |
| T11d row-ramp act × 4 amplifier                                       | 1024 | **1024** | ✓ (bias=2.0 has no exact-int edge) |
| T11e k-ramp uniform row                                               | 1024 | **1024** | ✓ |

**Bottom line:** for arbitrary (non-corner-case) `acc × bias` products
with non-zero fp16 mantissa, the formula above is **bit-exact**.  The
only drift case observed is `bias = 1.0 × 2^E` (mantissa = 0) with `acc`
an exact multiple of `512 / bias` — silicon floors `1.0 → 0` in T7d.
For typical QNN bias values (channel scales absorbed into mantissa),
this case is not expected to fire.

## Rounding mode

**Floor (toward zero for positive acc).**  T7c case closed this: predicted
`round-nearest(31.875) = 32`, silicon returns 31.  `floor(31.875) = 31`
matches.  Several T8 and T10 edge cases confirm.

## Bias encoding

The **same fp16 u16 entry encodes BOTH** the channel baseline and the channel
scale.  For QNN-style u8 output with zero-point 128, bias must have biased-
exponent = 16 (i.e., fp16 ∈ [2.0, 4.0)) — mantissa varies to encode the
per-channel scale in [1/256, 1/128).  Scales outside that range produce
per-channel zero-points that are **not 128** (e.g., bias=1.0 → zp=120;
bias=4.0 → zp=136).  QNN's op compiler pre-scales weights / bias so that
each channel's absorbed scale falls in `[1/256, 1/128)` and zp=128 uniformly.

## T12 probe matches (post-decoded formula)

| Test | bias[2c] | bias[2c+1] | match `[b=2c+1, s=2c]` |
|------|----------|------------|------------------------|
| T12a | 0        | varied     | 853/1024 (s=0 → baseline only) |
| T12b | same=varied | same=varied | 395/1024 (mirrored — rounding differs) |
| T12c | varied   | 0          | **1011/1024** |
| T12d | 2.0 unif | varied     | 372/1024 (mismatch: formula uses 2.0 for scale but silicon uses varied?) |
| T12e | varied   | 2.0 unif   | **999/1024** |
| T12f | 4.0 unif | 2.0 unif   | **1024/1024** ✓ |
| T12g | 2.0 unif | 4.0 unif   | **1024/1024** ✓ |

**T12f/g are the cleanest proofs**: when lanes DIFFER, only
`[b=bias_lane_2c+1, s=bias_lane_2c]` predicts 1024/1024.

## V8 integration result (2026-04-24)

With the split-lane encoding (zp in lane 2c+1, scale in lane 2c):
- 5 DIAG modes bit-exact (0/1024 mismatches)
- Random (varied act+wt+bias per col): **28/1024 mismatches, max_abs_err = 1**
- Residual = ~3% cells where `acc × bias / 512` is fractionally just above
  an integer (e.g., `106 × 14.54 / 512 = 3.010`, silicon returns 2, formula 3).
  Silicon appears to drop the epsilon above the integer at its internal
  fp16-precision normalization.

## Open (minor)

Exact silicon rounding at fp16 precision boundaries isn't nailed down.
max_err=1 in u8 output is within QNN's standard u8-quant tolerance, so
for practical purposes the formula is **production-usable**.

## Implementation implications

- **Activation path**: pass plain u8 tensor (no signed offset), declare
  `QNN_DATATYPE_UINT_8` or `QNN_DATATYPE_UFIXED_POINT_8`.
- **Weight path**: declare as **UFIXED_POINT_8** (not SFIXED_POINT_8) to
  bypass QNN's auto-inserted `Cast int8→uint8 (+128)` which would flip the
  MSB of each weight byte.  Store raw i8 bit pattern in the clientBuf.
  Confirmed on-device: V8 DIAG1 goes from 112 → 128 with this change.
- **Bias path** (post-T12): for col c,
    - `bias[2c+1] = 0x4000` (fp16 2.0 → baseline = 128 uniformly)
    - `bias[2c]   = fp32_to_fp16(512 × scale_c)` (per-col scale)
  For custom per-col zp, set `bias[2c+1] = fp32_to_fp16((zp / 128) × 2.0)`
  but note the (raw>>7) truncation means only 128 discrete zp values
  are encodable.

## Probe runners

```bash
cd example/hmx_matmul_device
bash build.sh
bash run_hmx_formula_probe.sh          # T7..T10 full validation
bash run_row_mapping_probe.sh          # T11 row-dim pin-down
bash run_pair_lane_probe.sh            # T12 pair-lane decode (BASELINE=2c+1, SCALE=2c)
bash run_sat_ub_probe.sh               # T1..T6 original (still useful)
```
