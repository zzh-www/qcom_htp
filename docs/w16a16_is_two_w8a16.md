# w16a16 = two w8a16 passes (there is no single-pass int16×int16 HMX matmul)

**Conclusion.** A `w16a16` MatMul (uint16 activation × int16 weight → uint16) is **not a
distinct kernel**. The HMX array is an integer **byte-MAC** and `int16×int16` overflows the
int32 accumulator, so the operation is **exactly two 8-bit-weight passes**: the int16 weight
is split into a signed high byte and an unsigned low byte, each multiplied against the int16
activation (i.e. two `w8a16`-class matmuls), drained separately, and combined with a ×256
shift. This is precisely how QNN lowers it on HTP (a 2-`ConvLayer` + `Concat` graph).

A single fused kernel that accumulates all four byte-products into one int32 and drains once
**cannot reproduce native** — it either overflows (large weights) or cannot represent the
requant scale (the drain is power-of-2 only). See "Why a single kernel fails" below.

## The decomposition (byte-exact)

Split the int16 weight by 2's-complement bytes (exact for the full int16 range):

```
q16 = hi·256 + lo        hi = q16 >> 8   (int8,  [-128,127])
                          lo = q16 & 0xff (uint8, [0,255])
```

Then the matmul is an exact integer identity:

```
(act − zp) @ q16  ≡  ((act − zp) @ hi)·256  +  (act − zp) @ lo
```

* The **high** pass `(act−zp) @ hi` is an int8-weight × int16-act matmul = **w8a16** (= the
  byte-exact `convhbh` kernel).
* The **low** pass `(act−zp) @ lo` is a uint8-weight × int16-act matmul — same shape/cost,
  unsigned carrier.

`round( ((act−zp)@q16) / 32767 + zp )` reproduces the on-device QNN-native `w16a16` within
the op's own drain rounding (≤ a few LSB; that delta is the requant, not a decomposition
error).

## Why it must be two separate passes (not one fused drain)

* **int32 overflow.** `max|(act−zp)@hi · 256|` exceeds 2³¹ for normal weights, so the ×256
  high contribution cannot be carried in the shared int32 accumulator. The two passes must
  be **drained before the combine**.
* **Drain is power-of-2.** The HMX `cvt.uh = acc(r31):2x2` drain applies a gain of
  `2^(exp-16)` taken from the bias control word's exponent field (bits 14:10); the mantissa
  is ignored (measured: gain ratio is exactly 1:2:4 across exponents, not fp16's 1:2:4.125).
  A single power-of-2 drain on the combined accumulator cannot hit the non-power-of-2
  `1/32767` scale. Native gets the fine scale from **two** power-of-2 drains at different
  byte scales (hi at ×256), combined — not from an fp16 multiply.
* Both passes use the **full int16 activation** (both activation bytes contribute).

## Cost

`w16a16` ≈ **2× the HMX work of `w8a16`** (two 8-bit-weight passes + a combine). There is no
single-pass shortcut: fusing the two passes into one int32 drain is mathematically lossy.

## Usage / recommendation

* **If int8-weight precision is enough** (for GDN it is: D&C `oc` 1.4e-2 ≈ shipped 1.22e-2),
  use the byte-exact **`w8a16` / `convhbh`** kernel directly. Do **not** pay for `w16a16`.
* **If you genuinely need int16-weight precision**, implement it as two 8-bit-weight passes
  (high = `w8a16`/`convhbh`, low = uint8-weight variant) with separate drains + ×256
  combine — i.e. run the proven `w8a16` path twice. That is what QNN-native does.
* Do **not** pursue a single fused `w16a16` HMX kernel — it cannot be byte-exact for
  arbitrary weights (overflow + power-of-2 drain).

## Reproduce

```bash
# Byte-exact decomposition gate (offline, all shapes):
python3 scripts/verify_w16a16_two_w8a16.py
python3 scripts/verify_w16a16_two_w8a16.py --M 256 --K 256 --N 256

# QNN-native w16a16 on device (lowers to the 2-ConvLayer graph) — true ground truth:
#   gen_onnx.py w16a16 <dir> --m 64 --k 64 --n 64
#   -> qairt-converter (no op package) -> qairt-quantizer 16 16 32 0
#   -> qnn-context-binary-generator (needs htp_config with soc_id) -> qnn-net-run
# (driven by scripts/run_qnn_kernel_e2e_ci.sh)

# The int8-weight building block (w8a16) is itself byte-exact vs native:
#   scripts/run_qnn_kernel_e2e_ci.sh            (group w8a16)
#   example/handwritten_hmx_matmul  +  scripts/run_handwritten_artifact_body_sim.py --family w8a16
```
