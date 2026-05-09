# HMX Unknown Inference Notes

Use these examples when converting HMX/QNN kernel `.byte` replicas to readable
inline asm. Always verify candidates by assembling and comparing bytes.

## Method

1. Match ordinary Hexagon words around each unknown packet from disassembly.
2. Search nearby decoded kernels in `Agent/qnn_re/` for the same raw word.
3. Assemble candidate packets with Qualcomm clang:

```bash
tools/hexagon-sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/clang-19 \
  -target hexagon -mcpu=hexagonv75 -mhmx -c -x assembler
```

4. Compare the emitted word or whole `.text` against the native bytes.

## W4A16 HNH Deep 1x1 Examples

These came from `hmx_v73_convhnh1x1deep_stride1` at `0x2fdb80`.

| Raw word | Mnemonic |
|---|---|
| `ec 47 06 92` | `activation.ub = mxmem(r6,r7)` |
| `ec 47 15 92` | `activation.ub = mxmem(r21,r7)` |
| `ee e9 08 92` | `weight.n = mxmem(r8,r9):deep` |
| `ee e9 02 92` | `weight.n = mxmem(r2,r9):deep` |
| `e1 e9 08 92` | `weight.n = mxmem(r8,r9)` |
| `fe c3 03 92` | `bias = mxmem2(r3)` |
| `10 d9 f9 a6` | `cvt.uh = acc(r25):2x2` |
| `18 cb ea a6` | `mxmem(r10,r11) = cvt` |
| `11 c0 e0 a6` | `mxclracc` |

The key W4A16 distinction from the u8i8 `convbbb` body is `weight.n`, not
`weight.b`. The `n` route encodes the HNH/nibble weight path used by int4
weights.

## W4A8 BNB Scaled Cvt Drain Examples

These came from `hmx_v73_convbnb1x1_stride1` at `0x2f0780`. The important
lesson is that the raw `a6..dc/dd` words are not plain `cvt.ub = acc(rX)`;
they are the scaled W4A8 drain forms.

| Raw word | Mnemonic |
|---|---|
| `10 dc fb a6` | `cvt.ub = acc(r27):sc0` |
| `10 dd fb a6` | `cvt.ub = acc(r27):sc1` |
| `10 dc ff a6` | `cvt.ub = acc(r31):sc0` |
| `10 dd ff a6` | `cvt.ub = acc(r31):sc1` |

The adjacent control slots must be assembled in the same packet. Examples:

```asm
Lhmx_2f08cc:
{ r3 = add(r3,#0x100);
  r25 = add(r25,#-0x1);
  r31 = memw(r5++#0x4);
  bias = mxmem2(r3) }
{ p0 = cmp.gt(r25,#0);
  if (p0.new) jump:nt Lhmx_2f08cc;
  cvt.ub = acc(r31):sc1 }
```

```asm
Lhmx_2f0ba4:
.word 0xb0036003, 0xbff77ff7, 0x9b85403b, 0x9203c3fe
{ p0 = cmp.gt(r23,#0); if (p0.new) jump:nt Lhmx_2f0ba4;
  cvt.ub = acc(r27):sc1 }
```

For this family, first assemble candidate `cvt.ub = acc(rX):sc0/sc1` forms,
then prove the enclosing packet with a real local label. The same `cvt` word can
look correct while the branch word is wrong if the label target distance is not
the native distance.

## W4A8 BNB Nibble Weight Load Examples

The same W4A8 BNB slice uses `weight.n` with a `:2x` route. This is distinct
from U8I8/W8A16 `weight.b` even when the activation side still uses
`activation.ub`.

| Raw word | Mnemonic |
|---|---|
| `ed 58 06 92` | `activation.ub = mxmem(r6,r24):cm` |
| `ed 58 17 92` | `activation.ub = mxmem(r23,r24):cm` |
| `ed 47 06 92` | `activation.ub = mxmem(r6,r7):cm` |
| `ed 47 17 92` | `activation.ub = mxmem(r23,r7):cm` |
| `46 f9 08 92` | `weight.n = mxmem(r8,r25):2x` |
| `46 e9 08 92` | `weight.n = mxmem(r8,r9):2x` |
| `46 e9 02 92` | `weight.n = mxmem(r2,r9):2x` |
| `43 e9 08 92` | `weight.n = mxmem(r8,r9):2x:deep` |
| `43 e9 02 92` | `weight.n = mxmem(r2,r9):2x:deep` |

Some loop-tail packets require the `:endloop0` suffix to reproduce ordinary
control words in the same packet. For example:

```asm
{ r8 += add(r25,#1);
  nop;
  activation.ub = mxmem(r23,r24):cm;
  weight.n = mxmem(r8,r25):2x }:endloop0
```

Without `:endloop0`, the HMX load words may still look right while the ordinary
Hexagon control word differs.

## Packet-Level Pattern

Public objdump may print this whole packet as `<unknown>`:

```asm
{ r3 = add(r3,#0x100);
  r26 = add(r26,#-0x1);
  r25 = memw(r5++#0x4);
  bias = mxmem2(r3) }
```

Only the HMX word is undecoded; the ordinary words still follow normal Hexagon
encoding. Recover mixed packets by preserving all four instruction slots and
assembling the entire packet.

## Verification Standard

The final acceptance check is whole-function byte identity:

```text
compile temporary C wrapper
extract .text from object
extract native slice from libQnnHtp*Skel.so
cmp -s generated.text native.bin
```

If any byte differs, inspect `cmp -l` around the first offset and check branch
label placement, packet grouping, and HMX suffixes.
