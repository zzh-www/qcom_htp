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
