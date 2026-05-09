---
name: hmx-inline-asm
description: Recover, rewrite, and verify Qualcomm Hexagon HMX/QNN kernel byte replicas as readable inline asm. Use when working on qcom_htp files such as v73deep_conv1x1_kernel.inc, libQnnHtp*Skel.so slices, Agent/qnn_re disassembly, unknown HMX objdump packets, or byte-exact conversion from .byte directives to hand-written Hexagon inline asm.
---

# HMX Inline ASM

Use this skill to turn embedded HMX kernel `.byte` replicas into readable
Hexagon inline asm without changing the native bytes.

## Workflow

1. Identify the native source slice: symbol, VMA, size, SDK `.so`, and current
   embedded `.inc`.
2. Get disassembly from every available source:
   - public `llvm-objdump` for ordinary Hexagon instructions;
   - Qualcomm `hexagon-llvm-objdump` for packet boundaries and raw words;
   - nearby checked-in slices under `Agent/qnn_re/` for decoded HMX idioms.
3. Preserve packet boundaries. Convert one packet at a time, using labels for
   internal branch targets instead of absolute addresses.
4. Infer HMX unknowns by assembling candidate mnemonics with:
   `clang-19 -target hexagon -mcpu=hexagonv75 -mhmx`.
5. For `cvt`/drain packets, prove the whole packet, not only the HMX word.
   Branch/control slots often encode label distance in the same packet as the
   HMX conversion, so add local labels at the real native target address and
   assemble `cmp/jump/cvt` together.
6. Verify the whole function, not only individual instructions, by compiling a
   temporary C wrapper, extracting `.text`, and comparing against the native
   slice byte-for-byte.
7. Update adjacent comments, headers, and any analysis scripts whose conclusions
   mention the raw packets. Stale "unknown" conclusions are bugs once a mnemonic
   is byte-proven.

## Guardrails

- Do not accept a semantic-looking rewrite unless the final `.text` is byte
  identical to the native slice.
- Do not trust `<unknown>` packet boundaries from public objdump blindly. A
  single undecoded HMX word can cause the whole packet to print as unknown.
- Do not mark a raw-drain cleanup complete while the `cvt` word is only
  explained in comments. It is complete only when the file contains readable
  HMX mnemonics and the whole native slice still matches byte-for-byte.
- Keep raw-byte regeneration commands in comments so the readable form can be
  replaced with a byte fallback if needed.
- Use the repo's existing labels and comment style from neighboring
  `v73deep_conv1x1_kernel.inc` files when available.

## Common Commands

Show the native slice:

```bash
tools/hexagon-sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump \
  -d --start-address=0x2fdb80 --stop-address=0x2fdea4 \
  tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so
```

Verify a rewritten `.inc` against a native slice:

```bash
python3 .codex/skills/hmx-inline-asm/scripts/verify_hexagon_inline_asm.py \
  --inc example/qnn_hmx_matmul_w4a16/src/v73deep_conv1x1_kernel.inc \
  --so tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so \
  --vma 0x2fdb80 --size 804
```

Verify the W4A8 BNB slice:

```bash
python3 .codex/skills/hmx-inline-asm/scripts/verify_hexagon_inline_asm.py \
  --inc example/qnn_hmx_matmul_w4a8/src/v73deep_conv1x1_kernel.inc \
  --so tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so \
  --vma 0x2f0780 --size 2624
```

## References

- For opcode examples and inference notes, read
  `references/hmx_unknown_inference.md`.
