# Handwritten HMX MatMul Body Evidence

This page tracks the owned HMX body evidence for the QNN-free handwritten
MatMul runtime.  It records the current accepted body set and the checks that
must remain green before changing or promoting any body slice.

Machine-readable body check:

```bash
uv run python scripts/check_handwritten_hmx_body.py \
  --json-out /tmp/handwritten_hmx_body_check.json
```

By default the checker covers the active families: `u8i8`, `w4a8`, `w8a16`,
and `w4a16`.  W16A16 remains retained reference material and can be checked
explicitly with `--family w16a16`, but it is not part of the current active
promotion gate.

The checker verifies:

- owned inline asm compiles and generated `.text` is byte-identical to the
  native skel slice;
- the runtime ABI headers compile with Hexagon `clang-19 -mhmx`;
- descriptor `sizeof` and `offsetof` contracts are pinned by
  `handwritten_hmx_kernel_abi.h`.

Body-entry smoke is checked separately through H2/hexagon-sim:

```bash
uv run python scripts/check_handwritten_hmx_body_entry_sim.py \
  --json-out /tmp/handwritten_hmx_body_entry_sim.json
```

That smoke compiles
`example/handwritten_hmx_matmul/tools/body_entry_smoke.c`, acquires an HMX
context in the simulator, enters each active owned body with conservative
synthetic descriptors, and requires every body to return without a simulator
fault.

## Current Body Manifest

| Family | Body | Owned source | Native slice | Status |
|---|---|---|---|---|
| `u8i8` | `hmx_v73_convbbb1x1deep_stride1` | `example/handwritten_hmx_matmul/kernels/u8i8/v73deep_conv1x1_kernel.inc` | `libQnnHtpV75Skel.so@0x2ebe40`, 1132 bytes | byte-identical; active |
| `w4a8` | `hmx_v73_convbnb1x1_stride1` | `example/handwritten_hmx_matmul/kernels/w4a8/v73deep_conv1x1_kernel.inc` | `libQnnHtpV75Skel.so@0x2f0780`, 2624 bytes | byte-identical; active |
| `w8a16` | `hmx_v75_convhbh1x1deep_stride1` | `example/handwritten_hmx_matmul/kernels/w8a16/v73deep_conv1x1_kernel.inc` | `libQnnHtpV75Skel.so@0x2f5200`, 1348 bytes | byte-identical; active |
| `w4a16` | `hmx_v73_convhnh1x1deep_stride1` | `example/handwritten_hmx_matmul/kernels/w4a16/v73deep_conv1x1_kernel.inc` | `libQnnHtpV75Skel.so@0x2fdb80`, 804 bytes | byte-identical; active |
| `w16a16` | `hmx_v73_convhhh1x1_stride1` | `example/handwritten_hmx_matmul/kernels/w16a16/v73deep_conv1x1_kernel.inc` | `libQnnHtpV75Skel.so@0x2fa740`, 1800 bytes | byte-identical; retained reference |

The native library path for these slices is:

```text
tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so
```

## ABI Boundary

The owned runtime headers live under:

```text
example/handwritten_hmx_matmul/include/
```

Each active family keeps a small native descriptor/register contract:

| Register | Meaning |
|---|---|
| `r0` | output descriptor |
| `r1` | activation descriptor |
| `r2` | packed weight stream |
| `r3` | folded bias/control stream |
| `r4` | mask descriptor |
| `r5` | extra parameter words |

Shared descriptor layout assertions are centralized in
`handwritten_hmx_kernel_abi.h`:

- output descriptor: 24 bytes, fields at offsets `0`, `4`, `8`, `12`, `16`,
  and `20`;
- activation descriptor: 12 bytes, fields at offsets `0`, `4`, and `8`;
- mask descriptor: 28 bytes, fields at offsets `0`, `4`, `8`, `12`, `16`,
  `20`, and `24`.

## W4A16 Acceptance Boundary

The W4A16 body is no longer promoted through the old residual/CVT exploration
route.  Current acceptance is the device gate's chain8 custom-baseline path:

- direct-HMX W4A16 body output is byte-exact against
  `example/qnn_matmul_profile/output_w4a16_aligned_e2e_256/device_out/out.raw`;
- `scripts/summarize_w4a16_custom_baseline_native_bridge.py` verifies the same
  custom public raw output against the QNN native raw output after the existing
  `native_transpose_2d` transform;
- `tests/qnn_kernel_e2e/handwritten_hmx_matmul/run_all.sh` requires this bridge before W4A16
  is promoted.

Do not reintroduce old descriptor sweeps, residual-ranking scripts, or
standalone CVT microprobe artifacts into this active body evidence path.  New
body work should start from a named byte-identity or runtime-gate failure.
