# V73DEEP wt layout DECODED via ctx-binary diff (2026-04-28)

## Method
Built two 256³ native MatMul ctx-binaries with marker weights:
- `s256_marker_K/`: `wRaw[k,n] = (k % 128) - 64`  (each row = same value)
- `s256_marker_N/`: `wRaw[k,n] = (n % 128) - 64`  (each col = same value)

Diffed the two ctx-binaries; differing bytes = wt blob (since graph
structure / quant params are identical).

Generators: `phaseA_native/gen_marker_wt_256.py`.
Ctx-binaries: `phaseA_native/s256_marker_{K,N}/ctx/matmul_native_ctx.bin` (110592 bytes each).

## Wt blob location
**Offset 0x9100..0x19100 = 65536 bytes** = exactly K*N int8.

## Layout: `[K_t=8, N_t=8, 1024 bytes]` (K-major outer)

Verified via tile-by-tile signature inspection:
- Tiles 0..7 (offset 0x9100 to 0xa900):  K-tile 0 (k=0..31), N-tiles 0..7
- Tiles 8..15 (0xb100..0xcd00):           K-tile 1 (k=32..63), N-tiles 0..7
- Tiles 16..23: K-tile 2, N-tiles 0..7
- ... etc.

Total: K_t * N_t = 64 tiles × 1024 bytes = 65536 ✓

## Within-tile: SAME 4-row interleave as V8C8

`dst_byte_in_tile = (r // 4) * 128 + c * 4 + (r % 4)`
where `r` = K-row 0..31, `c` = N-col 0..31 within the tile.

Verified: byte 0 of tile = (r=0, c=0); bytes 0..3 = (r=0..3, c=0); bytes 4..7 = (r=0..3, c=1); etc.

## Difference vs our V8C8 layout

| | Outer dim | Inner dim | Within-tile |
|---|---|---|---|
| **V8C8 / V73 non-deep** (our current) | N_t | K_t | (r//4)*128 + c*4 + (r%4) |
| **V73DEEP (native)** | K_t | N_t | (r//4)*128 + c*4 + (r%4) |

ONLY the outer dim ordering is flipped.

## Why the disasm wt-walk numbers fit

V73DEEP disasm:
- `r22 = (alt_rt+1)/2 * N_t = 4096` (wt advance per K-MAC)
- `r2 += alt_rt+1 = 1024` (wt advance per outer iter)

For `[K_t, N_t, 1024]`:
- Walking K-direction (next K-tile, same N) = N_t * 1024 = 8192 bytes. But r22 = 4096 = HALF.
- Walking N-direction (next N-tile, same K) = 1024 bytes. ✓ matches r2 advance.

The K-step (4096 = N_t/2 * 1024) hint: deep variant K-fanout 2 internally. Each `mxmem(r8,r9):deep` wt packet covers 2 K-tiles' worth in 1024 bytes (compressed K-doubling). So K_t MAC packets = K_t/2 K-tile-pairs × 2 K-tiles each = K_t K-tiles ✓ all-K coverage.

Wait: K_t MAC packets stepping 4096 bytes each cover K_t * 4096 = 32768 bytes from r2. With layout `[K_t, N_t, 1024]`, 32768 = 4 K-tiles' worth = K_t/2. Combined with K-fanout 2 internal to each packet, we cover K_t = 8 K-tiles ✓.

So inside one outer iter, we touch 4 K-tile slots in memory (× 2 internal K-doubling) for the active N-tile-pair.

## Per-outer-iter coverage

Per outer iter:
- Outputs: 2 N-tiles × M_t M-tiles drained
- Wt range: r2 walks 4096 bytes per K-MAC, K_t MAC packets, K_t/2 = 4 starting offsets per K-MAC chain
- r2 advances 1024 bytes per outer (= 1 N-tile)

Total outer iters: N_t/2 = 4 (since deep variant produces 2 N-tiles per outer).

After 4 outers: r2 advanced 4096 bytes = 4 N-tiles × 1024 = covers N=0..3 directly. Combined with 2-wide internal N-fanout per MAC packet: covers all N=0..7. ✓

## Other unknowns (still need to verify)

- **Bias layout**: deep loads bias from `r3+0x101` and `r3+0x200`. Per outer iter r3 advances 0x200 = 512 bytes (vs non-deep 256 B/N-tile). So deep bias = **512 B per N-tile-pair** (vs non-deep 256 B/N-tile).
- **mask_desc[+0x30] bit 5** must be set for v73 dispatcher to take deep branch.
- **extra_param[]** array — at least 1 + (per-drain-count) entries, contents TBD.

## Action plan

1. ★ **Test #1 (DONE 2026-04-28)**: changed gen_v8c8_chain.py to support
   `--wt_layout=kmaj` (packs `[K_t, N_t, 1024]` instead of `[N_t, K_t, 1024]`).
   Used asymmetric `wRaw[k,n] = ((31k + 13n) % 15) - 7` (the previous symmetric
   `(13(k+n)) % 15` produced byte-identical kmaj/nmaj — invalid test).
   Result at 256³ chain hot:
     - V73DEEP nmaj: 45.7% bit-exact
     - V73DEEP kmaj: 50.4% bit-exact
   **Both essentially saturation noise** — reference is 100% saturated (0 or
   255 only) due to chain-8 propagation. Output has 6.6% mid-range values
   meaning kernel IS computing partial accumulators wrong; "matches" come
   from saturation coincidence (output happens to clip to right endpoint).
   K-major helps only marginally (+5pp) over N-major. **wt layout alone
   is not sufficient** to make V73DEEP work.

2. **Other unknowns confirmed needed**:
   - Bias access semantics differ: deep loads bias from `r3+0x101` then `r3+0x200`
     (2 loads per outer, vs non-deep 1 load). With our 256B/N-tile layout, this
     reads bias[1, 2] for outer 0 (vs needed bias[0, 1] for nt=0,1). Either:
       (a) `mxmem2` has special address semantics (low bits = HMX register select?)
       (b) Native bias layout has 256-byte header padding offset (bias[0] at +0x100)
   - `extra_param[]` array contents — likely per-K-iter scale/cvt config
   - `mask_desc[+0x30]` bit 5 = deep-dispatch toggle (set by us = irrelevant since
     we call deep variant directly, but other bits of mask_desc might matter)
   - `set_hmx_params_conv1x1` symbol has NO `_deep` variant — single entry takes
     5 uint32 args. Native MUST be calling it with different arg values for deep
     mode. Need to find caller in `libHtpPrepare.so` to learn the canonical args.
   - Call site address in libHtpPrepare.so: 0xd998f0 (= set_hmx_params_conv1x1).
     Static RE: search x86 .text for `e8 XX XX XX XX` instructions that reach
     this address; check the immediate args they pass.

3. **What works without V73DEEP**: V73 non-deep (`hmx_v73_convbbb1x1_stride1`
   with extra_param[2]={1,0}) is 100% bit-exact at 256/512/1024³ and gives
   1.17–2.27× speedup. That's the production-ready path.

## Saturation noise floor

For our chain-8 256³ matmul test, reference output is 100% saturated due to
how int32 accumulators of 8 chained matmuls explode quickly. So any output
where we randomly pick endpoints (0 or 255) gets ~50% match by coincidence.

For a more sensitive test, use a single matmul (chain=1) where the reference
output has more mid-range values. The 45-50% match floor would drop signif-
icantly when most cells require correct mid-range computation, exposing the
true computation bug.
