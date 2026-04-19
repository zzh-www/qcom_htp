# int16 per-tensor quantized matmul on HMX

32×32×32 fixed tile. Symmetric per-tensor quantization (zero-point = 0).
End-to-end bit-exact against an int64 reference across constant-small,
uint16-wrap, and random-int16 scenarios.

## Approach

HMX u8×i8 with `:after.uh=acc:2x1` stores `acc mod 2^16` as int16 —
which truncates for our partial-product magnitudes (max ≈ 2^20 for
K=32).  We recover the full int32 accumulator via **dual-scale
readback**: one full-K MAC followed by two convert-stores on the same
acc, enabled by `:after:retain.uh=acc:2x1`.

- Convert 1 with bias scale 1.0  → `OUT_LO = acc mod 2^16`
- Convert 2 with bias scale 2^-8 → `OUT_HI = floor(acc/256) mod 2^16`,
  interpreted as int16 for sign-correct bits [8..23].
- Reconstruction: `acc = ((int16_t)OUT_HI << 8) | (OUT_LO & 0xFF)`.

**12 HMX packets per 32×32×32 tile** (4 MAC + 8 convert, one full-K
MAC per partial).  See `hexagon_hmx_matmul_native_int.md` for probe
data and calibration.

Full int16 decomposition:

```
a_u = a_q + 32768                     (uint16, ∈ [0, 65535])
a_u = 256·A_h + A_l                   (A_h, A_l uint8)
w_q = 256·W_h + W_l                   (W_h int8, W_l uint8)
a_q · w_q = 65536·A_h·W_h + 256·(A_h·W_l + A_l·W_h) + A_l·W_l − 32768·w_q
```

Four partial products:

| Partial | Operands | HMX? | Correction |
|---------|----------|------|------------|
| M1 = Σ A_h·W_h | u8·i8  | native | — |
| M3 = Σ A_l·W_h | u8·i8  | native | — |
| M2 = Σ A_h·W_l | u8·u8  | u8·(i8 reinterp) | `+256·Σ A_h·topbit(W_l)` (scalar) |
| M4 = Σ A_l·W_l | u8·u8  | u8·(i8 reinterp) | `+256·Σ A_l·topbit(W_l)` (scalar) |

Combine: `65536·M1 + 256·(M2+M3) + M4 − 32768·ColSum_W`, then
`sat_i16((sum · rq.mul) >> rq.shift)`.

## HMX tile layout (decoded empirically)

- **Activation tile** (2 KiB):
  `A_byte(phys_row, K, stream)` at `128·phys_row + 4·K + (stream ? 3 : 1)`
  with `phys_row ∈ 0..15`, `K ∈ 0..31`, `stream ∈ {0,1}`.
  Bytes at `4·K + 0/2` are ignored (must be zero).
- **Weight tile** (1 KiB):
  `W_byte(K, col)` at `128·(K>>2) + 4·col + (K&3)`.
- **Output tile** (1024 u16):
  `out[phys_row·64 + 2·col + stream]`.
- **Logical 32×32 mapping**:
  `ir → phys_row = ir & 15, stream = ir >> 4` (stream 0 covers logical
  rows 0..15, stream 1 covers 16..31).  One HMX call produces a full
  32×32 logical output.
- **Bias** (identity convert): 128 × `f16(2.0) = 0x4000` u16s, filled
  into a 2 KiB scratch.  With this bias and clean (zeroed) ignored-byte
  positions, HMX outputs `acc mod 2^16` (no extra 257/256 scale).

See `hexagon_hmx_matmul_native_int.md` for the full reverse-engineering
log and worked probes.

## Files

- `int16_matmul.h`              — public API
- `int16_matmul_ref.c`          — int64 oracle + top-byte model
- `int16_matmul_hmx.c`          — HMX K-sliced kernel
- `test_int16_matmul.c`         — harness (S1, S2, S3 scenarios)
- `probe_hmx_acc.c`             — diagnostic probes used during decoding
- `build.sh`                    — hexagon-clang compile for sim
- `../../tests/test_hmx_matmul_int16.sh` — E2E test (sim + bit-exact)

## Build / run

```sh
bash scripts/install.sh                          # first-time only (SDKs)
source scripts/env.sh
# H2 hypervisor (first-time only):
(cd tools/hexagon-hypervisor && make ARCHV=75 TARGET=ref USE_PKW=0)
ln -sfn hexagon-hypervisor/install tools/h2-install

bash tests/test_hmx_matmul_int16.sh              # build + sim + validate
```

Expected tail:

```
--- S1 constant small  (A=256,  W=256,  shift=16) ---
  [PASS] HMX == ref bit-exact (1024 / 1024)
--- S2 constant mid    (A=-10000, W=20000, shift=25) ---
  [PASS] HMX == ref bit-exact (1024 / 1024)
--- S3 random int16    (full range, shift=20) ---
  [PASS] HMX == ref bit-exact (1024 / 1024)
ALL PASS
PASS: HMX int16 matmul E2E (3/3 scenarios bit-exact)
```

## Throughput note

**12 HMX packets per 32×32×32 tile**: 4 full-K u8·i8 MAC packets + 8
`.uh` convert packets (2 per partial, at scales 1.0 and 2^-8).  Down
from 128 in the earlier K-sliced kernel — a ~11× reduction while
staying bit-exact.  Hard floor would be 8 packets (4 MAC + 4 convert)
if a single-convert int32 read were possible, but HMX integer output
caps at 16 bits per cell, so two converts per partial is the minimum.
See `hexagon_hmx_matmul_native_int.md` "Final kernel (2026-04)" for
calibration data.
