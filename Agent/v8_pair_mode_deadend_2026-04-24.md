# V8 mxswapacc pair-mode — not applicable (2026-04-24)

## Hypothesis

Use HMX's dual-accumulator + `mxswapacc` to pipeline two N-tiles:
- Acc A receives MACs for tile i
- Acc B receives MACs for tile i+1
- Drain A and B via `:after:cm:sat.ub`; overlap drain with next pair's MACs

Expected gain: amortize the ~6K cyc/tile `:after:cm:sat.ub` drain latency
across two tiles (~2× on mmv8).

## Why it doesn't work

### 1. `:after:cm:sat.ub` has no `:retain` variant

Per probe_dualacc_device.c (Agent/qnn_hmx_pipelining.md §"Followup"):

> - `store :after.uh` — destructive to **both accs** (implicit clear on store).
> - `store :after:retain.uh` — preserves both accs.

Silicon tested:
- `mxmem(...):after:cm:sat.ub = acc` — clears both A and B.
- `mxmem(...):after:retain:cm:sat.ub = acc` — compiler accepts the token
  but silicon ignores `:retain` in this form and still clears both.

Implication: draining acc B (the current acc) wipes acc A before we can
read it.  Both tiles cannot be read out from the pair.

### 2. RE already showed pair-mode doesn't speed up MAC loop

`probe_pipeline_device.c` P5 (alternating act ptrs in 2-MAC loop) gave
7.98 cyc/MAC — the **same** as P4 plain single-ptr (8.03).  Conclusion
from that probe:

> QNN's 2-MAC-per-iter structure is there to **reuse the weight tile
> across 2 output rows** (amortizing weight load); it is **not** a
> pipelining mechanism on its own.

We are already at the 7.9 cyc/MAC HMX ceiling in V8's single-acc loop.

### 3. Measured attempt confirms the theory

Implemented pair-mode V8 with `:after:retain:cm:sat.ub` on B first, then
`mxswapacc` + non-retain drain A:

| Shape | mmv8 single-acc | mmv8 pair-mode | DIAG bit-exact? |
|-------|----------------:|---------------:|:---------------:|
| 32³   |          6,200 |          6,150 | ✓ (only 1 tile, no pair) |
| 512³  |      1,607,000 |      1,585,000 | **✗** max_err=64 |
| 1024³ |      7,540,000 |      7,452,000 | **✗** max_err=127 |

~1.2% nominal speedup at large shapes — within noise — **AND** bit-
exactness breaks because B's non-retain drain erases A.

## Correct conclusion

V8's per-tile overhead (~6K cyc) is dominated by the HMX `:after:cm:sat.ub`
instruction's intrinsic latency (fp16 requant pipeline + 1 KiB VTCM
write).  This latency is **serial** within a single HMX stream and
**cannot be hidden** by pair-mode because:
- `:after:cm:sat.ub` has no `:retain` form — draining one acc clears the
  other.
- HMX is a single issue unit — two MAC streams in the same loop don't
  speed up scalar MAC throughput.

**V8's per-tile cost is architecturally bounded by `:after:cm:sat.ub`
itself.**  The only ways to close the 4.7× gap to V6 at 512³:
1. Fuse pack_act + mmv8 in one op (overlap HVX pack with HMX MAC via
   operation interleaving — estimated ~30% win).
2. Abandon `:after:cm:sat.ub` and use V6's dual-scale + HVX requant
   path (which defeats V8's "HMX-only" design goal).

Neither changes the MAC itself.  The cycles are spent in HMX's
built-in requant, which is a fixed hardware cost per tile.
