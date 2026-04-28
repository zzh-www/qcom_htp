# V73DEEP `hmx_v73_convbbb1x1deep_stride1` analysis (2026-04-28)

Goal: figure out why our 6-arg call lands at 55.1% bit-exact at 256³, with
the per-tile match pattern (out of 1024 cells) constant across mt rows:

```
nt=  0    1    2    3    4    5    6    7
   128  640  512  640 1024  544  448  576
```

## Disasm artefacts
- `hmx_v73_convbbb1x1deep_stride1_2ebe40.S`  (full disasm)
- `hmx_v73_convbbb1x1_stride1_2eadc0.S`      (non-deep v73, for diff)
- `hmx_convbbb1x1_stride1_FULL_decoded.S`    (baseline)

## Loop structure decoded

```
Preamble:
  r28 = K_t/2                          (loop0 trip — 2 MACs/iter)
  r20 = M_t                            (loop1 trip)
  r22 = (alt_rt+1)/2 * N_t = 4096       (wt stride per K-MAC, with our params)
  r13 = N_t                            (outer iter counter; -=2 each iter)
  m0  = (out_table_stride*4) - 4

  if (extra_param[0] == 1)             → fast path  (our case: extra={1,0})
    r25 = extra_param[1] = 0           (r25 used as `acc(r25)` arg in drain)

Outer iter (N_t/2 trips, since r13 -= 2 from N_t):
  r3 += 0x101; bias = mxmem2(r3)        ── bias load 1
  r3 += 0xff;  bias = mxmem2(r3)        ── bias load 2 (so r3 = orig + 0x200)

  loop1 (M_t trips):
     r0 = r18; r1 = r19; r8 = r2

     loop0 (K_t/2 trips, 2 MACs/iter):
        r6 = act_pairs[i]; r21 = act_pairs[i+1]; r1 += 8
        r8 += r22 (=4096)
        activation.ub = mxmem(r6, r7):deep:cm
        weight.b      = mxmem(r8, r9):deep
        r8 += r22
        activation.ub = mxmem(r21, r7):deep:cm
        weight.b      = mxmem(r8, r9):deep
     endloop0

     r10 = memw(r0++#4)                 ── r10 = *r0_orig, r0 += 4

     ── Drain 1 (with r25 bit 12 CLEAR):
     cvt.ub = acc(r25)
     r10' = memw(r0++m0); mxmem(r10, r11):cm = cvt    ── store at r10_old
     r25 |= bit 12

     ── Drain 2 (with r25 bit 12 SET, then clear for next iter):
     cvt.ub = acc(r25)
     mxmem(r10', r11):cm = cvt                        ── store at r10' (= next entry)
     r25 &= ~bit 12; r8 = r2                          ── reset wt for next M-iter
  endloop1

  r2 += alt_rt+1 = 1024                  ── r2 advance per OUTER iter
  r18 += 8                              ── out_tbl base advances 2 entries
```

### Output write pattern (verified)

Per loop1 iter writes 2 consecutive `out_tbl[]` entries (drain 1 → idx, drain 2 → idx+1).
Per outer iter, M_t loop1 iters cover M_t rows of N_t entries (r0 advances N_t*4 per iter).
So one outer iter writes 2*M_t entries at columns `(2k, 2k+1)` of all M_t rows
where `k = outer_index` (since r18 += 8 per outer = +2 entries).

For our 256³ test with M_t=N_t=8, this gives:
- Outer 0 → out_tbl[mt*N_t + 0..1]   (nt=0,1)
- Outer 1 → out_tbl[mt*N_t + 2..3]   (nt=2,3)
- Outer 2 → out_tbl[mt*N_t + 4..5]   (nt=4,5)
- Outer 3 → out_tbl[mt*N_t + 6..7]   (nt=6,7)

Total = 8 outputs/row * 8 rows = 64 = full coverage. ✓

So our **out_tbl layout `[mt*N_t + nt]` IS compatible with the deep variant.**

## What's wrong then?

**Wt walk does not match `[N_t, K_t, 1024]` layout.**

Per outer iter:
- r2 advances by 1024 per outer (total: 4 outers * 1024 = 4096 bytes total)
- 8 K-MACs walk r2 + [4096, 8192, ..., 32768] (8 stops, stride 4096)
- Outer covers 2 N-tiles' worth of OUTPUTS

For our `[N_t=8, K_t=8, 1024]` layout (= 65536 bytes), 2 N-tiles span 2*K_t*1024 = 16384 bytes. So r2 SHOULD advance by 16384 per outer to walk N-tiles. Instead it advances 1024 = 16× too small.

This means the deep variant expects a TOTALLY DIFFERENT wt layout. The 4096-byte K-MAC stride and 1024-byte outer-iter stride do not fit any simple `[N_t, K_t, 1024]` or `[K_t, N_t, 1024]` interpretation.

### Hypothesis: `:deep` weight load fans out internally

The `:deep` modifier likely makes one mxmem packet read wt covering MULTIPLE
N-tiles or K-rows simultaneously (HW-internal fan-out), so the SOFTWARE-level
walk pattern looks weird but the engine reads the right data.

For example, if `:deep` reads 8 N-tiles' wt for a single K-row in one packet
(N-fanout=8 internally), then:
- Per packet covers K-row-K_t for all N at once
- K-MAC stride = next K-row stride = some value

But this doesn't fit either, because we'd need only K_t=8 MACs to cover K
fully, not interleaved with N.

### Hypothesis: wt layout is `[K_t * N_groups, N_t/N_groups * 1024]`

If layout is somehow K-deep-paired with smaller N stride, we could see
4096-byte K-MAC stride and 1024-byte outer-iter stride. But the exact
bytes-per-row don't fit any clean rearrangement of 65536 bytes that we
can construct from `wRaw_KN` without further RE.

## Why nt=4 specifically is perfect

In Outer 2 (writing to nt=4,5), r2 = wt_base + 2048. Wt walks at
r2 + [4096..32768] = [6144, 10240, ..., 34816].

Modulo our `[8,8,1024]` layout:
| MAC | offset | (N,K)     |
|-----|--------|-----------|
|  1  | 6144   | (0,6)     |
|  2  | 10240  | (1,2)     |
|  3  | 14336  | (1,6)     |
|  4  | 18432  | (2,2)     |
|  5  | 22528  | (2,6)     |
|  6  | 26624  | (3,2)     |
|  7  | 30720  | (3,6)     |
|  8  | 34816  | (4,2)     |

Random-looking; no clean answer for why nt=4 specifically gets 1024/1024
without knowing what `:deep` actually reads from each packet.

## Path forward

Two viable routes:

### Route A — Runtime dump native deep call (recommended)

Use ld_preload or memhook on device to intercept a real
`hmx_v73_convbbb1x1deep_stride1` call by native QNN's `q::ConvLayer_s1.opt`.
Dump:
- `out_desc[64 B]`, `act_desc[16 B]`, `mask_desc[64 B]`
- `wt_base` and the first ~2KB of wt bytes (to compare against our wRaw_KN)
- `bias_base` and first 1KB of bias bytes
- `extra_param[]` array contents (size unknown — probably 16+)

Compare each to our V73DEEP call site to find the exact mismatches.

### Route B — Static RE descriptor builder

Find QNN compiler's deep-variant descriptor builder in `libHtpPrepare.so`
(prepare-time; produces `out_desc`/`act_desc`/`mask_desc`/`extra_param[]`
bytes that get loaded into VTCM). The string we'd grep is something like
`bias_to_vtcm_deep` or similar deep-specific helper. Then read the C++
that constructs each field.

This is HOURS of static RE without high confidence.

## Conclusion

V73DEEP is a 3× speed upside but the wt layout transform required to use
it is non-trivial and not derivable from disasm alone. **Route A (runtime
dump) is cleaner and faster** — defer this to a session with device-side
debug tooling set up.

V73 (non-deep) gives 1.17–2.27× already and IS production-ready. Commit
that win first.

---

## Quick win attempts that did NOT pan out (this session)

Tried mentally working through what variations might fix the 55% match:

- **Halve loop1 trip** (`n_tiles_pow2 = M_t * 4` so r20 = M_t/2):
  output write pattern would have us writing only M_t/2 rows × 2 entries
  per outer = M_t entries per outer × N_t/2 outer = M_t * N_t / 2 = HALF
  out_tbl coverage. So loop1 trip = M_t (current) is correct — drains land
  at 2 separate N positions, not 2 M positions.

- **Halve K_t in act_desc** (`n_act_pairs = K_t/2`):
  loop0 trip becomes K_t/4 = 2, with 2 MACs/iter = 4 K-MACs total per
  loop1, half of the K=K_t we need. Underaccumulation.

- **Different m0 / out_table_stride**: stride controls horizontal step in
  out_tbl. Drains land 2 entries apart so stride=N_t (current) is correct.

The fundamental issue is that `:deep` weight load semantics consume MORE
data per MAC packet than the 1024 bytes the address would suggest,
in a way the disasm can't reveal. Need either:
  (a) the HMX `:deep` weight modifier spec, or
  (b) runtime dump of native's call args + first 4KB of wt bytes that
      `q::ConvLayer_s1.opt` passes when triggering deep dispatch.
