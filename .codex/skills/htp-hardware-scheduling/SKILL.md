---
name: htp-hardware-scheduling
description: How to hand-write EFFICIENT Hexagon HTP kernels by orchestrating the hardware units (DMA engine, HVX, HMX) and VTCM into an overlapped pipeline. Use this BEFORE designing/optimizing any hand-written DSP op in qcom_htp — bare-metal FastRPC HAPs, dspqueue kernels, custom QNN ops doing their own VTCM/HMX work, or any "make this op faster / why is my op data-movement-bound" task. It pins the methodology (NOT a specific op or dtype): the 3 non-negotiable principles (one RPC = whole graph never per-op; if it fits put ALL data resident in VTCM; hide every byte move under compute), the 4 scheduling layers, the DMA double-buffer that hides data movement (17×), the bottleneck hierarchy (DDR I/O ≫ readback ≫ HMX compute), and the HMX-vs-HVX decision rule. The point is the FLOW, not fp16/int8.
---

# HTP Hardware Scheduling

Use this when **hand-writing or optimizing a Hexagon DSP kernel** and you need it
to be fast. The skill is about **how to schedule DMA / HVX / HMX / VTCM**, not
about any one operator or numeric format. Process guide only — do not turn it
into an end-to-end tool unless asked.

**The one rule:** the bottleneck is almost always *data movement*, not compute.
Pure HMX matmul on VTCM-resident data is ~free (~1µs, ~28k GFLOPS). Your job is
to keep every hardware unit busy at once and hide the moving of bytes.

**Three non-negotiable principles (apply these first, always):**
1. **One RPC = the whole graph.** After the single FastRPC `start`, process the
   ENTIRE graph / composite computation on the DSP before returning — NEVER one
   RPC per op. Per-op round trips pay the comms tax (and a DDR round trip) every
   op; one dispatch amortizes it once. (Qwen3-0.6B: 196 ops → 71ms per-op-FastRPC
   vs 12ms one-dispatch-dspqueue.)
2. **If it fits in VTCM, put ALL of it in VTCM** — *for data touched by HVX / HMX
   / DMA*. Make the working set resident to eliminate DDR round trips between
   ops/stages and feed HMX (mandatory). Fall back to streaming/double-buffer
   (Layer 3) only on overflow.
   ⚠️ **TWO measured gotchas (learned the hard way on the GDN solve):**
   - **VTCM must be acquired ONCE per process and SHARED** (acquire on the main
     thread, give each worker a `slot*REGION` slice). A per-worker
     `HAP_compute_res_acquire` SERIALIZES the workers — the resource manager
     grants VTCM per-context, so concurrent acquires block on each other.
     (Fixing this: GDN 4-thread 270K→215K, scaling 1.83×→2.19×.) Canonical:
     SDK `incs/HAP_compute_res.md` + tutorial ch04 `demo_vtcm_alloc.c`.
   - **Do NOT put SCALAR / data-dependent scratch in VTCM.** VTCM scalar access
     is catastrophically slow (it's built for HVX-vector + HMX + DMA, not scalar
     loads/stores). Moving the GDN merge/forward-subst scratch (hammered by
     scalar code) into VTCM made the solve **7× SLOWER** (471K→3.48M). Keep
     scalar-accessed scratch in DDR (L2 prefetch handles it); only put
     vector/HMX/DMA-accessed buffers in VTCM.
3. **Hide every unavoidable byte move** under compute (DMA double-buffer, Layer 3).

(Nuance, not an exception: VTCM is not faster than DDR for a *single sequential*
HVX pass — L2 prefetch saturates that. Residency still wins because it kills the
cross-op DDR traffic and feeds HMX; design for the graph, not one vadd.)

Full manual: `docs/htp_hardware_scheduling_flow.md`. ASM building blocks:
`docs/hexagon-tutorial/hmx-tutorial/ch05-hmx/src/exp5_standalone_asm.c`.

## The 4 scheduling layers (design top-down, optimize bottom-up)

1. **ARM↔DSP = dspqueue, not per-call FastRPC.** One FastRPC `start` (pass queue
   id / n_hvx / use_hmx); all compute via a shared-memory queue (`rpcmem_alloc` +
   `rpcmem_to_fd` + `fastrpc_mmap` zero-copy; DSP runs a `dspqueue_read_noblock`
   loop → `switch(op)` → `dspqueue_write`). **61µs vs 364µs/op.** Fuse a whole
   composite computation into ONE message when you can. Keep the DSP busy — if it
   idles, VTCM is reclaimed by camera/audio clients and your data is silently
   corrupted ("VTCM disappears").

2. **Multi-HVX worker pool** for data-parallel HVX work (one slot per HVX thread).
   ⚠️ **HMX does NOT parallelize across threads** (single matrix unit; the
   process-exclusive lock serializes workers — verified bare-metal AND in QNN).
   HMX "parallelism" comes from Layer 3 overlap, never from threads. Do not chase
   threaded HMX.

3. **Intra-op DMA∥HVX∥HMX pipeline + VTCM ping-pong double-buffer — THE HEART.**
   Four stages: `DMA(DDR→VTCM)` → `HVX(dequant / repack into 32×32 tiles)` →
   `HMX(mxmem matmul, VTCM-resident)` → `HVX/DMA(readback)`.
   - UDMA: `Q6_dmstart_A(desc)` (type0 linear descriptor, `srcbypass=1`),
     `Q6_R_dmwait()` / `Q6_R_dmpoll()`. The engine moves bytes in the background;
     CPU/HVX/HMX do not stall on it.
   - **Cache trap:** DMA bypasses L2 and reads physical DDR → you MUST
     `qurt_mem_cache_clean(src, n, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE)` the DDR
     source first. VTCM is not cached, so as a DMA *target* it needs no flush.
   - **Ping-pong:** two VTCM scratch buffers. Loop: `dma_start(next→buf[nxt])`
     (non-blocking) → compute current `buf[cur]` (HVX+HMX) → `dma_wait()`. The
     move of chunk N+1 hides under the compute of chunk N. **Measured 17×**
     (217µs overlap vs 3701µs serial) — this is the single biggest lever.

4. **VTCM residency / persistent tile format (NativeKV).** Keep reused data
   permanently in HMX tile (WH) layout so HMX consumes it with zero conversion
   (v75+ only; and on v75 f16 AH==WH, so a matmul's output tile directly feeds the
   next matmul). VTCM has no malloc — bump-allocate, 128B aligned.
   ⚠️ VTCM is **not** faster than DDR for *sequential* HVX (L2 prefetch hides the
   latency). Use VTCM only because (a) HMX requires it and (b) it is a DMA target.

## Bottleneck hierarchy (memorize; measure against it)

```
DDR↔VTCM I/O      ~1000 µs   ← the real enemy; hide with DMA double-buffer
hexkl readback   120-3000 µs ← bypass it: use hmx_store_acc + HVX vdeal
HVX vdeal deint.    5-126 µs  ← VTCM→VTCM, ~24× faster than hexkl readback
pure HMX compute      ~1 µs   ← essentially free when VTCM-resident
```
Rule: **never touch DDR mid-op**; bypass hexkl readback; chain stage outputs in VTCM.

## HMX-vs-HVX decision (the sober line)

- HMX wins only on **large** matrices (≥512 dim): e.g. 256×1024×4096 → ~166× CPU.
- **Small / narrow** matrices (≤128, or 64×64 blocks) are *fixed-overhead
  dominated* — comms + tile-format conversion + readback drown HMX's throughput
  (MNIST 832×128: HMX only 1.1× HVX). For small blocks, EITHER make the overhead
  vanish via zero-conversion tile-residency (Layer 4), OR just use **HVX qf32**.
- Always ask "is my output narrow?" — narrow output ⇒ readback dominates ⇒ HMX loses.

## Design checklist for a new fast kernel

1. Comms: one FastRPC start, then **process the whole graph in one DSP dispatch** via dspqueue — never per-op RPC.
2. Working set: **does it fit in VTCM? If yes, put ALL of it resident** (one bump layout for the whole graph) and skip streaming entirely. Only if it overflows → go to steps 3–4.
3. If overflowing: pick the parallel axis (batch/head/row-strip) → that axis is your ping-pong unit; VTCM layout = weight/act tiles + output acc + **two** DMA scratch buffers + scales.
4. Stage the pipeline: `dma_start(next)` → HVX prep → HMX compute → HVX readback → `dma_wait`.
5. Keep intermediates in VTCM tile format across ops; bypass hexkl readback (hmx_store_acc + vdeal); never round-trip DDR between ops.
6. HMX init once: `bias = mxmem2(scales)` (scale=1.0 / bias=0) or HMX returns garbage.
7. Measure against the hierarchy above: if data-movement-bound, either your set isn't resident or your DMA isn't overlapping.

## Common mistakes (seen in this repo's GDN solve)

- Reading DDR synchronously / one-shot memcpy to VTCM with **no DMA overlap** →
  data-movement-bound (GDN diag 373K bare-metal vs 48K in QNN's layout).
- Chasing **threaded HMX** (wrong layer; HMX can't thread).
- No dspqueue fusion, no persistent tile → per-call + repeated conversion overhead.
- Concluding "GOAL unreachable" from a build that never used this pipeline.

## References
- `docs/htp_hardware_scheduling_flow.md` — the full manual (4 layers + numbers + GDN post-mortem).
- `docs/hexagon-tutorial/hmx-tutorial/` — ch03 dspqueue, ch04 DMA pipeline (17×), ch05 HMX ASM /
  bottleneck hierarchy, ch06 NativeKV persistent tile, ch07 llama.cpp full orchestration.
- Related skills: `hmx-inline-asm` (HMX kernel bytes ↔ asm), `qnn-htp-profiling` (measure before claiming).
