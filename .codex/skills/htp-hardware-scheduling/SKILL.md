---
name: htp-hardware-scheduling
description: How to hand-write EFFICIENT Hexagon HTP kernels by orchestrating the hardware units (DMA engine, HVX, HMX) and VTCM into an overlapped pipeline. Use this BEFORE designing/optimizing any hand-written DSP op in qcom_htp — bare-metal FastRPC HAPs, dspqueue kernels, custom QNN ops doing their own VTCM/HMX work, or any "make this op faster / why is my op data-movement-bound" task. It pins the methodology (NOT a specific op or dtype): the 4 scheduling layers, the DMA double-buffer that hides data movement (17×), the bottleneck hierarchy (DDR I/O ≫ readback ≫ HMX compute), and the HMX-vs-HVX decision rule. The point is the FLOW, not fp16/int8.
---

# HTP Hardware Scheduling

Use this when **hand-writing or optimizing a Hexagon DSP kernel** and you need it
to be fast. The skill is about **how to schedule DMA / HVX / HMX / VTCM**, not
about any one operator or numeric format. Process guide only — do not turn it
into an end-to-end tool unless asked.

**The one rule:** the bottleneck is almost always *data movement*, not compute.
Pure HMX matmul on VTCM-resident data is ~free (~1µs, ~28k GFLOPS). Your job is
to keep every hardware unit busy at once and hide the moving of bytes.

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

1. Comms: dspqueue (or fuse into one op), not per-call FastRPC.
2. Pick the parallel axis (batch/head/row-strip) → that axis is your ping-pong unit.
3. VTCM bump layout: weight/act tiles, output acc, **two** DMA scratch buffers, scales.
4. Stage the pipeline: `dma_start(next)` → HVX prep → HMX compute → HVX readback → `dma_wait`.
5. Keep intermediates in VTCM tile format; bypass hexkl readback (hmx_store_acc + vdeal).
6. HMX init once: `bias = mxmem2(scales)` (scale=1.0 / bias=0) or HMX returns garbage.
7. Measure against the hierarchy above: if data-movement-bound, your DMA isn't overlapping.

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
