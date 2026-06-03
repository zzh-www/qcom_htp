# HTP scheduling — concrete building blocks (copy-paste idioms)

Verified on v75 (Snapdragon 8 Gen 3). DMA from tutorial ch04-vtcm-memory; dspqueue from ch03-dspqueue.
These are the dtype-agnostic hardware INVOCATION mechanisms. (HMX matmul itself = our existing int8
v73deep kernel in `example/gdn_native/solve_br_op/`; this file is about scheduling the hardware, not the
kernel.)

## HMX enable (invocation mechanism — our route's int8 v73deep mxmem)
HMX requires, in order, on the using thread:
1. power vote: `HAP_power_set_core_corner(TURBO)` + `HAP_power_set` with `type=HAP_power_set_HMX, hmx.power_up=TRUE` (else mxmem HANGS — unpowered).
2. VTCM acquire (all HMX in/out + descriptor tables MUST be in VTCM; mxmem faults on DDR addresses).
3. `compute_resource_hmx_lock(hctx)` — the v1 lock QNN's libQnnHtpV75Skel uses (verified via `nm`).
   NOT: `qurt_hmx_lock` (unlinkable in unsigned PD), `HAP_compute_res_hmx_lock3` (NOT_SUPPORTED),
   `lock4` (no-op). ⚠️ Lock is process-EXCLUSIVE; held across a region it serializes other threads.
Readback: bring mxmem output out of VTCM with HVX `vdeal` (VTCM→VTCM, ~24× faster than hexkl acc_read).

## UDMA — async DDR→VTCM (overlap data movement with compute)
```c
typedef struct { void*next; unsigned len:24,desctype:2,dstcomp:1,srccomp:1,/*...*/; void*src,*dst; }
  __attribute__((aligned(64))) dma_desc_type0_t;
dma_desc_type0_t d = {.next=NULL,.length=n,.desctype=0,.srcbypass=1,.src=ddr,.dst=vtcm};
qurt_mem_cache_clean((qurt_addr_t)ddr,n,QURT_MEM_CACHE_FLUSH,QURT_MEM_DCACHE); // MUST flush DDR src
Q6_dmstart_A((void*)&d);
int st = Q6_R_dmwait();   // 0 = ok  (or Q6_R_dmpoll() to poll without blocking)
```
Ping-pong double-buffer (the 17× lever):
```c
for (int c=0;c<n;c++){ int cur=c&1,nxt=(c+1)&1;
  if (c+1<n) dma_start(scratch[nxt], ddr+(c+1)*sz, sz);   // start next, non-blocking
  hvx_prep(scratch[cur], tiles); hmx_compute(tiles, act, out); // compute current, overlaps DMA
  if (c+1<n) dma_wait(); }
```

## VTCM + HMX acquire/power (bare-metal HAP)
```c
HAP_power_set_core_corner(ctx, TURBO, TURBO, MAX);                 // clock
HAP_power_set: req.type=HAP_power_set_HMX; req.hmx.power_up=TRUE;  // power HMX (else mxmem hangs)
// acquire VTCM and HMX SEPARATELY, each on the using thread (combined acquire fails):
compute_res_attr_t a; HAP_compute_res_attr_init(&a);
HAP_compute_res_attr_set_vtcm_param(&a, BYTES, 1);  // ... acquire -> get_vtcm_ptr
HAP_compute_res_attr_set_hmx_param(&a, 1);          // REQUIRED or acquire returns 0
// then enable mxmem with compute_resource_hmx_lock(hctx) once per slot (the v1 lock QNN uses;
// lock3 NOT_SUPPORTED, lock4 no-op, qurt_hmx_lock unlinkable in unsigned PD).
```

## dspqueue (ARM↔DSP, replaces per-call FastRPC)
- ARM: `rpcmem_alloc` + `rpcmem_to_fd` + `fastrpc_mmap` (zero-copy) → `dspqueue_create` →
  `dspqueue_export` → one FastRPC `start(queue_id)` → `dspqueue_write(req+bufs)` per op.
- DSP: `dspqueue_import` → callback loops `dspqueue_read_noblock` → `switch(req.op)` → kernel →
  `dspqueue_write(rsp)`. Buffer flags: `REF | FLUSH_SENDER | INVALIDATE_RECIPIENT`.
- 61µs/op vs 364µs FastRPC. Keep the DSP loop busy so VTCM isn't reclaimed.
