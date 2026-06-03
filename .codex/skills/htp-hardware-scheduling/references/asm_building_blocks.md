# HTP scheduling — concrete building blocks (copy-paste idioms)

Verified on v75 (Snapdragon 8 Gen 3). Source: htp-ops-lib / llama.cpp Hexagon backend, mirrored in
`docs/hexagon-tutorial/hmx-tutorial/ch04-vtcm-memory` (DMA) and `ch05-hmx/src/exp5_standalone_asm.c` (HMX).

## HMX f16 — the 5 ASM ops (no hexkl in the hot path)
```c
static inline void hmx_clear_acc(void){ asm volatile("mxclracc.hf":::"memory"); }
static inline void hmx_set_scales(const void*s){ asm volatile("bias = mxmem2(%0)"::"r"(s):"memory"); } // scale=1.0(0x3C00 f16)/bias=0, ONCE after lock — else garbage
static inline void hmx_load_tiles(const void*act,const void*wt,uint32_t n){ // :deep MACs up to 32 tile pairs in one instr
  uint32_t lim=n*2048-1;
  asm volatile("{ activation.hf = mxmem(%0,%1):deep\n  weight.hf = mxmem(%2,%3) }\n"
               ::"r"(act),"r"(lim),"r"(wt),"r"(lim):"memory"); }
static inline void hmx_store_acc(_Float16*out){ // acc -> f16 AH tile in VTCM (arg 2 = f16 format)
  asm volatile("cvt.hf = acc(%0)\n  mxmem(%1,%2) = cvt\n"::"r"(2),"r"(out),"r"(0):"memory"); }
```
- Tile = 32×32, 2048 bytes, **2-row interleaved**. On v75 f16 the activation (AH) and weight (WH) tile
  bytes are IDENTICAL → an `hmx_store_acc` output is directly reusable as the next matmul's input.
- Per output tile: `mxclracc` → one `:deep` load of K (=k/32) tile pairs → `hmx_store_acc`. WH tiles are
  stored column-major (tile_idx = ct*K + kt) because `mxmem:deep` wants same-column tiles contiguous.

## Tile <-> row-major (HVX vshuff / vdeal, granularity -2 = halfword)
```c
// RM -> AH/WH tile (32 rows): interleave row pairs
HVX_VectorPair vp = Q6_W_vshuff_VVR(v_row_odd, v_row_even, -2);  out[rr] = Q6_V_lo_W(vp);
// AH tile -> RM: deinterleave
HVX_VectorPair d = Q6_W_vdeal_VVR(Q6_V_vzero(), tile[rr], -2);   v_even=Q6_V_lo_W(d); v_odd=Q6_V_hi_W(d);
```
Readback rule: use this `hmx_store_acc + vdeal` path (VTCM→VTCM, ~24× faster), NOT hexkl acc_read/ah_to_rm.

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
