# Step 4 · 完整 int4×int8 32³ tile：把 step 2+3 合起来

**Demo 文件**：[`demo08_i4xi8_tile.c`](../../../example/hmx_programming_guide/demo08_i4xi8_tile.c)

（这和 step 3 的 demo 是同一份。本章是对整个 kernel 的导读与结构图。）

## 完整数据流图

```
Host (CPU)                                 HMX
┌─────────────────────┐                   ┌──────────────────┐
│ A_i8 [M][K]         │                   │                  │
│ W_i4 [K][N]         │                   │                  │
├─────────────────────┤                   │                  │
│ A_u8 = A_i8 + 128   │                   │                  │
│ ColSumW[j] = ΣW[k,j] │                   │                  │
├─────────────────────┤                   │                  │
│ pack_act(A_u8)→act  │ ──write VTCM──→   │ act tile (2KiB)  │
│ pack_wt_n(W_i4)→wt  │ ──write VTCM──→   │ wt tile (512B)   │
│ bias[128]=0x4000    │ ──write VTCM──→   │ bias (256B)      │
└─────────────────────┘                   └──────────────────┘
           │                                       │
           ▼                                       │
    mxclracc         ────────────────────→       清 int32 acc
    bias=mxmem(bias) ────────────────────→       载 bias
    { act.ub; wt.n } ────────────────────→       MAC: acc += a_u·w_q
    mxmem(out):after.uh = acc:2x1 ────────→      convert + 写 out
                                                   │
           ┌─────────────────────┐ ←──read VTCM──  │ out u16 tile
           │ unpack → Chmx[M][N] │
           │ Cfinal = Chmx - 128·ColSumW │  (每 col 一次)
           └─────────────────────┘
                   │
                   ▼
           C[M][N] bit-exact ref
```

## 完整 kernel 源码（demo08 精简版）

```c
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_vecaccess.h>
#include <h2_mxaccess.h>
#include <hexagon_types.h>

#define M 32
#define N 32
#define K 32

/* ---- Pack helpers ---- */
static void pack_act(uint8_t *tile, const uint8_t A[M][K]) {
    memset(tile, 0, 2048);
    for (int ir = 0; ir < M; ir++) {
        int pr = ir & 15, st = ir >> 4, off = st ? 3 : 1;
        for (int k = 0; k < K; k++)
            tile[128 * pr + 4 * k + off] = A[ir][k];
    }
}
static void pack_wt_n(uint8_t *tile, const int8_t W[K][N]) {
    memset(tile, 0, 512);
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++) {
            int off = 128 * (k >> 3) + 4 * j + ((k >> 1) & 3);
            int hi = k & 1;
            uint8_t nib = (uint8_t)(W[k][j] & 0x0F);
            if (hi) tile[off] = (tile[off] & 0x0F) | (nib << 4);
            else    tile[off] = (tile[off] & 0xF0) | (nib & 0x0F);
        }
}
static void unpack_out(const uint16_t *out, int32_t C[M][N]) {
    for (int i = 0; i < M; i++) {
        int pr = i & 15, st = i >> 4;
        for (int j = 0; j < N; j++)
            C[i][j] = (int32_t)(int16_t)out[pr * 64 + 2 * j + st];
    }
}

/* ---- Main kernel ---- */
void i4xi8_matmul_32_tile(
    const int8_t A_i8[M][K],
    const int8_t W_i4[K][N],   /* 值限制在 [-8, 7] */
    int32_t      C[M][N],
    uint8_t     *vtcm_base)
{
    uint8_t  *act  = vtcm_base + 0 * 2048;
    uint8_t  *wt   = vtcm_base + 2 * 2048;
    uint16_t *bias = (uint16_t *)(vtcm_base + 4 * 2048);
    uint16_t *out  = (uint16_t *)(vtcm_base + 6 * 2048);

    /* CPU prep */
    uint8_t A_u8[M][K];
    int32_t col_sum_w[N];
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            A_u8[i][k] = (uint8_t)(A_i8[i][k] + 128);
    for (int j = 0; j < N; j++) {
        col_sum_w[j] = 0;
        for (int k = 0; k < K; k++) col_sum_w[j] += W_i4[k][j];
    }

    pack_act(act, A_u8);
    pack_wt_n(wt, W_i4);
    for (int i = 0; i < 128; i++) bias[i] = 0x4000;
    memset(out, 0, 2048);

    /* HMX */
    asm volatile("mxclracc" ::: "memory");
    asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");
    asm volatile("{ activation.ub = mxmem(%0,%1)\n"
                 "  weight.n      = mxmem(%2,%3) }"
                 :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
    asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
                 :: "r"(out), "r"(0) : "memory");

    /* Unpack + correction */
    int32_t Chmx[M][N];
    unpack_out(out, Chmx);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i][j] = Chmx[i][j] - 128 * col_sum_w[j];
}
```

## HMX packet 数量

一个 32³ tile 用了 **4 条 HMX 指令**：
- 1 × `mxclracc`
- 1 × `bias = mxmem`
- 1 × `{ activation.ub; weight.n }`（一个 VLIW packet 里 2 条）
- 1 × convert `mxmem(...):after.uh = acc:2x1`

加起来 4 条 VLIW packet（其中 MAC packet 含 2 条 slot-0 伴生）。

## 为什么 K=32 不用 dual-scale

acc max = `135 · 8 · 32 = 34560 < 65536`。16-bit convert 可以完整装下。
一旦 K ≥ 32 还继续加或者输入范围扩大到超过这个阈值，就得切 dual-scale。step 5
演示。

## 踩坑提醒（合并版）

| 阶段 | 错法 | 症状 |
|------|------|------|
| pack_act | 忘 memset act 到 0 | VTCM 残留污染 |
| pack_wt_n | 忘 `& 0x0F` | 负数 weight 填错 |
| pack_wt_n | tile alloc < 512 B | 踩下一区 |
| HMX | 忘 `bias = mxmem` | out 全 0 |
| combine | 忘 `-128·ColSumW` | 结果差一常数 |
| unpack | 用 `out[i*32+j]` 平铺 | 只 col=0 stream=0 那一组对 |

## 下一步

[Step 5 · K > 32](step5-extend-full-k.md) —— 用 K-accumulate + dual-scale 扩到 K=128。
