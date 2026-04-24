# Step 5 · 扩到 K > 32：K-accumulate + dual-scale readback

**Demo 文件**：[`demo09_i4xi8_fullK.c`](../../../example/hmx_programming_guide/demo09_i4xi8_fullK.c)

## 两个新概念

K=32 一次 MAC 解决。K > 32 时要：

1. **K-accumulate**：在**同一 acc**里连续发 `K/32` 条 MAC packet，不清 acc
2. **dual-scale readback**：K 大时 acc 超 2^16，单次 convert 会 wrap

## K-accumulate 结构

```
mxclracc                                  ← 清一次
bias = mxmem(bias_lo)                     ← 配 scale
for k_tile = 0, 32, 64, ...:              ← 每 32 K 一次
    pack_act (A 的 K slice [k_tile, k_tile+32))
    pack_wt_n (W 的对应 slice)
    { activation.ub; weight.n }           ← MAC: acc += 这段的 outer product
convert (readback acc)
```

每条 MAC packet 启动一个 multi-cycle HMX task，核继续发下一 packet 不等（credit driven）。
硬件在后台消费累加。

## Dual-scale readback 原理

当 `acc max > 2^16`：
- `.uh:2x1` 只读低 16 bit，wrap 了
- 解决：两次 convert，用不同 bias scale，各读 acc 的一段，CPU 拼回

```c
/* bias_lo: f16(2.0) -> scale 1.0 -> OUT_LO[j] = acc[j] mod 2^16 */
for (int i = 0; i < 128; i++) bias_lo[i] = 0x4000;
/* bias_hi: f16(2^-7) -> scale 2^-8 -> OUT_HI[j] = floor(acc[j]/256) mod 2^16 */
for (int i = 0; i < 128; i++) bias_hi[i] = 0x2000;

/* 第一次: 低 bit, :retain 保留 acc */
asm volatile("mxmem(%0,%1):after:retain.uh = acc:2x1"
             :: "r"(out_lo), "r"(0) : "memory");

/* 第二次: 换 bias 读高 bit */
asm volatile("bias = mxmem(%0)" :: "r"(bias_hi) : "memory");
asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
             :: "r"(out_hi), "r"(0) : "memory");

/* CPU 拼 int32: 高 16 bit 当 int16 (自动带符号)，左移 8；低 8 bit 来自 OUT_LO */
int32_t acc = ((int32_t)(int16_t)out_hi[idx] << 8) |
              ((int32_t)out_lo[idx] & 0xFF);
```

### 为什么是 `acc_hi << 8 | acc_lo & 0xFF`？

acc_hi 的含义是 `floor(acc / 256)` 的低 16 bit（但当 int16 解释自动取到负数范围）。
把 acc 拆成 32 bit `AAAABBCCDD`：
- acc_hi = `AAAABBCC`（高 16 bit 的低 16 bit，相当于 bit [8..23] 的 signed 解读）
- acc_lo = `CCDD`（低 16 bit）

但是 `CCDD` 和 `BBCC` 在 bit 8..15 位置冲突：
- acc_lo 的 bit [8..15] 是 `CC`
- acc_hi 的 bit [0..7] 也是 `CC`（因为 acc_hi = acc >> 8 的低 16 bit）

所以拼法是：**acc_hi 取完整 16 bit 左移 8, acc_lo 只取低 8 bit**。
这样：
```
bit [23..8]  来自 acc_hi (包括符号 sign-extended)
bit [7..0]   来自 acc_lo 低 8 bit
```

完美重建 int24（对 acc 大到 ±2^23 以内的都 bit-exact）。实际 acc 范围远小于 2^23，够用。

## Step 5 的数值范围

K=128, A_i8 ∈ [-8, 7], W_i4 ∈ [-8, 7]:
- a_u ∈ [120, 135]
- acc_hmx max = `135 · 8 · 128 = 138240` > 2^16 ✓（dual-scale 有必要）
- acc_hmx < 2^18，int24 够装

## 完整 kernel 段（demo09 核心）

```c
#define FULL_K 128
#define KT 32

/* bias tiles (分开放) */
static uint16_t bias_lo[128] __attribute__((aligned(128)));
static uint16_t bias_hi[128] __attribute__((aligned(128)));

void setup_biases(void) {
    for (int i = 0; i < 128; i++) {
        bias_lo[i] = 0x4000;
        bias_hi[i] = 0x2000;
    }
}

/* 一次 K = FULL_K 的 kernel */
void i4xi8_fullK_kernel(
    const int8_t A_i8[M][FULL_K],
    const int8_t W_i4[FULL_K][N],
    int32_t      C[M][N],
    uint8_t     *vtcm)
{
    uint8_t  *act     = vtcm + 0*2048;
    uint8_t  *wt      = vtcm + 2*2048;
    uint16_t *out_lo  = (uint16_t *)(vtcm + 6*2048);
    uint16_t *out_hi  = (uint16_t *)(vtcm + 8*2048);

    /* CPU prep */
    static uint8_t A_u8[M][FULL_K];
    static int32_t col_sum_w[N];
    for (int i = 0; i < M; i++)
        for (int k = 0; k < FULL_K; k++)
            A_u8[i][k] = (uint8_t)(A_i8[i][k] + 128);
    for (int j = 0; j < N; j++) {
        col_sum_w[j] = 0;
        for (int k = 0; k < FULL_K; k++) col_sum_w[j] += W_i4[k][j];
    }

    /* HMX */
    asm volatile("mxclracc" ::: "memory");
    asm volatile("bias = mxmem(%0)" :: "r"(bias_lo) : "memory");

    for (int kt = 0; kt < FULL_K; kt += KT) {
        pack_act_u8_slice(act, A_u8, kt);
        pack_wt_n_slice(wt, W_i4, kt);
        asm volatile("{ activation.ub = mxmem(%0,%1)\n"
                     "  weight.n      = mxmem(%2,%3) }"
                     :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
    }

    /* Dual-scale */
    asm volatile("mxmem(%0,%1):after:retain.uh = acc:2x1"
                 :: "r"(out_lo), "r"(0) : "memory");
    asm volatile("bias = mxmem(%0)" :: "r"(bias_hi) : "memory");
    asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
                 :: "r"(out_hi), "r"(0) : "memory");

    /* Stitch + correct */
    for (int i = 0; i < M; i++) {
        int pr = i & 15, st = i >> 4;
        for (int j = 0; j < N; j++) {
            int idx = pr * 64 + 2 * j + st;
            int32_t acc = ((int32_t)(int16_t)out_hi[idx] << 8) |
                          ((int32_t)out_lo[idx] & 0xFF);
            C[i][j] = acc - 128 * col_sum_w[j];
        }
    }
}
```

## HMX packet 数量

K=128（4 tile）：
- 1 × `mxclracc`
- 1 × `bias = mxmem(bias_lo)`
- 4 × `{ activation.ub; weight.n }`（每 K tile 一条 MAC）
- 1 × `:after:retain.uh` convert
- 1 × `bias = mxmem(bias_hi)`
- 1 × `:after.uh` convert

共 **9 条 VLIW packet**（其中 MAC packet 4 条，每条含 2 条 slot-0 伴生指令）。

## 何时切 dual-scale？

判断公式：`max |acc_hmx| = max|a_u| × max|w_q| × K > 2^15`

K 的阈值大约是：
- u8×i8 全范围：K > 2 就溢出 → 永远 dual-scale
- 受限输入（如本教程的 [-8,7] range）：K ≤ 64 一般单次 convert 够

用 `:sat` 替代 dual-scale 在很多场景也能（只要下游不需要精确 int32），但
本教程坚持 **bit-exact**。

## demo09 输出

```
--- demo09: int4×int8 full K=128 with dual-scale readback ---
  0 mismatches / 1024 cells
  [PASS] demo09
```

## 扩展思路

- **M > 32 / N > 32**：外层再加一层 (m_tile, n_tile) 循环，每 tile 一次上述 kernel。
- **Weight pre-pack**：如果 kernel 要反复对同一 W 做多次 matmul，把 pack_wt_n 的
  结果缓存到 VTCM，避免重复 pack（`Agent/int4_matmul_optimization_log.md` iter 3 讲过）。
- **Activation pre-pack**：同理，A 如果要配多个 W，pack 一次缓存。
- **`weight.n:2x`**：理论上再 2× 吞吐，但当前 sim/硅语义未确认，不推荐。

## 本教程到此为止

到这里你已经实现了一个完整、可扩 K 的、bit-exact 的 int4×int8 matmul kernel。
下一步可能的方向：
- 接到 QNN 做 OpPackage（`example/hmx_matmul_qnn/` 已有工程）
- 真机测吞吐（`example/hmx_matmul_device/` harness）
- 性能优化迭代（`Agent/int4_matmul_optimization_log.md`）

## 回到主指南

[README](../README.md) · [章节地图](../README.md#章节)
