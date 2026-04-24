# 14 · HVX 和 HMX 的协作

**TL;DR**：HVX 是 Hexagon 的 **128-byte SIMD 向量单元**，负责 pack/unpack/
correction 这类"HMX 周围的 glue 计算"。本指南每个 demo 都提供两个
路径——**CPU + HMX** 和 **HVX + HMX**——两路都 bit-exact 过 C reference。

## 分工原则

| 任务 | 适合 | 原因 |
|------|------|------|
| MAC（矩阵乘加）| **HMX** | HMX 阵列是专门为此设计的，1024 cell 并行 |
| 大块 fill / zero | **HVX** | 单条 vsplat 一拍写 128 B |
| per-column 修正（-128·ColSumW）| **HVX** | `Q6_Vw_vsub_VwVw` 一拍对 32 个 int32 |
| Activation offset（+128）| **HVX** | `Q6_V_vxor_VV` XOR 0x80 翻转符号位 |
| Tile pack（byte layout 重排）| **scalar u32 写** | HVX `vshuffe` 在 v75 上**产生错 tile 字节**（生产经验，见下） |

## 重要坑：HVX vshuffe 不能用来做 tile pack

本仓库 `example/hmx_matmul_qnn/kernel/hmx_int4_matmul.c:114` 的注释记录：

> "Kept scalar (u32-packed writes): HVX shuffle rewrite was tried but
> produced wrong HMX tile content (vshuffe_b semantics de-interleave
> EVEN bytes only, dropping odd-indexed bytes). Since this function is
> called once per m_tile (amortized <1% of total runtime after P3c),
> HVX is not a win here."

所以本指南的 `hvx_pack_act_u8_32x32` / `hvx_pack_wt_b_32x32` 实际用
**u32 packed writes**（和 CPU 版逻辑一致，仅文件位置在 `hmx_hvx_common.h`
以示整合）。pack 是 <1% 的开销，不值得用 HVX。

**真正给吞吐的 HVX 加速点**：fill/zero（VTCM 初始化）、offset（激活前处理）、
correction（输出后处理）。这些是线性 O(M·K 或 M·N) 的操作，HVX 带来 32×
速度提升的地方。

## 本指南的 HVX helpers（hmx_hvx_common.h）

```c
/* Fill / zero */
void hvx_zero(void *dst, int bytes);
void hvx_fill_u8(void *dst, int bytes, uint8_t v);
void hvx_fill_u16(void *dst, int count, uint16_t v);

/* Tile pack (内部 scalar u32 写，保持 API 形式统一) */
void hvx_pack_act_u8_32x32(uint8_t *tile, const uint8_t *A);
void hvx_pack_act_u8_32xKslice(uint8_t *tile, const uint8_t *A, int row_stride);
void hvx_pack_wt_b_32x32(int8_t *tile, const int8_t *W);
void hvx_pack_wt_n_32x32(uint8_t *tile, const int8_t *W);
void hvx_pack_wt_n_32xKslice(uint8_t *tile, const int8_t *W, int row_stride);

/* 真正的 HVX 向量加速点 */
void hvx_add_i8_plus_128(uint8_t *dst, const int8_t *src, int n_bytes);
void hvx_col_sum_w(int32_t col_sum[32], const int8_t *W);
void hvx_apply_col_sum_correction(int32_t *C, const int32_t *cs, int scale, int rows);
```

## 每个 demo 的 HVX 版在做什么

| demo | CPU 版做的事 | HVX 版**替换的**部分 |
|------|-------------|--------------------|
| demo01–05 | `memset` + for-loop 填 act/wt/bias | `hvx_zero` + `hvx_fill_u*` + `hvx_vstu` splat |
| demo06 (u8·i8) | 同上 + scalar pack | + HVX fill |
| demo07 (i8·i8 offset) | + scalar `+128` + scalar correction | + `hvx_add_i8_plus_128` (XOR 0x80) + `hvx_apply_col_sum_correction` |
| demo08 (i4·i8) | 同 demo07 + int4 pack | 同 demo07 |
| demo09 (K=128) | 4 次 K-slice pack 循环 + correction | 同 demo08 |

HMX MAC 指令两路完全相同。Pack 内部两路也相同（u32 packed writes）。差别
仅在"HMX 前后的 glue"。

## 为什么两路都要跑

- **证明 HVX 加速不改变数值正确性**：bit-exact 相同结果
- **教学**：读者能对比"全 scalar 版"和"HVX 加速版"的代码结构
- **regression 保护**：如果后续改 HVX helper 产生数值偏差，test harness 立刻 FAIL

## HVX 常用 intrinsic 速查

| Intrinsic | 作用 |
|-----------|------|
| `Q6_V_vsplat_R(word)` | 把 32-bit 值 splat 到整个 128-byte 向量 |
| `Q6_V_vzero()` | 全零向量 |
| `Q6_V_vor_VV(a, b)` | vector OR |
| `Q6_V_vand_VV(a, b)` | vector AND |
| `Q6_V_vxor_VV(a, b)` | vector XOR（用于 +128 offset 等等）|
| `Q6_Vb_vadd_VbVb(a, b)` | 128 个 int8 同时加 |
| `Q6_Vh_vadd_VhVh(a, b)` | 64 个 int16 同时加 |
| `Q6_Vw_vadd_VwVw(a, b)` | 32 个 int32 同时加 |
| `Q6_Vw_vsub_VwVw(a, b)` | 32 个 int32 同时减（correction 用）|
| `Q6_Vb_vshuffe_VbVb(a, b)` | **de-interleave 偶数字节**（不是 interleave，易踩坑！）|

## 编程模式：CPU / HMX / HVX 混合

典型的"全路径"kernel 结构：

```c
/* 1. HVX: 数据预处理 */
hvx_add_i8_plus_128(A_u8, A_i8, M * K);  // HVX: 激活偏移

/* 2. scalar: tile pack (pack 是 layout-specific，不用 HVX) */
pack_act(act_tile, A_u8);
pack_wt_n(wt_tile, W_i4);

/* 3. HVX: 其他 fill */
hvx_fill_u16(bias, 128, 0x4000);

/* 4. HMX: 真正的 MAC */
asm volatile("mxclracc" ::: "memory");
asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");
asm volatile("{ activation.ub = mxmem(%0,%1); weight.n = mxmem(%2,%3) }" ...);
asm volatile("mxmem(%0,%1):after.uh = acc:2x1" ...);

/* 5. scalar: unpack + CPU: col_sum_w */
unpack_cpu(out, C_hmx);
for (int j = 0; j < N; j++)
    col_sum_w[j] = sum_k W_i4[k][j];

/* 6. HVX: correction */
hvx_apply_col_sum_correction(C_hmx, col_sum_w, 128, M);
```

## 参考

- HVX 完整 ISA：v75 HVX Programmer's Reference Manual
- 本仓库 HVX 使用示例：`example/hmx_matmul_qnn/kernel/hmx_int4_matmul.c`
- 本指南 helpers 源码：`example/hmx_programming_guide/hmx_hvx_common.h`
