# Step 2 · signed int8 activation：+128 偏移 + 列和修正

**Demo 文件**：[`demo07_i8xi8_offset.c`](../../../example/hmx_programming_guide/demo07_i8xi8_offset.c)

## 动机

真实量化模型的 activation 通常是 **signed int8**（范围 `[-128, 127]`），不是
uint8。HMX 只支持 `activation.ub`，所以我们必须**在 CPU 侧先把 a_q 转成 a_u**，
再修正 MAC 的输出。

## 数学变换

```
a_q ∈ [-128, 127],  w_q ∈ [-128, 127]
a_u = a_q + 128                            ∈ [0, 255]

a_q · w_q = (a_u - 128) · w_q
          = a_u · w_q - 128 · w_q

Σ_k a_q[k] · w_q[k] = Σ_k a_u[k] · w_q[k] - 128 · Σ_k w_q[k]
                    = acc_hmx[i,j] - 128 · ColSumW[j]
```

HMX 负责算 `acc_hmx = Σ_k a_u · w_q`，CPU 侧减 `128 · ColSumW[j]`。

- `ColSumW[j] = Σ_{k=0..K-1} W[k][j]`，一列一个 int32，在 CPU 侧算。
- 修正只在 unpack 之后的标量域做一次加减，一点也不贵。

## 数值范围

Step 2 的范围选择：`A_i8 ∈ [-8, 7]`（→ a_u ∈ [120, 135]）, `W_i8 ∈ [-8, 7]`。

- `acc_hmx max = 135 · 8 · 32 = 34560` < 2^16 ✓（不用 dual-scale）
- `ColSumW[j]` ∈ `[-256, 224]`（每 col 32 个 [-8,7] 之和）
- `corr = acc_hmx - 128·ColSumW` ∈ 差不多 `[-33 000, 40 000]`，**超过 int16 范围**

所以 unpack 出来的 int16 加上 CPU 修正后会超 int16 界，但仍在 int32 内，不是 bug。

## 代码段

### CPU 侧 prep

```c
/* 1. 生成 signed int8 输入 */
for (int i = 0; i < M; i++)
    for (int k = 0; k < K; k++)
        A_i8[i][k] = ...;    /* [-8, 7] */

/* 2. 偏移成 uint8 */
for (int i = 0; i < M; i++)
    for (int k = 0; k < K; k++)
        A_u8[i][k] = (uint8_t)(A_i8[i][k] + 128);

/* 3. 算 col_sum_w */
for (int j = 0; j < N; j++) {
    col_sum_w[j] = 0;
    for (int k = 0; k < K; k++)
        col_sum_w[j] += W_i8[k][j];
}
```

### HMX MAC

用 pack_act_u8 + pack_wt_b（把 A_u8 和 W_i8 装进 tile）。MAC 序列不变：

```c
asm volatile("mxclracc" ::: "memory");
asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");
asm volatile("{ activation.ub = mxmem(%0,%1)\n"
             "  weight.b      = mxmem(%2,%3) }"
             :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
             :: "r"(out), "r"(0) : "memory");
```

### CPU 侧 combine

```c
unpack_out(out, Chmx);              /* int32 Chmx[M][N] */

for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
        Cfinal[i][j] = Chmx[i][j] - 128 * col_sum_w[j];   /* 修正 */
```

## Bit-exact 验证

demo07 的 oracle 直接算 `Σ A_i8 · W_i8`，和 `Cfinal` 比较，应全对。

```
--- demo07: i8 x i8 via +128 offset + col_sum_w correction ---
  0 mismatches / 1024 cells
  [PASS] demo07
```

## 为什么这种分解是通用的

这个技巧叫 **zero-point offset 修正**。对任意"非零 zero-point 的量化"：
- `a_q = a_u - zp_a`
- `w_q = w_u - zp_w`
- `a_q · w_q = (a_u - zp_a)(w_u - zp_w)`
                = `a_u · w_u - zp_a · w_u - zp_w · a_u + zp_a · zp_w`

展开后有 4 项，其中 `a_u · w_u` 走 HMX MAC，其它三项都是 CPU 侧 scalar 修正
（一列和、一行和、一常数），用 `n_K = K` 乘后加回。

本 tutorial 因为 weight 是 signed int4（zp_w = 0）所以只有前两项，简化成一项修正。

## 踩坑提醒

1. **col_sum_w 用错精度**：一定 int32，不能 int16（32·127 会溢出 int16）。
2. **A_u8 溢出 uint8**：`A_i8 + 128` 对 A_i8 = -128 是 0，对 127 是 255，都在 uint8 内，
   但如果你用 `int8_t + 128` 先做 int 运算再强转会有符号位问题。用 `(uint8_t)((int)A + 128)`
   更安全。
3. **忘了修正就直接比 C ref**：结果会偏差 128·ColSumW[j]，乍看像"公式错"。

## 下一步

[Step 3 · 切到 weight.n](step3-weight-n-int4.md) —— 把 weight 从 `.b` 换成 `.n`，
tile 字节数减半，packer 也要改。
