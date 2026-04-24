# Step 1 · u8×i8 最小 kernel

**Demo 文件**：[`example/hmx_programming_guide/demo06_u8xi8_tile.c`](../../../example/hmx_programming_guide/demo06_u8xi8_tile.c)

## 目标

写一个 **32×32×32 的 u8·i8 matmul**，跑 hexagon-sim 结果和 C 参考 bit-exact。
这是所有后续步骤的骨架：pack → MAC → unpack → verify。

## 数学

```
A ∈ u8^{32×32},  w ∈ i8^{32×32}
C[i][j] = Σ_{k=0..31} A[i][k] · W[k][j]    (int32 accumulate)
```

## 数值范围控制

K=32 + u8·i8 的 acc 最大值 = `32 · 255 · 127 ≈ 2^20`，超过 16 bit → 单次 convert 会 wrap。

**工程办法**：把输入限制到 `A ∈ [0, 16), W ∈ [-8, 7]`。acc max = `32 · 16 · 8 = 4096 < 2^16`，
**单次 convert `.uh:2x1` + bias=0x4000 可直接读回 int16**。

这是教学简化——后续 step 5 会解除这个约束（用 dual-scale）。

## Pack / Unpack 代码

pack 函数把 logical `A[i][k]` / `W[k][j]` 映射到 VTCM tile 字节。布局公式和
ch03 cheatsheet 一样：

```c
/* Pack activation: A[ir][k] -> act tile byte */
static void pack_act(uint8_t *tile, const uint8_t A[32][32]) {
    memset(tile, 0, 2048);
    for (int ir = 0; ir < 32; ir++) {
        int phys_row = ir & 15;
        int stream   = ir >> 4;
        int byte_off = stream ? 3 : 1;
        for (int k = 0; k < 32; k++)
            tile[128 * phys_row + 4 * k + byte_off] = A[ir][k];
    }
}

/* Pack weight: W[k][j] -> wt tile byte */
static void pack_wt(int8_t *tile, const int8_t W[32][32]) {
    memset(tile, 0, 1024);
    for (int k = 0; k < 32; k++)
        for (int j = 0; j < 32; j++)
            tile[128 * (k >> 2) + 4 * j + (k & 3)] = W[k][j];
}

/* Unpack output: out_u16 -> C[i][j] */
static void unpack_out(const uint16_t *out, int16_t C[32][32]) {
    for (int i = 0; i < 32; i++) {
        int phys_row = i & 15;
        int stream   = i >> 4;
        for (int j = 0; j < 32; j++)
            C[i][j] = (int16_t)out[phys_row * 64 + 2 * j + stream];
    }
}
```

## HMX kernel 段

和 demo01 一样的 5 条指令：

```c
asm volatile("mxclracc" ::: "memory");
asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");       /* 0x4000 identity */
asm volatile("{ activation.ub = mxmem(%0,%1)\n"
             "  weight.b      = mxmem(%2,%3) }"
             :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
             :: "r"(out), "r"(0) : "memory");
```

## C reference (oracle)

```c
for (int i = 0; i < 32; i++)
    for (int j = 0; j < 32; j++) {
        int32_t s = 0;
        for (int k = 0; k < 32; k++)
            s += (int32_t)A[i][k] * (int32_t)W[k][j];
        Cref[i][j] = s;
    }
```

## Bit-exact 验证

跑 demo06，期望：

```
--- demo06: u8 x i8 bit-exact 32x32x32 tile ---
  0 mismatches / 1024 cells
  [PASS] demo06
```

全部 1024 个 cell 都 `hmx[i][j] == ref[i][j]`。

## 你学到了什么

1. HMX u8·i8 32³ 的完整 kernel 骨架
2. pack / unpack 的具体实现（不只是公式）
3. 如何用 C oracle 做 bit-exact 验证

## 下一步

[Step 2 · signed int8 activation](step2-signed-i8-activation.md) —— 把 A 从 uint8
换到 signed int8（加 +128 偏移 + 列和修正）。
