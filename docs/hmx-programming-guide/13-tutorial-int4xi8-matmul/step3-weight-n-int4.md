# Step 3 · 切到 weight.n (int4)

**Demo 文件**：[`demo08_i4xi8_tile.c`](../../../example/hmx_programming_guide/demo08_i4xi8_tile.c)
**Probe 文件**：[`demo_probe_n.c`](../../../example/hmx_programming_guide/demo_probe_n.c)

## 动机

把 weight 从 int8 换成 int4 主要收益是 **VTCM 带宽 / tile 字节减半**。对 v75 这种
带宽 bound 的 HMX matmul kernel 直接给 2× 实际速度（见 `Agent/int4_matmul_optimization_log.md`）。

ISA 层面用 `weight.n = mxmem(...)` 就行；坑在 **tile 字节布局不同于 int8**。

## int4 tile 字节布局（probe 反推）

跑 `demo_probe_n` 可以自验证这个 layout：

```
byte_off(K, col) = 128·(K>>3) + 4·col + ((K>>1) & 3)
hi_nibble        = K & 1     (K=偶 -> 低 nibble, K=奇 -> 高 nibble)
```

直观理解：
- 每 **128 B 一条线** 仍然是 32 cols × 4 bytes-per-col (同 int8)
- 但**每条线现在装 8 K**（不是 4），因为每 byte 装 2 个 nibble
- 512 B 总 tile = 4 条 128-B 线 × 8 K = K=32 ✓

## Probe 的结果

跑 `demo_probe_n` 看到的前几行：

```
--- probe_n_layout: single hot nibble in weight.n tile ---
byte  nib  col  cells  val
   0  lo     0     32    7
   0  hi     0     32    7
   1  lo     0     32    7
   1  hi     0     32    7
   2  lo     0     32    7
   2  hi     0     32    7
   3  lo     0     32    7
   3  hi     0     32    7
   4  lo     1     32    7
   ...
```

- byte 0..3 都映射到 col=0（4 bytes = 一 col 的一个 4-byte 段）
- byte 4..7 → col=1
- ...
- byte 124..127 → col=31（一条线完）

byte 0..3 内部：每 byte 的 lo/hi nibble 对应 K 不同值。具体分配可以通过 K-specific
probe（A=1 except 某 K）再探，但通过"tile 总共 32 K × 32 col = 512 nibble × 2 =
1024 nibble"的计数约束 + 跨线 step 已经足以确定公式 `byte_off(K, col) = 128·(K>>3) + 4·col + ((K>>1)&3)`。

这个公式在 demo08 里**直接端到端 bit-exact 验证过**，所以保证正确（sim 上）。

## Pack 代码

```c
/* int4 weight packer: W[k][j] is signed int4 stored in int8 container */
static void pack_wt_n(uint8_t *tile, const int8_t W[K][N]) {
    memset(tile, 0, 512);   /* 只 512 B */
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++) {
            int byte_off = 128 * (k >> 3) + 4 * j + ((k >> 1) & 3);
            int hi = k & 1;
            uint8_t nib = (uint8_t)(W[k][j] & 0x0F);   /* 低 4 bit，其余舍 */
            if (hi) tile[byte_off] = (uint8_t)((tile[byte_off] & 0x0F) | (nib << 4));
            else    tile[byte_off] = (uint8_t)((tile[byte_off] & 0xF0) | (nib & 0x0F));
        }
}
```

关键点：
- `W[k][j] & 0x0F` 把 signed int4 的两补码保留（`-1 → 0x0F`, `-8 → 0x08`, `+7 → 0x07`）
- 读回时 HMX 自动把 nibble 作 signed int4 解释
- tile 只 alloc 512 B

## HMX MAC（indentical 到 step 2，只换 weight type）

```c
asm volatile("{ activation.ub = mxmem(%0,%1)\n"
             "  weight.n      = mxmem(%2,%3) }"   /* 唯一变动: .b -> .n */
             :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
```

## 数值范围

- `A_i8 ∈ [-8, 7]` → `a_u ∈ [120, 135]`
- `W_i4 ∈ [-8, 7]`
- acc_hmx max = `135 · 8 · 32 = 34560 < 2^16` ✓ 单次 convert 够

修正项仍然是 `-128 · ColSumW[j]`，其中 `ColSumW[j] = Σ_k W_i4[k][j]`（就是 int4 list 的和，
值范围 [-256, 224]）。

## demo08 完整输出

```
--- demo08: int4×int8 32³ bit-exact ---
  0 mismatches / 1024 cells
  [PASS] demo08
```

## 踩坑

1. **忘了 `& 0x0F`**：直接写 `W[k][j] << 4` 会把负 int8 的高 bit 带进来，变成 8-bit 值。
2. **tile 只 alloc 512 B 但 memset 1024 B**：越界踩到下一个 VTCM 分区。
3. **忘把 tile memset 成 0 再 OR**：残留 nibble 会污染结果。
4. **把 int4 填 W ∈ [-16, 15] 范围**：int4 范围 `[-8, 7]`（4-bit 两补），超了会被截断
   丢位。oracle 也得限制输入在 [-8, 7]。

## `weight.n:2x` 为什么不用？

ISA 有 `weight.n = mxmem(...):2x` 变体，理论上吞吐 2×。但：

- sim 和硅上都给异常值（`byte=0x11` → out=544，不是 32 或 64）
- 比普通 `.n` 慢 16%（VTCM 带宽更紧）
- 语义没完全解码

本 tutorial 不用。生产 kernel 也不推荐。详见 ch11。

## 下一步

[Step 4 · 把 step 2+3 合一](step4-full-int4xi8-tile.md) —— 整合完整 kernel 源码 + 结构图。
