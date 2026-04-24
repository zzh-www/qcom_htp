# 13 · Tutorial：`int4 (weight) × int8 (activation)` 完整 matmul kernel

**目标**：从 u8×i8 最小 kernel 出发，逐步推到一个**bit-exact、K 可扩展**的
`int4 × int8` matmul kernel。全部在 hexagon-sim 上 end-to-end 验证通过。

**受众**：做过 demo01–05，已经理解 tile 布局 / bias / MAC packet / convert 的人。

## 总体思路

v75 HMX 的约束：
- **activation 只能是 `.ub`**（uint8），**不能是 signed int8**
- **weight 可以是 `.n`**（signed int4），原生支持
- acc 是 int32，convert 只能读 16 bit

把 int4×int8 matmul 拆成两个变换叠加：

1. **signed int8 activation → uint8 activation**：加 +128，最后减 `128 · ColSumW[j]` 修正
2. **使用 `.n` 直接喂 signed int4 权重**，tile 字节数自动 1/2

数学：
```
a_q (int8) ∈ [-128, 127],  w_q (int4) ∈ [-8, 7]
a_u = a_q + 128                                      // uint8
a_q · w_q = (a_u - 128) · w_q = a_u·w_q - 128·w_q   // 乘开
        = a_u·w_q - 128·ColSumW[j]                   // j 维求和后
```

HMX 算 `a_u·w_q`（MAC in HMX），CPU 侧减 `128·ColSumW[j]`（scalar in C）。

## 教程拆 5 步

每步一个独立 demo，PASS 才进下一步。

### [Step 1：u8×i8 骨架](step1-u8xi8-minimal.md) (demo06)

先跑通"纯 u8·i8"的 32³ matmul bit-exact。确认：
- pack / unpack 公式对
- 5 条 HMX 指令串起来
- C oracle 对比 PASS

**不涉及**：量化、偏移、sub-byte。

### [Step 2：signed int8 activation 靠 +128 偏移](step2-signed-i8-activation.md) (demo07)

在 step 1 基础上加 `a_u = a_q + 128` + CPU 侧 `-128 · ColSumW[j]` 修正。
这是从"无量化"到"真实量化 int8 × int8 matmul"的关键一步。

### [Step 3：切到 weight.n（仍然 32³ tile）](step3-weight-n-int4.md) (demo08)

把 weight 从 `.b` 换到 `.n`：
- tile 字节数变 512 B（int4）
- packer 要重写（每 byte 装 2 个 signed nibble）
- 修正项自动适配（col_sum_w 范围现在是 `[-256, 224]`）

**需先确定 int4 tile 的 (K, col) → byte/nibble 精确映射**。Step 3 用 probe 结果给
出 layout：
```
byte_off(K, col) = 128·(K>>3) + 4·col + ((K>>1) & 3)
hi_nibble = K & 1   (K=偶 用低 nibble, K=奇 用高 nibble)
```

### [Step 4：完整 int4×int8 32³ tile](step4-full-int4xi8-tile.md) (demo08)

其实 step 3 已经是这个了。本节讲 **为什么 K=32 这个尺寸不用 dual-scale**（acc 范围
够小），以及给出完整 kernel 源码的导读。

### [Step 5：扩到 K > 32（K-accumulate + dual-scale）](step5-extend-full-k.md) (demo09)

真实模型的 K 通常远大于 32。做法：
- 在**同一 acc** 里连续发 `K/32` 条 MAC packet（不清 acc）
- 单次 convert 可能溢出 → 用 retain + 两次 convert 重建 int32（dual-scale）

demo09 用 K=128 演示。

## 为什么这样拆？

每一步只引入**一个新概念**，pass 之后下一步不会因为多处出错难以 debug：

| 步骤 | 新概念 |
|------|--------|
| 1 | 首次 bit-exact kernel 本身 |
| 2 | zero-point 偏移修正 |
| 3 | sub-byte weight tile 布局 |
| 4 | 合并 step 2+3 的完整量化 matmul |
| 5 | K-accumulate + dual-scale |

## 关键数值范围

| 步骤 | A 范围 | W 范围 | acc max | dual-scale？|
|------|--------|--------|---------|:-----------:|
| 1 (u8·i8)    | [0, 16)  | [-8, 7] | 4 096  | 否 |
| 2 (i8·i8)    | [-8, 7]  | [-8, 7] | 34 560 | 否 |
| 3/4 (i4·i8)  | [-8, 7]  | [-8, 7] | 34 560 | 否 |
| 5 (i4·i8 K=128) | [-8, 7]  | [-8, 7] | 138 240 | **是** |

Step 5 正好越过 `2^16 = 65 536` 的阈值，所以是引入 dual-scale 的自然时机。

## 完整 kernel 参考

最终 kernel 源码：`example/hmx_programming_guide/demo08_i4xi8_tile.c` + 
`demo09_i4xi8_fullK.c`。

验证命令：
```sh
bash tests/test_hmx_programming_guide.sh
# 期望 demo06..demo09 全 PASS
```

## 不涉及的主题

- **真机验证**：本教程只 sim-only。真机流程见 `example/hmx_matmul_device/`。
- **QNN OpPackage**：如何把这个 kernel 接进 QNN 图，见 `example/hmx_matmul_qnn/`
  （已有工程，其 HMX 路径在硅上崩 err 1003 还没解决）。
- **int4×int16 activation**：比 int4×int8 复杂一层（activation 要拆 hi/lo 两 byte 做 2
  partial），见 `example/hmx_matmul_int16/` 以及 `Agent/hmx_int4_combos_analysis.md` §5.3。
- **性能优化**：pack hot loop、VTCM double-buffer、K-tile prefetch 等，见
  `Agent/int4_matmul_optimization_log.md`。

---

## 下一步

打开 [Step 1](step1-u8xi8-minimal.md) 开始读。
