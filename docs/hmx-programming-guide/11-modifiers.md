# 11 · 修饰符大全

**TL;DR**：HMX 指令带一串修饰符（`:above :dilate :deep :drop :single :cm :sat :retain
:pos :2x :2x1 :2x2`）。matmul 能用上的不多；大多数是给 conv / 浮点 pipeline 用的。
这一章给每个修饰符 **一句话说明 + 用场景 + 证据级别**。

## 一览表

| 修饰符 | 属于哪族 | 什么作用 | matmul 用？ | 级别 |
|--------|---------|----------|:----------:|:----:|
| `:above` | activation load | "读上方空间位置"（conv）| ✗ | F (限于 conv) |
| `:dilate` | activation / weight load | Conv2D dilate=1 标记 | ✗ | F |
| `:deep` | activation / weight load | "深通道" pipelining | ✗ | P |
| `:drop` | activation / weight load | 丢最后一线 | ✗ | P |
| `:single` | activation / weight load | Conv 单空间点 | ✗ | F |
| `:cm` | activation load / convert | compression-mode | ✗ | P |
| `:sat` | convert | 饱和到 u16/u8 | ✓ | F |
| `:retain` | convert | 保留 acc 给下次读 | ✓ (dual-scale) | F |
| `:pos` | convert (.hf) | ReLU 融合（正数通过）| 条件 | F |
| `:2x` | weight.n | int4 双倍宽度读 | **不推荐** | F 功能 + P 语义 |
| `:2x1` | convert | 标准 output geometry | ✓ | F |
| `:2x2` | convert | 2×2 output geometry | ✗ | P |
| `:after` | convert / weight | 在 MAC 之后（默认）| ✓ | F |
| `:before` | convert | 在下次 MAC 之前 | ✗ | F (pipeline 用) |

## matmul 必用的修饰符

### `:after` (convert 默认)

`mxmem(...):after.uh = acc` —— 等 MAC 完成再 convert。**matmul 所有 convert 都该
带 `:after`**。

### `:2x1` (convert 标准 geometry)

`mxmem(...):after.uh = acc:2x1` —— 输出 32×32×u16 的标准布局 2 KiB。matmul 必带。

### `:sat` (convert 饱和)

`mxmem(...):after:sat.uh = acc:2x1` —— int acc 超 [0, 2^16) 时 clamp 到边界。
单次 convert 读 acc 时想防 wrap 用 `:sat`；dual-scale 读时 **不要** `:sat`
（会破坏低位信息）。demo02 展示了两种差别。

### `:retain` (convert 保留 acc)

`mxmem(...):after:retain.uh = acc:2x1` —— convert 后 acc 不被消耗。配合 dual-scale
的第一次读。demo03 和本仓库 int16 kernel 大量用。

## matmul 可能用的修饰符

### `:pos` (convert 融合 ReLU)

浮点路径专属（`.hf` 输出），把 "负数截断为 0" 融进 convert，省一遍 HVX
`vmax(v, 0)`。

```c
/* 不带 ReLU: */
asm volatile("mxmem(%0,%1):after.hf = acc" ...);
/* 带 ReLU: */
asm volatile("mxmem(%0,%1):after:pos.hf = acc" ...);
```

matmul 后接 ReLU 的场景适用。单独 matmul 不用。

## matmul 不用的修饰符

### `:single` / `:dilate` / `:deep` / `:drop` / `:above` / `:cm`

这些都是 **conv 专用**的 tile 遍历策略。matmul 读整 tile 的行为 ≈ `:dilate` 的
特例，但你**不写** `:dilate`，硬件按"无修饰符默认" = 读整 tile 正好就是
matmul 要的。

**你唯一会碰到它们**的场景：
- 反汇编 QNN skel 看 conv kernel 想理解语义
- 自己写 HMX conv kernel 而非 matmul

QNN 典型 pattern（见 `libQnnHtpV75Skel.so::hmx_convhnh_5x5_stride1`）：
```
{ activation.ub:above  = mxmem(r0, r6)    ; 从上一空间位置读 act
  weight.n:dilate      = mxmem(r23, r9) } ; 带 dilate=1
```

### `:before` (convert 早发)

`mxmem(...):before.uh = acc:2x1` —— 在下一条 MAC 开始之前就 convert，让 HMX pipeline
overlap。在吞吐敏感的 conv kernel 里用，matmul 用 `:after` 即可。

## `:2x` —— `weight.n` 独有的修饰 (F 功能 + P 语义)

### ISA 现状

`weight.n` 是唯一有 `:2x` 变体的 weight load：

```
weight.n = mxmem(Rs, Rt):2x
weight.n = mxmem(Rs, Rt):2x:after
weight.n = mxmem(Rs, Rt):2x:dilate
weight.n = mxmem(Rs, Rt):2x:deep
weight.n = mxmem(Rs, Rt):2x:drop
weight.n = mxmem(Rs, Rt):2x:single
```

### 实测观察

1. v75 skel 里用了 81 次（`libQnnHtpV75Skel.so`）
2. cycle：比普通 `.n` 慢 ~16%（`probe_subbyte_device.c` / HAP_perf_get_pcycles）
3. 功能：`byte=0x11, tile=1KiB` 给 `out = 544`（而不是 `.n` 同输入的 32）

544 = 32 × 17，不是 32 的整数倍（64/128 之类）。sim 和硅给一样的 544，说明不是
sim bug。可能是 `:2x` 需要特定的 activation Rt 配置 + tile 几何才能正常工作。

### 结论

**matmul 不要用 `:2x`**。本指南 tutorial 完全不涉及。要用时先参考 QNN 已验证的
Rt / tile 配置，不要自己发明。

## 完整修饰符语法总结

根据 ISA 头观察到的组合规则：

```
activation.<type> = mxmem(Rs, Rt)[<pos>][<cm>]
    pos ∈ { (none), :above, :deep, :dilate, :single }
    cm  ∈ { (none), :cm }
    type ∈ { ub, hf, f8 }

weight.<type> = mxmem(Rs, Rt)[<width>][<pos>][<extra>]
    width ∈ { (none), :2x }           -- 只 .n 支持 :2x
    pos   ∈ { (none), :after, :deep, :dilate, :drop, :single }
    type  ∈ { b, n, c, sc, ubit, sbit, sm, hf, f8 }

mxmem(Rs, Rt):<when>[<retain>][<extra>].<otype> = acc[:<geom>]
    when    ∈ { after, before }
    retain  ∈ { (none), :retain }
    extra   ∈ { (none), :sat, :pos, :cm, :sat:cm, :retain:sat, ... }
    otype   ∈ { uh, ub, hf }
    geom    ∈ { (none), :2x1, :2x2 }  -- uh 和 hf 需要 geom
```

## 参考

- 所有修饰符具体 intrinsic：`hmx_hexagon_protos.h`，按关键词 grep
- `:2x` 异常值 544 的记录：`Agent/hmx_int4_combos_analysis.md` §6
- QNN conv kernel 的修饰符使用：`Agent/hmx_int4_combos_analysis.md` 附录 A

## 下一章

v3 开始：[13 tutorial int4×i8 matmul](13-tutorial-int4xi8-matmul/)。
