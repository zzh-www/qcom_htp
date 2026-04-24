# 06 · `activation.*` load

**TL;DR**：HMX 只支持 **3 种激活类型**：`.ub`（uint8）、`.hf`（fp16）、`.f8`（fp8）。
**没有 int4、没有 signed int8、没有 int16 activation**——想用这些必须在 CPU 侧先
偏移/拆分成 uint8 再喂（详见 ch13 tutorial）。

## 三种激活类型 (F)

### `.ub` — uint8 activation

最常用的整数路径。

```c
asm volatile("{ activation.ub = mxmem(%0,%1)\n"
             "  weight.b      = mxmem(%2,%3) }"
             :: "r"(act_ptr), "r"(rt_ctrl),
                "r"(wt_ptr),  "r"(rt_ctrl)
             : "memory");
```

- `Rs` = act tile 起始地址（VTCM），128 B 对齐
- `Rt` = 控制字；`2047` = "读完 2 KiB tile"

**配对 weight**：可以配 `.b / .n / .c / .ubit / .sbit / .sc / .sm / .f8`（整数 + fp8）。

### `.hf` — half-float activation

浮点路径，和 `weight.hf` 配对。

```c
asm volatile("mxclracc.hf" ::: "memory");
asm volatile("{ activation.hf = mxmem(%0,%1)\n"
             "  weight.hf     = mxmem(%2,%3) }"
             :: "r"(act_ptr), "r"(2047),
                "r"(wt_ptr),  "r"(2047)
             : "memory");
asm volatile("mxmem(%0,%1):after.hf = acc"
             :: "r"(out), "r"(0) : "memory");
```

- tile geometry 同 `.ub`（2 KiB），但每 4 字节槽装 2 个 fp16 元素（stream 0/1 各一个）
- 必须用 `mxclracc.hf` 清浮点 acc
- readback 用 `.hf`，不是 `.uh`

### `.f8` — fp8 activation（QNN 未使用）

ISA 里列了，但 QNN 在 v75 上没有 emit 任何 `.f8` 指令（`libQnnHtpV75Skel.so` 中 0 次出现）。
本指南不展开。若要自研 fp8 kernel，参考 `hmx_hexagon_protos.h` 的 `_f8` 系列 intrinsic。

## 修饰符族 (F + P 部分)

HMX 给 `activation.*` load 配了一套"位置/模式"修饰符，决定**在 tile 里怎么跳**。
下面按"本指南已验证 / 推荐用"→"ISA 列但未实测"分组。

### `(无修饰符) / default` (F) — matmul 用

标准 "读整个 tile 一次 MAC" 模式。matmul 只用这个。demo01–05 都是这种。

### `:single` (F) — conv 只做 1 个空间位置

对应 Conv2D 的 kernel-size=1 × stride=1 模式。matmul 用不到。`libQnnHtpV75Skel.so` 里
有用（conv 1×1）。

### `:dilate` (F) — MatMul-as-Conv 路径

Conv2D 带 dilate=1 时走这个。QNN 的 `hmx_convhnh_*` 和 `hmx_convhbh_*` 都带 `:dilate`。
`weight.n:dilate` 是 QNN int4 conv 的标准 pattern。对 matmul 本身不必，但你写 conv kernel 时会碰到。

### `:above` (F，有限实测)

conv 空间卷积里 "这拍读上面一个空间位置的 activation"。`hmx_convhnh_5x5_stride1` 的
第 1 个 MAC 用 `activation.ub:above`。matmul 不用。

### `:deep` (P) — "深"通道流水

ISA 列，具体语义未在本仓库实测。QNN 有 `_deep` 的 intrinsic。matmul 不用。

### `:drop` (P) — 丢弃最后一条线？

猜测是"读 tile 但最后一条 128 B 不进 MAC"。未实测。

### `:cm` (P) — "compression-mode"？

带压缩激活的加载。和 `weight.c / .sc / .sm` 一类相关。未实测。

**生产建议**：matmul **只用无修饰符的 `activation.ub` / `activation.hf`**。其他修饰符
留给 conv 场景，需要时查 `hmx_hexagon_protos.h` + 参考 QNN skel 反汇编。

## Rt 控制字的含义 (F + P)

`activation.* = mxmem(Rs, Rt)` 的 `Rt` 是一个打包了多个字段的 immediate/register 值。
目前我们知道：

- `Rt = 2047`（= `0x7FF` = 11 位全 1）= **"读完整个 tile"**。所有 demo 都用这个。
- 其他值编码 stride / 部分读取 / conv 的空间偏移。具体位字段我们没完全解码。

**实践**：**matmul kernel 里 `Rt = 2047` 永远对**。其他值只在复现 QNN conv pattern 时研究。

## 完整 intrinsic 速查表

全部 `activation.*` intrinsic（从 `hmx_hexagon_protos.h` 提取）：

```
activation.ub  = mxmem(Rs, Rt)                ; Q6_activation_ub_mxmem_RR
activation.ub  = mxmem(Rs, Rt):cm
activation.ub  = mxmem(Rs, Rt):deep
activation.ub  = mxmem(Rs, Rt):deep:cm
activation.ub  = mxmem(Rs, Rt):dilate
activation.ub  = mxmem(Rs, Rt):dilate:cm
activation.ub  = mxmem(Rs, Rt):single
activation.ub  = mxmem(Rs, Rt):single:cm
activation.ub  = mxmem(Rs, Rt):above
activation.ub  = mxmem(Rs, Rt):above:cm
activation.hf  = mxmem(Rs, Rt)                ; Q6_activation_hf_mxmem_RR
activation.hf  = mxmem(Rs, Rt):deep
activation.hf  = mxmem(Rs, Rt):dilate
activation.hf  = mxmem(Rs, Rt):single
activation.hf  = mxmem(Rs, Rt):above
activation.f8  = mxmem(Rs, Rt)                ; Q6_activation_f8_mxmem_RR
activation.f8  = mxmem(Rs, Rt):deep
activation.f8  = mxmem(Rs, Rt):dilate
activation.f8  = mxmem(Rs, Rt):single
activation.f8  = mxmem(Rs, Rt):above
```

> Intrinsic 命名规律：`Q6_activation_<type>_mxmem_RR[_<mod>]`。比如
> `Q6_activation_ub_mxmem_RR_dilate` = `activation.ub = mxmem(Rs, Rt):dilate`。

## 踩坑提醒

1. **跟错 weight 类型搭配**（如 `activation.hf` 配 `weight.b`）→ sim 报非法 packet 或硬件行为未定义。
   固定搭配：`.ub × {整数/f8}`、`.hf × .hf`、`.f8 × .f8`。
2. **`Rs` 不是 VTCM 地址** → 权限错。HMX 的 mxmem 只读 VTCM。
3. **Rs 未 128 B 对齐** → 读 misaligned，sim 报错或读到脏数据。
4. **忘了 pack**（直接把 row-major array 当 tile 喂）→ 全错，见 ch03 pack 错法对照。

## 参考

- QNN int4 conv 的 `activation.ub:above + weight.n:dilate` 用法：`Agent/hmx_int4_combos_analysis.md` 附录 A
- Intrinsic 头：`tools/hexagon-sdk/.../hmx_hexagon_protos.h`，grep `activation`

## 下一章

[07 weight load](07-instr-weight-load.md) —— weight 所有类型 + `.n:2x` 的坑。
