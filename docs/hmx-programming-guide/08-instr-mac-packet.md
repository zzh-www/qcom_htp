# 08 · VLIW MAC Packet

**TL;DR**：HMX 的 MAC 是**一个 activation load + 一个 weight load 同 VLIW packet
发射**触发的。分开发会行为未定义。定点 + 定点、浮点 + 浮点各自合法，跨精度非法。

## Packet 语法 (F)

Hexagon 的 VLIW packet 写法：花括号 `{ ... }`。

```c
asm volatile(
    "{ activation.ub = mxmem(%0, %1)\n"
    "  weight.b      = mxmem(%2, %3) }"
    :: "r"(act), "r"(2047),
       "r"(wt),  "r"(2047)
    : "memory");
```

两条指令写在同一对花括号里 = 同 packet。这是 Hexagon 汇编的标准语法，不是
HMX 特色。

## 为什么必须同 packet (F)

HMX 的 MAC 阵列是 **双 operand 同拍可用**才能触发乘加：
- `activation.* = mxmem` 把 128 B act 线喂进 **act FIFO**
- `weight.* = mxmem` 把 128 B wt 线喂进 **wt FIFO**
- 两个 FIFO 在**同一拍**都有数据 → 启动一次 MAC 周期，acc += outer-product(A 线, W 线)

分开两个 packet 发射 → 前一拍只有 act 线进 FIFO，后一拍 act 线已过期但新 packet 又换了 act，HMX 看到 **非时序对齐的 operand**，结果未定义（sim 上可能直接读旧 acc，硅上可能 pipeline stall 或读错数据）。

## 合法 operand 组合 (F)

整数路径：

```
{ activation.ub = mxmem(...); weight.b     = mxmem(...) }  ; u8 × i8
{ activation.ub = mxmem(...); weight.n     = mxmem(...) }  ; u8 × i4
{ activation.ub = mxmem(...); weight.c     = mxmem(...) }  ; u8 × i2
{ activation.ub = mxmem(...); weight.sc    = mxmem(...) }  ; u8 × i2 alt
{ activation.ub = mxmem(...); weight.ubit  = mxmem(...) }  ; u8 × u1
{ activation.ub = mxmem(...); weight.sbit  = mxmem(...) }  ; u8 × s1
{ activation.ub = mxmem(...); weight.sm    = mxmem(...) }  ; u8 × sparse
```

浮点路径：

```
{ activation.hf = mxmem(...); weight.hf    = mxmem(...) }  ; fp16 × fp16
{ activation.f8 = mxmem(...); weight.f8    = mxmem(...) }  ; fp8 × fp8
```

混合（稀少，查 QNN）：
- `activation.ub × weight.f8` —— ISA 允许，QNN 未用，语义推测为 "fp8 weight 解成 int8 等价"

## 非法组合（报错或行为未定义）

```
{ activation.ub = ...; weight.hf = ... }   ; int × fp 跨路径
{ activation.hf = ...; weight.b  = ... }   ; fp × int
{ weight.b = ...;      weight.n = ... }    ; 两个 weight 无 activation
{ activation.ub = ...; activation.ub = ... } ; 两个 activation
```

sim 表现：hexagon-sim 可能直接 assert，或者放过去让你读到全 0 / 脏数据。
硅级行为：未定义。**不要尝试**。

## Packet 里可以同时放什么？(F)

Hexagon VLIW packet 最多 4 条指令占 4 个 slot。HMX 的 `mxmem` 都是 slot-0（`activation.*`）
和 slot-0（`weight.*`），但它们有"伴生 packet"特批——两条 slot-0 允许同 packet。

**推论**：你可以在同一 packet 里再放 1-2 条 HVX 或 scalar 指令占 slot-1/2/3：

```c
asm volatile(
    "{ activation.ub = mxmem(%0, %1)\n"
    "  weight.b      = mxmem(%2, %3)\n"
    "  r4            = add(r4, #4) }"        /* scalar 指令同 packet */
    :: "r"(act), "r"(2047),
       "r"(wt),  "r"(2047)
    : "memory", "r4");
```

对 matmul 优化有用：pack 下一 K-tile 的地址算术可以藏进 MAC packet 的
scalar slot，无代价。但 v1/v2 的 demo 里用不到。

## `mxclracc` / `bias = mxmem` / `convert` 能不能和 MAC 同 packet？

- **不能和 MAC 同 packet**：`mxclracc` 是另一条独立的 slot-0 指令。
- `bias = mxmem` 同理——独立 packet。
- `mxmem(...):after.uh = acc:2x1`（convert）也独立。

HMX kernel 的典型序列：

```
packet 1:  mxclracc
packet 2:  bias = mxmem(...)
packet 3:  { activation.*; weight.* }    ← MAC
packet 4:  mxmem(...):after.* = acc
```

如果要做多次 MAC 累加到同一 acc：

```
packet 1:  mxclracc
packet 2:  bias = mxmem(...)
packet 3:  { activation.*; weight.* }    ← MAC 1
packet 4:  { activation.*; weight.* }    ← MAC 2 (K 另一段)
packet 5:  { activation.*; weight.* }    ← MAC 3
...
packet N:  mxmem(...):after.* = acc
```

每条 MAC packet 启动一个 multi-cycle HMX task，但 Hexagon 核立刻继续发下一
packet（credit-driven，见 `Agent/hmx_u8xi8_matmul_layers.md` §L4）。累积的 MAC
都会落到同一个 acc，直到 convert 时统一读回。

## 踩坑提醒

1. **花括号写错位置**：`asm volatile("mxclracc { ... }")`——把 mxclracc 塞进 packet，
   packet 内包含了非 slot-0 合规的指令，编译会报非法。
2. **两条 asm statement 中间有 C 代码**：
   ```c
   asm volatile("activation.ub = ..." ...);
   int tmp = something();  /* 这里插了一条! */
   asm volatile("weight.b = ..." ...);
   ```
   编译器把它们拆成两个 packet —— activation 和 weight 分拍发射 = MAC 未定义。
   **必须写在同一个 `asm volatile("{...}")` 里**。
3. **Memory clobber**：`:memory` 必须加，否则编译器可能把 VTCM 写重排到 mxmem load 之后。

## Debug 技巧

验证 packet 是否真的同发射：用 `hexagon-llvm-objdump` 反汇编 ELF，看花括号对齐：

```sh
hexagon-llvm-objdump -d --mcpu=hexagonv75 --mattr=+hmxv75 demo01_hello_hmx \
  | grep -B 1 -A 1 "activation.ub\|weight.b"
```

正确的 packet 在反汇编里会连续两行，第二行行尾有 `}` 闭合：

```
1234:  01 02 03 04   activation.ub = mxmem(r0, r6)
1238:  05 06 07 08   weight.b      = mxmem(r8, r9) } 
```

如果第一行末尾也有 `}`，说明编译器把两条拆成了独立 packet。

## 参考

- Hexagon VLIW packet 规则：v75 Programmer's Reference Manual Ch 3.3
- HMX 伴生 packet 规则：ISA 头 `hmx_hexagon_protos.h` 的 "Execution Slots: SLOT0" 注释
- 反汇编工具：`tools/hexagon-sdk/.../Tools/bin/hexagon-llvm-objdump`

## 下一章

[09 convert](09-instr-convert.md) —— `mxmem(...)=acc` 全家族 + dual-scale readback。
