# 04 · `mxclracc` / `mxclracc.hf` / `mxswapacc`

**TL;DR**：HMX 有两套独立累加器（int32 定点 + xfp 浮点），两条清零指令各管
一套。`mxswapacc` 在两个 int32 acc slot 之间交换，给 conv 的 double-buffer 用。

## `mxclracc` —— 清零 int32 累加器 (F)

**汇编**：`mxclracc`
**Intrinsic**：`Q6_mxclracc()`

**语义**：把 HMX 内部 32×32 个 int32 累加器 cell 全部置为 0。

**必要性**：HMX 的 MAC 是**累加**（`acc += A·W`）——如果上次 kernel 残留的
acc 没清，新 kernel 算出来的数会叠上去。

**调用时机**：每次 MAC 前（如果是全新的累加）。如果你在一个 kernel 内分多条
MAC 累加同一块 acc，**只在第一次**清。

```c
asm volatile("mxclracc" ::: "memory");
```

**典型 pattern**（见 demo01 段 6 ①）：
```c
mxclracc;                                  // 清
bias = mxmem(bias);                        // 配 convert scale
{ activation.ub = ...; weight.b = ... };   // MAC, acc += A·W
mxmem(out):after.uh = acc:2x1;             // convert + read
```

## `mxclracc.hf` —— 清零 xfp 浮点累加器 (F)

**汇编**：`mxclracc.hf`
**Intrinsic**：`Q6_mxclracc_hf()`

**语义**：清零 xfp（extended fp）累加器，用于 `activation.hf × weight.hf` 浮点路径。

```c
asm volatile("mxclracc.hf" ::: "memory");
```

**和 `mxclracc` 的关系**：两个累加器**物理上独立**，互不影响，但**不能同一
kernel 混用**——一次 kernel 内只能走 int 路径或 fp 路径，两套的 `mxmem` MAC 发
射会看到对方的残留。切换模式必须各自 `mxclracc*`。

```c
/* 错误: 混用 */
mxclracc;                                   // int acc 清
{ activation.hf; weight.hf };               // 错！fp MAC 但 fp acc 没清
```

```c
/* 正确: 同一 kernel 只走一条路 */
mxclracc.hf;
{ activation.hf; weight.hf };
mxmem(out):after.hf = acc;
```

## `mxswapacc` / `mxswapacc.hf` —— 双 acc slot 交换 (F)

**汇编**：`mxswapacc`（int），`mxswapacc.hf`（fp）
**Intrinsic**：`Q6_mxswapacc()` / `Q6_mxswapacc_hf()`

**语义**：HMX 其实有 **两个** 32×32 acc slot（A 和 B），`mxswapacc` 让后续的
MAC+convert 指令作用在"另一个 slot"上。主要用途是 **double-buffered conv**
kernel：一个 slot 边算 row k, 另一个 slot 边 convert 上一个 row k-1 的结果，
两条流水重叠。

**设备证据** (F)：QNN 的 `hmx_convhnh_5x5_stride1`（v75 skel 0x213e60）的内层
循环用 `mxswapacc` 在 5×5 conv 的前 2 MAC 和后 2 MAC 之间切换，实现 overlap-add。

**对 matmul 用不用？** —— 不需要。matmul 的 acc 单一累加直到读回，不用交换。

```c
/* matmul 永远不用 mxswapacc；这里只是示意语法 */
asm volatile("mxswapacc" ::: "memory");
```

**注意**：`mxswapacc` **不清零任何 slot**——只是切换 "current slot" 指针。如果另一个
slot 里有旧数据，你切过去后 MAC 会接着累加。

## 汇总速查

| 指令 | Intrinsic | 清哪个 / 做什么 |
|------|-----------|----------------|
| `mxclracc` | `Q6_mxclracc` | 清 int32 acc slot（当前 slot）|
| `mxclracc.hf` | `Q6_mxclracc_hf` | 清 xfp acc slot（当前 slot）|
| `mxswapacc` | `Q6_mxswapacc` | 切换 int acc slot（A↔B）|
| `mxswapacc.hf` | `Q6_mxswapacc_hf` | 切换 xfp acc slot |

## 踩坑提醒

1. **忘了 `mxclracc`** → `out[0]` 是"这次 + 上次"的叠加，数值看起来接近预期但偏大。
2. **int 路径用了 `mxclracc.hf`**（或反之）→ 累加器根本没清。
3. **`mxswapacc` 之后忘记对新 slot 再做 `mxclracc`** → 读到旧数据污染。

## 参考

- 硬件双 acc slot 的架构位置：`Agent/hmx_u8xi8_matmul_layers.md` §L7
- 两套 MAC 基元（fxp 和 xfp 走独立 accumulator）：`Agent/hmx_int4_combos_analysis.md` §3.1

## 下一章

[05 instr-bias](05-instr-bias.md) —— bias load 的细节 + f16 scale 选择指南。
