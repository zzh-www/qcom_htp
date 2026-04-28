---
name: P1 ctx-binary extraction findings
description: Result of NEXT_STEPS_v73deep_gap.md P1 — extracted native ctx-binary for 256³ and compared to ours. No inline kernel embedded; both call public symbols. Apples-to-apples chrometrace comparison shows native does same cpp (~4) but half the packets (346 vs 747).
type: project
---

# P1 — Ctx-binary 字节提取（2026-04-29）

执行 `NEXT_STEPS_v73deep_gap.md` Phase 1 P1：从 native 256³ ctx-binary
里看 native ConvLayer.opt 用的 kernel 字节，和我们的 V73DEEP 对比。

## 关键发现

### 1. 没有 inline kernel 嵌在 ctx-binary 里 ✗

搜索常见 mxmem opcode pattern (`0x9203c3xx` bias load, `0x920647xx`
activation MAC, `0x9208e9xx` weight MAC) 在 native ctx-binary
(`example/hmx_matmul_phase3/standard_flow/phaseA_native/s256_chain8_compare/ctx/matmul_native_ctx.bin`)
全部 0 命中。

```bash
LC_ALL=C grep -aob -P '\xfe\xc3\x03\x92' "$CTX"  # bias = mxmem2(r3)
LC_ALL=C grep -aob -P '\xe1\x47\x06\x92' "$CTX"  # act.ub = mxmem(r6,r7):deep:cm
# all → 0 matches
```

**结论：** native 不嵌内联 kernel；和我们一样走 dlsym 调 libQnnHtpV75Skel.so
里的公开符号。`mxmem` 字节只在 `.so` 里，不在 ctx-binary 里。

### 2. Native 的 lowered graph node template 名称揭示 dispatch 逻辑

Native ctx-binary 字符串里看到的 graph nodes：

```
*InputSlice@FB.s4*6.
ForceFormat_Crouton_f2c@CB.FB
ConvLayer_s1.opt@CB*2.Xsb.Fi.t.fi    ← 256³ matmul 用的模板
ForceFormat_Flat_c2f@FB.CB
*OutputSlice@FB.s4*3.
```

libQnnHtpV75Skel.so 里也能找到该模板族的所有变体：

```
ConvLayer_s1.opt@CB*2.Xwb.Fi.t.fi
ConvLayer_s1.opt@CB*2.Xpb.Fi.t.fi
ConvLayer_s1.opt@CB*2.Xsb.Fi.t.fi   ← native 用这个
ConvLayer_s1.opt@CHB.CB.Xsb.Fi.t.fi
ConvLayer_s1.opt@CH*2.Xwb.Fi.t
ConvLayer_s1.opt@CH*2.Xpb.Fi.t
```

后缀解读：
- `@CB` Crouton byte 输入
- `*2` 2-tile fan-out 变体
- `Xsb` signed-byte 权重
- `Fi` flat-int 输出/中间
- `.t` 转置 / type modifier
- `.fi` flat-int folded bias（=我们 V8C8 的 bias-int32-fold 一致）

### 3. 关键 hint：native ConvLayer_s1.opt 的 Description

从 chrometrace `q::ConvLayer_s1.opt` event 的 args Description 字段：

> "Conv2d with strides=(1,1) but there's an extra CTRL input that
> controls whether we do a depthwise/grouped conv or use the 4/8/16bit
> kernels."

**这说明 ConvLayer_s1.opt 是一个 dispatcher**，根据 CTRL 输入选择
4-bit / 8-bit / 16-bit kernel。对于 u8×i8 → u8 (8-bit)，最终还是调
hmx_v73_convbbb1x1_stride1 家族的某个 kernel。

### 4. Native vs 我们：input ops 个数不同

Native ConvLayer_s1.opt 输入 ops 数量 = **5**：
```
"Inputs Ops": ["0x000011ed00000012","0x0000121100000016",
               "0x000011a700000016","0x0000100200000008",
               "0x0000123200000012"]
```

我们 BbbKMajor 输入 ops = **3**（act, wt, bias）。

Native 的额外 2 个输入很可能是：
- 1 个 "CTRL" tensor（4/8/16 bit dispatch select）
- 1 个 shape/scale meta tensor

这两个对 packet count 影响不会是核心 — 都只是 setup 数据。

## 5. 真实 chrometrace 对比（hot 实例，apples-to-apples）

| 指标 | Native MatMul_1 | Ours bbb_chain1 (V73DEEP production) |
|------|---:|---:|
| dur (cyc) | 1446 | 3042 |
| cpp | 4.18 | 4.07 |
| **pkts** | **346** | **747** |

**cpp 一致（~4 cyc/pkt）— 每 packet 用时一样，gap 完全在 packet count（2.16×）。**

## 当前最强 hypothesis

**Native 用 `ConvLayer_s1.opt@CB*2.Xsb.Fi.t.fi` 这个特定 template
instance dispatch 到一个 kernel 路径，packet 数比我们调
`hmx_v73_convbbb1x1deep_stride1` 直接少一半。**

可能机制：
1. **Native 调的是 `*_unaligned` 变体或我们没识别出的专用变体** —
   不是 `_deep_`。
2. **Native 通过 CTRL 输入触发 kernel 内部的 "fast path"**，跳过
   prologue 和某些 setup packet。该路径在我们的 disasm 里可能存在但
   我们没设对触发条件。
3. **Native 用 set_hmx_params_conv1x1 的某个参数组合**触发 kernel
   内部的更紧凑 loop（我们尝试过 mask args ∈ {0x700, 0x70b, 0x73b,
   0x780}，但可能不够）。

## P1 不能彻底 close gap，需要进一步 RE

P1 排除了"native 用嵌入式 inline kernel"假设。Gap 真实存在且在 kernel
call 内部，但需要进一步实验定位：

### P1.1 — 跑 native 模板的 dlsym 实验（如果有 callable 符号）

`q::ConvLayer_s1.opt` template 名只在 strings 里，没有 dynamic symbol。
但 libHtpPrepare 里有相关的 dispatcher 代码。需要找到对应的可调函数。

### P1.2 — 试 `hmx_v73_convbbb1x1_stride1_unaligned`

我们至今未测过 `*_unaligned` 变体。如果 native 走 unaligned 路径
（比如因为 act/out 指针对齐位不同），可能解释 gap。

### P1.3 — 静态 RE libHtpPrepare.so 找 dispatcher

具体看 `libHtpPrepare.so` 里 `@CB*2.Xsb.Fi.t.fi` template 实例化对应
哪个 kernel 调用 + 传什么参数。

### P1.4 — 在 native run 时 patch kernel 入口 dump 真实参数

NEXT_STEPS Phase 2 一部分。Patch `hmx_v73_convbbb1x1deep_stride1` 入口
（或 wrapper）记录真实 native 调用参数，对比我们的。

## 接下来推荐

P1.2 (`*_unaligned` 变体测试) 最便宜，~ 1-2 hours。可以现在做。

P1.3 / P1.4 都是多天的工作。

P1 单独做 → 排除了"inline kernel"假说 + 找到 ConvLayer.opt dispatcher
线索（CTRL 输入 + template @CB*2.Xsb.Fi.t.fi）。Gap 来源缩小但未定位。
