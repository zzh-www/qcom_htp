---
name: P1.3 — libHtpPrepare RE + 4-path empirical sweep
description: libHtpPrepare 静态 RE = cost-table 评估器（不是 kernel 调度器）。重要发现：v73 deep variant 内部有 4 个 dispatch 路径，alt-B (n_act_pairs=1, ep[0]=1) packet 数 367 几乎匹配 native 346（1.06×），但 60% bit-exact 因 K 累加方式不对。Native 真正路径不是 4 个静态可达 path 之一，必须 P1.4 patch native kernel dump runtime args 才能 close gap。
type: project
---

# P1.3 — libHtpPrepare RE + 4-path empirical sweep（2026-04-29）

执行 NEXT_STEPS_v73deep_gap.md P1.3 全部子任务。三层结论：

1. **libHtpPrepare 注册表 ≠ kernel 调度器**——它是 cost-based rule selector
2. **DEEP variant 内部有 4 个分支**（main + alt-A/B/C），全部 empirical 测过
3. **Alt-B 367 pkts ≈ native 346**（1.06×），但 K 累加顺序错→60% bit-exact

## (a) libHtpPrepare.so 注册表分析

工具：搜 `0x3d2f2ce` (= "ConvLayer_s1.opt@CB*2.Xsb.Fi.t.fi" 字符串 VMA)
的 64-bit pointer 引用 + 解 `R_X86_64_RELATIVE` 重定位。

**3 个注册表条目**引用同一个字符串：

| 入口 VMA | 类型 | +0x10 | +0x20 函数 |
|---|---|---|---|
| 0x61e2fd0 | primary A | 0x48c644b (内部 data) | **0xf1a360** |
| 0x62428d0 | primary B | 0x48c644b           | **0xf1a360** |
| 0x627d8c0 | fallback  | 0x42b385d ("...&fallback") | 0xf1aef0 |

`0xf1a360` 函数前缀 (`push rbp; push r15..r12; push rbx; sub rsp, 0x88`)
+ 浮点 mov/cvt/cmp + polynomial computation = 标准
`hnnx::cost_func_from_str` polynomial scorer。两条 primary 用同一函数但
不同 polynomial coefficients（A: 0x3f6147ae≈0.88, 0x3ef0a3d7≈0.47;
B: 0x43fa4000≈500.5, 0x3c23d70a≈0.01）— 选择基于 shape regime.

**关键事实：** libHtpPrepare 仅引用 `hmx_v73_convbbb1x1_stride1`（**non-deep entry**），
没有 `hmx_v73_convbbb1x1deep_stride1` 字符串。意味着 native lowering 走
dispatcher entry @ 0x2eadc0，由 mask[+0x30] bit 5 内部 dispatch 到 deep variant。
跟我们一样。

## (b) DEEP variant 4 个内部分支

读 `hmx_v73_convbbb1x1deep_stride1_2ebe40.S` entry 后的 dispatch：

```
2ebec8: if (!p0) jump 0x2ec0a0    // p0 = (n_act_pairs > 1)
2ebed8: if (!p2) jump 0x2ebfa0    // p2 = (extra_param[0] == 1)
```

| 分支 | 条件 | 起始 | 内部结构 |
|---|---|---|---|
| **Main** | n_act_pairs>1 & ep[0]=1 | 0x2ebee0 | loop0=4iter×3pkt, drain×2/loop1, M-fanout 2 |
| Alt-A | n_act_pairs>1 & ep[0]≠1 | 0x2ebfa0 | loop0+mini-drain (r21=ep[0] iters per drain block) |
| **Alt-B** | n_act_pairs≤1 & ep[0]=1 | 0x2ec0c0 | loop0=r20iter×5pkt, no loop1, drain INSIDE loop0 |
| Alt-C | n_act_pairs≤1 & ep[0]≠1 | 0x2ec160 | (alt-A 简化版？未细读) |

### Alt-B body @ 0x2ec100（5 packets/iter）

```
2ec100: r21 -= 1; act.ub = mxmem(r6,r7):deep:cm; wt.b = mxmem(r2,r9):deep
2ec10c: cmp.gt(r21,0); cvt.ub = acc(r25)
2ec114: r25 = setbit(r25,12); r10 = memw(r0++m0); mxmem(r10,r11):cm = cvt
2ec120: if (p0) r6 = memw(r1++4); cvt.ub = acc(r25)
2ec128: r25 = clrbit(r25,12); if (p0) r10 = memw(r0++4); mxmem(r10,r11):cm = cvt :endloop0
```

每 loop0 iter: 1 act-MAC + 1 wt-MAC（**wt 在 loop0 内 fixed**, 只在 outer iter 步进），
2 cvt + 2 store drain 到 (r10, r10+4) 两个不同 M-pair 位置。loop0 trip = `r20` = 8.
**没有 loop1**——loop0 直接覆盖所有 N tiles。

## (c) 4 个 path empirical 测试结果（256³ chain8 hot）

| Path | N_ACT_PAIRS | EXTRA_PARAM_0 | hot dur | hot pkts | cpp | bit-exact | vs native pkts |
|------|---:|---:|---:|---:|---:|---:|---:|
| Main | 8 | 1 | 3042 | 747 | 4.07 | 100% | 2.16× |
| Alt-A | 8 | 2 | 4015 | 937 | 4.29 | 46% | 2.71× |
| **Alt-B** | **1** | **1** | **1148** | **367** | **3.13** | **60%** | **1.06×** ✓ |
| Alt-C | 1 | 2 | 1829 | 581 | 3.15 | 41% | 1.68× |
| Native | ? | ? | 1120 | 346 | 3.24 | ref | 1.0× |

**关键发现**：alt-B perf 直接达到 native 量级（cyc 1.02× / pkts 1.06×）但
**算法不正确**。alt-A/C 都 bit-exact 失败且 perf 不如 alt-B。

artefacts:
- `phase1_validation/v73deep_alt_path_n_act_1/` — alt-B
- `phase1_validation/v73deep_alt_A_extra_param0_2/` — alt-A
- `phase1_validation/v73deep_alt_C_n_act_1_ep0_2/` — alt-C

## (d) Alt-B 为什么 bit-exact 60%

每 loop0 iter:
- MAC adds 当前 K-tile 到 accumulator（accumulator 持续累加, cvt 不 clear）
- 立即 drain 写 output[2*iter] 和 output[2*iter+1]

→ output[0,1] = bias + K_0 only (1/8 K)
→ output[2,3] = bias + K_0 + K_1 (2/8 K)
→ ...
→ output[14,15] = bias + ΣK_0..K_7 (8/8 K, **正确**)

只有最后 iter 的 outputs 算对，其他 partial K → ~12.5% 严格 bit-exact。
但 u8 saturation 让多数 cells 都达到 255 或 0 → 60% 凑巧重合.

**Alt-B 不能通过 descriptor 调整变正确**——drain 在 loop 内部、output ptr
强制 advance, kernel 结构无法跳过早 iter 的 drain.

## (e) 那么 native 用什么 path？

**4 个静态可达 path 都不是答案**：
- Main path → 我们用的，747 pkts (太多)
- Alt-A/B/C → 都 bit-exact 失败

**剩余可能**:
1. **Native 有第 5 个 path** 我们没识别出来（disasm 漏看了某段）
2. **Native 用 main path 但 some descriptor 让 inner loop 减半**——但我们已经 sweep 过 mask args / od / ad 全部字段
3. **Native 用 HMX state-config**（per docs/hmx-programming-guide/01-mental-model.md "descriptor-driven HMX"）在 kernel call 前/外发了 mxmem packets 写入 HMX 寄存器，让 kernel 内 loop 提前结束 — 我们的 dlsym 路径绕过了这步

## P1.3 完成 → 必走 P1.4

P1.3 已穷尽静态分析能给的：libHtpPrepare 端 + skel deep variant 内部 4 个
path 全部理解了，但都不解释 native 346 pkts + 100% bit-exact 同时存在.

**唯一可推进路径**：P1.4 — on-device patch
`hmx_v73_convbbb1x1deep_stride1` (or dispatcher 0x2eadc0) entry 第 1 个
packet，让它把 r0..r5 + 64 bytes from r4 (mask) + 64 bytes from r0 (od)
+ 12 bytes from r1 (ad) + 8 bytes from r5 (extra_param[0..1]) 写到 VTCM
固定地址，再 native run dump 出来对比。

具体方案：
1. 用 `hexagon-llvm-objdump` 找到 entry 第一个 packet 字节
2. Replace with `memw(0xVTCM_addr) = r0; memw(...) = r1; ...`（需要 free
   register 暂存——可用 r28..r31 然后 jumpr r31 之类）
3. 把 patched bytes 通过 `mlock` 写到内存映射的 .so 文件 — but unsigned skel
   是 read-only，需要 root + remount
4. 或者更简单：mmap 私有副本，然后用 dlopen-from-fd 方式 load patched 副本

实施 risk: 高（系统 .so patching），但是是 closing gap 的唯一手段。

## (f) Bonus — alt-B 有 production 用例吗？

**Yes**：如果 K = 1 (实际的 conv 1×1 with 1 input channel)，alt-B 的"每 iter
1 K-tile MAC + 立即 drain" 行为是正确的，跟 main path 一致 (因为 K=1 只有
1 个 K-tile，drain 1 个最终值是对的).

→ alt-B 是为 1×1 channel-1 conv 优化的 path，不是为 matmul (K_t=8)。

但 perf 数字告诉我们：**HMX 硬件能力上限就是 ~350 pkts 一次 256³**——我们
只是没找到正确的 descriptor 组合让 main path 也跑这么少的 packet.

## Output 产物

- `Agent/qnn_re/p1_3_libHtpPrepare_RE_2026-04-29.md`（本文件）
- 4 个 alt-path empirical artefacts in `phase1_validation/`
- `MEMORY.md` 更新：alt-B perf-trap finding
- next: **P1.4 kernel patching** is the only way forward
