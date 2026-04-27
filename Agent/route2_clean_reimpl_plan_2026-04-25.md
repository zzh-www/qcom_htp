---
name: Route 2 — Clean Reimplementation Plan (fallback if Route 1 dlsym fails)
description: Stage A-D plan to reimplement each QNN MatMul L1 primitive as our own QHPI custom op without touching libQnnHtpV75Skel.so internals; activate this plan if Route 1 (dlsym into skel) hits an unrecoverable wall
type: project
---

# Route 2 — 干净复现 QNN MatMul L1 原语图（不用 libQnnHtpV75Skel 内部导出）

> 选择背景 (2026-04-25): 用户决定走路线 1（dlsym 直插 `libQnnHtpV75Skel.so`
> 内部导出符号）。**路线 2 在这里冷藏**，作为以下场景的 fallback：
> 1. dlsym 在 graphFinalize/load 时不解析（符号未出现在 op-pkg 命名空间）
> 2. 内部函数有未公开的 hnnx context 依赖（如 VTCM allocator state）
> 3. Qualcomm 升级 QNN SDK 把符号改名/inline 化
>
> 目标：在不依赖 `libQnnHtpV75Skel.so` 任何内部函数的前提下，**自己用 HVX/HMX
> 重新实现** QNN MatMul lowering 出来的全套 L1 原语，匹配 QNN 在 32³ → 4096³
> 全 shape 的图拓扑和性能。

## 背景与硬约束

QNN MatMul (`qti.aisw::MatMul`) 在 `QnnGraph_finalize` 被 lower 成 10+ 种
`q::*` HTP 原语。这些原语：
- 不在 public QnnOpDef API 里
- 不能从 op-pkg 用 QnnOpPackage_reference() 调
- 都在 `libQnnHtpV75Skel.so` 内部 hnnx op-def registry

路线 2 = **"L1 原语都自己写"**。每个 q:: 原语我们写一个等价的 QHPI op，组装成
QNN 同样拓扑的 ONNX 子图。

参考完整论证：
- `Agent/qnn_matmul_as_composition_2026-04-25.md` §3.3 + §4
- `Agent/qnn_matmul_design_principles_2026-04-25.md` §4
- `Agent/forceformat_crouton_re.md` (HVX vshuff(-32) 算法已 RE)

## QNN 原语 ↔ 我们 custom op 对应表

| QNN HTP 原语                       | 我们的 custom op           | 现状                  | 路线 2 工作       |
|-------------------------------------|----------------------------|-----------------------|-------------------|
| `q::*InputSlicePad`                 | `InputSlicePadV2`          | 🔴 没做               | 写 HVX 切+pad     |
| `q::ForceFormat_Crouton`            | `PackActivationU8RowMajor` | 🟡 600K cyc vs 5K    | 用 vshuffvdd(-32) 重写（5×-10× 提速）|
| `q::ConvLayer.opt.weights_to_vtcm`  | `WeightDmaToVtcm`          | 🟡 在 PackWeight 里  | 拆成 DMA-only op  |
| `q::ConvLayer.opt.bias_to_vtcm`     | `BiasDmaToVtcm`            | 🟡 在 mmv8 里读      | 拆成 DMA-only op  |
| `q::ConvLayer_s1.opt`               | `MatMulV8`                 | ✅ 已追平             | 无               |
| `q::Concat`                         | QNN 内置 Concat            | ✅ 直接用             | 无               |
| `@Spill`                            | `VtcmSpillToDdr`           | 🔴 没做               | 写 HVX vmem 拷贝  |
| `@Fill`                             | `VtcmFillFromDdr`          | 🔴 没做               | 写 HVX vmem 拷贝  |
| `q::*OutputSlice`                   | `TcmDramCopy`+`UntileToRowMajor` | 🟡 已有，慢       | 优化 HVX vmem 路径|

## Stage A — gen_v8_graph(M, K, N, vtcm_mb) 升级（无依赖）

实现 `qnn_matmul_design_principles_2026-04-25.md` §4 的 `plan_matmul_graph`：
- M_TILE = 256（QNN 固定）
- N_TILE = `min(256, floor((0.6×VTCM - M_TILE×K) / (K + M_TILE)))`，向下取整 32 倍数
- M_ROUNDS = `ceil(M/M_TILE)`，N_ROUNDS = `ceil(N/N_TILE)`
- HVX pack 实例数：`{≤2:1, ≤8:2, else:min(4,M_ROUNDS)}`
- 工作集 > VTCM 触发 `need_spill`

输出：M_ROUNDS × N_ROUNDS 个 MatMulV8 实例，每个独立 pack/concat 子图。

512³ 应跑出 4 rounds、2 HVX pack threads；4096³ 跑 1024 rounds、4 threads（与
QNN 实测拓扑一致）。

## Stage B — pack_act 改用 HVX vshuff(-32) 拓扑

参考 `Agent/forceformat_crouton_re.md` §4。当前 `pack_act_rm_hvx.c`
600K cyc/call @ 512³，QNN ForceFormat_Crouton 5K cyc/call。120× 差距是 V9 vs
QNN 总差的主因。算法已 RE 完，写一遍即可。

预期：单 pack 实例 600K → 60K（10×），结合 Stage A 多线程化再降 4×。

## Stage C — InputSlicePad（M_offset, M_len, zero-pad to 32）

让 PackActivationU8RowMajor 接受 M 维子区间参数，输出 zero-padded 到 32 倍数。
之前硬约束 M%32==0 借此放开。

## Stage D — VtcmSpill / VtcmFill custom op

当 plan_matmul_graph 检测到 `need_spill=True` 时，子图中显式插 Spill/Fill 节点：
- VtcmSpill：[NHWC] in VTCM → DDR（保留输出 tile）
- VtcmFill: DDR → VTCM（取回 next round 用的 tile）

实现：HVX vmem loop 拷贝（参考已有的 `tcm_dram_copy_hvx.c` —— 它就是 256KB
HVX vmem 拷贝，12K cyc，已在 V8 用到）。

## 验证序列

1. Stage A 完成：512³/1024³ 全跑通且 cyc 不退化
2. Stage A+B：512³ pack_act 显著下降，总 cyc 接近 QNN
3. Stage A+B+C：32³/128³/256³（M/N 不是 32 倍数的）跑通
4. Stage A+B+C+D：2048³/4096³ 跑通，无 fail
5. 全部完成：5 个 shape (32/128/256/512/1024/2048/4096) 的 V8 vs QNN
   chrometrace 对比，确认拓扑一致 & cyc 数 ≤ 1.2× QNN

## 工作量估计

- Stage A: 1-2 天（图生成器逻辑直接 port §4 算法）
- Stage B: 1 天（vshuff 算法已 RE，重写一个文件）
- Stage C: 0.5 天（InputSlicePad 较简单）
- Stage D: 1-2 天（Spill/Fill 单纯，但调度依赖图的正确性）
- 验证 + 调试：3-5 天

总：~1-2 周。比路线 1 慢，但完全自主可控、可维护。

## 路线 2 vs 路线 1 决策点

如果路线 1 在以下任一节点失败，**降级到路线 2**：
1. dlsym spike：op-pkg 加载时 `convert_to_crouton_b` 不解析
2. 调用 spike：参数对了但 graphExecute 立刻 crash（说明依赖未公开 context）
3. 结果对了但 perf 没变好（说明这些导出函数走的是 stub/fallback 路径）
4. QNN SDK 未来升级把符号改名 → 维护负担过高

降级时直接读这份文档执行 Stage A-D。
