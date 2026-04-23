# Next session starting point — Phase 3C 并行化 + ISA 探索 (2026-04-23)

> 最新 commit: Phase 3B V3 纯 HMX op 至 **0.143 cyc/MAC @ 512³ bit-exact**。
> 177× gap 到 QNN w8a16（v8.09e-4 cyc/MAC）仍在。本 doc 明确下阶段要探索什么。

## 0. 当前位置（一句话）

V3 kernel 已经剥干净 —— op 里只剩 HMX MAC + 小 scalar decode + scatter。
但 pack/gather/combine 都让 host 代劳了（没接 Agent B 的上游 HVX op）。
所以**架构对了、接线没做完**。

## 1. 你问"为什么没对齐 QNN"—— 直接答案

177× 差距里可明确归因：

| 来源 | 比例 | 能否做 |
|---|---:|---|
| **Graph 并行未接通**（Agent B 上游 HVX + HMX 并发） | ~3-4× | ✅ 明确可做 |
| **`:cm` 零 pack 未启用**（`:cm` row-major layout 最后 bug） | ~1.3-1.5× | ✅ 明确可做 |
| **mxmem pipeline 停顿**（VTCM 读取 pattern 次优化） | ~1.5-2× | ⚠️ ISA 文档 gated，靠探针 RE |
| **Native `weight.n` int4 HMX** | ~2× | ❌ ISA 文档未公开 |

**能做的 3-4 × 1.3 = ~5×**，把 177× 压到 **~35×**。
**探针能拿到的 1.5-2×**，进一步到 **~20×**。
**文档门槛内**，剩下 ~10× 到 QNN。

## 2. 明确探索方向（按优先级）

### 方向 A — Path W 图级并行 (第一优先，明确可做)

**目标**：把 V3 kernel 的 host-side pre-pack 替换成 Agent B 已经写好的 HVX ops 作为 graph 节点，让 QNN scheduler 把 pack 分派到 4 个 HVX 线程、HMX 单独跑 MAC。

**要做的**：
1. 读 `example/hmx_matmul_phase3/kernel/pack_act_hvx.c` 和 `pack_wt_hvx.c` 的 signature（输入 tensor shape + 输出 tensor shape）
2. 改 `run_matmul_v3.cpp` → `run_matmul_graph.cpp`，构图为：
   ```
   raw uint8 act   →  PackActToHmxTile  →  packed_act tensor  ┐
   raw int8 wt     →  PackWeightToHmxTile →  packed_wt tensor ┴→  MatMulV3  → int32 out
   ```
3. 每个 op 是独立 graph 节点，QNN scheduler 自然把 PackAct + PackWt 派到 HVX 线程
4. 测量 `cyc/MAC` 对比 V3（host-pre-pack）。若 ≤ 0.05 说明并行有效；若 ≈ 0.143 说明 scheduler 串行化了（之前 P8 auto-tile 的情形）

**回退**：如果 QNN scheduler 不给 HVX ops 并发调度：
- 尝试 V3 op 自己声明 `QHPI_RESOURCE_HVX + multithreaded=true`（撒谎说是 HVX op），让 QHPI self-slicing 生效
- 或：手写 qurt_thread_spawn 在 kernel 内部多线程（绕过 QHPI）

**关键文件**：
- `Agent/phase3b_path_w_impl.md` — Agent B 的 op API 设计
- `example/hmx_matmul_phase3/kernel/{pack_act_hvx,pack_wt_hvx,combine_hi_lo_hvx,int4_expand_hvx}.c`
- `example/hmx_matmul_phase3/src/run_matmul_v3.cpp` — 参考，改造起点

### 方向 B — `:cm` row-major 零 pack 最后一公里 (第二优先)

**目标**：完成 `:cm` + row-major 的 bit-exact 实现。Agent A 硅探针证明 7.92 cyc/MAC 可达，比 V3 的 2-stream 9.03 快 1.1×。若用上则 V3 本身从 0.143 再降。

**要做的**：
1. 写新的定向硅探针 `example/hmx_matmul_device/probe_cm_weight_layout.c`
2. Test 1：weight[k=0, n=0]=1，其余 0。activation 全 1. 预期 output[m, 0]=1 for m in 活跃 rows。**从输出哪些行非零，推 weight 的 K-ordering**。
3. Test 2：weight[k, 0]=k+1, k=0..31, 其余 0. activation 全 1. 预期 output[m, 0]=32*33/2=528 for 活跃 m. **从实际值推 K-sum 是否正确**。
4. Test 3：变 weight 布局（row-major vs Phase 2 packed vs 其他）测哪个给正确输出
5. 得出 `:cm` + 正确 weight layout 后，改 V3 kernel 用 `:cm`，bit-exact 全量 shape

**参考**：
- `Agent/cm_row_major_re.md` — Agent A 做的 activation layout RE
- `Agent/phase3b_cm_readback_layout.md` — 之前 debug 过程中得到的部分 findings
- `example/hmx_matmul_device/probe_cm_row_major.c` — 探针模板

### 方向 C — mxmem pipeline RE（ISA 文档 gated，但硅可探）

**假设**：V3 实测 75,000 cyc/(m,n)tile vs 理论 130 cyc (silicon ceiling)。99.8% 时间不是 HMX MAC 本身，是 VTCM 读取停顿、pipeline bubble、scheduler 开销。

**要做的探针** (写到 `example/hmx_matmul_device/probe_mxmem_pipeline.c`)：
1. **Tile prefetch**：连续 16 K-iter 读 **同一个** wt tile (hot) vs 16 **不同** wt tiles (cold)。量化 cold-read penalty per packet。
2. **`:dilate` modifier**：Phase 1 probe 测 `:dilate` 在 2-stream 上是 no-op，但没在 `:cm` + row-major 上测过。试 `weight.b = mxmem(..., 0x3FF):dilate` 看是否改变吞吐。
3. **连续 tile 地址模式**：weight tile address 按 stride 递增 vs 随机。找"QNN 风格"(连续 1KB 步长) 是否更快。
4. **bias reload 间距**：Phase 1 看到 QNN 在某些点 combine(r9, r7) 改 Rt 值。试在 inner loop 某些位置 reload bias，看是否影响 throughput。

**期望收益**：1.5-2×，若探出 hot prefetch pattern 或 dilate 的隐藏效果。

### 方向 D — `weight.n` 硅级探针（ISA 文档 gated，低成功率但值得 1-2 天投入）

**背景**：`libQnnHtpV75Skel.so` 里有 232 次 `weight.n` 使用，都在 Conv2D + LPBQ 路径。理论 2× 带宽优势对 int4 重要。

**要做的**：
1. 反汇编 `libQnnHtpV75Skel.so` 里所有 `weight.n` 出现的 op kernel，收集指令上下文（Rt 值、紧随 `:dilate` / `:cm`、tile 地址计算 pattern）
2. 写 `probe_weightn_native.c` 硅探针，按观察到的 pattern 发 MAC 指令，看是否能产出正确 int4 × int8 结果
3. 如果硅接受：将 V3 kernel MAC 指令从 `weight.b = mxmem(...)` 改成 `weight.n = mxmem(...)`，同步改 weight tile pack 格式（int4 nibble pack，半 VTCM 带宽）
4. bit-exact + 测性能

**失败的话**：确认 Qualcomm 把 `weight.n` 和特殊 Rt 编码绑定，需要内部文档解码，停在这里。

### 方向 E — w4a16 / w4a8 端到端（应用层）

上面 A/B 做完后，扩展到完整应用：
- w4a16: 加 `Int16HiLoSplit` + `CombineHiLo` HVX ops (Agent B 已写好) 到 graph
- w4a8: 加 `Int4Expand` HVX op 到 graph

这是套用架构；没新问题。

## 3. Session 起手动作

```bash
source scripts/env.sh

# 1. Regression: V2/V3 仍然 bit-exact
cd example/hmx_matmul_phase3
bash build.sh
bash run_v2_on_device.sh --shape 512,512,512   # expect 0.23 cyc/MAC, 0 mismatches
bash run_v3_on_device.sh --shape 512,512,512   # expect 0.143 cyc/MAC, 0 mismatches

# 2. 方向 A 起手：读 Agent B 的 op signatures
cat kernel/pack_act_hvx.c | head -100
cat kernel/pack_wt_hvx.c  | head -100

# 3. 写 graph-wired host harness
# （新文件 src/run_matmul_graph.cpp，参照 run_matmul_v3.cpp，
#   用 g_qnn.graphAddNode 添加多个节点）
```

## 4. 数字目标（明确）

| 里程碑 | cyc/MAC @ 512³ | vs QNN w8a16 |
|--------|---------------:|-------------:|
| 当前 V3 | 0.143 | 177× |
| 方向 A 完成 (Path W graph 并行) | ≤ 0.05 | ≤ 60× |
| 方向 A+B | ≤ 0.03 | ≤ 40× |
| 方向 A+B+C | ≤ 0.015 | ≤ 20× |
| 方向 A+B+C+D | ≤ 0.008 | ≤ 10× |
| 理论 HMX-only ceiling | ~2.4e-4 | 0.3× (超过 QNN) |

"对齐 QNN" 现实目标：≤ 5× of QNN (即 ≤ 4e-3 cyc/MAC)。需要 A+B+C 三者都成。

## 5. 状态快照

```
Commits today (新→旧):
  a8137db  NEXT_STEPS.md Phase 3 plan
  861ccd3  Phase 3A Crouton probe
  5551504  Phase 3B Path X/W initial
  e4ae8f7  :cm readback layout RE
  146d8e5  :cm stride-2 iteration
  3ef0434  Phase 3B bit-exact main
  9d75a35  Phase 3B @ Phase 2 parity
  23ae15f  NEXT_STEPS.md update
  (latest) Phase 3B V3 pure HMX op 0.143 cyc/MAC

Registered ops in example/hmx_matmul_phase3/:
  MatMulInt8xInt8Crouton  -- Phase 3A probe (diagnostics)
  MatMulV2                 -- gather + HMX + decode in one op (0.23)
  MatMulV3                 -- pure HMX, host pre-packs (0.143) ★
  PackActivationToHmxTile  -- Agent B HVX op (built, not wired)
  PackWeightToHmxTile      -- Agent B HVX op
  CombineHiLo              -- Agent B HVX op
  Int4Expand               -- Agent B HVX op

Phase 2 baseline 仍在 example/hmx_matmul_{qnn,w4a8}/ 不动.
```

## 6. 不要做的事

- 不要再在单个 op 内堆优化。V2→V3 已证明拆 op 更有效。
- 不要动 `example/hmx_matmul_qnn/` 和 `example/hmx_matmul_w4a8/`（Phase 2 baseline，冻结）。
- 不要从零写新 HVX ops；Agent B 的 4 个已经过 build 测试。
- 不要忽略 host-side pack 的成本 —— graph 并行后"accelerator cycles"包含 HVX pack，数字会变化但是更真实的。

## 7. 最后：诚实的限制

即使方向 A+B+C 全部做完到 20×，仍未"对齐"QNN。差的 20× 里包含：
- 我们看不到的 HMX 指令 pipeline tricks
- `:dilate` 在 1×1 MatMul 路径下的真实语义
- Qualcomm 对 QNN 框架做的 per-shape optimization

不要期待 2× 以内对齐 QNN 没有 Qualcomm 内部文档泄漏。**10× 以内对齐已是公开文档能达到的较高水平**，这是下阶段的合理终点。
