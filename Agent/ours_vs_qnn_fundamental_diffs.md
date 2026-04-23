# 我们 vs QNN built-in MatMul — 根本差异分析（2026-04-23）

Post T0 + T1b baseline。目标：拆清楚剩下 600-800× 差距的结构性来源，为后续 roadmap 提供判断基础。

## 事实对照表

| 维度 | QNN built-in | 我们 (post T0+T1b) |
|---|---|---|
| 入口 op 类型 | `q::ConvLayer_s1.opt` (1×1 conv = MatMul) | `MatMulInt4xInt16` / `MatMulInt4xInt8` 自定义 |
| 图节点数 (512³) | **14 个** (切分+pack+4×ConvLayer+Concat) | **1 个** |
| 热 kernel 里的 HVX pack | **零** — 所有 pack 在上游 op 完成 | 有（T1b 后：`pack_weight_32x32` + scalar `pack_activation` / col_sum / readback / combine） |
| 上游 pack op | `ForceFormat_Crouton_f2c@CB.FB` / `@CH.FH`（HVX `V6_vshuffvdd`，已 RE） | 无（kernel 自己做） |
| 并行度 (512³) | **4 个 ConvLayer 并发**在 4 HVX 线程 | **1 个** (multithreaded=false) |
| HMX tile 数据布局 | **stride-32 depth-32 lane**（HMX 原生） | **stride-4 u32-lane**（我们自定义 u8·i8 分解产物） |
| w8 权重路径 | `weight.b`（int8 HMX 原生） | 我们是 int4 → int8 扩展（`weight.b` 带 -128 offset 修正） |
| w4 权重路径 | `weight.n`（v75 ISA 有，232 次 uses；QNN 只在 Conv2D+LPBQ 启用） | 没用到（我们把 int4 扩到 int8） |
| a16 activation 分解 | hi+lo 两 MAC 通 (同我们) | hi+lo dualacc 融合成单次 K-loop（T1b 前已完成） |
| HMX 调用模式 | `{activation.ub = mxmem(p,Rt_act\|0x1C):cm + weight.b = mxmem(q,0x3FF)}`，2 MAC packet/loop iter | 同样模式 (Phase 1 RE 完)，fused dualacc 1 MAC per iter + swapacc |
| 每 packet HMX 周期 | ~7.9 硅级 ceiling (探针 P2 数字) | 同 ceiling（Rt_wt=0x3FF 已应用） |
| 数据流 | 上游 ForceFormat 做 pack → VTCM → ConvLayer 只做 MAC → Concat 粘回 | DDR wu → BSS w_col 缓存（T0）→ 每 K-tile HVX pack (T1b) → VTCM wt_tile → HMX mxmem |

## 为什么 QNN 更快 — 拆到 cycle 层面

QNN w8a16 @ 512³ = **108,573 cycles**（QHAS timeline）。拆：
- HMX issue 总数：4096 packets × 2 passes (hi/lo) = 8192 packets
- 单线程硅 ceiling：8192 × 7.9 = ~64,700 cycles
- 4 线程理论下限：64,700 / 4 = ~16,200 cycles
- 实测 108,573 ≈ 单线程 ceiling × 1.68（所以每线程跑到 ~13.2 cyc/packet，比硅 ceiling 7.9 慢 67%）
- 4 线程实际吞吐 ≈ 4096 packets × 26.5 cyc/packet ÷ 4 ≈ 108K ✓

QNN 没达到 4× 完美并行，但 4-way + per-thread ~13 cyc/packet 组合压到 108K。

我们 w4a16 @ 512³ = **87,105,246 cycles** (0.65 cyc/MAC) = 807× slower。拆：
- HMX issue 总数：同 QNN 的 8192 packets（可能更多如果 fused dualacc 增加额外 packet）
- 单线程 HMX 硅下限：~64,700 cycles（同 QNN 单线程）
- 87M / 65K = **1347× 过 HMX floor**

差距全在 HMX 以外 —— pack、col_sum、readback unpack、combine、gather_w_col 这些 scalar / HVX 工作 + VTCM 停顿。

## 根本差异 taxonomy (按结构深度)

### 差异 A — 图级并行（4×，**最大可及增益**）

QNN 的 MatMul@512³ 被切成 **4 个独立 ConvLayer 子 kernel**，数据路径完全解耦（InputSlicePad 切 M，weights_to_vtcm 切 N，Concat 粘回）。QNN scheduler 发到 4 个 HVX 线程。单 HMX 单元时分复用，但因为每线程里大部分时间花在 HVX pack/unpack/concat，HMX 的"闲时"正好被其他线程填上。

我们目前 multithreaded=false，**没有任何图级并行**。QHPI auto-tile 尝试过 regression（P8 日志）；host 端 2×2 tiling (T3) 还没试。

**要追平这一点**：需要 host emit 4 个 sub-op 节点 + QNN 自动并发（T3）。**但前提**：sub-op 内必须有可并行的 HVX 工作来 overlap HMX — 否则还是串行化。我们 post-T1b 的 HVX 工作量只有 `pack_weight` 这一处，在整体时间占比 <15%，overlap 潜力有限。

### 差异 B — Pack 在上游 vs 内嵌（结构性设计差异）

QNN 的 pack 是**独立的 `ForceFormat_Crouton_f2c` 图节点**，输出的是 HMX mxmem **原生消费格式**（stride-32 depth-32 lane）。ConvLayer 热循环只做 mxmem load + MAC，完全不知道 pack 存在。

我们的 pack 是**kernel 内嵌代码**。即使 T1b 把 `pack_weight_32x32` 做成了 HVX 只花 16 cycles/tile，它还是在 HMX 热路径里占用 scalar / HVX 指令槽。更关键的是，我们的 pack 产出**和 QNN 不一样的 layout**（stride-4 u32-lane），因为我们用了自定义的 u8·i8 4-channel 分解（见 `Agent/hmx_u8xi8_matmul_layers.md` L1）。

**要追平这一点**：
- 方案 1（激进，~5-7 天）：抛弃 stride-4 u32-lane 分解，改用 QNN 同款 stride-32 lane binding。要重写 HMX issue pattern + VTCM layout + 所有 pack 函数。风险高，但理论上能像 QNN 那样让 pack 变成 upstream op 候选。
- 方案 2（现实）：保持 stride-4，把 pack 搬到独立 QHPI op（`Op1=pack_weight_u32lane`, `Op2=matmul_mn`），让 QNN 调度它们成管道。需要验证 QNN 是否能把我们自定义 op 间的 dataflow 管道化。

### 差异 C — w4 处理路径（潜在 2×，ISA 限制）

v75 ISA 有原生 `weight.n` int4 HMX（`libQnnHtpV75Skel.so` 里出现 232 次）。但 QNN 只在 Conv2D+LPBQ 图上激活这条路径 — MatMul 不走。我们**也没用**，而是 int4 → int8 sign-extend，然后走 `weight.b`。

**要启用 `weight.n`**：需要 RE 其 tile layout + mxmem 绑定方式。P10/P11 已经尝试 `:dilate` 和 `mxswapacc` 但 ISA 文档没公开（Qualcomm 内部），三次尝试都错。**这是文档 gated 的硬瓶颈**，除非我们通过硅探针穷举拿到准确语义。

### 差异 D — Rt_wt / scalar overhead per packet

这一层**已经基本追平**：
- Rt_wt=0x3FF 已应用（Phase 1 RE）→ HMX packet 本身跑到硅 ceiling
- 双缓冲 wt_tile + fused dualacc 已应用
- T1b HVX pack_weight 已减少 scalar 占比

剩余 scalar hot paths（T1d 待做）：col_sum_w 循环（K×32 iters），readback unpack（32×32 iters），combine（32×32 iters），gather_w_col（可 HVX 化但已在 T0 后仅 N/16 次）。每个 1-3% 级别。

### 差异 E — Activation 分解数

QNN w8a8 = 单 MAC pass（a 和 w 都 int8 直入）。
QNN w8a16 / 我们 w4a16 = hi+lo 两 pass，MAC packet 翻倍。
我们 w4a8 = 也两 pass（因为 activation 是 int8，但我们 mxmem 读 uint8 + col_sum_w 修正 -128 的 offset）？wait — actually w4a8 应该只需要 1 pass ，因为 int4→int8 扩展 + int8 activation 都 byte-width。需要 check。

实际上 w4a8 是 int4×int8 → u8·i8 HMX，单 pass 即可（activation 无需 hi/lo 分解）。对比 w4a16 需要两 pass，这是 w4a8 比 w4a16 更快的根本原因（0.48 vs 0.65）。

### 差异 F — 多分片 gather/concat 的 DDR 带宽利用

QNN 的 `weights_to_vtcm` 显式把权重切片预载到 VTCM，让 4 个 ConvLayer 零延迟读。我们 T0 后把 w_col 预缓存到 BSS（DDR），第一次访问要 DDR→L2。QNN 的 VTCM 预载让 4 线程零 DDR 延迟。

---

## 把差距归因到每项差异的权重估算

按 **"追平这一项能关掉多少 × "** 排：

| 差异 | 结构深度 | 预估关差距 | 工作量估 | 风险 |
|------|----------|-----------:|---------:|-----|
| **A 图级并行 4-way** | 高 | **~3-4×** (最大) | 2-4 天 (T3) | 中 — 前提 HVX 工作量足 |
| **B pack 上游化 + stride-32 binding** | 最高 | **~2-5×** (若彻底重构) 或 ~1.5×（方案 2 独立 op 不改 binding） | 5-7 天（方案 1）/ 2 天（方案 2） | 高 / 中 |
| C weight.n 原生 int4 HMX | 最高 | ~2× | 不可估 (ISA 文档 gated) | 极高 |
| D 剩余 scalar HVX 化 (T1d) | 中 | ~1.2-1.4× | 1-2 天 | 低 |
| E 单 MAC pass (w4a8 已占便宜，w4a16 不适用) | 低 | 我们已享受过 | 0 | - |
| F DDR 预载到 VTCM | 中 | ~1.1-1.3× | 1 天 | 中（VTCM 容量/bank 冲突） |

**可达上限**：如果 A+B(方案2)+D+F 都做了，预计 800× → 800 / (3 × 1.5 × 1.3 × 1.2) ≈ **100×** gap（= ~10 M cycles vs QNN 108K）。

继续 C（weight.n）或 B 方案 1（binding 重构），理论可再拿 2-5×，接近 **20-50× gap**。

---

## 结论和下一步判断

**根本区别一句话**：QNN 是"把 pack 拆成上游 HVX op，然后 HMX 热 kernel 只做 mxmem MAC，再图级切 4 份并发" — 我们是"单 op 里自己 pack + MAC + combine"。

**这不是算法差异** — HMX ISA 使用已经 Phase 1 追平。是**架构风格差异**：
- QNN 用 HTP 作为 dataflow graph，每一阶段一个专用 HVX/HMX op
- 我们把整个 matmul 塞进一个巨单元

**下一步投入建议**（按 ROI × 可行性）：

1. **短期（1-2 天）**：T1d 完成剩余 scalar HVX 化 — 顺便验证 HVX 工作量提升后 T3 是否不再 regression。
2. **中期（2-4 天）**：T3 retest — 有足够 HVX work 则激活 4-way，立即吃 3-4× 最大红利。
3. **长期决策点**：B 方案 1（重构 binding 成 stride-32）vs B 方案 2（pack 作为独立 custom op）。方案 2 更稳妥，可在图里先试。

**永远无法追平的部分**：C（weight.n native int4 HMX）需要 Qualcomm HMX ISA 文档。没有硅探针穷举或泄露文档的情况下，2× 硬上限永远存在。

## 如果目标是"接近 QNN 3×-5× 以内"

只要 A + D + F 做完，理论可达到 **~100× QNN w8a16 gap** —— 这已经足够**"数量级内"**（10^2 vs 10^0 of parity）。但要**打平**或进**ConvLayer 级 (3× 内)**，必须啃 B 方案 1（重构 HMX binding 成 stride-32 lane 原生）或 C（weight.n 探针 RE）。这是 7+ 天工程 / 不可估风险。

判断建议：**先以"数量级内"为阶段目标**（~100× gap），做完后评估 B1 或 C 是否值得投入。
