# Phase 3 — 依 QNN 架构第一性原理重写 int4 matmul kernel（2026-04-23）

> 前作 commits `6a8fc00 → e691ee1 → 6b8b267 → 05aacfb` 把 kernel 做到 w4a16=0.32/w4a8=0.22 cyc/MAC (6.5×/8.7× vs baseline). 这是**在错误架构里把错工作做到极致**。Phase 3 是架构重构。

---

## 0. 指导原则（QNN 底层原理第一性）

QNN HTP 后端是一个**按硬件资源调度的 dataflow graph**。v75 上硬件是：
- **1 个 HMX 单元** — matmul 计算
- **4 个 HVX 线程** — 所有数据整形/搬运/reduction
- **DDR / VTCM** — 存储层级

QNN scheduler 的并发是**跨 op 节点**的，不是跨 op 内部。要充分利用 4×HVX + 1×HMX，**必须把数据流拆成多个 op 节点，每个节点只声明一种硬件资源**，让 scheduler 自然把 HVX ops 分派到 4 个线程，HMX op 在它们之间时分复用。

结论：**正确的 custom MatMul op 只写 HMX MAC，其余（pack / split / gather / reduce / combine）全部拆成独立上游 HVX ops 或依赖 QNN 框架 ops**。这正是 `q::ConvLayer_s1.opt` 热循环零 HVX 零 scalar 的根源。

---

## 1. 当前 Phase 2 架构诊断

当前 `HmxInt4MatMulPackage::MatMulInt4xInt16` **单 op 包办 6 件 QNN 本该做的事**：

| # | 工作 | QNN 框架等价 | 当前代码 |
|---|------|--------------|----------|
| 1 | weight DDR → VTCM | `weights_to_vtcm` | `gather_w_col` |
| 2 | weight Crouton pack | `ForceFormat_Crouton_b` | `pack_weight_32x32` (HVX) |
| 3 | int16 hi/lo split | `Cast` + split | `prepack_activation_fused` (HVX) |
| 4 | int32 col sum reduce | ReduceSum HVX op | `col_sum_w` 循环 |
| 5 | HMX dual-scale readback 解码 | HVX decoder op | scalar readback unpack |
| 6 | `(hi<<8)+lo-128·cs` 线性组合 | HVX combine op | scalar combine |

根因：
- 输入签名选了 `QHPI_Layout_Flat4 + QHPI_Storage_Direct` → QNN 给我们**生张量**，我们只能自己 pack。
- HMX binding 选了**自定义 stride-4 u32-lane**（非 QNN 原生 stride-32 depth-lane）→ 2026-04-20 P9 Crouton probe 得出 "dead end for int16×int8" 的结论，把我们锁死在自定义 pack 路径。

后果：
- Op 内 HVX + scalar + HMX 挤在**同一线程**里串行
- QHPI `QHPI_RESOURCE_HMX + multithreaded=false` 阻止 QNN scheduler 并发我们的 op
- 我们只用 **1 个 HVX 线程**（QNN 用 4 个）
- 模型解释不了的 96% cycle 都是 pipeline stall / 单线程资源冲突

---

## 2. 目标架构

```
╔════════════════════════════════════════════════════════════════════╗
║  int16 act tensor            int4 wt tensor (packed as uint8 nibbles)║
║        │                              │                             ║
║        ▼                              ▼                             ║
║  ┌─────────────┐              ┌─────────────┐                       ║
║  │ QNN Cast    │              │ Int4Expand  │   ← 我们 op 1         ║
║  │ (built-in)  │              │ (HVX, MT=T) │   (只有 int4 路径需)  ║
║  │→ uint16     │              │→ int8 (2×)  │                       ║
║  └──────┬──────┘              └──────┬──────┘                       ║
║         │                            │                              ║
║         ▼                            ▼                              ║
║  ┌──────────────┐             ┌──────────────┐                      ║
║  │HiLoSplit     │             │ QNN auto     │                      ║
║  │(HVX,MT=T)    │             │ ForceFormat  │                      ║
║  │←我们 op 2    │             │ _Crouton_b   │ ← 框架 ops 自动插    ║
║  │→(act_hi,act_lo)│           │→Crouton8 wt  │                      ║
║  └──┬───┬───────┘             └──────┬───────┘                      ║
║     │   │                            │                              ║
║     ▼   ▼                            ▼                              ║
║  ForceFormat_Crouton_b (auto) → Crouton8 act_hi + act_lo            ║
║                                                                     ║
║                ╚══════════════════════════════════════╝             ║
║                  ┌─────────────────────────────────┐                ║
║                  │  MatMulHmxDual (HMX, MT=F)      │ ← 我们 op 3    ║
║                  │  仅 mxmem + mxclracc + mxmac    │  （核心！）     ║
║                  │  + dual-scale readback          │                ║
║                  │  输入: 3 个 block_table         │                ║
║                  │  输出: (P_hi, P_lo) int32 partials│              ║
║                  └─────────────┬───────────────────┘                ║
║                                │                                    ║
║                                ▼                                    ║
║     ┌─────────────────────────────────────────┐                     ║
║     │ Combine (HVX, MT=T)  ← 我们 op 4        │                     ║
║     │ cs = ReduceSum(wt_int8, axis=K)        │                     ║
║     │ out = (P_hi<<8) + P_lo - 128·cs        │                     ║
║     │   (for w4a16 activation offset term)   │                     ║
║     └──────────────┬──────────────────────────┘                     ║
║                    ▼                                                ║
║                [int32 output]                                       ║
╚════════════════════════════════════════════════════════════════════╝
```

**关键属性**：
- 4 个 custom ops + 2-4 个 QNN 自动插的框架 ops = 6-8 个图节点
- **每个 op 单一 resource**（HMX or HVX），scheduler 可自由并发
- HVX ops 全 `multithreaded=true` → QHPI 自动 4-way self-slicing
- HMX op kernel 体**零 scalar 零 HVX**，只 `mxmem + mxmac`
- 数据流走 Crouton + Indirect → QNN 自动管 VTCM + pack

---

## 3. 分阶段执行计划

### Phase 3A — Crouton consume probe（0.5–1 天）

**目的**：验证 Crouton_8 输出能**直接**喂 HMX mxmem，即不再需要自定义 stride-4 binding。

**前置阅读**：
- `Agent/forceformat_crouton_re.md:168-240` — Crouton_b 的 4×128 byte transpose 图解
- `Agent/int4_matmul_optimization_log.md:401-421` — P9 Crouton dead-end 结论（前提是我们保留自定义 binding；Phase 3 不保留，所以此结论需要重新验证）

**执行**：
1. 新建 `example/hmx_matmul_phase3/` 目录（保 Phase 2 完整）
2. 写最简 `probe_crouton_consume.cpp`：
   - OpPackage 声明 `QHPI_Layout_Crouton_8` + `QHPI_Storage_Indirect` for weight
   - Op kernel: `qhpi_tensor_block_table()` → 取第一个 block 的 VTCM 地址
   - 直接 `mxmem(block_addr, 0x3FF)` issue 一次 MAC
   - 对照 scalar reference
3. 32³ shape，int8 weight，跑 bit-exact 检验

**出口**：
- PASS bit-exact → Phase 3B
- FAIL → 反推实际 Crouton_8 byte layout（byte-dump），可能需要一个最小 HVX glue（仍是独立上游 op，不是我们 matmul 内的代码）

**关键文件**：
- `tools/qnn-sdk/include/QNN/HTP/core/memory_layout.h:311` R4CroutonLayout
- `tools/qnn-sdk/include/QNN/HTP/core/qhpi.h` — 查 `qhpi_tensor_block_table`, `qhpi_tensor_block_count` 的准确 API

### Phase 3B — 最小 HMX MatMul op（1–2 天）

**目的**：把 Phase 3A 成功的机制扩成完整 32×32×K 矩阵乘，移除所有 pack/gather/col_sum/readback/combine 代码。

**执行**：
1. 基于 Phase 3A 的签名/signal：
   ```c
   sig_inputs[] = {
       {QHPI_QUInt16, QHPI_Layout_Crouton_16, QHPI_Storage_Indirect, ...},  // act
       {QHPI_QInt8,   QHPI_Layout_Crouton_8,  QHPI_Storage_Indirect, ...},  // wt
       // no scratch — QNN/VTCM 已在 Crouton block 里
   };
   sig_outputs[] = {
       {QHPI_Int32,   QHPI_Layout_Crouton_32, QHPI_Storage_Indirect, ...}, // out
   };
   ```
   （Crouton_32 = int32 HMX output native layout，需实验确定）
2. Kernel 体（简化后大概 50 行）：
   ```c
   for each (m_tile, n_tile):
       mxclracc
       for each k_tile:
           act_block = qhpi_tensor_block_table(inputs[0], ...)
           wt_block  = qhpi_tensor_block_table(inputs[1], ...)
           activation.ub = mxmem(act_block, Rt_act|0x1C)
           weight.b      = mxmem(wt_block, 0x3FF)
       readback into out_block
   ```
3. 开始用 int8 activation (w8a8 等价) 先验证，再处理 w4a16 的 hi/lo（Phase 3D）

**出口**：
- 32³/128³/512³ bit-exact + cyc/MAC 测量
- 期望：w8a8 等价路径 cyc/MAC 接近 QNN w8a8 (5e-4) 的同数量级（10× 内）

**文件**：
- `example/hmx_matmul_phase3/kernel/hmx_matmul_crouton.c`
- `example/hmx_matmul_phase3/src/MatMulHmxOp.cpp`
- `example/hmx_matmul_phase3/test_sim.c` + `test_device.cpp`

### Phase 3C — Int4Expand 独立 HVX op（0.5–1 天）

**目的**：int4 path 需要 int4→int8 sign-extend。作为独立 HVX op，不放 matmul 内。

**执行**：
1. 写 `Int4ExpandOp` 纯 HVX op
   - 输入：int4 packed (uint8 with 2 nibbles/byte)
   - 输出：int8 (signed, 每个 nibble 扩展到一个 int8 byte)
   - `QHPI_RESOURCE_HVX + multithreaded=true`
2. HVX kernel 用 `Q6_Vb_vshuffoe` 或类似分离 high/low nibble，然后 sign-extend
3. 注册到同一 OpPackage

**出口**：
- int4 graph 端到端：Int4Expand → QNN ForceFormat → Phase 3B MatMul → bit-exact
- Perf: Int4Expand 应该快到可忽略（几 % of MAC cost）

**键文件**：
- `example/hmx_matmul_phase3/kernel/int4_expand.c`

### Phase 3D — int16 activation (w4a16) support（1–2 天）

**目的**：w4a16 需要 int16 activation hi/lo 分解成两条 int8 activation 流。每条独立跑 HMX MAC，最后 combine。

**执行**：
1. 写 `Int16HiLoSplit` HVX op
   - 输入：uint16 (post-Cast 的 signed+32768)
   - 输出：2 个 int8 tensors (act_hi, act_lo)
   - 用 `Q6_Vh_vasr`/`vand` 提 byte
2. MatMul graph 里 emit 2 个 Phase 3B op 节点（或一个带 2 partial 输出的 op）
3. 写 `CombineHiLo` HVX op
   - 输入：(P_hi int32, P_lo int32, col_sum_w int32 from ReduceSum of weights)
   - 输出：`(P_hi<<8) + P_lo - 128·col_sum` int32
   - QNN 有 built-in ReduceSum？查`QnnOpDef.h` — 若有直接用，否则写 ReduceSumInt8 HVX op
4. 若无 built-in ReduceSum，加 `ColSumOp` HVX op（输入 int8 weight，沿 K 轴 sum）

**出口**：
- w4a16 端到端 bit-exact @ 512³
- cyc/MAC 目标：≤ 2× QNN w8a16 = ≤ 1.6e-3 cyc/MAC（现 Phase 2 是 2.4e-3）

### Phase 3E — Profile + Gap reconcile（0.5 天）

**执行**：
1. 跑 `example/qnn_matmul_profile/profile_all.sh` + 新 OpPackage，对比：
   - Phase 3 完整图 (w4a16, 6-8 节点)
   - Phase 2 单 op baseline
   - QNN built-in w8a16 (no int4 support, closest comparator)
2. 解 chrometrace：确认 sub-ops 在 4 HVX 线程上真并发
3. HMX utilization % 期望 70-90% (vs QNN w8a16 85.6%)

**出口**：
- gap to QNN w8a16 ≤ 10×：架构正确，continue to 3F
- gap > 10×：需深挖 chrometrace 找瓶颈（可能是 QNN 的 Crouton pack 不如我们预期高效，或 HMX 硬件外调度开销）

### Phase 3F — w4a8 / w16a16 类推（1–2 天）

- **w4a8**: Phase 3C (Int4Expand) + Phase 3B (HMX MatMul, int8×int8) + 无 combine. 最简单 case.
- **w16a16**: 2×HiLoSplit + 4×HMX partial + 2-level CombineHiLo. 最复杂 case.
  - 或者用 `mxacc:scale` 多 accumulator 语义在一个 op 里做 4 partial？需要 ISA RE.

---

## 4. 每阶段的回退/验证

| Phase | pass gate | fail recourse |
|-------|-----------|---------------|
| 3A | Crouton_8 probe 32³ bit-exact | 做 byte-dump 反推实际 layout, 写最小 HVX glue upstream op |
| 3B | 512³ bit-exact + cyc/MAC < 0.1 (w8a8 path) | profile 定位瓶颈，可能重新审视 Crouton_X 选择 |
| 3C | int4 graph 端到端通 | 若 Int4Expand 太慢, 考虑用 QNN 内建 Cast (如支持 int4) |
| 3D | w4a16 端到端 bit-exact + ≤ 2× QNN w8a16 | 保 Phase 3B w8 路径, 研究 hi/lo 折叠进 HMX MAC 的可能 |
| 3E | gap ≤ 10× | continue 3F 但 gap expectations 下调 |
| 3F | 三 kernel 都 ≤ 5× QNN | 接受结果 commit |

**硬失败**（需要停下重新评估）：
- Crouton block 布局对 HMX 完全不兼容（Phase 3A FAIL 且 byte-dump 显示没有合理映射）
- `qhpi_tensor_block_table` API 不稳定 / 没文档

---

## 5. 保留的 Phase 2 成果（作为对照）

- Phase 2 代码 (w4a16=0.32/w4a8=0.22) 作为 baseline commit `05aacfb`，**不动**
- Phase 3 放在 `example/hmx_matmul_phase3/`
- Phase 1 RE 永久有效：`Rt_wt=0x3FF`, dualacc semantics, HMX 7.9 cyc/packet silicon ceiling
- 所有 Agent/*.md RE 文档保留

**从 Phase 2 丢弃的**（Phase 3 不要）：
- T0 `gather_w_col` hoist 代码（QNN `weights_to_vtcm` 替代）
- T1b HVX `pack_weight_32x32`（QNN `ForceFormat_Crouton_b` 替代）
- T1d-1 `col_sum_w` hoist（独立 ReduceSum op 替代）
- T2a HVX combine loop（独立 HVX op 替代）
- T2c HVX `prepack_activation_fused`（`Cast + HiLoSplit + ForceFormat_Crouton` 替代）
- Path B wu-keyed 跨 inference cache（QNN `weights_to_vtcm` 内部做更优）

**可能保留的**：
- Dualacc + `mxswapacc` + `:retain` 语义（Phase 1 RE 成果，继续用在 Phase 3B kernel）
- fused hi+lo K-loop 结构（Phase 3D 可选）

---

## 6. 硬限制（无法绕过）

| 限制 | 影响 | 对策 |
|------|------|------|
| v75 只有 1 个 HMX 单元 | MatMul 本身最大 1 thread | Phase 3 目标是**最大化 HMX 利用率**, 不是 HMX 并行 |
| `weight.n` int4 HMX native 无公开文档 | int4 必须扩到 int8, VTCM 带宽 2× | Int4Expand 解决 (accept 2× VTCM cost) |
| QNN scheduler 黑箱 | 不能直接指派 thread | 靠声明 + 拓扑 hint |
| Crouton 精确 byte layout 仅可从 probe 推 | Phase 3A 是真实险 | 承担, byte-dump 回滚 |

---

## 7. 成功指标

Phase 3 完成后**期望** (SM8650 v75, 512³)：

| kernel | Phase 2 | Phase 3 目标 | vs QNN built-in |
|--------|--------:|-------------:|----------------:|
| w4a16  | 0.32   | **0.01–0.03** | 10–30× w8a16    |
| w4a8   | 0.22   | **0.005–0.02**| 5–20× w8a8      |
| w16a16 | n/a    | **0.05–0.15** | 10–30× QNN w16a16|

"持平"需要 native `weight.n` path（Qualcomm 内部）。"10× 数量级内"已是架构胜利。

---

## 8. 启动即行动

Auto-mode 下逐阶段执行。每阶段：
1. 独立 commit 命名 `Phase3A/3B/...`
2. 命令 / 验证步骤写进 commit message
3. fail 时立即回退到前一 phase，更新本 doc 记录 learning

**今天（2026-04-23）就开始 Phase 3A**。

---

## 附：关键命令清单（快速上手）

```bash
# 环境
source scripts/env.sh

# Phase 3A 起步：复制 Phase 2 skeleton, 改签名
mkdir -p example/hmx_matmul_phase3/{src,kernel}
cp example/hmx_matmul_w4a8/build.sh example/hmx_matmul_phase3/
# ... 编辑 build.sh 改 PACKAGE_NAME, 路径

# Phase 3A probe（目标）
cd example/hmx_matmul_phase3
bash build.sh
bash run_on_device.sh --shape 32,32,32

# chrometrace 查 QNN scheduler 行为
bash example/qnn_matmul_profile/profile_all.sh \
    --device oneplus --connect ssh --arch v75 --shape 512,512,512 \
    --custom-op example/hmx_matmul_phase3/build
# → 读生成的 chrometrace_htp.json, 看 sub-ops 时间轴

# 反汇编 QNN 对照 (Phase 3B fail 时)
hexagon-llvm-objdump -d --mattr=+hmxv75,+hvxv75,+hvx-length128b \
    tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so \
    | grep -A 40 "convert_to_crouton_b"
```

---

**本 doc 起，Phase 3 为当前工作主线。每个 Phase 完成后在此 doc 更新结果表。Phase 2 blocks 所有 "HVX-ify more scalars" 类优化 —— 架构根因不解，单点优化到头**。
