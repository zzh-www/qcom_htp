# GDNSolveHVXMixHMX — HVX-feed + HMX-matmul producer/consumer pipeline (matmul-portion squeeze)

## 🎯 目标架构（权威，solve 重写照此）：静态量化 int16（2026-06-05，sim 验证可行）

**决定:最终走静态量化(固定/分层可预测标度,非运行时 maxabs),精度 int16。** 这是效率最优:一切可提前预处理、
运行时只剩纯向量整数 matmul + 纯 int 加(无量化、无 rescale、无动态扫描、无标量)。一举消掉 merge-glue 的
quant(14%)+ acc-rescale(36% 主项)。

**sim 验证(`scripts/gdn_solve_static_quant_probe.py`,真实 golden p29_L00,端到端 off-diag oc):**
| | oc mean | oc max-head |
|---|---|---|
| int8 dynamic(现状) | 0.0287 | 0.060 |
| int8 **static** | 0.0545 | 0.160 ← 太松(~4× 差于出货) |
| int16 dynamic | 0.00012 | 0.00024 |
| **int16 static** | **0.00024** | **0.00066 ← 比出货 ~1.2e-2 好 50×** |

**结论:int8-static 太松,int16-static 完美**(用户"int8 大不了 int16"精确命中)。值有界(|A|≤0.94,|T|≤1,随
block-distance 衰减 dist0=1.0→dist3=0.20)→ 单一全局 int16 静态标度就够;分层标度(按 block-dist)可更紧但非必需。
**实现注意**:int16×int16×64-deep 最坏 2³⁶ 溢出 int32 → 用 int64 acc 或下移(多数块码值小不溢出,仅 max 块需处理);
或 w8a16(int8 算子 + 16-bit 累加)折中。
**这定调了 solve 重写**:静态 int16 量化 + 全预处理 + vmem 向量读([[向量读 matmul]] 已通)+ pure-add acc。

## ✅ 向量读 matmul 解锁 VTCM operand caching（2026-06-05，用户路线验证成功）

**之前的"vrmpy quant 不可折叠 / 全塞 VTCM 有标量例外"是因为 matmul 用标量读 A —— 正向解决 = 把标量改成向量读。**
方法:看 QNN `matmul_qu8xqi8_32.S`(vmem A + vdelta 广播 + valign,零标量)+ 查 Hexagon HVX 手册 p202 确认
vdelta 语义。
- **`gdn_matmul_i8_vrmpy_vec`**:`vmem` 载 A 整行 + `Q6_V_vdelta_VV(vA, ctrl)` 把 word-g 4 字节广播到全 32 lane
  (ctrl[i]=i&0x7C,手册 p202:vdelta=大步长优先 butterfly,设 stride 4/8/16/32/64 位即跨字复制)+ `valign(·,4)`
  推进。**隔离实测 4.10×(8322→2031),BIT-EXACT**(`first_mismatch=-1`)。坑:vrdelta 广播的是 byte0 不是 word0
  (手册 p202 反向网络)→ 必须用 **vdelta**。
- **解锁 VTCM operand caching**:A 现在向量读 → `GDN_BR_PREQUANT_A` 把 A 块缓存进 VTCM(`vt->acache`)从之前的
  **+112K 灾难** 变成 **−3K 收益**(vmem 读 VTCM 快)。
- **全 solve 实测(4-thread,int8,min)**:scalar-splat **97030** → VEC_MM **93084(−4%)** → +PREQUANT_A
  **89880(共 −7.4% / 1.08×)**,全 bit-exact。

**教训修正**:① "全塞 VTCM 有标量硬例外" 仍真,但**正确做法是消灭标量访问(向量读),而非接受例外**;② operand caching
对 vrmpy 也能赢了(前提:向量读 + VTCM)。flag:`GDN_BR_VEC_MM`、`GDN_BR_PREQUANT_A`(默认关)。
**下一步**:T_kj 算子也向量读+VTCM 缓存(同法);把 acc/requant 的中间量也留 VTCM 向量格式(VTCM-first 全链路)。

## 全局 roofline：32-head solve 的真实瓶颈分解（CLEAN WALL，2026-06-05）

**先纠正一个度量陷阱**：`GDN_BR_PROBE_CYCLES` 的 per-stage `C15:14` 计数器**不可靠**(per-hardware-thread,
QuRT 迁移线程 → 垃圾值)。它报 `zero=52%`,但**消阶段测 wall 的 ablation 证明 zero 只 ~5-7%**。**唯一可信
= wall-delta(`-D<flag>` 消掉一个阶段,测 cyc/head 变化)。** 别再信 PROBE 的 per-stage 数。

**真实分解(干净 wall,int8 vrmpy 路径 `-DGDN_BR_MM_I8`,4-thread min-of-3,baremetal):**

| 部分 | cyc/head | 占比 | ablation 测法 |
|---|---|---|---|
| **merge GLUE(quant + acc + requant,off-diag 块)** | **~49,700** | **51%** | `SKIP_MM` − `DIAG_ONLY` |
| diag(fwd-subst + 对角 requant + zero) | ~28,500 | 29% | `DIAG_ONLY` |
| **matmul compute(vrmpy)** | **~20,000** | **20%** | full − `SKIP_MM` = 98157−78173 |
| —— 其中 zero-fill | ~6,600 | 7% | full − `NO_OUTPUT_ZERO` |
| **full int8 solve** | **98,157** | | (int16 默认路径 = 129,073) |

测量 flag(都在 `GdnSolveBROp.cpp`,默认关,不影响 QNN op):`GDN_BR_NO_OUTPUT_ZERO`(消 zero-fill)、
`GDN_BR_SKIP_MM`(消 vrmpy)、`GDN_BR_DIAG_ONLY`(只 diag)。

**全局结论(权威,覆盖之前所有错误的局部判断):**
1. **matmul compute 只占 20%。** 即使 HMX 把它打到免费(20%→~3%),全局也只 ~1.2×,且 HMX-feed 还引入 pack +
   SMT 争用(前几轮的发现)。**所以纯抠 HMX matmul 不是全局最优。**
2. **zero-fill 只占 7%(不是 PROBE 骗我的 52%)。** "DMA 后台化输出写"只值 ~7%,优先级低。
3. **真正的王是 merge GLUE(51%)= 每个 off-diag 项的 quant(A_ik u16→i8 + T_kj codes→i8)+ acc(K 个 int32
   项带 per-term rescale 累加)+ requant(写结果块)。** 这才是全局最优该攻的地方。
4. **用户的"稀疏下三角省略"洞察应作用在这 51% 上**:off-diag 块 T_ij = −A_ii⁻¹·Σ_k A_ik·T_kj,其中 **T_kj 本身
   是下三角(对角块求逆结果)→ 一半 MAC/quant/acc 在结构性零上 → 可省**。这是降 51% glue 的真杠杆,远胜抠 matmul。

**merge-glue ceiling 分解(干净 wall,ablation flag):**
- per-term quant(A+T)= ~13K(14%)= full − `SKIP_QUANT`(97004→83757)
- **acc + off-diag requant + B-pack = ~34.7K(36%)← merge-glue 最大子项**
- matmul = ~20.6K(21%)

**折叠 quant — 实测 DOUBLE FAILED(负结果,2026-06-05),根因明确**:加 `GDN_BR_PREQUANT_A`(输入 A 每块
预量化一次/head 缓存,merge 读缓存):a8c 进 **DDR scratch(冷)** → 98312→**112227(+14K)**;a8c 进
**VTCM(`vt->acache`)** → **210570(+112K,更糟 2.1×)**。**根因**:`gdn_matmul_i8_vrmpy` 读 A 算子是
**SCALAR splat**(`Q6_V_vsplat_R(aw[g])`,aw=int32* 标量读),而 **scalar 访问 VTCM 是 catastrophically slow**
(skill/[[reference_htp_hardware_scheduling_flow]] 警告"scalar scratch 进 VTCM → 7× 慢")。所以 A 算子**必须在
热 L1**;现状"re-quant 进小而热的 `sc->a8`"对标量 splat 读已是最优——**quant 不是浪费,它把 A 保持在热 L1**。
⇒ **vrmpy 路径的 quant 不可折叠**。**HMX 路径能缓存算子到 VTCM,是因为 HMX 把算子当 crouton VTCM tile 读
(向量访问,快),不是标量** → operand caching 只对 HMX 有效。**通用教训:"全塞 VTCM" 有一条硬例外——SCALAR
访问的数据进 VTCM 是灾难,必须留 L1/L2-cached DDR。** flag `GDN_BR_PREQUANT_A` 留作负结果(默认关,baseline 98179 不受影响)。

**下一步(全局最优,修正后)**:真正的大头是 **acc(36% 里的主项)= 每个 term 的 fixed-point rescale 累加**(因每
term 标度不同)。杠杆:**统一标度让 acc 退化成纯 int32 加**(无 per-term rescale 乘),或**利用 T_kj 三角性跳过
零块的 matmul/acc**。这两个直接打 36%,且不踩"冷缓存比重算贵"的坑。matmul 的 HMX 化排最后。

## 验证用户假设"攒指令一次发射" + QNN native HMX 逆向（2026-06-05）

**用户假设**：让 HMX 不抢 HVX 的关键 = 把要做的指令攒起来、一次 burst 发射(descriptor-driven 自治),
HMX 线程只占极少 round-robin 轮次。

**逆向验证结论:方向对,但字面"一次发射"在 HMX 上不成立;可落地的精确版 = "用最少的 packet,每个 packet
做最多的自治工作,放进 zero-overhead 硬件 loop"。** 证据(`Agent/qnn_re/`):
1. **没有"一条指令做完整个 matmul"**。64³ 必须**流式喂 K 个 weight/activation tile**:`activation.ub=mxmem(r6,r7)`
   + `weight.b=mxmem(r8,r9)` 一次各喂一个 tile。v73deep 和 v75 native 都这样。
2. **两者都用 Hexagon zero-overhead 硬件 loop**(`loop0(addr,cnt)`/`loop1`)流式发射 mxmem —— 这是最接近
   "burst 无开销"的形态:硬件管 loop 计数+分支,**每迭代无 SW 分支开销**,mxmem 背靠背发射。✓ 用户直觉的硬件落地。
3. **QNN 预建 descriptor/param 结构体**(`descriptor_builder` 把 stride/mask/tile-count 一次算好写进结构,kernel
   读它驱动硬件 loop 边界)—— 这是"攒设置"。我的 v73deep 已有(od/ad/mask/ep)。✓
4. **真正可拉的杠杆 = 每 packet 的自治工作量**(决定总 packet 数 → 占多少轮次):
   - **v75 native**:`cvt.uh=acc:2x2`(**一次 drain 2×2=4 个输出 tile**)+ 开头 burst 预取 4 个 `bias=mxmem2`。
   - **我的 v73deep(int8)**:`cvt.ub=acc(r25)` 平凡版(**1 tile/cvt**)+ `:deep:cm`(已有多通道自治)。
   - 实测:P=1 一个 64³ matmul 提交 ~**203 个 packet**(COMMITTED_PKT_ANY)→ consumer 发射密集 → 占很多轮次。
   **若 int8 路径能用更密的 drain/自治模式 → 每 matmul 发射更少 packet → 占更少 round-robin 轮次 → 与 HVX 争用更低。**

**⇒ 关键发现(已逆向确认):int8 输出没有更密 drain 的 headroom。** 全仓库所有 int8/byte-输出 HMX kernel
(`convbbb` v73、`convbbb` v75、`convbbh` v75)**一律用平凡 `cvt.ub=acc`(1 tile/cvt)**;**`:2x2`(4 tile/cvt)
只出现在 fp16/16-bit 输出的 `convhbh`(`cvt.uh=acc:2x2`)**。即:HMX 的 4-tile 密集 drain 是 **16-bit 输出专属**,
int8 输出本质 1-tile/cvt。我的 v73deep 已是 QNN int8 的最密形态 —— **靠"更密 int8 自治模式"压 packet 数这条路不存在。**

**⇒ 但这揭示了一条连贯的真路径:把 HMX 输出从 int8 改成 16-bit(int16/fp16)。** 一举两得:
(a) 解锁 `cvt.uh:2x2` → **每 matmul 的 cvt packet 数 ÷4** → consumer 占更少 round-robin 轮次 → SMT 争用降;
(b) **彻底免掉 int8 的 multi-pass gain-search 税**(文档已证:GDNSolveHVX 的 int16 路径无 multi-pass)。
代价:输出 2× 字节(更多 VTCM 流量,但已知不是瓶颈——瓶颈是 SMT)、下游需消费 16-bit。**这把"HMX 不该和 HVX 抢"
和"免 multi-pass 税"统一成同一个改动:GDNSolveHVXMixHMX 应走 16-bit 输出 + `:2x2` drain。** 这是
[[reference_hmx_dsp_vs_descriptor]] 的核心,也是 solve 重写时 HMX 路径的首选形态。下一步逆向:`convhbh` 的
`:2x2` 配置(descriptor 怎么设)+ HMX 是否支持 int8×int8→int16 累加+`:2x2` drain。

## 探索 lever A（线程→cluster 放置 / 让 HMX 不与 HVX 抢）— 2026-06-05

**结论先行：v75 上无法 pin cluster；优先级无效；真正的杠杆是把 HMX consumer 做成 descriptor-driven（发射少），
而非线程放置。** 这条路径 = QNN 的机制（逆向确认）。

**实测 + 文档 + 逆向：**
1. **无 thread→cluster affinity API**（v75 QuRT `qurt_thread_attr_t` 无 affinity 字段，只有 priority /
   bus_priority / group_id）。**逆向 QNN 确认**：`libQnnHtpV75Skel.so` 导入 `qurt_thread_set_priority` +
   `qurt_sysenv_get_hw_threads/max`,但**无 `set_affinity`/hwthread-pin 符号** → QNN 也 pin 不了 cluster。
2. **`max_hw_threads = 6`**（device 实测 `qurt_sysenv_get_max_hw_threads`）。5 个 SW 线程（4 producer +
   consumer）≤ 6 → **无 oversubscription**，所有线程并发跑。
3. **优先级 = no-op**（实测）：consumer prio=64(高)/producer prio=192(低) vs baseline，consumer-busy ~425
   不变。因为无 oversubscription 时，**Hexagon in-order SMT 的发射仲裁是硬件级 round-robin/公平的**，SW
   priority 只管"oversubscription 时谁上 HW 线程",管不了 issue-slot 轮次分配。
4. **PMU 证实争用机理是 round-robin 轮次摊薄，不是发射饱和、不是内存**：P=4 时 aggregate IPC 仅 **0.85**
   (4-wide 机器,远未饱和)、`SMT_BANK_CONFLICT≈0`、`CYCLES_5T=385/503`。即 5 个线程轮流发射，consumer 的
   packet 流被摊薄 → wall 拉长（busy 214→418）。
5. **QNN 怎么解（逆向 + `docs/qnn_htp_scheduling_and_custom_op_limits.md`）**：静态编译期 per-unit runlist；
   HMX 在独立 thread context（optrace tid 256），HVX 在 512-515；HVX∥HMX 重叠 17-22% **只**靠编译器 supertile
   fusion，且对 custom/plugin op 被 `is_plugin_op` 排除。QNN 不"pin"——它让 **HMX descriptor-driven（发射少、
   让出轮次，HMX 单元自治运行）** + 编译器在指令级交织。

**→ 真正的杠杆(待探,深)**：consumer 抢是因为 v73deep kernel 是 **packet-streaming**(持续发射 mxmem+控制包,
持续占 round-robin 轮次)。把它做成 **descriptor-driven**(一次描述符 kick off 更大的自治 HMX 运行,每 matmul
发射更少包)→ consumer 让出轮次 → producer 填上 → HMX 执行真正与 HVX 并行。这要改 HMX kernel(N-tile 描述符 /
更大 per-descriptor 工作量),是 [[reference_hmx_dsp_vs_descriptor]] 的核心。**注意**:v73deep 是 QNN 出货 kernel,
本身已是 K-MAC 循环;要更 descriptor-driven 需理解 QNN native HMX op 怎么做大块自治(下一步逆向方向)。

**禁止重犯**:别再试 cluster-pin(无 API)或线程优先级(round-robin 公平,no-op)——已实测否决。

## ⛔ 工作方式（权威，置顶，勿违反）

**只跑微基准（microbench）。不要再跑完整 GDNSolveHVXMixHMX 全 solve。**

- 当前**只有微基准是跑通且可信的**（matmul-portion 4P、HMX_BENCH ceilings、P-sweep）。
- 完整 solve 路径 `-DGDNBM_HMX_MERGE_PATH` 在 **open 阶段就 SSR**（`gdnbm_open rc=0x80000406`，
  2026-06-05 复现确认；HVX 基线同设备 rc=0x0 正常 → 是该 build 自身问题，不是 DSP 挂了）。它的
  "434K/head" 数字**无法在当前代码复现**，已作废，别再引用、别再尝试跑它。
- **下一步不是修这个全 solve，而是据微基准结论彻底重写整个 solve**（2-head pack / fused depack /
  2-head eff+bias / #1c / 正确多线程一次性进新实现）。在重写之前，唯一可信的性能依据是微基准。
- 因此所有"GDNSolveHVXMixHMX vs GDNSolveHVX 全 solve 快/慢"的结论**暂缺**，等重写完成后用同 harness、
  同线程数才能下；现在不要据残缺/SSR 的全 solve 数字下任何结论。

## 命名定义（GDN 求逆三种实现）

全仓库统一用这三个名指代 GDN 三角求逆的三种实现，**勿用其他叫法**（如 "HVX-merge"、"HMX-feed"、
"HMX 加速版"、"HVX 直算版"、"pure HMX" 等旧叙述名一律弃用）。注意：这是**叙述性名词**的统一；
代码标识符（宏 `GDN_BR_HVX_MERGE`、函数 `gdn_merge_packed` / `gdn_merge_hvx` 等）保持原样不变。

| 标准名 | 定义 | 代码路径 | 性能 |
|---|---|---|---|
| **GDNSolveHVX** | 纯 HVX 实现（基线，已出货）。64³ 矩阵乘用 HVX int16 vrmpy；对角块求逆（diag forward-subst）用 HVX。**正确的多线程方式：4 个 HVX worker 按 head 并行（#HVX-locked ≤ 4 单元）。** | 定义 `GDN_BR_HVX_MERGE` 时走的 `gdn_merge_hvx` | baremetal 实测 **4-thread 157K cyc/head**（1-thread 414K）。这是当前已验证的基准。 |
| **GDNSolveHVXMixHMX** | HVX 喂数 + HMX 算 64³ 矩阵乘；对角块求逆仍用 HVX。**HMX=1 单元 process-serial，绝不 thread HMX**（skill `htp-hardware-scheduling`）；正确多线程 = HMX consumer 在 main + HVX 并行。 | 默认（不定义 `GDN_BR_HVX_MERGE`）走的 `gdn_merge_packed`；baremetal `GDNBM_FEED_PIPE`/`GDNBM_FEED_4P` 微基准流水线 | **已验证**：matmul 子环节 578 cyc/matmul（2.77× vs vrmpy 微基准）；向量化 bias 已进完整 solve（bit-identical）；multi-pass 真实成本 4190→1218（向量化bias+#1c，oc 安全）。**未做**：把微基准 pack/depack 优化集成进完整 solve + 正确多线程实现。**完整-solve 多线程性能 = 未知，尚无 vs 基线的公平结论**（见文末 "Full-solve status"）。 |
| **GDNSolveHMX** | 全程 HMX（连三角求逆本身也用 HMX，如 divide-conquer）。 | 已探索并否决 | 多算 28–60× 矩阵乘，实测慢 4–6×，三者中最差 |

本文档主要讲 **GDNSolveHVXMixHMX**。

## VERIFY 复测结果（2026-06-05，real v75 `ssh oneplus`，本次 session 全部重跑）

微基准全部复现，与文档记录一致或更好；全 solve 路径 SSR（见置顶禁令）。

| 量 | 文档记录 | 本次实测 | 判定 |
|---|---|---|---|
| GDNSolveHVX 4-thread | 157K | **128–146K**（127541/128317/131675/140586/146189） | ✅ 复现，实测**更快**，157K 偏保守/过时 |
| GDNSolveHVX 1-thread | 414K | **406K**（405895） | ✅ 复现，略快 |
| matmul-portion 4P (cyc/matmul) | ~578/585 | **587/588/603** | ✅ 复现（噪声内） |
| 4P consumer spin /matmul | ~167 | **166/181** | ✅ 复现 |
| P-sweep P2/P3/P4 | 1034/721/585 | **1030–1043 / 713–730 / 588–612** | ✅ 复现，缩放曲线吻合 |
| P-sweep spin (64%→46%→29%) | 667/334/167 | **666–703 / 344–350 / 172–183** | ✅ producer-feed-bound 结论成立 |
| HMX kernel floor (stats[2]) | 215 | **214** | ✅ 复现 |
| bit-exact 门 dep_mism / ovr_mism | 0 / 0 | **0 / 0** | ✅ 复现 |
| consumer ceiling (HMX_BENCH stats[5]) | 388 | **749** | ⚠️ 对不上，但文档自己已声明 388 被 4P 实际 floor ~418/585 取代，属过时次要 ceiling，不影响结论 |
| producer pack floor OLD/NEW (stats[3]/[4]) | ~3139 | **2561 / 2333** | ℹ️ 比记录低；向量化 eff+bias 省 ~228 cyc（isolated 单 head） |
| 全 solve nthreads=1 `-DGDNBM_HMX_MERGE_PATH` | 434K | **SSR `rc=0x80000406`（open 即挂，无法复现）** | ❌ 作废，见置顶禁令 |

**结论：微基准侧的性能优化分析（producer-feed-bound、HMX kernel 215、4P≈585、缩放曲线、bit-exact）全部
站得住；全 solve 的任何数字（含 434K）不可信。**

## NEXT SESSION — START HERE (集成状态 + 计划, 2026-06-05)

**真实基线（已验证，apples-to-apples 用它）**：GDNSolveHVX baremetal **4-thread = 128–146K cyc/head**
（本次实测；旧记 157K 偏保守。1-thread 406K）。这是当前最优。
**禁止跑全 solve（SSR + 即将重写）——只跑微基准，见置顶禁令。**

**GDNSolveHVXMixHMX 在基准代码 (`gdn_merge_packed`) 的集成状态：**
- ✅ **已集成**：向量化 `gdn_pack_bias`（128 scalar→4 vector，bit-identical，commit `2cc2884`）；
  **#1c** 省 PASS2（范数预估 gain，`gdn_effective` 顺带算 colabsmax，oc 安全，**工作区未 commit**）。
- ❌ **未集成**（只在 baremetal 微基准 `gdnbm_imp.cpp`）：2-head pack `fp_pack_act2/wt2`、
  fused depack `fp_depack`、2-head eff+bias `fp_pack_effbias2`。完整 solve 的
  `gdn_get_act_A`/`gdn_get_wt_T` 仍用旧 1-head `gdn_pack_act_crouton8`/`gdn_pack_w8_kmajor` +
  `gdn_depack_out_fast`。
- ❌ **未做**：diag 优化；GDNSolveHVXMixHMX 正确多线程实现。

**完整 solve 现状（实测，别再用错误估算）**：GDNSolveHVXMixHMX nthreads=1 C=256 = **434K/head**（旧 1-head
pack）；per-stage diag 132K / fold 122K（虚高，旧 pack）/ merge 61K / quant 53K / other 59K。
⚠️ 之前文档里的 "65K 三块构成 / 2.06×→2.35× / 输 2.77× / 不能并发" **都是错误估算或不公平比较，已废弃**
（原因见文末 "Full-solve status — CORRECTED"）。

**计划（按优先级）：**
1. **集成微基准 pack/depack 优化进 `gdn_merge_packed`/getter**（2-head `fp_pack_act2/wt2` + fused
   `fp_depack` + `fp_pack_effbias2`）→ 完整 solve 的 fold(122K) 应大降。重测 nthreads=1，`gdn_br.sh` 验 oc 不退化。
2. **commit #1c**（已在工作区，oc 安全），C=256 重测确认收益。
3. **正确多线程 GDNSolveHVXMixHMX** —— 按 skill `htp-hardware-scheduling`：**HMX=1 单元 process-serial，
   绝不 thread HMX**；HMX consumer 在 MAIN，HVX work 多线程（head 并行）。测 4-thread。
4. **apples-to-apples vs GDNSolveHVX**（同 baremetal harness、同线程数、C=256）才能下"快/慢"结论。
5. （独立）**diag 优化分析** —— diag 两路线共享，优化它绝对提速两者（相对倍数几乎不变，但 prefill 墙钟有用）。

**禁止重犯**（见 memory `feedback_read_scheduling_skill_before_hmx_hvx`）：读 skill 先；never thread HMX（→SSR
`rc=0x80000406` 需 reboot）；公平比较（同线程数）前不下结论；`pkill -9 gdnbm` between device runs。

**Reproduce（baremetal，`ssh oneplus` via `scripts/dssh.sh`）：**
```bash
cd example/gdn_native/baremetal
# GDNSolveHVX 基线（4-thread 157K）：
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDN_BR_PROBE_CYCLES" bash build.sh                       # ./gdnbm 4 A_u16_h32.raw /dev/null 32 256 32768 32768 2.77e-05 6.10e-05
# GDNSolveHVXMixHMX 完整 solve（nthreads=1 SSR-safe；多线程是 TODO，勿乱开多 HMX worker）：
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_HMX_MERGE_PATH -DGDN_BR_PROBE_CYCLES" bash build.sh # ./gdnbm 1 ...
# matmul-portion 微基准（578）：
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_FEED_PIPE -DGDNBM_FEED_4P" bash build.sh            # ./gdnbm 4 ...
```
PROBE stats（HMX_MERGE_PATH/HVX 两路通用）：[3]=diag [4]=mergeRun [5]=mergeGlue [6]=fold [7]=quant [8]=other [9]=wall/head。

**Goal:** speed up the GDN triangular-inverse *merge-matmul* portion by running the 64³ matmuls on the
(idle) HMX unit, fed by HVX producers — instead of the GDNSolveHVX baseline's HVX `vrmpy` matmul. Bare-metal only
(QNN custom ops get NO HVX∥HMX overlap).

**Result (real v75 `ssh oneplus`, C=256-shaped 64³ matmuls, chain8-style steady, DUMMY data):**
matmul-portion throughput **1208 → ~507 cyc/matmul = 2.4×**, and **~3.2× vs the shipped-best vrmpy
4-thread (1601 cyc/matmul throughput)**. Journey: 1208 → ~585 (2-head pack + fused depack + 2-head eff+bias)
→ **~507 (lever #A: fuse act-crouton ∥ wt-kmajor, 2026-06-05)**. Now ~80% consumer-bound (spin 97/507).
Consumer-side levers (descriptor hoist, barrier removal, kernel-setup amortize) ALL MEASURED NO-OP /
REFUTED — the ~410 consumer-busy is VTCM-bandwidth CONTENTION (rises 315→410 with P=1→4), not kernel setup
(isolated kernel=214 floor). The only real lever left is reducing VTCM traffic via operand caching in the
SOLVE rewrite. See "Where it's still bound".

## The pipeline

`P HVX producer threads` pack each 64³ matmul's operands into a VTCM ring (crouton act + k-major wt +
bias), a `main-thread HMX consumer` drains the ring (the `our_v73deep_kernel` mxmem). Static-stripe job
distribution, CAS-free volatile slot flags (CAS `__sync` is unreliable on this target — use volatile +
static striping). Reproduce: `example/gdn_native/baremetal`, flags below.

## Clean ceilings (single-thread, wall-based — the only trustworthy timing)

| quantity | cyc/matmul | note |
|---|---|---|
| HMX kernel (continuous, descriptors set once) | **215** | mxmem floor |
| **consumer drain ceiling** (zero+kernel+depack, hot loop) | **388** | the pipeline's theoretical floor |
| producer 1-head pack floor (eff+act+wt+bias) | **3139** | the wall — pack is 8× the HMX consumer |

⚠️ **Per-thread `pcyc()` sub-timings are GARBAGE** — `C15:14` PCYCLE is per-hardware-thread and QuRT
migrates SW threads. Only total wall + single-thread isolated loops are reliable.

## Optimization journey (each step measured)

| step | cyc/mm | lever |
|---|---|---|
| orig pipeline (3 producers, 1-head, scalar eff+bias) | 1208 | |
| **+ vectorized eff+bias** | 833 (1.45×) | `gdn_pack_bias`'s 128 SCALAR int32 VTCM writes were a hidden 35% (eff+bias 1093→215). Scalar VTCM writes = the documented pathology. Replaced with HVX col-sum + **4 vector stores** (`fp_pack_effbias`). Also un-stuck 2-head. |
| + 2-head producers (3P) | 713 | act/wt pack ILP now visible (was masked by the scalar eff+bias) |
| **+ 4th producer (3-stage)** + 2-head | **~635 (1.9×)** | consumer made PURE-HMX → frees its HVX unit → a 4th HVX producer becomes legal |
| **+ fused one-pass depack** (`fp_depack`) | ~612 | kill the `surf_sub` VTCM round-trip: fold base-subtract into the de-crouton loop (read `surf` once, sub inline). Isolated depack 512→~420 (~1.2×); bit-exact (dep_mism=0). |
| **+ 2-head interleaved depack** (`fp_depack2`) | ~592 | interleave the pair's two depack streams (ILP, mirrors `fp_pack_act2`); distinct out buffers avoid store aliasing |
| + drop producer out-zero | ~592 (neutral) | verified `our_v73deep_kernel` fully OVERWRITES the out surface (ovr_mism=0) → pre-zero unnecessary. No wall gain (was overlapped) but removes 32 vec-stores/matmul; kept as verified-safe simplification. |
| **+ 2-head interleaved eff+bias** (`fp_pack_effbias2`) | **~585 (2.06×)** | interleave the two column-sum chains; consumer spin 184→168/matmul (producer genuinely faster); bit-exact (eff_mism=0) |
| **+ FUSED act-crouton ∥ wt-kmajor** (`fp_pack_actwt2`, lever #A, 2026-06-05) | **~507 (1.16× over 585; 2.4× overall)** | crouton is ALU(vand/vor/vror), kmajor is permute(vshuff) — SEPARATE HVX resources. Fusing the two packs into ONE 32-iter loop co-issues ALU∥permute in the same VLIW packet. **Isolated: act+wt 2038→1563 = 1.30×, BYTE-IDENTICAL** (o[14]/o[15] mism=0). Pipeline 4P: spin 167→97 (producer feeds faster) → ~573→~507. Build flag `-DGDNBM_FUSED_ACTWT`. |

## Three findings that unlocked it (all device-verified)

1. **Diagnosis inversion:** consumer ceiling **388 ≪ pipeline 1196** ⇒ the pipeline is **producer-feed-bound**,
   consumer idle ~68% — NOT consumer-bound. (Earlier "consumer-bound" was wrong; the clean ceiling
   measurement settled it.) And 2-head not helping at first = the feed wasn't compute-bound *because*
   the scalar eff+bias dominated.
2. **The HMX kernel needs only `hmx_lock`, NOT `qurt_hvx_lock`** (verified: pure-HMX consumer → `open rc=0`,
   no fault). So the consumer can be HVX-lock-free ⇒ it doesn't consume an HVX unit ⇒ 4 HVX producers
   + 1 pure-HMX consumer = legal (vs the 3+1 cap when the consumer locks HVX for zero/depack).
3. **HVX-unit rule (root cause of the thread cap):** `#HVX-LOCKED threads ≤ #HVX units (=4)`, because
   `qurt_hvx_lock` blocks when no unit is free (`qurt_hvx.h:145-148`). Oversubscribing (5 HVX-locking
   threads) hangs and **SSRs the cDSP** (`gdnbm_open` → `rc=0x80000406` until reboot; non-root can't
   force-clear). Landed in `.codex/skills/htp-hardware-scheduling` + `docs/htp_hardware_scheduling_flow.md`.

## 4-producer 3-stage state machine (CAS-free)

Slot states `g_fp_ready[k]`: `0=free` / `1=packed` / `2=hmx-done`. 4 HVX producers (static stripe) each:
spin until target slot ∉ {1}; if state==2 **depack** the previous occupant (cold cross-thread read of the
HMX output) and zero; pack (2-head pair); set state 1. Main pure-HMX consumer (in job order): spin
state==1; run kernel; set state 2 (a producer reuses+depacks it later).

## Where it's still bound + why we stopped

**UPDATE 2026-06-05 (lever #A landed):** the act∥wt fuse cut the producer's exposed pack cost → 4P is now
**~507** and the bottleneck SHIFTED to the consumer: spin fell 167→**97** (19% idle), consumer-busy
(=cyc−spin) is ~410 both before/after — i.e. we're now ~80% consumer-bound, approaching the pure-HMX
consumer floor ~418 (kernel 215 + descriptor-build + 2 barriers ~203/iter). **The earlier "act/wt
crouton-pack can't be cut without more HVX units" was WRONG** — it assumed the two packs must run
serially; fusing them overlaps ALU∥permute and bought 1.16× with ZERO extra HVX units.

**Consumer-side levers — BOTH MEASURED NO-OP (2026-06-05), don't retry:**
- *descriptor hoist* (pre-build all FP_K `od`/`ad` once vs per-iter struct build): **no-op** (507→518,
  within noise). The per-iter 6-field struct build is NOT the cost — the expensive mxmem descriptor setup
  is INSIDE `our_v73deep_kernel` (re-configured every call), which hoisting my outer struct can't touch.
- *barrier removal* (drop the 2 `__sync_synchronize`): **no-op on throughput.** Removing the release barrier
  cut consumer-busy 406→362 but raised spin 97→144 by the same amount (cyc stayed ~503) — the barriers are
  already fully hidden under producer feed. So they're free; keep them (correctness).

**Why both fail — DECISIVE measurement (2026-06-05): the consumer-busy is VTCM-BANDWIDTH CONTENTION, NOT
kernel setup.** Swept P on the FUSED build, consumer-busy (=cyc−spin) rises MONOTONICALLY with producer
count: **P=1→315, P=2→355, P=3→394, P=4→410** (each added producer slows the consumer's kernel ~30-40 cyc).
The isolated kernel (HMX_BENCH, NO producer) = **214** = the mxmem config-once hardware floor for a 64³
matmul. So consumer-busy ≈ 214 (HMX floor) + contention (~100 at P=1 → ~196 at P=4 from producers hammering
the shared VTCM/L2 bus).

⇒ **Lever "#1 amortize the kernel's mxmem setup" is REFUTED before touching asm:** the kernel is ALREADY at
its 214 config-once floor in isolation; the pipeline's ~410 is pure shared-bus contention. No v73deep
kernel-asm change can cut the mxmem hardware floor OR the inter-thread VTCM contention. (My earlier "needs a
kernel-internal change" was wrong — corrected.) This re-confirms the doc's "average-bandwidth bound" thesis
and the scheduling skill's "bottleneck is data movement, not compute."

**The ONLY real lever left = reduce VTCM TRAFFIC (less contention), and it lives in the SOLVE, not the
kernel or pipeline:** the microbench re-packs both operands every matmul (worst case → max producer
traffic → max contention). The real block-recursive solve REUSES T_kj / A_ik across many i,j, so an
operand cache (pack once, reuse) cuts producer VTCM writes → less contention → consumer-busy drops toward
214. So the microbench's 507 is contention-PESSIMISTIC; the rewrite's operand caching is where the next
real gain is. Secondary producer levers (fuse eff+bias / depack into the actwt loop) only cut the 97 spin,
capped by the contention-inflated consumer-busy.

**Producer eff+bias fusion — MEASURED NO-OP (2026-06-05, don't retry).** Built a 3-way fused
`fp_pack_actwteff2` (crouton + kmajor + eff col-sum, 2 wt rows/iter) to hide eff+bias's ~250/head under
the actwt loop: 512 → 512, spin ~97 → ~97 (unchanged). Two reasons, both predicted: (1) eff's col-sum is
`vadd` = ALU, the SAME unit crouton already saturates (lever #A only won because crouton-ALU ∥ kmajor-PERMUTE
are DIFFERENT units) → no co-issue; (2) the extra wt loads add VTCM traffic = the contended resource. So
the 97 spin is NOT cheaply removable from the producer side — confirms we're at the contention-bound floor.
The only remaining traffic lever is operand caching in the solve rewrite (kmajor + eff both read the full
64×64 wt — sharing that load is a rewrite-time win, not a microbench loop tweak).

### Operand-cache ceiling + the REAL floor is SMT thread contention, NOT VTCM traffic (2026-06-05)

Built an OPCACHE measurement mode (`-DGDNBM_OPCACHE`): pack every slot ONCE (operands resident), then the
steady loop's producers ONLY depack + re-arm — modelling the real solve reusing T_kj/A_ik across i,j.
Measured (real v75, 4-thread, min-of-4):

| mode | cyc/matmul | spin | consumer-busy (=cyc−spin) |
|---|---|---|---|
| FUSED (pack+depack every matmul) | 518 | ~100 | **~418** |
| OPCACHE (cached operands, depack only) | **469** | ~50 | **~418** |
| OPCACHE + NODEP (no depack either) | 473 | ~50 | **~420** |
| OPCACHE + NODEP, **P=1** | 439 | ~131 | **~309** |

**Two decisive reads:**
1. **Operand caching = real but MODEST ~1.1× (518→469).** The win is ALL in spin (100→50): producers do
   less work → feed faster → consumer waits less. Worth banking in the rewrite, but not transformative.
2. **The consumer-busy ~418 floor is HW-THREAD (SMT) CONTENTION, not VTCM data traffic — PROVEN:**
   - Removing depack (NODEP) did NOTHING (473≈469) → data traffic isn't the bound.
   - FUSED (producers MAX busy packing) and NODEP (producers idle-spinning) give the SAME consumer-busy
     ~418 → if it were traffic, max-traffic FUSED would be worse. Equal ⇒ it's thread *presence*, not bytes.
   - consumer-busy scales with thread COUNT: P=1→309, P=4→418 (the isolated kernel with ZERO producers=214).
   So the consumer's HMX thread loses issue-slots/memory-ports to the 4 producer HW threads regardless of
   what they move. **No data-movement lever (operand caching, depack elimination, crouton-out direct
   transcode) can cut the ~418 consumer-busy** — only fewer contending threads could, which trades against
   feed (P-sweep already showed P=4 optimal). This is the architectural floor of the 4P pipeline.

**Net:** the microbench is at ~470 (cached) / ~507 (uncached) and that IS the 4-HVX-silicon floor for this
producer/consumer topology. Operand caching banks ~1.1× for the rewrite; the deeper floor is SMT contention,
not addressable by traffic reduction. (Reproduce: `-DGDNBM_OPCACHE [-DGDNBM_OPCACHE_NODEP]` on the FUSED 4P
build; off by default, baseline unaffected.)

#### Hexagon-doc cross-check of the SMT-contention root cause (V75 PRM, verified 2026-06-05)
The "consumer-busy rises with thread count, independent of data traffic" finding is the documented v75
microarchitecture, NOT a guess. The Hexagon manuals confirm it three ways:
- **It's SMT with shared per-cluster execution resources.** V75 *HVX* PRM §1.2.2 (p9-10): "Multiple hardware
  threads execute in parallel, each with a different vector context… number of vector contexts is
  implementation-defined" (=4 on this SKU). V75 PRM PMU (p128): each cluster has **"cluster (private)
  execution resources"** — threads on the SAME cluster share the issue slots / register ports; v75 has
  **2 clusters** (PMU events count "cluster 0"/"cluster 1", p138-139).
- **The exact conflict counters exist** (V75 PRM ch.9 PMU, p135-140): `SMT_PKT_SLOT_CONFLICT_*` = "In-cluster
  SMT thread is **not picked due to a slot conflict between the primary thread and SMT thread**";
  `SMT_CONFLICT_FOR_REG_READ` = blocked on a register-read port; "**inter-thread SMT bank conflicts**" (p135,
  memory banks); "thread was not picked or there was an **inter-cluster resource conflict**". These are
  exactly the issue-slot / reg-port / bank contention that inflates the consumer's kernel when producer
  threads co-run — and they fire regardless of how many BYTES the producers move (matches NODEP==depack).
- **The hardware meters throughput by concurrent-thread count:** `CYCLES_N_THREAD_RUNNING` and
  `COMMITTED_PKT_N_THREAD_RUNNING` (N=2..6, p126/139) — i.e. committed-packets/cycle is defined as a function
  of how many threads are simultaneously running, which is precisely our consumer-busy 214(1 thr)→309(2)→418(5).
**Implication:** since v75 has 2 clusters and contention is worst *within* a cluster (primary+SMT share
slots), the only architectural lever would be thread→cluster *placement* (keep the HMX consumer on a cluster
by itself), which QuRT controls, not us — a deep, separate probe. Confirms: no data-movement optimization
can break the ~418 consumer-busy.

#### DIRECT PMU silicon measurement of the SMT floor (2026-06-05) — the third, definitive proof
Read the v75 PMU on-device via the **QuRT API** (`qurt_pmu_set(QURT_PMUEVTCFG, …)` + `qurt_pmu_get(QURT_PMUCNTn)`
— note: `HAP_user_pmu.h` is NOT linked in the bare-metal skel, `__HAP_register_pmu_group` is a null weak symbol;
`qurt_pmu_*` resolves at load like `qurt_thread_*`). PMUEVTCFG packs four 8-bit raw event codes from PRM
Table 9-1. Build `-DGDNBM_PMU` on the OPCACHE+NODEP path (producers do ZERO data work — pure thread-presence).
Thread-occupancy spectrum, per matmul:

| P (producers) | threads | cyc | busy | CYC_2T | CYC_3T | CYC_4T | CYC_5T |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 439 | 312 | 14 | 0 | 0 | 0 |
| 2 | 3 | 393 | 332 | 13 | **341** | 0 | 0 |
| 3 | 4 | 439 | 393 | 6 | 0 | **352** | 0 |
| 4 | 5 | 486 | **441** | 4 | 0 | 0 | **386** |

The diagonal lights up exactly: at P producers (=P+1 threads) the `CYCLES_(P+1)_THREAD_RUNNING` counter (raw
codes 0x3c/0x3d/0x3e/0x0a) owns the window — **at P=4, 386/486 = 79% of every matmul has EXACTLY 5 threads
running concurrently**, and consumer-busy climbs in lock-step (312→332→393→441). A separate run measured
`SMT_BANK_CONFLICT` (raw 0xb9) ≈ **0-1/matmul** → it is NOT memory-bank contention. So the silicon directly
confirms: the consumer's HMX thread is co-executing with the producer HVX threads ~79% of the time, the
co-running count is what inflates its busy cycles, and it is execution-resource (SMT issue-slot) sharing —
the PRM's `SMT_PKT_SLOT_CONFLICT` mechanism (raw 0x320, a 10-bit code needing extended PMUEVTCFG encoding,
not read here) — NOT data movement (bank-conflict ≈ 0). Reproduce: `-DGDNBM_PMU` on the OPCACHE+NODEP 4P build.

### Best practices distilled (v75 HVX-feed / HMX-consume pipelines)
1. **The floor of a producer/consumer HVX→HMX pipeline is SMT execution-resource contention, not VTCM
   bandwidth.** Verify with the PMU (`CYCLES_N_THREAD_RUNNING` + `SMT_BANK_CONFLICT`) before chasing a
   data-movement lever — if bank-conflict≈0 and CYCLES_(maxthreads)_T owns the window, you're SMT-bound and
   traffic optimizations (operand caching, depack elimination, layout) will only shrink the *spin*, never the
   consumer-busy.
2. **More producer threads buys feed, but each co-running thread costs the consumer ~30-40 cyc/matmul of
   busy** (SMT slot sharing). The optimum is the smallest P that drives spin→~0; past that, added threads only
   raise contention. Here P=4 (spin already ~50-100) is the knee.
3. **The only sub-floor lever is thread→cluster placement** (v75 = 2 clusters, "cluster-private execution
   resources", PRM p128): isolating the latency-critical HMX consumer on its own cluster would cut in-cluster
   SMT slot conflicts. QuRT-controlled (priority / `qurt_sysenv` / hardware-thread pinning) — an open probe.
4. **PMU is your ground truth on this silicon.** `qurt_pmu_*` works from the unsigned user PD; program
   PMUEVTCFG with raw 8-bit event codes from PRM Table 9-1 (≤0xff), read PMUCNT0-3 deltas around the window.
   `HAP_user_pmu.h` (which takes itrace 0x8xxx IDs) is unavailable here.

### (superseded) original stop-point at ~585
Producer-bound at ~585 (vs the 388 HMX floor); consumer spun ~168/585 (~29% idle). The conclusion "this is
the practical limit of the 4-HVX-unit silicon, can't be cut without more HVX units" was **refuted by lever
#A** — same silicon, 585→507 via ALU∥permute co-issue.

**Two levers that did NOT pay off (measured, then reverted/neutral):**
- *Pack-before-depack reorder* (to hide the cross-thread `.out` read latency under pack compute): no-op,
  587/601/611 vs 592/598/601 — within noise. The cross-thread VTCM read isn't an exposed gap; reverted.
  (Note: l2fetch is the WRONG tool here — `.out` is VTCM-resident, not L2/DDR-cached.)
- *Drop the producer out-zero*: wall-neutral (was already overlapped), but kept because it's a verified-safe
  simplification removing real work.

### Scaling curve (P=2/3/4 sweep) — corrected bottleneck model (2026-06-04)

Swept producer count P in 4P mode (`./gdnbm <P> ...`); each row min-of-2, real v75:

| P (HVX producers) | cyc/matmul | consumer spin | consumer busy (=cyc−spin) | per-producer cost (=cyc×P) |
|---|---|---|---|---|
| 2 | 1034 | 667 (64%) | 367 | 2068 |
| 3 | 721  | 334 (46%) | 387 | 2163 |
| 4 | **585** | 167 (29%) | 418 | 2340 |
| 5 (extrapolated) | ~495 | — | ~432 | ~2476 |
| 6 (extrapolated) | ~435 | — | ~446 | ~2612 |

**Three reads of this curve:**
1. **Producer-feed-bound, confirmed independently:** consumer spin falls 64%→29% as P grows 2→4; adding
   producers directly buys throughput. Scaling P=2→4 is 1034→585 = **1.77× (88% of the ideal 2.0×)** — mildly
   sublinear, not a wall yet.
2. **Light VTCM/L2 bandwidth contention (the sublinearity):** *both* the per-producer cost (2068→2340) and
   the consumer's own busy time (367→418) rise with P — each added producer slows every thread ~13% / ~14%.
   This is the bandwidth fingerprint the route doc predicted; the bound is VTCM/L2 *bandwidth*, not capacity.
3. **The real floor is the PURE-HMX consumer at ~418, NOT 388.** The 4P consumer does only kernel(215) +
   descriptor-build + 2× `__sync_synchronize` (~203 fixed/iter) — it can never go below ~418 even with
   infinite producers. The old "388 ceiling" was the *non-parallel* consumer's zero+kernel+depack hot loop;
   it does NOT apply to the 4P pure-HMX consumer. Extrapolating the two lines, the producer-feed rate would
   only cross the consumer floor at **P≈6** (~440 cyc/matmul) — so even a 6×128B SKU lands near ~440, not 388.
   On this 4-HVX-unit part, ~585 is the practical optimum.

**Next lever IF the consumer ever becomes the bound (P≥5 hardware):** cut the consumer's ~203 fixed/iter —
hoist the `od`/`ad` descriptor build out (only `outtab`/`acttab` change per slot; pre-store them in the slot)
and replace the 2 full barriers with one-way volatile acquire. Pointless today (consumer has 167 slack).

### Can we re-schedule the 4 HVX to "use bandwidth better"? — NO (tested 2026-06-04)

The ~13% contention is **conserved average-bandwidth, NOT burst collision** — so reordering the work can
only *move* the contention between threads, never reduce the total. Two experiments settle it:
- **Phase-stagger** (even producers depack→pack, odd pack→depack, so ~half read / ~half write VTCM at any
  instant): no-op on throughput (575→578). It DID cut consumer spin 168→158 (producers fed sooner) but the
  consumer's own busy time ROSE 412→427 by the same amount — the contention just shifted producer→consumer.
  This is the signature of an average-bandwidth bound, not a peak-collision one.
- *(small real win, kept)* one acquire barrier after the spin instead of one per depack: ~585 → **~578**.

**Implication:** scheduling/relayout (phase-stagger, bank-aware slot placement, read/write specialization)
cannot beat ~578 here — the VTCM traffic is already the HMX-mandated minimum *for one HMX run*. The ONLY way
down is to **reduce VTCM traffic itself** (algorithm-level). See the next section for the real lever.

## Algorithm-level traffic-reduction map (2026-06-04) — the microbench HIDES a 2-3× factor

**Headline finding:** the microbench measures ONE HMX run per "matmul", but the real `gdn_merge_packed`
(GdnSolveBROp.cpp:854) runs the HMX kernel **2-3× per logical 64³ matmul** — a dynamic-quant gain search:
- **PASS 1** (`gdn_hmx_run_only`, loose gain g1) → `gdn_surf_maxabs(out)` → **result thrown away**, only max|P| kept
- **PASS 2** (refine gain gr, almost always taken since code1>0) → maxabs → **thrown away**
- **PASS 3** (tight gain g2 = 127/maxP) → `gdn_depack_out_fast` → the ONLY useful output

Why: int8 HMX output must be scaled to fill [-127,127], which needs max|P|, which needs P itself → chicken-
and-egg, "solved" by running the matmul 2 extra times just to measure its own output magnitude. oc is
hypersensitive to this scale (comment: dist-1 blocks oc 0.73 vs 0.01 if scale is ~20% off), so the passes
can't just be dropped.

**VTCM traffic per LOGICAL matmul (corrected):** ~3 runs × (read act+wt+bias 8.5K + write out 4K) + 2×
maxabs-read out (8K) + depack-read out (4K) ≈ **~50K**, of which **PASS1+2 ≈ 33K (~66%) is pure
scale-probing that is discarded.** So the real GDNSolveHVXMixHMX matmul-portion is ~2.5-3× the microbench's 578 — and
the #1 traffic lever is killing the probe passes, NOT the depack round-trip or scheduling.

**Optimization points, ranked by traffic saved:**
1. **Kill / cheapen the multi-pass gain search (≈2-3× — by far the biggest).** Options, each oc-gated:
   (a) *predict* max|P| from input norms (‖A_i,:‖·‖T_:,j‖ bound) → 0 probe runs, but oc-risky (scale must be
   ~1% accurate); (b) replace the PASS-1/2 *full HMX runs* with a cheap HVX sub-sampled dot-product estimate
   of max|P| (keeps accuracy, drops 2 HMX runs + 2 out-writes); (c) tighten PASS-1's initial gain (norm-based)
   so PASS 2 is unnecessary → 3 passes → 2 (saves 1/3). **Must validate oc vs golden** (scale-sensitive).
2. **HMX-accumulate the inner-product** `S_ij=Σ_k A_ik·T_kj` (10 matmuls/head): accumulate K terms into one
   `out` → depack once per block, not per term. Bounded by (i) `extra_param` accumulate-mode availability
   (today ep={1,0}=overwrite, verified) and (ii) per-term scale alignment; NB=4 is shallow so ~small.
3. **crouton-out → k-major-wt direct transcode** (out codes are reused as the next stage's wt): skip the
   row-major intermediate. Tangled — codes also feed int32 widen + requant. Small-medium.
4. **Already correct, keep:** operand cache (act A_ik / wt T_kj packed once, reused across i/j — amortizes the
   pack writes); intermediate scratch in DDR not VTCM (keeps it OFF the contended VTCM bus).

**Note:** the microbench's single-run 578 is the right number for the *steady GDNSolveHVXMixHMX throughput of one run*;
the real-solve speedup vs GDNSolveHVX (151K/head) must account for the 2-3× multi-pass factor — i.e.
GDNSolveHVXMixHMX only wins once lever #1 lands. This reframes "is GDNSolveHVXMixHMX worth it": the squeeze done so far
is necessary but not sufficient; the multi-pass gain search is the gate.

### SIM VERDICT on lever #1 (scripts/gdn_solve_maxp_probe.py, real golden p29_L00, 2026-06-05)

Replayed the real block-recursive forward-subst, measured actual max|P| vs norm predictors at all 512 merge
matmuls. Output-quant relerr (lower=better): **PASS3-exact (true max|P|, today) p50~0.022 / p90~0.037**;
**pure norm-prediction as the final scale p50~0.078 / p90~0.168** (K=1 Holder bound, never saturates but
3-4x worse; block-recursive propagation would push oc up — too lossy to BE the output scale).

**BUT lever #1c is SAFE and lands ~33%:** the Holder bound's output FILL is ~37 (>=5-bit), enough for PASS-1's
integer maxabs to be accurate in ONE shot. So replace PASS-1's LOOSE constant (fill~1 -> needs PASS-2) with
the Holder norm-predicted gain -> PASS-1 measures max|P| accurately -> **drop PASS-2**; PASS-3 still uses the
MEASURED max|P|, so **oc is unchanged** (relerr stays 0.022, not the predicted 0.078). Saves 1 HMX run +
1 maxabs/matmul ~= 33% of the path's VTCM traffic. The Holder norm (act row-1-norm max x wt max) is ~free:
`fp_pack_effbias` already computes wt column-sums; act row-sums piggyback on the act-pack.

**STRATEGIC fact uncovered:** the shipped/baremetal GDNSolveHVX path (`gdn_merge_hvx`, int16) has NO multi-pass
at all — int16 output doesn't need the int8 gain search. Multi-pass is a tax UNIQUE to GDNSolveHVXMixHMX (int8-out).
So even with #1c (3->2 passes) GDNSolveHVXMixHMX still carries 1 probe pass GDNSolveHVX doesn't. The
GDNSolveHVXMixHMX-vs-GDNSolveHVX decision weighs: HMX raw matmul is faster, but pays a ~2x (post-#1c) probe-traffic tax GDNSolveHVX
avoids. #1c lives in `gdn_merge_packed` (the QNN `solve_br_op` DEFAULT path; baremetal forces GDNSolveHVX);
oc-validate via `solve_op/standalone` net-run vs golden.

## Bigger-picture context (does the solve even need to be this fast?)

- **Whole C=256 GDN chunk ≈ 2 ms is hopeful** — IF the chunk is fused into one bare-metal kernel and the
  **HVX-bound solve overlaps the HMX-bound big matmuls (A/U/W/P/v_new/oc/S_out) across heads**. Then
  wall ≈ HMX work (~1.9 ms), and the solve's HVX work HIDES under it. So the solve may not need to be
  squeezed at all — but this 2.52× is banked. See `reference_htp_hardware_scheduling_flow`.
- Alternatives explored and REFUTED for the *inverse itself*: Taylor/Neumann/squaring/Newton iterative
  matmul (‖A‖≫1 explodes), GDNSolveHMX (divide-conquer; 28–60× more matmuls → more glue, HMX never
  saturates, ~4–6× slower). Hybrid (HVX forward-subst diagonals + HMX/vrmpy merges) is optimal. See
  `gdn_solve_taylor_probe.py` / `gdn_solve_divconq_probe.py`.
- Hardware what-ifs: more HVX units → sublinear (~1.4× at 8, bandwidth + single-HMX walls); a 6×128B SKU
  exists. **VTCM capacity adds ~0** at C=256 (8MB ≫ working set; the bound is VTCM/L2 *bandwidth*).

## Reproduce

```bash
cd example/gdn_native/baremetal
# ceilings + producer floor + vectorized-eff+bias gain:
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_HMX_BENCH" bash build.sh   # gdnbm 1 ... -> stats[2]=HMX kern, [3]/[4]=pack OLD/NEW
# the pipeline (matmul-portion throughput), pick one:
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_FEED_PIPE"                 bash build.sh   # 3P 1-head  -> ./gdnbm 3 ...
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_FEED_PIPE -DGDNBM_FEED_2H" bash build.sh   # 3P 2-head (713)
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_FEED_PIPE -DGDNBM_FEED_4P" bash build.sh   # 4P 3-stage 2-head (~635)
#   deploy build/{libgdnbm_skel.so,gdnbm} to $HOME/gdnbm_run; pkill -9 gdnbm BEFORE each run; cap threads ≤4
#   ./gdnbm <P> A_u16_h32.raw /dev/null 32 256 32768 32768 2.770166930875267e-05 6.103701895199438e-05  -> stats[0]=cyc/matmul
```

## Depack-squeeze DONE (2026-06-04) — plateaued ~635 → ~585

The matmul-portion squeeze the user requested (2026-06-04) is complete; landed in `gdnbm_imp.cpp`:
- **`fp_depack`** (one-pass fused depack) + **`fp_depack2`** (2-head interleaved) — replace the producer's
  `gdn_depack_out_fast` call. `GdnSolveBROp.cpp:gdn_depack_out_fast` is UNCHANGED (still the bit-exact path
  for the real solve op).
- **`fp_pack_effbias2`** (2-head interleaved eff+bias) — replaces the two serial `fp_pack_effbias` in
  `fp_pack_slot2`.
- Producer out-surface **zero removed** (verified kernel overwrites).
- **HMX_BENCH stats fixed + extended** (stats[3] double-write gone): stats[5]=consumer ceiling,
  stats[6]=old two-pass depack, stats[7]=new fused depack, stats[8]=depack-mismatch (==0),
  stats[9]=overwrite+effbias2-mismatch (==0). All three correctness gates PASS on device.

**Verdict:** plateaued — 1c (reorder) was a no-op, lever 2 (no-zero) wall-neutral, lever 3 (~2%). Going
below ~585 needs >4 HVX producers (impossible) or cheaper crouton pack. The remaining stretch lever (act/wt
crouton-pack lane waste, 32/128 masked) is near the floor and not worth it.

## DEFERRED / TODO (the real next steps)

- **Integrate the matmul-portion microbench wins into the real `gdn_merge_packed` + getters** (2-head
  `fp_pack_act2/wt2`, **fused `fp_pack_actwt2` [lever #A, byte-identical, 1.16×]**, fused `fp_depack`,
  `fp_pack_effbias2`, #1c) — they are validated in isolation but the full-solve path still uses the OLD
  1-head pack + `gdn_depack_out_fast`. Re-check oc vs golden after.
- **Implement GDNSolveHVXMixHMX multithreading per the skill** then measure (see status below).

## Full-solve status (2026-06-05) — CORRECTED (prior "loses 2.77× / can't parallelize" was WRONG)

⚠️ An earlier version of this doc concluded "GDNSolveHVXMixHMX loses to GDNSolveHVX by 2.77× and can't
parallelize." **That was wrong — removed.** The errors:
1. **Unfair comparison** — GDNSolveHVXMixHMX nthreads=1 (434K) vs GDNSolveHVX nthreads=4 (157K). Not apples
   to apples. GDNSolveHVXMixHMX's 4-thread number was never correctly measured.
2. **Un-integrated pack** — the full-solve `gdn_merge_packed` still uses the OLD 1-head pack, so its
   fold/pack is inflated (fold 122K). The 2-head pack / fused-depack microbench wins are NOT yet integrated.
3. **Skill violation → SSR misread as "can't parallelize"** — I spawned 4 HMX workers, which the skill
   `htp-hardware-scheduling` explicitly forbids (HMX = 1 unit, process-serial, NEVER thread HMX; put the HMX
   consumer on MAIN, HVX work multithreaded). That oversubscription SSR'd the cDSP; it is an implementation
   bug, NOT evidence that GDNSolveHVXMixHMX can't be threaded. (There is 1 HMX unit, not 2.)

### Confirmed-effective (device-verified, real)
- **matmul-portion microbench**: HMX matmul + 2-head pack + fused depack + 2-head eff+bias =
  **578 cyc/matmul** (2.77× vs the vrmpy microbench 1601). commit `2cc2884`.
- **vectorized `gdn_pack_bias` LANDED in the full solve**: 128 scalar VTCM writes → 4 HVX vector stores,
  **T relerr BIT-IDENTICAL** (gdn_br.sh C=128: 8.538e-3, same to every digit; HMX path only). commit `2cc2884`.
- **multi-pass real cost** (MULTIPASS microbench, P=3, consumer-bound): scalar-bias 3-pass **4190** →
  vectorized-bias 3-pass **1720** (vectorizing the per-pass bias saves ~60% — scalar `gdn_pack_bias` was the
  pathology) → 2-pass #1c **1218** cyc/matmul.
- **#1c** (norm-predicted PASS-1 gain via `gdn_effective` colabsmax → drop PASS-2): in `gdn_merge_packed`;
  sim + gdn_br.sh verified **oc unchanged** (off-diag 21.2%→21.5%, baseline already 21%). Working tree (uncommitted).

### Full-solve current measurement (reference only — NOT GDNSolveHVXMixHMX's potential)
GDNSolveHVXMixHMX, nthreads=1, C=256, **OLD 1-head pack**, baremetal `-DGDNBM_HMX_MERGE_PATH`: 434K/head —
diag 132K / fold 122K (inflated by old pack) / merge 61K / quant 53K / other 59K. Single-thread +
un-integrated, so this number does NOT represent the path's potential.

### Baseline reference (GDNSolveHVX, this baremetal harness, C=256)
nthreads=4 = **157K/head**; nthreads=1 = 414K/head. (Shipped QNN-op numbers + metric alignment:
[[project_gdn_solve_handwritten_route_2026-06-03]], `docs/cycle_metric_alignment.md`.)

### Honest verdict
**GDNSolveHVXMixHMX's full-solve multithreaded performance vs GDNSolveHVX is UNKNOWN** — it needs (1) the
microbench pack/depack wins integrated into `gdn_merge_packed`, and (2) correct multithreading per the skill
(HMX-on-main consumer + parallel HVX, never threaded HMX). Only then is a fair verdict possible. What IS
proven: the matmul-portion is 2.77× faster on HMX, and the vectorized bias + #1c cut the multi-pass tax
~3.4× (4190→1218) oc-safely.
