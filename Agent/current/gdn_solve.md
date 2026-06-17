# GDN 三角求逆 solve（统一文档：三路线 + 现状 + 实现 + 口径）

GDN/KDA linear-attention 在 v75 HTP 上的核心难点 = 每 head 对 C×C 下三角矩阵 `L=I-A` 求逆。
本文是这件事的**唯一权威文档**：三种路线、当前进展、实现、度量口径、经验。
矩阵乘 kernel 本身（u8i8 / w8a16 / **w16a16**）见独立文档 `docs/w16a16_is_two_w8a16.md`。

---

## 0. 当前最佳（权威数）

> **当前最快路线 = pure-HMX `GDNSolveHMX`，~1.258M（VTCM-only，−26% vs native 1.703M，cron#89，生产默认 lean+wt-cache+B，oc 4.238e-3 synthetic / 1.107e-2 真 GDN，bit-exact-to-native）。release doc = `docs/gdn_inverse_pure_hmx.md`；过程/极限/dead-end = `pure_hmx_solve_build.md`。** 下面 §0 详情是**另一路线** GDNSolveHVXMixHMX（精度最佳档 oc 3.10e-3 / 速度 ~1.79M）的出货定稿，作并行路线保留（release doc `docs/gdn_inverse.md`）。**📌 2026-06-17 更正:三方案 QNN-标准同口径稳态对比 + trace 口径统一(MM 只在 consumer)+ 旧 "HVXMix HVX 利用率低 / 已到地板" 措辞更正,见 §1.2;权威稳态表 = `Agent/current/perf_3impl_cron82kqie.txt`。**

**GDNSolveHVXMixHMX，`GdnSolveBR16.cpp` int16 静态 solve，producer-consumer pipeline（P=4 HVX 生产者 + 1 主线程 PURE-HMX 消费者）。出货旗组 7 个（见 §3）。**

| 指标 | 值 |
|---|---|
| **精度 oc**（设备 T vs host fp64 `inv(I−A)`，32 头） | **3.10e-3**（vs 旧 u8i8 int8-drain 基线 1.37e-2 = **4.4×**；由 SBOOST+FBOOST 达成,见 §1.1） |
| **32-head TOTAL wall**（= QNN `graph wall` = `gdnbm wall=t1-t0`，PCYCLE） | 冷态 ~1.79M（≈1.12ms @1.594GHz）；**绝对值随设备态漂(冷~1.7-1.8M/热~1.9-2.1M),只作参考** |
| vs GDNSolveHVX 4-thread 基线 | **~2.2×**（基线 ~3.97M） |
| 正确性 | 对角块 bit-exact；DIAG_I16/REQ_FUSE/SBOOST 全 bit-exact 验证，FBOOST oc 兑现 oracle |

> **对外展示文档 = `docs/gdn_inverse.md`（图为主：架构/算法/性能）。本文 = 工程权威源。优化全台账 = `gdn_opt_ledger.md`。**
> **口径铁律（详见 §5.1，唯一标准 = QNN optrace 字段）：唯一权威终指标 = C=256, 32-head TOTAL wall（= QNN `graph wall`，PCYCLE）。** per-head（tiler 88K artifact）、min-of-reps、per-stage PROBE **全禁**。**只比 同字段+同 shape+同场景**（跨字段 `num_dominant_path` vs `cycles_used`、跨场景 单 conv vs 批摊销 = 假矛盾）。**wall 比对用交替全交织 A/B（ACAC… median）消热漂,不与固定常数比。**

### 1.1 本轮采纳的 4 项优化（出货默认开）

| 旗 | 作用 | 增量 |
|---|---|---|
| `GDN_BR_SBOOST` | Sacc drain gain 逐 d 标定 {5.5,12,20}（Hölder 界松 16×） | oc 1.37e-2→~5e-3，**零 wall** |
| `GDN_BR_FBOOST` | final drain @sTw/2（主导误差源）+ 下游 bit-exact ÷2 re-narrow | oc→**3.10e-3**(1.27×)，wall +1.3% |
| `GDN_BR_DIAG_I16` | 对角 forward-subst int16 直写,省 int32 widen/narrow round-trip | wall **−6.2%**，bit-exact |
| `GDN_BR_REQ_FUSE` | final-merge widen+requant 融合一遍读,省冗余 VTCM 读 | wall **−0.9%**，bit-exact |

可选 min-wall 档 `GDN_BR_SKIPFIN_D3`:跳 d=3 块 final merge,wall 再 −2.7%,oc 9.56e-3(贴 1e-2);非默认。

### 1.2 QNN-标准三方案稳态对比 + trace 口径统一（2026-06-17）

> **本小节是三方案（SHIP u8i8 / W16_N8 ARES / pure-HMX 生产）在 cron#82 同一稳态口径下的权威对比,作废所有从单-rep trace 图读出的利用率%(58%/82% 等全是 trace artifact)。** 权威表 = `Agent/current/perf_3impl_cron82kqie.txt`(reproduce 段在文件头)。

**(1) trace 口径统一(MM 只在 consumer)。** 三方案聚合 timeline 已对齐 pure-HMX 口径:producer **不再标 MM** —— 全代码库唯一 stage-3(MM)push = `gdnbm_imp.cpp:1521` 主线程 consumer tid(单 HMX 单元被 consumer 独占)。HVXMix(SHIP/ARES)producer 过去把整段同步 merge 握手(fill→spin→depack)误标为自身 tid 的 MM,双计 consumer 的 HMX 窗口、把 producer 行画成橙色并掩盖真 SPIN;现已改为 producer 只发 SPIN(11)+DEPACK(10)(+ACT/WT-PACK 预派发)。证据 = `Agent/current/timeline_3impl_aggregate.svg`(自检:1664 个橙 MM 矩形全在 CONS 行,producer 0 橙;CONS MM-Σ 逐字段对上 run 的 cons= 字段)+ `Agent/current/trace_kqie_production_unaffected.txt`(本轮 trace 改动全在 `#if defined(GDN_BR_TRACE)` guard 内或纯注释,SHIP 生产 TU 预处理 0 残留,生产字节不受影响已证)。**⚠️ 该 SVG 是单-rep trace,只看 STAGE 结构,per-thread 利用率/占用以下方稳态表为准。**

**(2) QNN-标准三方案稳态表**(cron#82 同口径:32-head TOTAL wall / VTCM-only / reps2-N median / P-sweep):

| 字段 | SHIP u8i8 | W16_N8 ARES | pure-HMX(生产) | 口径 |
|---|---|---|---|---|
| wall (P=4, reps2-N med) | 1.892M | 2.506M | 1.259M | graph-wall stats[0], VTCM-only |
| cbusy(HMX Σ) / HMX 闲 | 0.140M / 92%闲 | 0.297M / 88%闲 | 0.387M / 69%闲 | stats[3] consumer mxmem Σ |
| feed_Σ (HVX work-volume Σ) | 6.894M | 9.145M | 3.643M | stats[4] / pure 分解Σ |
| b_serial (常数-K 拟合 ⚠️) | 0.849M | 0.767M | 0.396M | wall=K/P+b 拟合值; ⚠️ ARES 此值混入 contention 曲率, 真固定底 honest-tail≈0.25M(见(3)) |
| K/feed_Σ (feed 并行占比) | 0.61 | 0.76 | 0.95 | slope ÷ 实测 feed_Σ |
| HVX-util/thread (稳态推导,非 PMU) | 96% | 94% | 83% | (feed/P)/lmax |

口径自洽校验:pure-HMX feed 几乎不随 P 涨(+4.2%),常数-K 拟合成立、b_serial 0.396M ≈ cron#82 独立测 0.376M;**ARES feed 随 P 涨 19.8%,常数-K 拟合不成立,其 b_serial 0.767M 是 artifact(见(3) honest-tail 修正)**。三方案都 feed-bound(HMX 闲 69-92%)。

**(3) 关键结论(2026-06-17 analyzer 收敛终判,3 实验 cron#82 口径钉死;dev 钉死数据 `perf_ares_bserial_pin.txt` + analyzer 对抗复算)。**
- 🔴 **更正(被推翻的 artifact 措辞):"HVXMix 的 HVX 利用率比不上 pure-HMX" 是 ERROR** —— 真值相反:HVXMix HVX 瞬时利用率反而**更高**(SHIP 96% / ARES 94% vs pure 83%)。旧单-rep trace 图的 "58% vs 82%" 是 trace artifact,**已作废**。
- 🔴 **"串行 merge-GLUE 是 ARES 候选杠杆" = REFUTED**。按 **honest-tail**(`wall − 实测feed_Σ(P)/P`,不假设常数 K)用 4-pt P-sweep 原始点复算:ARES 真·固定串行底 @P4 = **0.254M**,跨 P flat,**反而低于 pure-HMX 0.349M**(因 ARES HMX cbusy 0.289M < pure 0.387M)。上表 `wall=K/P+b` 回归得的 b_serial 0.767M 是 **contention 曲率被折进截距的口径 artifact**(feed_Σ 随 P 涨 19.8% 违反常数-K 假设),**ARES 没有"多出 pure 的串行胶水"**。(典型"两数差 ~2× 先疑口径":回归 b 0.77M vs honest-tail 0.25M 差 3× = 测量模型错,非真串行。)
- **自洽分账(ARES 2.522M vs pure 1.260M @P4,差 +1.262M 几乎全在 feed/P=K 段,固定底差 ≈ 0):**
  - raw work-volume **+0.982M(主因,在 K)**= W16 双-pass **2.08×** 单线程 HVX 活(P=1 干净 feed 7.57M vs pure 3.64M)。**这才是 ARES 慢的真·最大差异项**(非 contention、非串行底)。
  - contention **+0.375M wall-equiv(在 K)**= feed_Σ 随 P +19.8% = gather/SMT issue-slot 争用(pure 仅 +4.2%),抬 effective K、NOT P-可约。
  - 固定串行底 **−0.095M(在 b)**:ARES 反更低。
  - pack(wt-pack)per-inst 4113 ≈ pure 4026、在 K(cap-delta×4≈1.26M=K 的 18%)**不在 b_serial** —— dev 旧"pack 0.46M=b_serial 主成分"措辞数量+归属双错,撤销。
- **唯一干净杠杆 = `gdn_w16n8_untile_cv` 输出重排 64× vgather → vxor**(`GdnSolveBR16.cpp:709`):5 窗 ACAC cap 实测全消上界 **−4.24%**(真 vxor 实得 <4.2%);**推翻旧 EST −5~12%**。
- ⚠️ **"HVXMix 已到地板" 现钉死**:ARES **clean 地板 ≈ 2.41M**(保 oc 1.16e-3 bit-exact;= 唯一干净杠杆 untile≤4.2% 兑现后)。越过 → pure-HMX 1.259M **需精度/算法授权**(减 matmul/降 Taylor 阶/降 pack 精度 = 放弃 W16 高精度档,干净路到不了)。精确对角(EXP3 cap 1/16 残漏 −1.9%≈≤0.05M、mostly SMT 藏住)非干净路大杠杆。"off-diag merge MATMUL 对齐 pure-HMX" 维持(per-inst pack 贴平 pure)。(pure-HMX 自身 cron#86 地板不受影响。)
- 🟡 **仍 EST/待补**:真 vxor 实得增益(<4.2%,待实现 + ==BASE oracle bit-exact 门);对角残漏 ≤0.05M。

**(4) 复现。** 三方案 reproduce 段(SHIP/ARES/pure build+run flags、A_ares.raw 扩展 A 装配)= `Agent/current/perf_3impl_cron82kqie.txt` 文件头 REPRODUCE 段;trace blob 抓取 = `scripts/gdn_capture_3impl_traces.sh`;聚合 SVG 重绘 = `scripts/gdn_3impl_aggregate_timeline.py <ship.raw> <ares.raw> <purehmx.raw> <out.svg>`(blob 归档于 `Agent/current/trace_blobs_3impl/`)。

---

## 1. 三种路线（命名权威，全仓库统一）

| 标准名 | 定义 | 矩阵乘在哪 | 对角块求逆在哪 | 现状/角色 |
|---|---|---|---|---|
| **GDNSolveHVX** | 纯 HVX | HVX vrmpy(int8)/`gdn_matmul_i16`(int16) | HVX forward-subst | 基线 ~3.97M，只测不改 |
| **GDNSolveHVXMixHMX** | HVX 喂数 + HMX 算 matmul | **HMX**（u8i8 mxmem） | **HVX forward-subst（并行 4 单元）** | 备选路线；**SHIP = HVXMix 默认档**(u8i8, ~1.83–1.89M, oc 3.10e-3) / **ARES = W16_N8 备选高精度档**(oc 1.16e-3, ~2.5M, gated)（release doc `docs/gdn_inverse.md`） |
| **GDNSolveHMX** | 全程 HMX（连对角求逆都 matmul） | HMX | HMX（Taylor(3)/Newton=0 矩阵乘） | **最佳实践(推荐,最快) ~1.258M（VTCM-only，−26% vs native 1.703M，cron#89，生产默认 lean+wt-cache+B）；release doc `docs/gdn_inverse_pure_hmx.md`，过程/极限 `pure_hmx_solve_build.md`** |

> **角色定位（2026-06-17 用户裁定）：`GDNSolveHMX` pure-HMX = 最佳实践（最快 ~1.258M，推荐）；`GDNSolveHVXMixHMX` = 备选路线，其内 SHIP(u8i8) = HVXMix 默认档、ARES(W16_N8) = HVXMix 备选高精度档；`GDNSolveHVX` = 纯-HVX 基线（只测不改）。**

代码标识符不变：`gdn_merge_hvx`(`GDN_BR_HVX_MERGE`)、`gdn_merge_packed`、`our_v73deep_kernel` 等。

---

## 2. 算法（block-recursive 三角求逆，C=256, BL=64）

C=256 切成 4×4 个 64-块下三角。`L=I-A`：
- **对角块 `T_ii = L_ii^-1`**：
  - HVX 路 = forward-subst（串行递归，但跨 head/块在 4 个 HVX 线程并行）。
  - 纯 HMX 路 = Taylor+Newton-Schulz（p=3 Taylor + 4 Newton 步 → A^64=0 精确，**11 个 64³ matmul/块**，全 HMX；`gdn_solve_taylor_newton_probe.py`）。
- **off-diag merge `T_ij = T_ii @ Σ_k A_ik @ T_kj`**（i>j）：逐 k 的 64³ matmul，HVXMixHMX 放 HMX。

每 head 的 64³ matmul 计数：

| C | 对角块（Taylor+Newton 若全 HMX） | off-diag merge(Σ_k) | 纯 HMX 合计 | HVXMixHMX 放 HMX |
|---|---|---|---|---|
| 128 | 2×11=22 | 2 | **24** | 2（对角走 HVX） |
| 256 | 4×11=44 | 16 | **60** | 16（对角走 HVX） |

---

## 3. 实现（文件 + 构建 + 运行）

**三文件**：
- `solve_br_op/src/GdnSolveBR16.cpp` — **基线**，干净 int16-only 静态 solve（`gdn_br_one_head16`），后续在这改。
- `baremetal/src/gdnbm_imp.cpp` — FastRPC 驱动 + pipeline（`pipe_producer`×P + 主线程 PURE-HMX consumer，`g_hmx_dispatch` 钩子）。
- `solve_br_op/src/GdnSolveBROp.cpp` — 共享 helper（pack/effective/merge_packed/diag/HMX kernel）+ 旧 int32 solve。

**构建 / 运行**（默认即最佳，7 旗）：
```bash
cd example/gdn_native/baremetal
EXTRA_DEFS="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL \
  -DGDN_BR_SBOOST -DGDN_BR_DIAG_I16 -DGDN_BR_REQ_FUSE -DGDN_BR_FBOOST" bash build.sh
# 设备（dssh = ControlMaster 复用,见 scripts/dssh.sh；DSSH_HOST 覆盖设备名）
source scripts/dssh.sh; dssh_open "${DSSH_HOST:-<your-v75-device>}"; W=$(dssh 'echo $HOME/gdnbm_run')
dssh "cat > $W/libgdnbm_skel.so" < build/libgdnbm_skel.so
dssh "cat > $W/gdnbm" < build/gdnbm; dssh "chmod +x $W/gdnbm"
dssh "cd $W && GDNBM_REPS=8 LD_LIBRARY_PATH=$W:/vendor/lib64:/system/lib64 \
  ADSP_LIBRARY_PATH='$W;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' \
  ./gdnbm 4 A_u16_h32.raw T.raw 32 256 32768 32768 2.770166930875267e-05 6.103701895199438e-05"
# 签名: <nthreads> <A.raw> <T.raw> <H> <C> <zpA> <zpT> <sA> <sT>
```
- 数据：`A_u16_h32.raw`(u16,zpA=32768) / `T_ref_h32.raw`(fp32 golden,源自 Qwen3.5-4B GDN 层,`scripts/gdn_extract_golden.py` 重生成) / `/tmp/T_avtcm.raw`(bit-exact 基准)。
- 取数：warmup 一次丢弃 + 一个 FastRPC session 跑 `GDNBM_REPS=8`，**取 reps 2–4 median**（rep1 冷,rep≥5 节流）。**绝不取 min。**
- 精度校验(设备实测 oc 3.097e-3,2026-06-11)：拉回 `T.raw`,host `oc=‖T_dev−inv(I−A)‖/‖inv(I−A)‖`(numpy 片段见 `docs/gdn_inverse.md §5`)。

---

## 4. 路线速度对比（口径统一,2026-06-09）

### 单个 64³ matmul 的真实 HMX cycle
| kernel | 隔离 64³ | 流水内真实 mxmem/matmul |
|---|---|---|
| u8i8(int8×int8) | 216(设备纯)/417(sim) | **313** |
| w8a16(int8 wt×int16 act) | ~462(sim) | ~626 |
| **w16a16 = 2× w8a16** | — | **~626 只是 MAC 折算;真实单 64³ 调用 ~10.4K(固定开销主导,设备实测 2026-06-11,下节)** |

> **HMX matmul 只占 HVXMixHMX wall 的 7%**：512 个 merge matmul 真实 mxmem = 160K / 1.78M。每 matmul 握手 glue ≈ 2950 cyc = **9.4× 浪费**。**瓶颈是 glue，不是 matmul。**
>
> **（2026-06-13 复核 → 修正，权威 = `int16_matmul_cycle_model.md`）** 口径厘清:**纯 kernel 时间(latency=dominant-path) native int16 64³ = 256,u8i8 = 176,只 1.45×**——不是 6×。本节下文的 "1167/6×" 是 `HmxU16I16ToU16MatMul`(2×w8a16 软件分解)按 **HMX-busy 吞吐** 测的,不是 native 单 convhhh 的 latency(4 个 byte-pass 流水,latency≪throughput)。
> **裁决修正:int16-HMX 求逆"不重开"是 KILL 早了**——那 roofline 用了吞吐口径(6×/1365)。inverse 是 **producer-bound(HMX 仅 7% busy,大半空闲)→ merge 该用 latency 口径**;512 merge u8i8(176)→int16(256) 只给 HMX 关键路加 ~41K。决定成本是 **producer 的 weight-pack**:ledger #13/#18 实测 = **8.4%(~150K),在 wall 临界路(kmajor vshuff,非 SMT 隐藏,HW 不可约)**;int16 权重 2 字节 → pack ~翻倍 → **+~150K**(diag forward-subst 是 SMT 隐藏,不涨 wall)。**修正 roofline ≈ 1.78M + 150K + 41K ≈ ~1.97M(+11%),oc 1.16e-3** —— 不是 2.32M(KILL 用了吞吐,错),也不是免费的 1.78M。**这是 precision-Pareto 点,且优于现有 BP4 精度档(2.71M/4.90e-3)双指标**。手写 `GDN_BR_W16`=4.31M 之所以更差,是手写 int16 **没流水**(HMX-busy→92% 吞吐 bound),非核慢(同一 byte-identical convhhh,native 跑 256)。**待 S1**:手写 int16 流水到 native 的 256 latency + 实测 pack delta。下文"1167 地板→两端都输"按吞吐口径成立,但**对 producer-bound 的 merge 不适用**,见 cycle 模型 Decision 段。

### merge mm 换 w16a16（GDN_BR_W16，2026-06-11）— 精度 12× 兑现，速度否决

实现 `-DGDN_BR_W16`：HVXMixHMX 的 merge mm 全替换为 w16a16（`our_v73deep_kernel_i16`，K-stack n_act_pairs=2d、padded 2048B 64³ 描述符，act 流 ×128 / wt ×256 int16 codes，固定 1/32767 drain 落静态链 sTw/256）。设备 A_u16_h32 实测（vs u8i8 同状态对照 reps2-4 中位）：

| | u8i8（现行） | w16a16（GDN_BR_W16） |
|---|---|---|
| 32-head TOTAL wall | 1.69M | **4.31M（×2.55，否决）** |
| HMX busy | 132K（8%） | **3.98M（92%，瓶颈翻到 HMX）** |
| oc vs fp64 | 1.17e-2 | **1.16e-3（12×）** |
| Sacc bit-exact vs numpy | — | max\|code diff\|=3 ≤ 4 ✓ |

死因 = **字节流量，不是 MAC**。"313×4≈1.2K"的预期错在 ① padding ×4（2048B 块只活 512B，唯一字节双射证过的布局；compact 512B 已设备否决）② lo/hi 双 pass 让 act/out 各流两遍。单调用流量 u8i8 ~12KB vs padded w16a16 ~136KB（act 32K×2 + out 32K×2 + wt 8K），@~13B/cyc ≈ 10K——10.4K 全被流量解释。**形态阶梯（vs u8i8 313/64³，全设备实测）：**

| 形态 | per-64³ | vs u8i8 | 备注 |
|---|---|---|---|
| u8i8 in-pipe | 313 | 1× | 现行 merge |
| w16a16 native supertile 地板 | 1167 | **3.7×** | QNN 整图大 op 才有（act-stream/drain 叠去别的单元） |
| w16a16 M=256 carrier 实测 | 5844 | 18.7× | padding 消失，双 pass + drain 串行残留 |
| **w16a16 padded 64³ in-pipe（本轮）** | **10.4K** | **33×** | padding×4 + 双 pass + per-call setup |

"≈4×" 只在 supertile 极限成立；64³ 粒度 33×。384 调用 ≈4M HMX 串行 = wall 地板；即便拿到 1167 地板，16eq×32head=600K 也贴 40% 门。**勿再用 w16a16 做 64³ 粒度 merge。**

实现保留可复跑：`EXTRA_DEFS="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL -DGDN_BR_W16"`；bisect 旗 `-DGDN_W16_NOSTACK`（K-stack→d 次 K=64+HVX acc）、`-DGDN_W16_DBG_S`（dump Sacc/操作数 vs numpy）。

**下一步（supertile 思路自建，容忍精度损失）**：1167 地板的本质 = act-stream/byte-decompose/drain 不占 HMX。自建等价 = **拆成 u8i8 byte-pass、由 producer 喂、HMX 只跑 313/调用的 u8i8 kernel**：act/wt 各拆 hi/lo 字节 → 1 个 64³ = 4 次 u8i8 pass（≈1252/eq ≈ 地板），HVX 组合 `(Ah@Wh)·65536+(Ah@Wl+Al@Wh)·256+Al@Wl`；hi-pass int8 drain 误差 ×256 ⇒ 输出 ≈14–15 bit（vs fused 16 bit，损失可容忍）。半档 a16w8/w8a16 = 2 pass（626/eq，16eq×32 = 320K = HMX ~18%）。**Oracle 裁决（`scripts/gdn_solve_bp2_oracle.py`，32 头镜像静态链，2026-06-11）：u8i8 5.30e-2 / 2-pass 4.73e-2（仅 1.12×，不够）/ 4-pass B=4 **1.77e-2（3.0×）** → 直接做 4-pass**（inner HH+LH+HL 三 dispatch + final 三 dispatch ≈ 3× u8i8 HMX ≈ 400K ≈ 23% < 40% 门）。Oracle 调通的静态档（设备照抄，调错任何一档 oc 爆到 0.3）：hi-drain boost **B=4**（B=8 部分头 clip，max code 232）；LH 档 gain g·B/4（lo 字节近均匀，raw ~2× hi）；HL 档 g·B/8（colabs(Wl)~8×COLABS）；final lo 档 /32（实测 max code 2523 必 clip）；act +128 常数 = colsum 修正 **drain 后整数加回**（进累加器必 clip）。oc 比例映射设备 ≈4e-3，需设备验证。

### 设备裁决（GDN_BR_BP4 已实现，2026-06-11）：精度门 PASS（2.9×），速度门 REFUTE（×1.6）

| 指标 | u8i8（现行基线） | **GDN_BR_BP4** | 门 | 判 |
|---|---|---|---|---|
| oc vs fp64（A_u16_h32, 32头） | 1.37e-2 | **4.90e-3（2.9×）** | ≤1.05e-2 | ✅ |
| 32-head TOTAL wall（reps2-4 中位） | 1.69M | **2.71M** | ≤1.87M | ❌ |
| HMX busy | 8% | ~10% | ≤40% | ✅ |
| producer busy | ~49%（SPIN 51%） | ~95%+（timeline busy 58%+PREP 留白≈40%，SPIN 仅 3%） | >85% | ✅* |

死因 = **producer HVX 工作量本身**（非 glue、非 HMX）：byte-pass 每 merge 比 u8i8 多 ~8.4K cyc（双 surface depack ×2、Sacc/T 组合、wt 16-bit 拆+双 eff/pack、final wt 双发），32head/4producer ÷4 = +1.0M wall，无并行可借（u8i8 producer 51% SPIN 余量只折 ~0.86M）。已做的优化：全操作数 VTCM 缓存(slot 0xE0000)、L pass 融合(Al/Ah×Wh/Wl2 一次 K=2·64 dispatch)、pair-job(HH+L 单握手)、d>1 双 L pass sat-add——共 3.0M→2.71M，再深(组合直读 crouton、批 dispatch)只剩 ~0.2M 级，地板 ~2.5M ≫ 门。

实现复跑：`EXTRA_DEFS="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL -DGDN_BR_BP4"`；探针旗 `-DGDN_BP_DBG_S`(dump HH/Sacc(1,0))。已踩坑（写回 oracle 解释）：① `vshuffo/vshuffe` 拆字节 = 行对转置, 用 `vdeal -1`；② final 三 pass drain gain 含 ×65536(=256·256)；③ acttab+32 撞 outtab,pair-job 第二表用 acttab+64/+96；④ bpc 缓存区超 0x20000 踩邻槽 → slot 0xE0000。**用途定位：BP4 = oc <5e-3 的备选档,以 +60% wall 换 2.9× 精度,默认不出货。**

**后续翻案（SBOOST，2026-06-11 同日）**：BP4 的精度收益 ~98% 其实只来自 drain boost——u8i8 原路仅把 Sacc drain gain ×4（`-DGDN_BR_SBOOST`，1 行）即 oc 1.37e-2→**5.01e-3**，wall 与基线同日差 0（1.91M vs 1.91M 热态）。Holder 界 16× 松是真地板；maxSacc 码 87/127（32 头），B=8 clip。oracle F=1 B=4 预测 4.99e-3 兑现。**新出货候选 = u8i8+SBOOST**；台账见 gdn_opt_ledger.md。全部复用现有 u8i8 kernel + producer 基建,只加 hi/lo 打包与 HVX 组合。两个真坑（已修，勿回退）：① wt 栈序 = [nt][half][kt 0..2d-1]，d=1 缓存块按 16-vec quarter 重排；② act 块必须 2048B padded（compact 512B 设备否决，kernel 读越界），32K/act → A act 不缓存（use-once 临时槽×3），Tdiag 缓存 nat 8K 每用重打包。

### HVX vrmpy matmul ∝ N³（`scripts/gdn_mm_chunk_sweep.py`，sim）
BL=32→1068, 64→7311, 128→52239, 256→402447。HMX 随 size 暴跌（setup 摊销）→ **HMX 比 HVX 便宜 17.5×(64³) → 32×(256³)**；HMX 最小块=64（M_t 必偶,32 要 pad）。

### 纯 HMX 路线的算力地板与数值约束（极致设计与极限 = `pure_hmx_solve_build.md` §6）

纯 HMX 把对角块求逆也做成 matmul（**Taylor(3)/Newton=0**，设备上 Newton 反效=精度甜点），全程 **24 matmul/head**（8 对角 + 16 merge）= 32 头 **768 total**（代码 `gdn_pure_solve.cpp`，`GP_LEANCHK_LIVE` checked=768 佐证）。**旧 "60/head（44 对角 + 16 merge）" 是旧 Taylor+Newton 设计，已废。**
本节只钉**算力地板 + 数值约束**这两个不变量;完整的极致设计（4 杠杆 + roofline 阶梯 + 落地路径）在 `pure_hmx_solve_build.md` §6。

**HMX 算力地板（QNN 原生 256³ matmul `by_htp_type_cycles`，`example/qnn_matmul_profile/output_*_*_256/optrace/summary.json`）：**

| family | 原生 256³ HMX 算力 | per-64³(÷64) |
|---|---|---|
| u8i8（`ConvLayer_s1.opt`） | 12,435 | **194** |
| w8a16（`ConvLayer_s1.opt`） | 30,182 | **472** |
| w16a16（`HmxU16I16ToU16MatMul`） | 74,670 | **1167** |

w16a16 ≈ 2×w8a16 + drain（证实"w16a16 = 2×w8a16 + 排空"）。**全 int16×int16 的 HMX-busy 地板 = 1167/64³**；
非迭代的 merge 可降 w8a16/u8i8（§6 杠杆 3）。1167 是**像原生那样把子块摊进大 op**的吞吐地板（不是单调用 latency——
那是 256，见 `int16_matmul_cycle_model.md`）。

**数值约束（对角 dtype 的根因 = §6 杠杆 4 要攻的）：**
对角块 `A_ii` 严格下三角 nilpotent（`A^64=0`），Neumann `I+A+…+A^63` 有限精确；但 `A∈[-1,1]` 下 `‖A‖₂`
中位 2.36 / p90 8.14 / max 31（`gdn_solve_taylor_newton_probe.py`），中间幂 `A^k` **瞬态**爆掉 int16
（peak 1.9e13）→ **未预条件时对角必须 w16a16**。注意这是**非正规瞬态**（谱 ρ=0，不是真发散）：对角均衡
`D⁻¹AD`（D=diag(sⁱ)）把 ‖Ã‖<1 即可令 `Ã^k` 有界 → 换 u8i8/w8a16 对角。这是 §6 杠杆 4 的数值前提，待 probe 验。

**当前实测**：纯 HMX solve（`gdn_pure_solve.cpp`，`GDNSolveHMX`）= **生产默认 = lean+wt-cache+B，wall ~1.258M（VTCM-only，reps2-8 median，−26% vs native escape `-DGP_NO_LEANMM` 1.703M，cron#89，commit 89927c4）**。三杠杆均**已提为生产默认（不再 gated / 不再 pending）**:lean consumer 核 −18.3%(cron#83,commit d075ccd)+ wt-cache −7.6%(cron#87,commit 01c5376)+ lever B 提生产 −3.1% / cron#88 gated −3.9%(cron#89,commit 89927c4)。3 个正交 escape(全 `#ifndef` default-ON):`GP_NO_LEANMM` / `GP_NO_WTCACHE` / `GP_NO_INPLACE_RENORM`。全程 bit-exact（oc 4.238e-3 synthetic 逐位不变 / 真GDN 1.107e-2;PACKCHK=0;`GP_LEANCHK_LIVE` max|d|=0 checked=768；默认 vs `-DGP_NO_INPLACE_RENORM` int32 oracle 整 32-head 4MB T RAW md5 逐字节相同）。**干净(非精度)路已到结构地板(cron#86 钉死);≤1.0M 不在干净路可达,需精度/算法权衡(减 matmul / 降 Taylor 阶,越过 oc 甜点),需用户授权,当前未做。优化线收口。** 详情 + 极限/dead-end = `pure_hmx_solve_build.md`(cron#89 收口态);release doc = `docs/gdn_inverse_pure_hmx.md`。**📌 历史数据点(已被 cron#87/#89 取代,勿当现态):cron#82 lean gated 态 1.410M(−18.6%,`GP_LEANMM` gated,瓶颈翻转 FEED-BOUND 在此首测,证据 `Agent/current/perf_baseline_cron82_leanmm.txt`);❌ 推翻 ~~"consumer-occupancy-bound AT FLOOR 1.31M / producer-bound:lmax 1.53M > consumer 1.30M,wt-pack(2.52M Σ)#1"~~ + ~~"−42% ceiling"~~(保留作审计链)。** ⚠️ **cron#77 PMU+FANOUT 终定(推翻 cron#76 的 "复现 walk-流水到 230K"):(1) packet 不是问题 —— n_tiles=8 实测 111 packets < native 130(587 是 chain_qdq n_tiles=64 过切);(2) 真根因 cyc/packet stall(14 vs 2);(3) **(↓cron#78 推翻)** ~~native 290 经 M-fanout 实测 UNREACHABLE —— 批 1/2/4/8 块 per-64³ 收敛 1320 walk-floor NOT 290(批只摊 prologue,convhhh kernel 单次 invocation 内 walk 不能跨 conv 流水)~~ → **cron#78 证伪:290 是批里 ConvLayer 的 HMX 子-op `cycles`(warm 263)非 conv-wall;我们单 conv 1576 ≈ native single 1970,不慢;FANOUT 测的不是跨-conv 流水。** ⚠️ **cron#78 终修正(权威细节见 `pure_hmx_solve_build.md` ④f-cron#78):(a) 我们单 conv convhhh 1576 ≈ native SINGLE conv `mm_1x1x64x64` 1970,NOT 慢;"native 290" 是批 `mm_64` 里 ConvLayer 的 HMX 子-op `cycles`,不是 conv-wall(native 批整图 HVX-bound,每-conv 真 wall ~4087)。(b) consumer 真杠杆 = 写精简流式核 1576→~363/conv(convhhh 比干净 dilate 臃肿 4.3×),NOT "流水"——双缓冲 fill‖drain 跨-conv 流水实测 ZERO 增益(HMX 单 acc 无 bank);cron#77 的 "290 UNREACHABLE / consumer 地板 1.01M" 措辞均撤。(c) ~~但 producer-bound(slowest-prod-life 1.53M > consumer 1.30M)⇒ consumer 提速对 wall 仅值 ~11%。#1 杠杆 = producer feed(wt-pack vgather)~~ → ~~cron#81 SPEC:consumer-occupancy-bound at floor 1.31M(臃肿-bound)⇒ #1 杠杆 = 精简 consumer kernel(ceiling ~−42% 🟡 derived)~~ → cron#82 Stage B 实测:lean 核落地 = wall 1.733M→1.410M = −18.6%(🔴 NOT −42%);瓶颈翻转 = FEED-BOUND(consumer 0.376M ≪ feed/P 1.004M;b_serial=0.376M)→ **cron#84 Phase 3:feed 两条最大杠杆双否决(加 P=4-HVX 天花板 / wt-vec→vshuff 非 gather-bound)→ cron#86 结构地板判定 + cron#87/#88/#89 三杠杆全提生产默认 = lean+wt-cache+B,生产默认 wall ~1.258M = −26% vs native,全 bit-exact,优化线收口。** 完整 loop 状态/分解/NEXT = `pure_hmx_solve_build.md`(cron#89 收口态);release doc = `docs/gdn_inverse_pure_hmx.md`;cycle taxonomy 权威 = `docs/cycle_metric_alignment.md` §"Canonical cycle taxonomy"。(链:18.7M→cron#42/68/72/73 feed→cron#74 全 VTCM 2.32M→cron#75 VTCM-only 1.78M→cron#77 WTREUSE 1.738M→cron#82 lean 1.410M(历史 gated 数据点)→cron#83 lean 提生产 1.391M→cron#87 +wt-cache ~1.30M→cron#89 +B ~1.258M(生产默认)。)

---

## 5. 工具 + 铁律（优化前先读）

| 工具 | 用法 | 可靠性 |
|---|---|---|
| 32-head TOTAL wall | `gdnbm_solve` 报 `wall=t1-t0`(主线程 PCYCLE makespan) | ✅ 唯一权威 |
| 4列 instrumentation | stats[3]=总HMX/[4]=总HVX/[5]=domain/[6]=真µs | ✅ 看机制 |
| A/B 控热 | `git stash`→build A→run→pop→build B→run，各 REPS=8 median | ✅ 抗热噪 |
| bit-exact 门 | dump 设备 T，跟 `/tmp/T_avtcm.raw` `np.array_equal` | ✅ 正确性 |
| timeline | `-DGDN_BR_TRACE` → `scripts/gdn_pipe_timeline.py T.raw [W]` | ✅ 看分布(⚠️span 高估 SMT 隐藏阶段) |
| PROBE_CYCLES / per-head / min-of-reps | — | ❌ artifact，禁用 |

**铁律**：① **HMX = 1 单元,绝不 thread**（多 HMX worker → SSR；正确多线程 = HMX consumer 在 MAIN，HVX work 多线程,≤4 单元；先读 skill `htp-hardware-scheduling`）。② 基线 GDNSolveHVX **只测不改**。③ 公平比较（同线程数/harness）前不下"快/慢"结论；`pkill -9 gdnbm` between runs。④ 反直觉"优化更慢"先查自己实现（热循环分支/寄存器溢出/次优 intrinsic），别急着下结论。

## 5.1 口径（唯一标准 = QNN optrace 字段，cron#71 统一；其他口径名一律废弃）

**只用 QNN optrace summary 的 3 个字段，不再造任何并行口径名**（旧的 ①op-latency/②unit-busy/③graph-wall/④per-call-wall、"domain cyc" 等自创名 = 混乱之源，全废）：

| 唯一口径 = QNN 字段 | 取自 | 含义 |
|---|---|---|
| **`num_dominant_path_cycles`**（per-op） | optrace `htp_op_types[].num_dominant_path_cycles_htp_0` | 一个 op 的关键依赖路径（理论流水满） |
| **`cycles_used`**（per-unit HMX/HVX） | optrace `htp_overall_summary[].htp_resources[].cycles_used` | 该硬件单元**实际占用**（含流水气泡）= 真实成本 |
| **graph wall** | `max(end_cycle)−min(start_cycle)`（= `gdnbm_solve` 的 `wall=t1-t0` 同一 makespan） | 端到端 |

全是 **PCYCLE**；baremetal `C15:14` == QNN PCYCLE（同一计数器，skill 已证），所以 baremetal 读的数**直接归到上面字段的含义**（back-to-back per-call wall = `cycles_used`/occupancy；不要再叫"per-call wall"）。

**铁律**：**只比 同字段 + 同 shape + 同场景**。跨字段比（如 `num_dominant_path` 比 `cycles_used`）或跨场景比（单 conv 比 批处理摊销）= 必造假矛盾。

**单 conv ≠ 批处理摊销（两套场景，差一个量级，别混）**：
- 单个孤立 `[1,1,64,64]`（optrace 实测）：`num_dominant_path` 3543 / `cycles_used` 11176。
- 批 `[1,32,64,64]` supertile（n_tiles=8）摊销 per-conv：`num_dominant_path` 370 / `cycles_used` 1388。

**我们 w16a16 kernel = native（同字段同场景实测，cron#71）**：
| 场景 | 字段 | 我们 | native | 比 |
|---|---|---|---|---|
| 单 [1,1,64,64]（QNN optrace 两边） | num_dominant_path | 3789 | 3543 | 1.07× |
| 单 [1,1,64,64] | cycles_used | 11152 | 11176 | **0.998×** |
| 批 n_tiles=8 ~~摊销~~ | cycles_used | 1320(baremetal) | 1388 | ~~0.95×~~ |
| **单 conv(同核同口径,cron#78)** | **per-call cyc** | **1576/conv(convhhh)** | **1970(`mm_1x1x64x64` single)** | **0.80× 我们略快,NOT 慢** |
| ~~批吞吐 5.4× 慢~~(cron#76/77 错比,cron#78 撤) | ~~start_cycle retire~~ | ~~1577~~ | ~~290(mm_64 子-op)~~ | ~~5.4×~~ ← 290 是 HMX 子-op `cycles` 非 conv-wall |
**⚠️ 结论终定(cron#78 PMU+native single-vs-batch+从零 micro-kernel;权威细节见 `pure_hmx_solve_build.md` ④f-cron#78,推翻 cron#76/#77 的 "我们慢 5.4× / 290 UNREACHABLE / consumer 地板 1.01M"):** 单孤立 op cycles_used 持平(11152≈11176)真。**(a) 我们 consumer convhhh 单 conv 1576 ≈ native 同核 SINGLE conv `mm_1x1x64x64` 1970(我们略快,NOT 慢);"native 290" = 批 `mm_64` 里 ConvLayer 的 HMX 子-op `cycles`(warm 263),不是 conv-wall** —— native 批整图 HVX-bound(HVX Σbusy≈graph span;HMX Σbusy 仅 8.6%),每-conv 真 wall≈graph-span/32≈4087;cron#76 拿我们单 conv 1576 比 native 批子-op 290 = 错比,撤。**(b) consumer 真杠杆 = 写精简流式核 1576→~363/conv(同 16 tile-MAC:convhhh 1576cyc/113pkt/14cyc-pkt vs 干净 dilate micro 363cyc/81pkt/4.5cyc-pkt,核臃肿 4.3×),NOT "流水"** —— 双缓冲 fill‖drain 跨-conv 软件流水实测 ZERO 增益(SERIAL==PIPE;HMX 单 acc 无 bank);cron#77 的 "290 UNREACHABLE/consumer 地板 1.01M" 与旧 "复现 walk-流水到 230K" 均撤。**(c) ~~但 producer-bound(slowest-prod-life 1.53M > consumer 1.30M)⇒ consumer 提速对 wall 仅值 ~11%,#1 杠杆 = producer feed(wt-pack vgather)~~ → ~~cron#81 SPEC:consumer-occupancy-bound at floor 1.31M(臃肿-bound)⇒ #1 杠杆 = 精简 consumer kernel(ceiling ~−42% 🟡 derived)~~ → cron#82 Stage B 实测:lean 核落地,consumer 整核占用 1.309M→0.376M,wall 1.733M→1.410M = **−18.6%(NOT −42%,🔴 ceiling REFUTED)**;瓶颈翻转 = FEED-BOUND(consumer 0.376M ≪ feed/P 1.004M;b_serial=0.376M)→ **cron#84 Phase 3:feed 两条最大杠杆双否决(加 P=4-HVX 天花板,P=6 +25.6%;wt-vec→vshuff bit-exact 但 wall 零增益=非 gather-bound)→ cron#86 结构地板判定 + cron#87/#88/#89 三杠杆全提生产默认 = lean(cron#83,d075ccd)+ wt-cache(cron#87,01c5376)+ lever B(cron#89,89927c4):生产默认 wall ~1.258M = −26% vs native 1.703M,全 bit-exact;干净路到结构地板(cron#86),≤1.0M 需精度授权(未做),优化线收口。** 📌 cron#82 lean 1.410M(`GP_LEANMM` gated 态)= 历史数据点,已被 cron#87/#89 取代,勿当现态。权威 taxonomy = `docs/cycle_metric_alignment.md` §"Canonical cycle taxonomy";收口态详情 = `pure_hmx_solve_build.md`(cron#89)+ release doc `docs/gdn_inverse_pure_hmx.md`;cron#82 证据 = `Agent/current/perf_baseline_cron82_leanmm.txt`。**

**64³ custom op 进 optrace 的正确流程（cron#71；错过一次的教训）**：
- ✅ `native_record_256` profile（FORMULA_DESC 按 shape 算 descriptor）+ `MODE=chain_qdq` + `W16A16_NATIVE_ORACLE_DIR`（native 64³ oracle 出 weight sidecar）+ 设备 **HTP-only** op package（CPU package 注册失败）。一键：`SHAPES="64,64,64" bash scripts/w16a16_shape_sweep.sh`（CI 覆盖）。
- ❌ 死因（别再犯）：用 `accepted` profile（描述符 out_y/n_tiles **写死 256**）跑 64³ → HMX 按 256 宽读写 64 宽缓冲 → 越界 → optrace execute fault（plain execute 容忍）。**把"自己配错 profile"误判成"QNN 工具对小 shape 的限制"——这是借口不是诊断。**
- 细节 memory：`[[reference_64cube_conv_occupancy_vs_latency]]`。

---

## 6. 已确立的杠杆与教训

- **v75 这条管线瓶颈 = HVX issue/compute,不是 VTCM 带宽**（实测：int16-native 消 VTCM round-trip 反而慢 ~3%，已 revert；int32 round-trip 几乎免费）。→ 有效杠杆 = **更少 HVX op/lane**（int16-lane 减半 = 真减 compute），不是更少内存搬运。
- **MM 真正的杠杆 = 粗化握手粒度**（512 次独立 dispatch → 批量交 N 个 kernel 描述符,consumer 连跑 N 个 mxmem 再 signal,把 2 屏障+spin 摊到每 N matmul）。次选 POST barrier2 → release-store。
- **PREP 碎**（QUANT 的 A-fold 最大 ~10%，PACK ~7%，EFF ~3% 不碰）；A 驻 VTCM ping-pong = **1.93×**（DDR 371K→192K/head）。
- **crouton 图集成 ~10%**（A 以 crouton 进 DDR,设备端 fold+quant+crouton 全消失）= bit-exact 已验，但**只在真实图里上游 HMX op 出 crouton 才免费**，standalone 省不掉 → 记为未来收益。
- **HMX 两布局硬约束**：激活=crouton、权重=kmajor、输出=crouton；激活路可全程 crouton-native，权重路(kmajor)不可约。

---

memory：`[[project_gdn_pipe_beats_hvx_2026-06-08]]`、`[[project_gdn_solveop_bottleneck_2026-06-01]]`、
`[[reference_htp_smt_pmu_hardware]]`、`[[reference_htp_hardware_scheduling_flow]]`、`[[feedback_gdn_metric_32head_total_wall]]`、
`[[feedback_retimeline_after_every_change]]`。w16a16 kernel：`docs/w16a16_is_two_w8a16.md`。
