# GDN 三角求逆 solve（统一文档：三路线 + 现状 + 实现 + 口径）

GDN/KDA linear-attention 在 v75 HTP 上的核心难点 = 每 head 对 C×C 下三角矩阵 `L=I-A` 求逆。
本文是这件事的**唯一权威文档**：三种路线、当前进展、实现、度量口径、经验。
矩阵乘 kernel 本身（u8i8 / w8a16 / **w16a16**）见独立文档 `docs/w16a16_is_two_w8a16.md`。

---

## 0. 当前最佳（权威数）

**GDNSolveHVXMixHMX，`GdnSolveBR16.cpp` int16 静态 solve，producer-consumer pipeline（P=4 HVX 生产者 + 1 主线程 PURE-HMX 消费者）。**

| 指标 | 值 |
|---|---|
| **32-head TOTAL wall** | **~1.78M domain cyc**（≈ 1.12 ms @1.594 GHz；crouton 图集成 ~10% → ~1.63M，记为待图集成的未来收益，不计入 standalone） |
| vs GDNSolveHVX 4-thread 基线 | **~2.2×**（基线 ~3.97M） |
| 正确性 | 全程 bit-exact（精度 relerr ~1.4% = int8-drain 代价；merge 换 w16a16 可到 oc 1.16e-3 但 wall ×2.55 否决，见 §4 GDN_BR_W16） |

> **口径铁律：唯一权威终指标 = C=256, 32-head TOTAL wall（domain cyc）。** per-head（tiler 低估 artifact 88K）、min-of-reps（抓快异常值）、per-stage PROBE（C15:14 嵌套求和 3.5× wall）**全禁**。

---

## 1. 三种路线（命名权威，全仓库统一）

| 标准名 | 定义 | 矩阵乘在哪 | 对角块求逆在哪 | 裁决 |
|---|---|---|---|---|
| **GDNSolveHVX** | 纯 HVX | HVX vrmpy(int8)/`gdn_matmul_i16`(int16) | HVX forward-subst | **基线 ~3.97M，只测不改** |
| **GDNSolveHVXMixHMX** | HVX 喂数 + HMX 算 matmul | **HMX**（u8i8 mxmem） | **HVX forward-subst（并行 4 单元）** | **当前最佳 ~1.78M ✓，优化目标** |
| **GDNSolveHMX** | 全程 HMX（连对角求逆都 matmul） | HMX | HMX（Taylor+Newton 矩阵乘） | **否决（~3.6× 慢，下§4）** |

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

**构建 / 运行**（默认即最佳）：
```bash
cd example/gdn_native/baremetal
EXTRA_DEFS="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL" bash build.sh
# 设备（dssh = ControlMaster 复用,见 scripts/dssh.sh）
source scripts/dssh.sh; dssh_open oneplus; W=$(dssh 'echo $HOME/gdnbm_run')
dssh "cat > $W/libgdnbm_skel.so" < build/libgdnbm_skel.so
dssh "cat > $W/gdnbm" < build/gdnbm; dssh "chmod +x $W/gdnbm"
dssh "cd $W && GDNBM_REPS=8 LD_LIBRARY_PATH=$W:/vendor/lib64:/system/lib64 \
  ADSP_LIBRARY_PATH='$W;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' \
  ./gdnbm 4 A_u16_h32.raw T.raw 32 256 32768 32768 2.770166930875267e-05 6.103701895199438e-05"
```
- 数据：`A_u16_h32.raw`(u16,zpA=32768) / `T_ref_h32.raw`(fp32 golden) / `/tmp/T_avtcm.raw`(bit-exact 基准)。
- 取数：warmup 一次丢弃 + 一个 FastRPC session 跑 `GDNBM_REPS=8`，**取 reps 2–4 median**（rep1 冷,rep≥5 节流）。**绝不取 min。**

---

## 4. 路线速度对比（口径统一,2026-06-09）

### 单个 64³ matmul 的真实 HMX cycle
| kernel | 隔离 64³ | 流水内真实 mxmem/matmul |
|---|---|---|
| u8i8(int8×int8) | 216(设备纯)/417(sim) | **313** |
| w8a16(int8 wt×int16 act) | ~462(sim) | ~626 |
| **w16a16 = 2× w8a16** | — | **~626 只是 MAC 折算;真实单 64³ 调用 ~10.4K(固定开销主导,设备实测 2026-06-11,下节)** |

> **HMX matmul 只占 HVXMixHMX wall 的 7%**：512 个 merge matmul 真实 mxmem = 160K / 1.78M。每 matmul 握手 glue ≈ 2950 cyc = **9.4× 浪费**。**瓶颈是 glue，不是 matmul。**

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

### 纯 HMX vs HVXMixHMX（只看速度，2026-06-09 QNN 原生 optrace + 2026-06-10 设备实测）
对角 Newton 迭代会放大误差 → **纯 HMX 路对角必须 w16a16 精度**（u8i8 不够）。
> **独立第二否决（数值）**：对角块 `A_ii` 是严格下三角 nilpotent（`A^64=0`），Neumann 级数 `I+A+…+A^63` 虽有限精确，但 `A∈[-1,1]` 下 `‖A‖₂` 中位 2.36 / p90 8.14 / max 31（`gdn_solve_taylor_newton_probe.py`），中间幂 `A^k` 爆掉 int16（实测 peak 1.9e13，仅 81% 收敛）。**forward-subst（HVX 路）从不形成 `A^k`，所以 HVX 对角不只是省时间，是数值上的必需。**

**权威数 = QNN 原生 256³ matmul 的 HMX-compute cycle**（`example/qnn_matmul_profile/output_*_*_256/optrace/summary.json`，`by_htp_type_cycles`）：

| family | 原生 256³ HMX 算力 | per-64³(MAC 折算 ÷64) | vs u8i8 |
|---|---|---|---|
| u8i8（`ConvLayer_s1.opt`） | 12,435 | 194 | 1× |
| w8a16（`ConvLayer_s1.opt`） | 30,182 | 472 | 2.4× |
| **w16a16（`HmxU16I16ToU16MatMul`）** | **74,670** | **1167** | **6.0×** |

w16a16 ≈ 2.5× w8a16（74670 ≈ 2×30182 + drain 开销），证实"w16a16 = 2×w8a16 + 排空"原理；**原生最优 per-64³ = 1167**（= 像原生那样把 64 个子块摊进一个大 op 的理论地板）。

纯 HMX C=256 = **60 matmul/head**（44 对角 Newton + 16 merge），门槛 = `1.78M/(60×32) = 927 cyc/matmul`：

| 纯 HMX w16a16 per-64³ | 来源 | 总 = 60×_×32 | vs HVXMixHMX 1.78M |
|---|---|---|---|
| **1167**（HMX-compute-only 理论地板） | QNN 原生 256³ optrace `by_htp_type_cycles`（HMX 单元忙周期，act-stream/drain 被 supertiling 叠到别的单元） | **2.24M** | **输 1.26×** |
| **5844**（单线程真实 wall，最佳摊销） | **我们刚提交的 w16a16 kernel，真设备 256³（64 个 64³-MAC 摊一个大 op）实测 374024 PCYCLE，8 reps 374–376K 极稳，`run_w16a16_standalone_device.py --time-reps 8`** | **11.2M** | **输 6.3×** |

> 5844 vs 1167 的 5× 差 = **wall 含 convhhh kernel 内部的 HVX act-stream + 2-pass byte-decompose + cvt-drain，全部串在调用线程上**。这部分**无法甩给 4 个生产者**——w16a16 是单次 fused kernel call（≠ 可拆的 HVX/HMX 双 op）。要拆得重写 kernel 成「HVX 喂 + HMX 算」两 op，且**即便完美重叠也只能逼近 1167 地板 → 仍 2.24M > 1.78M**。

**裁定：两端都输（设备实测确认，非投影）。** 连 QNN 原生 supertiling 的最高 HMX 效率（1167/64³）都已超门槛（927）；我们能真在设备上跑的最佳摊销形态（256³ 大 op）= 5844/64³，输 6.3×。
- 瓶颈 = **HMX 算力本身**，不是胶水。HVXMixHMX 是 **HVX-bound**：HMX 只扛 12 个便宜 u8i8 merge（占 wall 7%），对角甩给 **4 路并行 HVX forward-subst**。纯 HMX 把对角搬到**唯一串行 HMX** 做 w16a16（每个贵 6×、数量 5×）。
- 「4×HVX ∥ HMX、胶水被 HMX 掩盖」**能满足但无关**——光 HMX 算力地板(2.24M)就已超过整个 HVXMixHMX wall(1.78M)。
- 唯一能让纯 HMX 翻盘的是 u8i8（60×194×32≈0.37M），但 u8i8 精度撑不住迭代对角——不可调和的取舍。

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
