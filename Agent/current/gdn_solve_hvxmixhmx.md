# GDNSolveHVXMixHMX — HVX 喂数 + HMX 矩阵乘流水线（持续优化目标）

**持续优化的就是这一个：GDNSolveHVXMixHMX。** 当前阶段目标 = 把它做成 **静态 int16 版**（见"目标架构"）。
基线 GDNSolveHVX 只用来对照测量，**不改**。

## 命名定义（GDN 三角求逆三实现，全仓库统一）

| 标准名 | 定义 | 代码 | 角色 |
|---|---|---|---|
| **GDNSolveHVX** | 纯 HVX（64³ 用 vrmpy int8/int16，对角块 HVX forward-subst）。已出货。 | `gdn_merge_hvx`（`GDN_BR_HVX_MERGE`）；baremetal 默认 solve | **基线 = 只测不改** |
| **GDNSolveHVXMixHMX** | HVX 喂数 + HMX 算 64³；对角块仍 HVX。 | `gdn_merge_packed`（QNN op 默认）；baremetal `gdnbm_imp.cpp` 的 FEED_4P / FEED_PIPE / HMX_BENCH 微基准；HMX kernel `our_v73deep_kernel` | **优化目标 = 改这里** |
| **GDNSolveHMX** | 全程 HMX（连三角求逆都 HMX）。多算 28–60× 矩阵乘,实测慢 4–6×。 | 已否决 | — |

（这是叙述名统一；代码标识符 `GDN_BR_HVX_MERGE`、`gdn_merge_packed` 等不变。）

## ⛔ 铁律（勿违反）

1. **只跑微基准,不跑完整 HVXMixHMX 全 solve。** `-DGDNBM_HMX_MERGE_PATH` 在 `gdnbm_open` 即 SSR
   (`rc=0x80000406`),已用 `#error` 禁掉(逃生开关 `-DGDNBM_ALLOW_BROKEN_HMX_MERGE`,会 SSR 需 reboot)。
   旧"434K/head"数字作废。下一步是据微基准**重写**整个 solve,不是修旧全 solve。
2. **基线 GDNSolveHVX 只测不改。** 不要在 `gdn_merge_hvx` / baremetal 默认路径上加优化。
3. **HMX = 1 单元,绝不 thread。** 多 HMX worker → SSR。正确多线程 = HMX consumer 在 MAIN,HVX work 多线程
   (head 并行,#HVX-locked ≤ 4 单元)。先读 skill `htp-hardware-scheduling`。
4. **度量只信 wall / compute-cycles,不信 per-stage PROBE。** `GDN_BR_PROBE_CYCLES` 的 `C15:14` per-stage 数
   是 per-hardware-thread 垃圾值(QuRT 迁移线程)。唯一可信 = 整 solve 的 `cyc/head`(stats[0])+ 单线程隔离
   loop + ablation(`-D<flag>` 消阶段测 wall 差)。
5. 公平比较(同线程数、同 harness)前不下"快/慢"结论。`pkill -9 gdnbm` between device runs。

## 🎯 目标架构（权威,solve 重写照此）：静态对称 int16

**决定:HVXMixHMX 走静态量化(固定/可预测标度,非运行时 maxabs),输出 int16。** 这是效率最优:一切可提前
预处理,运行时只剩纯整数 matmul + 纯 int 加(无运行时量化、无 per-term rescale、无动态 maxabs 扫描)。

注意:**HVXMixHMX 的 matmul 始终是 HMX(int8 mxmem)**,"int16"不是算子精度,是指：

| 当前(动态)| 静态 int16 版 |
|---|---|
| int8 输出 → **multi-pass gain search**(PASS1/2/3 跑 HMX 3 次扫 maxabs)= int8-输出税 | **静态输出标度 → 零 gain search** |
| int8 输出 `cvt.ub`(1 tile/cvt) | **int16 输出 `cvt.uh:2x2`**(4 tile/cvt,发射更少)— `:2x2` 是 16-bit 输出专属(逆向确认) |
| 算子动态 per-block 量化 | 算子**静态 int8 量化**(预处理一次) |
| acc 每 term fixed-point rescale | **纯 int32 加**(同标度);每块仅 1 次 int32→int16 requant |

**sim 验证(`scripts/gdn_solve_static_quant_probe.py`,真实 golden p29_L00,端到端 off-diag oc):**
- int8-static 0.0545(太松);**int16-static 0.00024**;**int16-static + 真实 int32-wrap 累加器 0.00019,溢出 0/head**。
- → "int8 大不了 int16" 命中。值有界(|A|≤0.94,|T|≤1,随 block-dist 衰减)→ 单一全局 int16 静态标度即可;
  **int32 累加器实际从不溢出(不需要 int64)**。oc 0.00019 = 比出货 int8(~1.2e-2)好 ~60×。

**核心缺件(已找到,见阶段2):int16-输出的 HMX kernel(`cvt.uh:2x2`)。**

## 阶段2 结论（int16-输出 HMX kernel 逆向）—— 2026-06-05：已存在 + byte-verified

**`convhbh` int16-输出 kernel 早已被逆向并 byte-verify,不需从零做。**
- 原生切片 = `hmx_v75_convhbh1x1deep_stride1` @ `libQnnHtpV75Skel.so:0x2f5200`,1348 B。反汇编在
  `Agent/qnn_re/hmx_v75_convhbh1x1deep_stride1.S`;可读 inline-asm 复制品在
  `example/qnn_hmx_matmul_w8a16/src/v73deep_conv1x1_kernel.inc`(`our_v73deep_kernel`,与 solve 的 u8i8 同 ABI)。
- **byte-exact 复验通过**(2026-06-05):`verify_hexagon_inline_asm.py --vma 0x2f5200 --size 1348` → OK。
- **语义**:`activation.ub`(u8)× `weight.b`(i8)→ `cvt.uh = acc(r25):2x2`(u16 输出)+ `mxmem = cvt`(**密 drain**)。
  **输入路径与现 u8i8 kernel 完全相同**(`activation.ub:deep:cm` × `weight.b:deep`);差异仅在 drain:
  | | u8i8(现 solve)| convhbh(int16-out)|
  |---|---|---|
  | drain | `cvt.ub = acc(r25)` | `cvt.uh = acc(r25):2x2` |
  | store | `mxmem(r10,r11):cm = cvt` | `mxmem(r10,r11) = cvt`(密) |
  | 输出 tile/ cvt | 1 | 4(`:2x2` fan-out) |
- **含义**:静态 int16 merge **保持同一 int8 算子打包,只换输出 drain** 即得 int16 输出 → 杀 multi-pass gain search
  (int8-输出税)。阶段3 = 在 FEED_4P 微基准把这个 kernel 接上 + 调输出 tile 布局(`:2x2` 4-tile/cvt)。
  现成驱动参考:`example/qnn_hmx_matmul_w8a16/src/HmxU16I8ToU16MatMulOp.cpp`(u16-act 走 hi/lo 2×u8 pass;
  GDN 静态版只需 u8 单 pass + int16 输出)。

## 阶段3 实测结论（静态 int16 的 matmul-portion 收益）—— 2026-06-05

**核心:静态 int16 输出 → 单 HMX pass(去 multi-pass gain search)→ matmul-portion 降 2.8–3.4×,device 实测。**

FEED_4P 微基准(`ssh oneplus`,DUMMY 数据,cyc/matmul = stats[0]),apples-to-apples 同线程数:

| 配置 | nthreads | cyc/matmul | 说明 |
|---|---|---|---|
| **MULTIPASS**(忠实 HVXMixHMX 现状,3 HMX pass)| 3 | **1731** | PASS1/2 扫 maxabs(gain search,输出丢弃)+ PASS3 真输出 |
| **单 pass**(静态 int16 输出形态)| 3 | **615** | 去 multi-pass → **2.82× 更便宜** |
| **单 pass + 纯-HMX consumer**| 4 | **509** | 静态版 consumer 不需 HVX maxabs → 释放第4 producer;**总 3.4×** |
| (对照)基线 GDNSolveHVX vrmpy 64³ matmul(GLUE_BENCH o[3],单线程隔离)| 1 | 8325 | fed-HMX 静态 int16 每 matmul 远低于 vrmpy |

- **为什么静态 int16 能单 pass**:输出标度固定(`sim oc 0.00019`,[[目标架构]]),无需运行时扫 maxabs 定 int8 输出增益。
  multipass 的 PASS1/2 纯粹是 int8-输出税;int16 静态输出一次到位。
- **为什么能 P=4**:单 pass 的 consumer 是纯 HMX(不锁 HVX 做 maxabs)→ 释放 1 个 HVX 单元给第4 producer。
  multipass consumer 必须锁 HVX 做 gain-search maxabs → 只能 P=3(P=4 会 5 个 HVX 锁 >4 单元 → 卡死,已验)。
- **convhbh int16-out kernel(阶段2,byte-verified)= 单 pass 正确输出的载体**(`cvt.uh:2x2` u16 输出)。
- **剩余生产集成(下一步,已完全刻画)**:把 convhbh 接进单 pass FEED_4P 并对 `gdn_matmul_i16` 验 bit-exact。
  **不是 drop-in**——经读 `HmxU16I8ToU16MatMulOp.cpp` desc 构造(L1190-1320)确认,int16 路径用 **Crouton_16
  row4-grouped 布局**,与现 u8i8 的 crouton8(2 项表)不同。64³(M_t=N_t=K_t=2)的 convhbh 描述符:
  - `row4_groups = M_t*8 = 16`;act/out 表各 `(16-1)*stride + tiles = 32` 项(stride=K_t/N_t=2)。
  - `act_desc`={tbl(32), n_pairs=K_t=2, y_stride=K_t=2};`out_desc`={tbl(32), table_stride=N_t=2, y_stride=N_t=2,
    n_tiles_pow2=M_t*4=8, m_total_minus_step=8, k_total_bytes=N_t*32=64};out tile ptr = `out_raw+row4*4*64+nt*32`(密 row4)。
  - mask arg1 `0x70b`(非 0x700);extra_param `{1,1025,524}`(非 {1,0})。
  - **需新写**:Crouton_16 激活 packer(32 项 row4 表,替现 crouton8)+ u16 row4-dense 输出 depack。
  - perf 收益已量化(509 vs 1731),kernel 已 byte-verified,描述符已解码 → 是机械集成,不含未知 RE。

复现:`cd example/gdn_native/baremetal;`
`EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_FEED_PIPE -DGDNBM_FEED_4P -DGDNBM_FUSED_ACTWT" bash build.sh; ./gdnbm 4 ...`(单pass 509)
`EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_FEED_PIPE -DGDNBM_FEED_4P -DGDNBM_FEED_MULTIPASS" bash build.sh; ./gdnbm 3 ...`(multipass 1731)

## 阶段1 实测结论（对角块求逆 / 前代回代）—— 2026-06-05

**前提被实测推翻:对角块的前代回代不是瓶颈,已在 HVX 下限,无需优化。** 决策依据 = 真实全 solve
ablation(`ssh oneplus`,baremetal 默认 GDNSolveHVX 基线,A_u16_h32.raw,H=32 C=256,nthreads=1):

| 度量 | cyc/head | 占比 |
|---|---|---|
| 全 solve | 405,912 | 100% |
| DIAG_ONLY(`-DGDN_BR_DIAG_ONLY`,跳 merge = diag+zero-fill+requant) | 70,671 | **17%** |
| → merge(= full − diag_only) | 335,241 | **83%** |

- **`gdn_solve_diag64` 三段拆解(都已是固定标度对称 int16,zp=0):** A 折叠到 `2^-15`,T 输出固定 `GDN_BR_TI=2/32767`
  —— **对角块"静态对称 int16"形态已具备,不必改。**
- **前代回代主循环(2016 条 `Q6_Ww_vmpyacc_WwVhRh`)已流水到底**;行间串行依赖**不是**瓶颈(goal 前提错)。
  连 DIAG_ONLY 的 70K 里,大头也是 zero-fill(128KB/head vsplat)+ requant,不是前代回代。
- **bit-exact 门通过**:offdiag-oc=1.35e-3、whole-oc=9.5e-5(对 `T_ref_h32.raw` fp64 逆),与现状一致。
- ⚠️ **方法论坑(已记)**:`GDNBM_GLUE_BENCH` 隔离 REPS 微基准报 diag64=11.7K、widen=91%——**全是假象**
  (隔离循环对 `Tblk[0]` 的冷 DDR 写 + pcyc 开销)。改 widen 后真实 solve 130K→131K **零变化**。
  **只信整 solve wall / ablation,微基准 REPS-loop 对带 DDR 写的阶段不可信。**
- **静态 int16 真正的杠杆在 merge(83%),不在 diag。** widen(int16→int32)只为喂 int32 merge 而存在;
  T 全程保持 int16 → 消除 widen + 喂 merge 直接吃 int16,是**阶段3** 的收益,不在 diag 隔离。

复现:`cd example/gdn_native/baremetal; EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT[ -DGDN_BR_DIAG_ONLY]" bash build.sh;`
`./gdnbm 1 A_u16_h32.raw T_out.raw 32 256 32768 32768 2.770166930875267e-05 6.103701895199438e-05`。

## 当前微基准状态（matmul-portion,已优化到 ~507 cyc/matmul）

**流水线**：4 个 HVX producer 把每个 64³ matmul 的算子(crouton act + k-major wt + bias)pack 进 VTCM ring,
1 个 **main-thread 纯-HMX consumer** 排空(`our_v73deep_kernel` mxmem)。静态条带分工,CAS-free volatile slot
标志(CAS `__sync` 在本目标不可靠 → 用 volatile + static striping)。

**优化结果(real v75,DUMMY 数据,单 HMX run/matmul):**
- 起点 1208 → **~507 cyc/matmul**(min 实测 ~503–520)。关键杠杆按贡献：
  - 向量化 eff+bias(128 标量 VTCM 写 → 4 向量写)1208→833；2-head pack;4th producer + 纯-HMX consumer;
    fused/2-head depack;**lever #A:融合 act-crouton(ALU)∥ wt-kmajor(permute)进一个循环,共发不同 HVX 单元
    → 585→507**(隔离 1.30×,bit-exact)。
- **HMX kernel floor = 214 cyc/matmul**(continuous,descriptor set once)。
- 微基准是单 HMX run;真实 HVXMixHMX 每 matmul 跑 2–3 次 HMX(multi-pass gain search)→ 真实成本 ~2.5–3×。
  **静态 int16 版要消掉这个 multi-pass(见目标架构)。**

## 关键硬件事实（load-bearing,本程实测/逆向/手册三证）

- **consumer-busy 的底是 SMT issue-slot 争用,不是 VTCM 流量。** 4P 时 consumer-busy ~410 = HMX kernel 214 +
  ~196 SMT 争用。**PMU 直接证明**(qurt_pmu,PRM Table 9-1 原始码):P=4 时 `CYCLES_5_THREAD_RUNNING`=386/486
  (79% 时间 5 线程并发),`SMT_BANK_CONFLICT`≈0(非内存)。consumer-busy 随并发线程数升(P=1:309→P=4:410)。
- **降 VTCM 流量(operand caching / depack 消除 / 调度重排)都动不了这个底**(实测:OPCACHE 仅 1.1×;phase-stagger
  no-op)。唯一能动的是减少并发线程数,但那牺牲喂数。
- **无法 pin 线程到 cluster**:v75 QuRT 无 affinity API(逆向 `libQnnHtpV75Skel.so` 确认 QNN 也没有,只 import
  `qurt_thread_set_priority` + `qurt_sysenv`)。**线程优先级实测 no-op**(SMT 发射仲裁硬件公平;6 HW 线程/2 cluster,
  5 线程无 oversubscription)。
- **v75:4 HVX 单元(128B)/ 1 HMX(process-serial)/ 6 HW 线程(2 cluster)/ VTCM 8MB / ~1.42GHz。**
- QNN 怎么让 HMX 低争用重叠 HVX:静态 per-unit runlist + 编译器 supertile(custom op 被 `is_plugin_op` 排除)+
  **HMX descriptor-driven(发射少、让出 round-robin 轮次)**。详见 `docs/qnn_htp_scheduling_and_custom_op_limits.md`。
- **VTCM-向量读原则**只适用 HMX crouton tile + int8 vrmpy;**不适用 int16 HVX matmul**(`Q6_Ww_vmpyacc_WwVhRh`
  的 Rh 寄存器广播读标量不 stall)。详见 [[reference_gdn_solve_global_roofline]]、[[reference_htp_smt_pmu_hardware]]。

## 下一步（建静态 int16 版,在 4P 微基准上）

1. **逆向 QNN int16-输出 HMX kernel**(`convhbh` `cvt.uh:2x2`)—— 静态版的核心缺件。
2. 在 FEED_4P 微基准上：静态 int8 算子(预量化)+ 静态 int16 输出(`:2x2`)+ **去掉 multi-pass**,测 matmul-portion
   掉多少;oc 用 `gdn_solve_static_quant_probe.py` / `gdn_br.sh` 验。
3. 正确多线程(HMX-on-main + HVX workers,绝不 thread HMX),apples-to-apples vs GDNSolveHVX 基线(同线程数,C=256)。

## Reproduce（baremetal,`ssh oneplus` via `scripts/dssh.sh`；`pkill -9 gdnbm` 每次前;线程 ≤4）

```bash
cd example/gdn_native/baremetal
# matmul-portion 微基准 4P(~507),FUSED act∥wt：
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_FEED_PIPE -DGDNBM_FEED_4P -DGDNBM_FUSED_ACTWT" bash build.sh
#   ./gdnbm 4 A_u16_h32.raw /dev/null 32 256 32768 32768 2.770166930875267e-05 6.103701895199438e-05  -> stats[0]=cyc/matmul
# HMX kernel floor(214)+ 各组件:
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_HMX_BENCH"  bash build.sh   # stats[2]=kernel
# 真实 multi-pass 成本(P=3 consumer-bound):
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_FEED_PIPE -DGDNBM_FEED_4P -DGDNBM_FEED_MULTIPASS" bash build.sh
# SMT PMU 证据(OPCACHE+NODEP):
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_FEED_PIPE -DGDNBM_FEED_4P -DGDNBM_FUSED_ACTWT -DGDNBM_OPCACHE -DGDNBM_OPCACHE_NODEP -DGDNBM_PMU" bash build.sh
# 基线 GDNSolveHVX(只测对照,4-thread ~128-146K cyc/head)：
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT" bash build.sh                      # ./gdnbm 4 ...
# 静态量化 oc sim：
source scripts/env.sh && python scripts/gdn_solve_static_quant_probe.py
```

## 否决记录（勿重试）

- **全 HMX(GDNSolveHMX)/ Taylor-Neumann-squaring-Newton 迭代矩阵乘**:‖A‖≫1 爆炸,慢 4–6×。
- **线程→cluster pin / 线程优先级**:无 API / SMT 公平,no-op。
- **operand caching 到 DDR/VTCM 给 vrmpy A**:vrmpy 标量 splat 读约束,基线路径,且不迁移到 HMX。
- **微基准 consumer 侧 descriptor hoist / barrier 去除 / kernel mxmem-setup 摊销**:no-op(底是 SMT 争用非 setup)。
