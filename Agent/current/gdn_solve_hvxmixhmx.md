# GDNSolveHVXMixHMX — HVX 喂数 + HMX 矩阵乘流水线（持续优化目标）

**持续优化的就是这一个：GDNSolveHVXMixHMX。**
> **🎯 当前阶段目标(2026-06-06 用户拍板)= 先用 u8i8(int8)kernel 把 producer-consumer 流水做成全 solve,精度先不管。**
> int8 HMX matmul 实测 4× 便宜于 vrmpy(FEED_4P 509 vs 2081)。微基准已跑通,缺口 = 全 solve 集成
> (SSR-at-open 根因已定位,见 `gdn_solve_NEXT_AGENT.md` 顶部)。下面"目标架构/静态 int16"是**未来精度升级**的设计,
> int16 升级在这硅上贵(convhbh 8-bit → int16²≈vrmpy,见 `gdn_merge_static_int_design.md`)——**先不做,不影响 u8i8 主线**。
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

### ★ 阶段3 MAKE-OR-BREAK 最终结论(hexagon-sim 实测,无设备/无 SSR)—— 2026-06-05
**int16-输出 HMX kernel(convhbh)在 sim 里跑通(PASS),64³ matmul cyc ≈ int8,远低于 vrmpy → GO。**
脚本 `scripts/gdn_hmx_convhbh_sim.py`(复用 `pack_a16_crouton16_row4_surface`+`pack_w8_kmajor`+`pack_native_a16_bias`
+ 解码描述符)。apples-to-apples 同 sim 64³:

| kernel | cyc/64³ | 备注 |
|---|---|---|
| **convhbh int16-out（`cvt.uh:2x2`）** | **462**(PASS) | MAC 主循环跑满(>u8i8 417);输出当前 drain ~1/4(`:2x2` 成对签名 `o[2k]==o[2k+1]`,row4-dense 布局,描述符需调)|
| u8i8 int8-out | 417（bit-exact 4096/4096）| 同 sim 基准 |
| HVX vrmpy | 8325 | 现 merge |

- **结论:int16-输出 matmul ≈ int8(462 vs 417,+11% 重 drain),~18× 便宜于 vrmpy。** cyc 由 MAC 主循环主导(与输出表无关)
  → partial-drain 不影响该结论(完整 ≈ 462 + 少量 drain ≈ 500–560,仍 ~15× < vrmpy)。**静态 int16 HMX-merge 的 matmul
  杠杆成立,值得做。**
- **剩余(纯 sim 可迭代,无 SSR)**:调描述符使 drain 写全 64×64 + nail `cvt.uh:2x2` row4-dense u16 depack → 对 numpy ref 验 bit-exact
  → 再上设备测 cyc → FEED_4P 集成 → 整 solve producer-consumer 重构。

### 阶段3 集成 BLUEPRINT(下一会话执行,分步 device 验证)—— 2026-06-05
**Make-or-break(已确认值得做)**:HMX-fed matmul 整片吞吐 **509 cyc/64³**(FEED_4P,4 HVX producer+1 HMX consumer)
vs vrmpy 4 路并行 ~**2081 cyc/64³**(8325/4)→ **~4× 更便宜**;merge 占 solve 83% → 量级 2–3× 主杠杆
(与 [[project_gdn_solve_nhead_hmx_feed_2026-06-04]] 一致)。
**性质**:这是多步、有 **SSR 风险**(HMX 描述符错→cDSP fault 需 reboot)的 RE+编码,**仓库无可复用 Crouton_16 packer**
(sim 只有 crouton8;Crouton_16 只活在 QNN `HmxU16I8ToU16MatMulOp` 消费 QNN-provided 块)。
**已 de-risk**:kernel byte-verified(0x2f5200,1348B);描述符全解码(见上 + mask 0x70b / extra {1,1025,524});
convhbh prologue 已读(`Agent/qnn_re/hmx_v75_convhbh1x1deep_stride1.S`:r0=out/r1=act/r4=mask/r5=extra desc,
act y-stride ×4B,bias=mxmem2 链 r3+=0x101)。
**分步计划**:
1. (安全,无设备)RE Crouton_16 act tile 字节布局(convhbh 读 plain `.ub`,act 表项=`block[(row4&7)*K_t+kt]+(row4>>3)*256`)
   + cvt.uh:2x2 输出 row4-dense 布局(out 项=`out_raw+row4*4*64+nt*32`)。产出 packer/depack 规格。
2. baremetal 加 convhbh kernel(`our_v73deep_kernel_u16`,含 w8a16 `.inc`,标签 `L_hmx_w8a16_*` 不冲突)。
3. 写 `GDNBM_MM_I16_TEST`:确定性 int8 act/wt → Crouton_16 pack → convhbh → u16 depack,对 `gdn_matmul_i16` 验 bit-exact + 测 cyc。**先单线程小心跑(SSR 风险),描述符确认对再上多线程。**
4. 接进 FEED_4P:静态 int8 算子 + 静态 int16 输出 + 纯 int32 加 + 去 multipass,测 matmul-portion apples-to-apples。
5. 重构整 solve 为 producer-consumer(4 HVX pack+diag / 1 main HMX),整 solve wall vs 基线 GDNSolveHVX。
- perf 收益已量化(509 vs 1731 / vs vrmpy 2081),kernel 已 byte-verified,描述符已解码;余下 = Crouton_16 布局 RE + packer/depack 编码 + 谨慎 device 迭代。

复现:`cd example/gdn_native/baremetal;`
`EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_FEED_PIPE -DGDNBM_FEED_4P -DGDNBM_FUSED_ACTWT" bash build.sh; ./gdnbm 4 ...`(单pass 509)
`EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_FEED_PIPE -DGDNBM_FEED_4P -DGDNBM_FEED_MULTIPASS" bash build.sh; ./gdnbm 3 ...`(multipass 1731)

## 多-HVX(4 线程)roofline + 扩展性上限 —— 2026-06-05

**问题:4 HVX 单元下 solve 到理论极限了吗?还有多少头空间?** 实测(baremetal GDNSolveHVX 基线,
H=32 C=256,`GDNBM_REPS=5` 稳态):

| 线程 P | cyc/head | 加速比 | 效率 |
|---|---|---|---|
| 1 | 414K | 1.0× | — |
| 2 | 214K | 1.93× | 97% |
| 3 | 157K | 2.63× | 88% |
| 4 | ~140K | 2.95× | **74%** |

- **没到 4× 理想极限**:HVX-bound,4 单元的理想下限 ≈ 414/4 = **103.5K**,实测 **~140K → 差 ~36K(26%)**。
- **扩展性损失定位**(DIAG_ONLY 消融):**DIAG 部分只扩展 2.29×(57%)**,merge 扩展 3.15×(79%)。
  DIAG 的差扩展 = **DDR-写带宽争用**(zero-fill 128KB/head + requant 写 T,4 线程抢同一 DDR 写带宽,不随单元扩展)。
  - **✅ 已修并设为默认:T 算进 VTCM,再 DMA VTCM→DDR(dstbypass),把字节搬运从 HVX store 路径挪到 DMA 引擎
    → 4 线程 P=4 ~140K(抖 132–147)→ ~122K(稳),bit-exact,扩展性 2.95×→3.39×(85%)。** 旧 per-head DDR-直写
    escape = `-DGDN_BR_T_DDR_DIRECT`。⚠️ UDMA 坑:两个并发 `Q6_dmstart_A` 会互相 clobber(device 实证损坏 heads)
    → T 写回必须在 A 预取 wait 之后发,不并发;完整 T-overlap(~119K)需 UDMA 链/多流,卡在 dmwait 链语义,暂不冒险。
  - **✅ zero-fill 只写上三角块(已做,bit-exact):T 块下三角,对角+下三角块都被 requant 全覆盖,只上三角 off-diag
    块需 zpT → NB²→NB(NB-1)/2 块(256:16→6)。** 在 VTCM-DMA 之上性能中性(省的 HVX store 在噪声内)但减浪费,保留。
    QNN op(gdn_br.sh,C=128 H=16)oc 门通过(bit-exact 改动,relerr 不变)。
- **两类优化上限:**
  1. **扩展性/饱和(~26%,~36K)**:难——SMT issue-slot 争用(PMU 实证的底,[[reference_htp_smt_pmu_hardware]])
     + 共享 DDR 写带宽 + spawn/sync。可摘的低果 = zero-fill 只写上三角(~3%)。**这不是大头。**
  2. **★减少 HVX 绝对工作量(大头)**:solve 是 HVX-bound,4 单元已近饱和 → 想更快只能**减少 HVX 工作**,不是加单元。
     单线程 414K 中 merge 占 342K(83%);其中 merge-glue ~51% + matmul ~20%([[reference_gdn_solve_global_roofline]])。
     **静态 int16 + HMX 把 matmul 搬离 HVX 单元 + 去 multipass/per-term-rescale glue**——这才是 2–3× 的所在
     (阶段3 已实测 matmul-portion 3.4×)。**真正的优化上限在 merge 重写,不在前代回代/扩展性微调。**

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
- **前代回代主循环 floor-vs-measured(深读 HVX 手册 + CSE-proof device 实测,2026-06-05):**
  - **理论下限 ≈ 2016 cyc**:halfword×halfword 乘法是 **double-vector 指令**(占两个 multiply 资源,手册 4.1.1)
    → 1/cyc;Σ_{i<64} i = 2016 条 vmpyacc。vasr narrow(shift 资源)、Tc16 行向量(复用/热)都不在关键路径。
  - **实测 ≈ 6402 cyc/块 = 下限的 3.2×**(`-DGDN_BR_DIAG_SPLIT` o[5];**注意:必须 CSE-proof**——const-input
    fwdsubst 被 -O2 hoist,旧测 ~300 < 2016 物理下限 = 假象;用 rep 间数据依赖才得真值)。
  - **瓶颈 = per-term scalar Afx load**:2016 个不同 A_ik 各用一次、编译器只把它调度到 vmpyacc 前 1 个 packet
    → ~3cyc scalar-load 延迟未隐藏。**行间串行依赖不是瓶颈(goal 前提错)。**
  - **纯 vmpyacc 吞吐探针**(inline-asm,寄存器常驻,无 load/依赖):~0.4–1 cyc/op → 乘法不是限制,3.2 cyc/op
    全是 load 喂数 → **load-bound 实锤**。
  - **idiom 全试过、device 实测、均不 beat 6402**(故保留 single-acc):4 独立累加器 6404(排除 acc→acc 延迟链)、
    scalar 预载本地数组 8387、**C 层 4-deep scalar-prefetch 软流水 6402**(-O2 调度器把它重排回 1-packet-ahead,需 inline-asm 才能强制)、
    const-scalar 289(折叠);2×2 块递归/byte-split 只**增加**乘法数 → 不可能 beat 2016。
  - **★决定性 cap-test(`-DGDN_BR_FWD_CAP` 把内层工作量砍到 1/16,timing-only):4 线程全 solve 没变快**
    (142.9K→148.3K,噪声内)→ **前代回代在真实负载里被完全隐藏**(单线程 load 停顿被 SMT(其它 3 线程 HVX 填空闲槽)
    + 与 merge HMX 交错吸收)。**所以 inline-asm scalar-prefetch 只救单线程 3×,对真实 4 线程 solve ~0%**(且 4 线程
    wall 有 ~10% 噪声 ≫ 任何 fwdsubst 收益)→ **不值得写复杂的变长 asm,保留干净形态**。`gdn_diag_fwdsubst` 已抽出 + bit-exact。
- **bit-exact 门通过**:offdiag-oc=1.35e-3、whole-oc=9.5e-5(对 `T_ref_h32.raw` fp64 逆),helper 抽出后全 solve 输出逐字节不变。
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
