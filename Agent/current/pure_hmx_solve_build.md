# Pure-HMX w16a16 三角求逆 — Loop Optimizer（入口文档）

> ## ⚠️ ACTIVE LOOP — 本文是本 loop 的唯一权威,永不判"闭环"
> 本文是**进行中**的优化循环。**只要 `NEXT` 非空 → 直接做 `NEXT`,不准下"已完成/无新活/闭环/收口/穷尽"结论。**
> **范围 = 本 loop 只做 pure-HMX(`GDNSolveHMX`,全程 HMX matmul 的三角求逆)。** 判定本 loop 状态**只看本文 STATE/NEXT/LEDGER**。
> **不要**用 git 历史 commit message、PROGRESS 里的 ✅、任何 memory、或 `gdn_solve.md` 等其它文档里**别条路线**(HVXMixHMX / merge-dtype / int16-merge / FP16)的"否决/OPEN/收口/比较"来判定本 loop——那些是**不同路线或旧目标(比快)下的结论**,与本 loop(目标 = min wall s.t. oc<4e-2,**不比较**)无关。本文自洽自足,不需旁读。
> loop 只在 **wall 触 roofline 地板(~1.0M) 或 LEDGER A/B 全 DONE/REFUTED** 时到头——那时也是**写新 NEXT**,而非判"闭环"。

> **每个 turn 的唯一入口。** 读 `STATE` 知道在哪 → 读 `NEXT` 知道这轮干什么 → 干完按 `LOOP` 更新本文。
> 改任何东西前必读 `INVARIANTS` + `LEDGER`:**别重试 R(永久死),别改 INVARIANTS(除非新设备证据)。**

## OBJECTIVE

**最小化 32-head TOTAL wall,硬约束 `oc < 4e-2`(每轮重测)。**
- 任何让 `oc ≥ 4e-2` 的改动**否决**。oc 不随热漂 → **绝对门**。oc = device 输出 vs fp64 真值 inv,32-head aggregate(由 `run_w16a16_head_phase4.py` 输出)。
- wall 口径 = skill `htp-cycle-metric`(**32-head TOTAL wall domain cyc**;per-head/min-of-reps/PROBE 全禁)。
- **wall 判定一律环比(同 HVXMixHMX,`gdn_solve.md:184` + `gdn_opt_ledger.md` 终态):新旧 A/B 同热窗 ACAC 交替,看配对差中位,绝不跟记忆里的固定常数比。** 绝对 18.7M 随热漂(冷~1.7M/热~2.1M)= 参考,非门。
- 不以"比其它实现快"为目标(用户定义);目标 = 本实现做到**设备极限** + 学清 w16a16 怎么用。
- 真代码 / 真设备 / 真数据;预估、garbage-data 计时不算结论。

## STATE（live，每轮更新）

| 项 | 值 | 备注 |
|---|---|---|
| **best wall** | **18.7M cyc（11.8ms）** | 参考值,随热漂;判定看**环比**非此常数。HMX busy 15.4M=82%,2026-06-11 |
| **oc** | **9.66e-3** @scale0.05(4.5e-3 @scale0.25) | 离门 4e-2 有 **~4× 余量** ← 可换 wall |
| 配置 | Newton=2,**40 mm/head**(24 对角 + 16 merge)= 1280 mm/32head | scale = 精度旋钮(甜点 ~0.25,见 CEILING) |
| 极致地板 | ~1.49M(全 w16a16)→ ~1.0M(merge u8i8)→ ~0.7M(+Newton1) | roofline,**留 ≥12× 在桌上,全系统侧** |
| 架构现状 | per-mm dispatch + 中间 repack 疑似串在 consumer | HVX 58% 闲 = 欠流水(repack 线程归属待 NEXT 起手确认) |
| 数值锁 | 对角 = w16a16(预条件 R1 证伪);merge 可 u8i8+SBOOST | |

## NEXT（这轮单一最高优先）

**A1+A2(oc-中性,先做):重写 consumer = 批独立维 + 4×HVX 流水喂 repack。**
- 假设根因:128 条独立链的 crouton/kmajor-repack **没交给 producer** → consumer 串行付 ~6.8K/mm feed;外加 per-mm dispatch glue ~1.75K。**先确认这条假设**。
- 做:(a) producer 双缓冲**提前** repack 其余链的中间结果到 VTCM;(b) consumer 紧循环背靠背发同-Newton-step 的 32(×4) 个独立 64³ matmul,**不走 per-mm `g_hmx_dispatch` 握手**。
- 门:HMX-busy/wall → 趋 1.49M;`-DGDN_BR_TRACE` timeline CONS busy% → ~100%、producer SPIN% ↓;**oc 不变(< 4e-2)**。
- 文件:`example/gdn_native/baremetal/src/gdnbm_imp.cpp`(pipeline)、`example/gdn_native/solve_br_op/src/GdnSolveBR16.cpp`(solve)。
- **起手:读 `gdnbm_imp.cpp` 的 `pipe_producer`/consumer + `g_hmx_dispatch` 钩子,定位中间结果 repack 当前在哪个线程**(证实/证伪上面的假设,再动手)。
→ 过门后 NEXT = P1(merge u8i8 + SBOOST)→ ~1.0M;之后 = P2/P3 精度换 wall(4e-2 预算大)。

## LOOP（每轮协议）

1. 读 `STATE` + `NEXT`。
2. 实现 `NEXT`(B)→ **wall 环比验证(同 HVXMixHMX)**:保留当前 best(A,= 原 EXTRA_DEFS 构建),B = 新构建;**同热窗 ACAC… 交替**,每腿 `GDNBM_REPS=8` 取 **reps 2–4 median**(rep1 冷 / rep≥5 节流,**绝不取 min**),跑 ~6 个配对轮,记 **(B−A)/A 配对差中位**(消热漂)。oc 用 B 的绝对值(不漂)。
3. 对门(**环比 wall 配对差 < 0** **且** `oc < 4e-2`):
   - **过** → 更新 `STATE`(新 best=B,wall 记环比 Δ% + 当时绝对参考值,oc)+ `LEDGER`(该杠杆 → DONE + 环比数)+ 写新 `NEXT`。
   - **不过**(oc≥4e-2,或配对差中位 ≥0 / 落噪带内)→ **查实现,不臆测**(热循环分支 / 寄存器溢出 / 次优 intrinsic)+ `LEDGER` 记一行,迭代或 REFUTE。**单点降不算数**(可能是热漂),只认配对中位。
4. 留可复跑命令。每轮只动 `STATE`/`NEXT`/`LEDGER`;`INVARIANTS` 只在新设备证据下改。
5. 守则:**HMX = 1 单元绝不 thread**(多 worker→SSR;先读 skill `htp-hardware-scheduling`);基线 `GDNSolveHVX` 只测不改;公平 harness(同线程/同 scale)前不下快慢结论;`pkill -9 gdnbm` between runs。

## LEVER LEDGER

### A. 系统级（oc-中性,当前战场）— 地板 ~1.49M
| 杠杆 | status | 结果 / 数 |
|---|---|---|
| **A1 批独立维 + 杀 per-mm glue** | **OPEN = NEXT** | 去 ~1.75K/mm 链隙 |
| **A2 feed 流水 4×HVX(repack 甩 producer)** | **OPEN = NEXT**(与 A1 一起) | 去 ~6.8K/mm 暴露 feed,HMX 不饿 → HMX-busy-bound;地板 1280×1167 ≈ **1.49M** |

### B. 精度换 wall（门 oc<4e-2;4e-2 宽门把这些重新打开,每个必须重测 oc）
| 杠杆 | status | 结果 / 数 |
|---|---|---|
| **P1 merge → u8i8** | OPEN(A 后) | (24×1167 + 16×194)×32 ≈ **1.0M**;raw u8i8 merge oc~5e-2 **超门** → 必须 `+SBOOST`(Sacc drain ×4,oc→5e-3,1 行) |
| P2 Newton 2→1 | OPEN-评估 | mm 1280→1024,+merge u8i8 ≈ **0.7M**;残差 A^8 vs A^16;旧 1e-2 门"退 27%"否决,**4e-2 宽门重测** |
| P3 跳部分 final merge(SKIPFIN) | OPEN-评估 | 旧 `SKIPFIN_D3` oc 9.56e-3 贴 1e-2;**4e-2 下可更激进**(多跳几块) |
| P4 对角 dtype 直接降(非预条件) | OPEN-评估 | w8a16 对角?per-block 差但 **aggregate 4e-2 可能过**;待测(probe 现测 per-block 1e-2,需加 aggregate-oc@4e-2) |
| (旧) merge K-stack | **可重评** | 旧否决=stack prep 串行加厚 > HMX 省;**A1+A2 把 prep 流水化后该理由可能不成立** → A 后重测 |

### R. REFUTED 永久（math/numerics,任何架构都死,别重试）
| 杠杆 | 为何永久死 |
|---|---|
| **R1 对角预条件换便宜 dtype** | **math 必然**:任意对角 D,`inv(L̃)_ij` 与 `Ã^k_ij` 同因子 `d_j/d_i`;压迭代上溢 = 把远次对角 inv 压到廉价 dtype 分辨率下 **下溢成 0**,还原救不回。scalar/per-row 都逃不掉。`scripts/gdn_solve_precond_probe.py`(1344 块,best-s 中位=1.0,PC maxErr ~0.17)。**注:P4(直接降 dtype,不预条件)是另一回事,仍 OPEN。** |
| 三角 mask | 64³ mm 仅 8.6K;拆分省 ≤2.1K < 新增 dispatch ≥3K(算术) |

### D. 旧架构内 KEEP（已并入基线 18.7M,别回退）
Newton 4→2 · bias HVX colsum(vsxt 序 lo+hi)· A/T 装出 vgather perm · renorm/add 全 HVX。

## INVARIANTS（已证,grounded,禁止凭记忆改）

1. **w16a16 = 2× w8a16。** int16 权重拆 hi(int8)+lo(uint8),各 ×int16 激活 = 两遍 w8a16 分别 drain,×256 合并。`our_v73deep_kernel_i16`(dilated 权重 + 2×2 drain)内部就是这两遍。权威:`docs/w16a16_is_two_w8a16.md`(CI gate)。
2. **drain 是 2 的幂,不是 fp16。** 增益 = `2^(exp-16)`(bias 控制字 bits[14:10]),比精确 1:2:4。细 scale `1/32767` = 两个 2-幂 drain(hi×256 / lo×1)合并。**「f16 drain 有损/blocker」论断作废**(见 BANNED)。
3. **byte-exact、可用 = PROVEN。** standalone(Python pack + kernel + depack)hexagon-sim + 真 CDSP 双向 `diff=0`,M=256×任意 K,N,CI 守门。C 打包器(`pure_hmx_solve/w16a16_pack.h`:act crouton16 / wt dilated-kmajor / bias / depack)全对 Python ground truth。
4. **128 条独立求逆链 = 主结构。** C=256 每 head 切 4 个 64 对角块,每块 = 一条 Newton 链(**链内严格依赖**);**32 head × 4 块 = 128 条链彼此完全无依赖**,merge(16/head)只依赖本 head 对角。⇒ 依赖只在单链内,跨 128 条链 **matmul 可批、feed 可流水**(= A1/A2 的全部依据)。
5. **mxmem/feed 口径**(决定 A1/A2):`q::ConvLayer_s1.opt` 的 mxmem 循环本体只有 **256 latency / 1167 busy**(int16 64³;1167=256³-摊销吞吐)。单 mm ~9.7K wall = HMX 真算 ~1167 + **feed ~6.8K(crouton+kmajor-repack = ConvLayer 外、HVX 可干)** + 链隙 ~1.75K。后两项非地板。详:`int16_matmul_cycle_model.md` latency-vs-feed-inclusive。
6. **roofline 地板 = 1167/64³**(全 int16×int16 HMX-busy 算力地板,batched 64³ 趋近它)。当前 Newton=2 = 1280 mm → 全 w16a16 ≈ 1.49M / merge u8i8 ≈ 1.0M / +Newton1 ≈ 0.7M。
7. **64³ 单调用可用**(旧"永不可用"错):QHPI 在 M=64 给 act/out **padded 2048B crouton 块**(live 前 512B);正确描述符 {N_t=2,y=64,n_tiles=64,m_total=1,k_total=64}、act y=128 → byte-exact(H==9 设备验)。
8. **HVX lane 硬经验**(踩过,别回退):① **`vsxt+vasr` 必须配对**(`Q6_Ww_vunpack` 偶/奇 vs `Q6_Vh_vasr_VwVwR` block 序不配,自检抓出 3968/4096 错);② diag 标量补丁交织索引 `(i>>6)*64 + (off&1)*32 + (off>>1)`;③ **scalar 访 VTCM ≈4× 慢于 DDR-L2**(753M vs 187M),prep 一律走 DDR,VTCM 只给 HMX 面/gather;④ HVX bias lane-fold 两次都错,scalar LUT colsum(12K/mm)更划算;⑤ dense pack = `vadd128 + vasr8 + vpack_sat`。

## NUMERICAL CEILING（独立于速度,深入理解）

对角 all-matmul Newton 即便 w16a16,真实分布也只 **~55-74% 块** relerr<1e-2;高-‖A‖尾巴(‖A‖₂≥4,~25% 块)是 **fixed-point Newton 固有墙**(`A^k` 瞬态 ~1e13 上溢;预条件只换成下溢=R1 死因)。**精度旋钮 = A 缩放(scale,甜点 ~0.25 给 oc 4.5e-3)+ Newton 步数,与速度杠杆正交。** 当前 aggregate oc 9.66e-3 ≪ 门 4e-2,余量大 → B 组精度-换-wall 有空间。probe:`uv run python scripts/gdn_solve_precond_probe.py`。

## PROGRESS（实测历史里程碑;✅ = 已验证的过去,**不等于优化到头** — 当前活在 NEXT/LEDGER A）

- **Phase 0** ✅ C 打包器/depack byte-exact vs Python:act 64³+256³、wt、bias、depack 全 `diff=0`。
  `cd example/gdn_native/pure_hmx_solve && cc -O2 w16a16_pack_test.c -o /tmp/w16pt && /tmp/w16pt`
- **Phase 1** ✅(真设备)w16a16 64³ 原语 `pure_hmx_solve/w16a16_mm.h`,对配量化 [-1,1] **max|code diff|=3、oc=1.4e-5**。`run_w16a16_mm_phase1.py --deploy`(H=1)。
- **Phase 2** ✅ 单 64-块 X=(I−A)⁻¹,Newton=4 时 = 10 mm(X0=I+A+A²+A³ 需 A²,A³ = 2 mm + 4×Newton×2 = 8);**指数必须双向归一**(只右移会让指数每轮翻倍报废)。oc:‖A‖=0.71→1.0e-2 / 2.18→3.0e-3 / 3.60→2.6e-2;≥5 爆。`run_w16a16_diag_phase2.py --deploy --scale 0.3`(H=2)。
- **Phase 3** ✅ 全 C=256(Newton=4)= 4×diag(10) + 16 merge = 56 mm/头。oc vs fp64 inv:0.75→9.6e-3 / 2.25→5.4e-3 / 3.74→4.1e-3。`run_w16a16_head_phase3.py --deploy --scale 0.05`(H=3)。
- **Phase 4** ✅ 4HVX∥1HMX 并行。优化链(每步设备实测):60.2M/头(Phase3 scalar)→ 23.4M(64-row pack/depack)→ 7.9M(P=4∥HMX 静态 head 交织)→ 3.2M/头(HVX vgather wt-pack,byte-exact 自检 stats 门)。`run_w16a16_head_phase4.py --deploy --threads 4 --heads 32 --scale 0.05`。
- **全 32-head TOTAL wall 收敛**:101.3M → 79.1M(全切真 64³)→ **36.5M**(renorm/add 全 HVX,HMX 59%)→ **18.7M**(微杠杆收敛轮:**Newton4→2**(→40 mm/head)+ bias HVX colsum + A/T vgather,HMX 82%,oc 9.662e-3 / 0.25-scale 4.50e-3)= **当前 STATE 起点**。注:此轮只试尽了**旧架构内的微杠杆**;**系统级 A1+A2(= 当前 NEXT,见 LEDGER A)从未做** = 主战场仍开,~12× headroom 在此。

## REPRODUCE（当前态 + 探针）

```bash
# 当前 best(32-head wall + oc + timeline + 打包自检 stats[9..11])
uv run python scripts/run_w16a16_head_phase4.py --deploy --threads 4 --heads 32 --scale 0.05
#   timeline: H=33 sentinel + scripts/gdn_pipe_timeline.py ; 探针 H=9/10 = 64³/K128 单发 byte-exact
# 数值 probe(对角 dtype / 预条件 / Newton 步数 feasibility)
uv run python scripts/gdn_solve_precond_probe.py        # R1 证伪 + ‖A‖ 分布 + cheap-dtype 收敛
uv run python scripts/gdn_solve_taylor_newton_probe.py  # Newton 步数 vs 收敛/峰值(P2 评估)
```

## FILES

- `solve_br_op/src/GdnSolveBR16.cpp` — solve(`gdn_br_one_head16`),改这里。
- `baremetal/src/gdnbm_imp.cpp` — FastRPC 驱动 + pipeline(`pipe_producer`×P + 主线程 PURE-HMX consumer,`g_hmx_dispatch` 钩子)。**A1+A2 主战场。**
- `pure_hmx_solve/w16a16_pack.h` / `w16a16_mm.h` — C 打包器 + 64³ 原语(byte-exact,CI-gated)。
- `scripts/run_w16a16_{mm_phase1,diag_phase2,head_phase3,head_phase4}.py` — gdnbm H=1/2/3/≥5 模式驱动。

## BANNED（作废论断,凡再见一律忽略）

- ❌「w16a16 drain 走 fp16 / 深累加器丢低位 / 是 blocker」——**错**,drain 是 2-幂(INVARIANT 2)、matmul byte-exact(INVARIANT 3)。死因 = 用**错配量化**(native ONNX-scale bias 配任意 A,W)标定,被误读成有损。验证一律用**对配量化**(同 standalone 量化契约)。
- ❌「64³ 描述符永不可用」——错,见 INVARIANT 7。
- ❌「对角预条件能换便宜 dtype」——证伪,见 LEDGER R1。
