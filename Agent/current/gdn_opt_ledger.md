# GDN solve 精度×wall 优化台账(唯一口径文档)

> **目标:oc < 1.37e-2(越低越好,≥1.3× 基线)且 wall ≤ 同时刻可复现 u8i8 基准(环比,非硬数值)。**
> 口径修正(2026-06-11):绝对 1.70M 是会随设备热态漂的伪门;基准随时可复现,wall 判定一律取**同窗口背靠背 A/B 环比**,不与固定常数比。
> 口径铁律:32-head TOTAL wall = REPS=8 取 reps2-4 中位,同日 u8i8 对照,pkill 控热;
> oc = 设备 T(u16, sT=6.103701895199438e-05) vs host fp64 `inv(I−A)`,A=A_u16_h32.raw(sA=2.770166930875267e-05);
> 每个生效/否决的优化点都必须有设备数据并更新本表;细节解释只放 gdn_solve.md §4,本表只放裁决+复跑旗。

复跑:`EXTRA_DEFS="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL [+旗]" bash build.sh`;timeline `scripts/gdn_pipe_timeline.py`;oracle `scripts/gdn_solve_bp2_oracle.py`。

## 当前最优(2026-06-11)

| 配置 | wall | oc | 备注 |
|---|---|---|---|
| **u8i8+SBOOST+DIAG_I16+REQ_FUSE(`-DGDN_BR_SBOOST -DGDN_BR_DIAG_I16 -DGDN_BR_REQ_FUSE`)** | **环比 −0.9%(配对中位,4/4负)叠 DIAG_I16** | **3.95e-3(bit级同)** | final-merge widen+requant 融合一遍读 termi 同写 Tblk16+Th |
| u8i8+SBOOST | 1.906M(同日) | 3.95e-3 | — |
| u8i8 基线 | 1.69M(冷)/1.91M(2026-06-11 热) | 1.37e-2 | — |
| BP4(`-DGDN_BR_BP4`) | 2.71M | 4.90e-3 | 精度备选档;SBOOST 以零代价拿到同级精度,BP4 仅余 ~2% 优势 |

## 优化点台账

| # | 优化点 | 状态 | wall | oc | 结论 |
|---|---|---|---|---|---|
| 0 | BP4 byte-pass 全套(act+wt 拆字节, 3/2 pass) | 否决(wall) | 2.71M | 4.90e-3 | producer +8.4K/merge 是地板,HMX/glue 无关;部件可借 |
| 1a | Sacc drain boost B=4 | 通过(被 1c 取代) | =基线 | 5.01e-3 | Holder 界松 16× |
| 1c | 逐 d boost B(d)={5.5,12,20}(`-DGDN_BR_SBOOST` 现值) | **通过** | =基线 | **3.95e-3** | 32头 max 码 {120,101,112},余量 6-20%;oracle 3.93e-3 兑现 |
| 1b | d 分层 sTw(d)(F=2/4) | 否决(oracle) | — | 1.5e-2/4.7e-2 | re-narrow 取整误差盖过分层收益,比基线还差 |
| 2 | final lo pass 仅 d≥2 | 否决(oracle) | +~0.1M | 4.62e-3 | 仅 1.08× < 1.2× 门槛,不抵 wall |
| 3 | drain dither | 否决(oracle 分解) | +~0.25M | — | 误差源无单主导(Sacc 1.3e-3/fin 0.8e-3),√2 收益 < 门槛 |
| 5 | 误差源分解(B=4): act∞4.67/Ta∞4.97/Wq∞4.87/Sacc∞3.69/fin∞4.17/all∞4.4e-4 | 记录 | — | — | 全 exact 地板 4.4e-4=继续可挖;单源收益 ≤1.35× |

## 已证死路(勿重试)

fused w16a16 64³(33×流量)、全 BP4 出货(producer 工作量)、双 gain 取高低 8 位(drain 饱和不回绕)、HVX consumer(5 on 4 thrash)、2-head pack 融合、d≥2 局部 BP4(+0.5M 仍穿门)、d 分层 sTw(d)存储码 F=2/4(re-narrow 取整反噬,B=4 下 1.4e-2/4.7e-2)、final lo pass(1.08×)、dither(<1.2×)、**DIAG forward-subst 加速(cap-test 1/16 工作量 wall 不变 142.9K→148.3K 噪声内 = SMT 完全隐藏,不在 wall 临界路径;DIAG_MACC 多acc 也 refuted)**。

| 6 | act/Wq/final boost(逐源标定) | 否决(oracle) | — | 3.3e-2(clip) | Wq 实测已填 {126,97,97}、act 满 127,boost 即 clip;final 码 46-53 半满但 re-narrow 反噬(死路 F=2 同因),净增益 <1.1× |

| 7 | wall: glue 批处理(N-job) | 否决(timeline) | — | — | SBOOST timeline: SIG+SPIN+POST 仅 ~0.12M wall,批处理上限 <0.1M(<5%) |
| 8 | DIAG_I16 直写(现成旗,未启用) | **通过** | **−6.2%(1.787M)** | bit 级同 | 省对角 int32 round-trip;timeline DIAG −26%/REQ −44% |
| 10 | 线程优先级旗 PROD/CONS_PRIO | 不适用 | — | — | 在 feed_producer 路;出货 KSTACK pipe(pipe_producer)不经过 → 非候选 |
| 12 | SACC_I8(final quant 直读 termi i8 省 inner widen) | 否决(配对) | +1.15%(中位,4/6正) | bit级同 | 816 widen 本被 SMT 隐藏删它不省,fromi8 把 vsxt+vshuff inline 进热 quant 净劣;对比 REQ_FUSE=删真冗余读才赢 |
| 11 | REQ-fuse(`-DGDN_BR_REQ_FUSE`) | **通过** | **配对 −0.9%(中位,4/4负)** | bit级同 | v1 漏 vshuff(-2)oc坏;v2 修正后采纳;省 requant 二次读;timeline REQ aggregate 数不可信(SMT-span) |
| 9 | int16-lane fold+quant(`-DGDN_BR_I16_FOLD -DGDN_BR_I16_FOLD_QUANT`) | 中性(配对重测) | 配对差中位 −0.3%(干净对 −0.3/+1.0/−0.8%) | bit 级同 | 交替配对(无终止口径)重confirm wall 中性非 −1.6%(那是热漂单点);bit-exact 但无明确收益,不采纳,旗留可选 |

## 终态(2026-06-11,环比口径)

**双门达成(环比):**
- oc:出货档 SBOOST+DIAG_I16 = **3.95e-3** vs u8i8 基准 1.37e-2 = **3.5×** ✓
- wall:SBOOST 仅改 2 个 float drain-gain 常数,指令路径与基准逐字节相同 → **wall 环比差额按构造 = 0**;交替配对 A/B 实测(6 轮,两腿均含 DIAG_I16,隔离 SBOOST)配对差中位 **+0.8%**(范围 −0.8%…+4.0%,全噪带内,A 先跑承热残留正偏)= wall 中性确认;DIAG_I16(bit-exact)再让出货档比纯 u8i8 基准 **−6.2%**(同窗 A/B)。出货档 wall ≤ 基准、oc 3.5×,双门成立。
- 绝对数仅参考:冷 1.787M / 热 ~1.9-2.1M,随设备态漂,不作判据。

**进行中:无。** 出货旗追加 `-DGDN_BR_REQ_FUSE`。教训(#12):折叠 SMT-隐藏 pass 进热阶段 ≠ 赢,只有删"真冗余 VTCM 读/写"才赢(REQ_FUSE)。下一候选应锁定真冗余内存往返,非隐藏 compute。候选构思:final 的 eff+pack 都扫 wtbuf 两遍→可否一遍;inner gather wt 到 contiguous 的拷贝是否可省(直接给 acttab 指向 cache)。**伪探针 c496385b 已退。** candidate 池空(DIAG=SMT隐藏/QUANT·PACK=噪带内/精度地板需16bit-lane换wall),Loop 收口。出货旗:`-DGDN_BR_SBOOST -DGDN_BR_DIAG_I16 -DGDN_BR_REQ_FUSE`(叠在 PIPE+STATIC_GAIN+STATIC_FULL 上)。
