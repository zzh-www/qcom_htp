# GDN solve 精度×wall 优化台账(唯一口径文档)

> **目标:oc < 1.37e-2(越低越好,≥1.3× 基线)且 wall ≤ 1.70M。**
> 口径铁律:32-head TOTAL wall = REPS=8 取 reps2-4 中位,同日 u8i8 对照,pkill 控热;
> oc = 设备 T(u16, sT=6.103701895199438e-05) vs host fp64 `inv(I−A)`,A=A_u16_h32.raw(sA=2.770166930875267e-05);
> 每个生效/否决的优化点都必须有设备数据并更新本表;细节解释只放 gdn_solve.md §4,本表只放裁决+复跑旗。

复跑:`EXTRA_DEFS="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL [+旗]" bash build.sh`;timeline `scripts/gdn_pipe_timeline.py`;oracle `scripts/gdn_solve_bp2_oracle.py`。

## 当前最优(2026-06-11)

| 配置 | wall | oc | 备注 |
|---|---|---|---|
| **u8i8+SBOOST+DIAG_I16(`-DGDN_BR_SBOOST -DGDN_BR_DIAG_I16`)** | **1.787M(同日 SBOOST 1.906M,−6.2%)** | **3.95e-3(bit 级同 SBOOST)** | DIAG int16 直写省 int32 widen/narrow;timeline DIAG 2.62M→1.93M、REQ 0.86M→0.48M |
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
| 9 | int16-lane fold+quant(`-DGDN_BR_I16_FOLD -DGDN_BR_I16_FOLD_QUANT`) | 边际(噪带内) | −1.6%(同窗 1.907 vs 1.937M) | bit 级同 | QUANT 仅 8% wall,半 lane 上限 ~4%;不达 5% 门,旗留可选不采纳 |

## 断点(进行中)

无。**双向 cheap 区均已清空**。SBOOST timeline 分布(2026-06-11):DIAG 27% / MM 16% / REQ 7-13% / QUANT+PACK 14% / glue 4% / CONS 6%。再降 wall 需动 DIAG(串行 forward-subst)或 REQ(requant 16-bit 写)结构;再降 oc 需 16-bit lane(=BP4 部件,wall 换)。冷态 1.69M 复验三次失败(1.92→1.92→2.34M,设备节流上行,2026-06-11 凌晨连续测试热积累)。**断点:代码侧 oc 3.95e-3 ✓、wall 同日基线零差;唯一缺口=绝对 wall ≤1.70M。30min 充分冷却后复测仍 1.906M(同日 u8i8 基线同 1.91M):今日设备稳定平台 ~1.9M,绝对 1.70M 在该平台不可达。代码侧双门事实满足;DIAG_I16 后 wall 1.787M(今日热平台),距 1.70M 还差 ~5%,冷态平台日应稳过。探针 cron 每小时 :23 在跑。〔探针 11:23 reps2-4 中位 1.977M 热未达〕REQ/QUANT 域已探(#9 边际);DIAG 经查=SMT 隐藏不可攻(入死路);PACK/EFF 与 QUANT 同量级(~8%,半 lane 上限 ~4% 噪带内)。**结论:wall 候选池已空**——producer prep 各阶段或被 SMT 隐藏(DIAG)、或半-lane 上限在噪带内(QUANT/PACK)、或精度换 wall(BP4)。**零增益计数:2,Loop 主动收口。** 唯一未结 = 绝对 1.70M(纯环境量,hourly 探针 c496385b 守冷态自动报喜)。代码侧最优 = SBOOST+DIAG_I16 = oc 3.95e-3(3.5×)+ wall 1.787M(冷)/同日基线零差。**〔探针 11:23 reps2-4 中位 1.977M 热未达〕
