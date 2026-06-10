# GDN solve 精度×wall 优化台账(唯一口径文档)

> **目标:oc < 1.37e-2(越低越好,≥1.3× 基线)且 wall ≤ 1.70M。**
> 口径铁律:32-head TOTAL wall = REPS=8 取 reps2-4 中位,同日 u8i8 对照,pkill 控热;
> oc = 设备 T(u16, sT=6.103701895199438e-05) vs host fp64 `inv(I−A)`,A=A_u16_h32.raw(sA=2.770166930875267e-05);
> 每个生效/否决的优化点都必须有设备数据并更新本表;细节解释只放 gdn_solve.md §4,本表只放裁决+复跑旗。

复跑:`EXTRA_DEFS="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL [+旗]" bash build.sh`;timeline `scripts/gdn_pipe_timeline.py`;oracle `scripts/gdn_solve_bp2_oracle.py`。

## 当前最优(2026-06-11)

| 配置 | wall | oc | 备注 |
|---|---|---|---|
| **u8i8+SBOOST(新出货候选,`-DGDN_BR_SBOOST`)** | **=基线(同日 1.91M vs 1.91M,差 0)** | **5.01e-3(2.7×)** | Sacc drain gain ×4,零结构改动;2026-06-11 热态,冷态复测应回 ~1.69M |
| u8i8 基线 | 1.69M(冷)/1.91M(2026-06-11 热) | 1.37e-2 | — |
| BP4(`-DGDN_BR_BP4`) | 2.71M | 4.90e-3 | 精度备选档;SBOOST 以零代价拿到同级精度,BP4 仅余 ~2% 优势 |

## 优化点台账

| # | 优化点 | 状态 | wall | oc | 结论 |
|---|---|---|---|---|---|
| 0 | BP4 byte-pass 全套(act+wt 拆字节, 3/2 pass) | 否决(wall) | 2.71M | 4.90e-3 | producer +8.4K/merge 是地板,HMX/glue 无关;部件可借 |
| 1a | Sacc drain boost B=4(`-DGDN_BR_SBOOST`) | **通过** | =基线 | **5.01e-3** | Holder 界松 16×;maxSacc 码 87(32头),B=8 会 clip;oracle F=1 B=4 4.99e-3 兑现 |
| 1b | d 分层 sTw(d)(F=2/4) | 否决(oracle) | — | 1.5e-2/4.7e-2 | re-narrow 取整误差盖过分层收益,比基线还差 |
| 2 | final lo pass 仅 d≥2(pair-job 复用,~+0.1M) | 待试 | — | — | SBOOST 后边际收益需重扫 oracle |
| 3 | drain dither(双发 ±0.5 LSB 均值,~+0.1M) | 待试 | — | — | drain 误差减半,地板收益,配 1/2 用 |
| 4 | Sacc drain boost(B≤4,u8i8 路) | 待试 | — | — | Holder 界松 16×,码界数据见 gdn_solve.md §4 |

## 已证死路(勿重试)

fused w16a16 64³(33×流量)、全 BP4 出货(producer 工作量)、双 gain 取高低 8 位(drain 饱和不回绕)、HVX consumer(5 on 4 thrash)、2-head pack 融合、d≥2 局部 BP4(+0.5M 仍穿门)。
