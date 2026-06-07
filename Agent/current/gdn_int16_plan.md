# GDN solve int32→int16 — 落地方案(2026-06-08,精度已确证)

承接 `gdn_hmx_solve_plan.md`。当前最佳 = PURE-HMX PIPE P=4 **~2.50M**(32-head total wall)。

## 为什么做(成本,timeline 实测 PURE-HMX P=4 aggregate)
| 阶段 | agg | /prod | int32 域? |
|---|---|---|---|
| QUANT(fold+quant-rescale) | 2.73M | ~27% | rescale 部分是 |
| DIAG | 2.00M | ~20% | 难动 |
| MM | 1.86M | ~17% | |
| PACK | 0.82M | ~8% | |
| REQ(requant+widen) | 0.63M | ~6% | **是** |
| ACC | 0.47M | ~4% | **是** |
| EFF(effective) | 0.40M | ~3-4% | 否 → **太小,不碰**(精度不划算) |

int 域 glue(quant-rescale + ACC + REQ + widen/narrow)是真目标;int16 让它们**向量 lane 减半 ~2×** + 带宽减半。

## 精度:≈无损,已用数据证实(这是 relerr 答案)
- diag 码实测 **−7716..16384,0% 超 int16**(`GDN_BR_TI=2/32767` 专为塞满 int16 设计)。
- Sacc 静态 **pure-add ≤3 个 i8 = ±381**(2^22 是动态路,静态不走)。
- merge 块 = HMX 出 i8(±127)。
- diag solve 本就在 int16(`Tc16`/`ei16`)算,只是 widen 进 int32 → 改 int16 反少一次 widen。
- **结论:值逐位不变,只换更窄容器 → relerr ≈ 0。** 降幅估 2.50M→~2.2M(需落地实测)。

## 改法(GDN_BR_I16 flag,struct + ~8 函数 + 30+ 调用点)
1. `gdn_scr_t`:`Tblk`、`Sacc` → `int16_t`(`qbuf` 当 widen 临时也要 i16,可复用现成 `Tc16`)。
2. `gdn_widen_i8_to_i32` → i8→i16(只需 `Q6_Wh_vsxt_Vb` + 一次 vshuff 复序,比 i32 版少一半输出)。
3. `gdn_acc_i8_to_codes`(STATIC pure-add):widen i8→i16,`Q6_Vh_vadd_VhVh` 加。
4. `gdn_narrow_i32_to_i8` → i16→i8(直接 `Q6_Vb_vpack_VhVh_sat`,更简单)。
5. `gdn_quant_i8_from_codes` / `gdn_quant_u8_from_codes`:`codes` 改 `const int16_t*`;pre-shift(`while m>=2^15`)静态下恒不触发可留;`code×Mg` int16×Mg→int32 不溢。
6. `gdn_requant_block_out`:`codes` 改 int16。
7. `gdn_maxabs_codes` → int16 版(`Q6_Vh` 归约)。
8. `gdn_solve_diag64` 输出:已是 int16 内部 → 直接写 int16 Tblk(去掉它内部到 int32 的 widen)。
9. 全部调用点(7+7+5+5+2…)类型对齐。

## 验证(铁律)
- 每步 build → **timeline 重画**量 int 域 glue 降幅 + **relerr**(`/tmp/T_avtcm.raw` 比;预期 maxdiff≈0)。
- 32-head total wall;warmup+REPS=8 稳态;A/B 控热。
- 风险:scale 敏感,一个 pre-shift/符号扩展错就 relerr 炸 → 逐函数改、每改完跑一次整 solve relerr。
- 退路:GDN_BR_I16 flag,坏了直接关。

## 已确证的交付(本轮)
- 成本:effective 仅 3-4%(不碰);int 域 glue 是目标。
- 精度:int16 ≈ 无损(数据证实 0% 超界)。commit fb752a7。
