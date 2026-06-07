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
- **结论:值逐位不变,只换更窄容器 → relerr ≈ 0。**

## ⚠️ 修正 benefit(2026-06-08,读完 quant 内循环)
- quant/requant 的乘法是**逐码一次**(`code×Mg`),int16 码要寄存器内 widen 回 int32 再乘 → **乘法次数不变,compute 不砍半**,只省读带宽。
- **真正砍半的只有纯格式/加法 op**:widen i8→i16、narrow i16→i8、acc pure-add、maxabs(int16 lane 减半 / 少一级 pack)。
- **修正降幅估 ~5-10%(2.50M→~2.3M),不是 ~12%**。quant-rescale(QUANT 2.73M 大头)基本不动。
- 性价比:大工程(~400 行 precision-critical clean-room)换 ~5-10% → 建议开新上下文专做,别在小预算里硬塞(易留隐蔽 bug)。
- 已起头:`gdn_scr_t` 加 `Tblk16/Sacc16/qbuf16`(GDN_BR_I16 gated,不影响现路)。

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

## 已确证的交付
- 成本:effective 仅 3-4%(不碰);int 域 glue 是目标。
- 精度:int16 ≈ 无损(数据证实 0% 超界)。
- **落地:`GdnSolveBR16.cpp` int16 静态 solve,2.45M→2.19M(~11%)bit-exact,已设 PIPE 默认**(commit 4e5111f)。

## ⚠️ Review 发现的反直觉结论(2026-06-08):VTCM round-trip 在 v75 是免费的,别去抠它
按"数据源头/不做自我抵消 round-trip"原则 review,发现 5 处 `i16→i32 widen-back`(为复用 int32 quant/requant)。
**实现了 int16-native(寄存器内 widen,无 int32 round-trip)→ A/B 实测反而慢 ~3%(2.21M→2.27M),已 revert。**
- 根因:**VTCM 带宽充足,int32 round-trip(写/读 Tc)几乎免费**;而 int16-native 把 vshuff widen 揉进 multiply + q4 寄存器 juggling → **HVX 调度更差**。两个干净的 streaming pass(单独 widen + int32 quant)比 fused 流水更好。
- **大教训:v75 这条管线瓶颈是 HVX issue/compute,不是 VTCM 带宽。** 所以有效杠杆是"**更少 HVX op/lane**"(int16 lane 减半、少 pass),不是"更少内存搬运"(round-trip 消除)。
  → 这也解释了为何 int16 有用(widen/narrow/acc lane 减半 = 真减 compute),而 round-trip 消除没用(compute 没减,只挪了内存)。
- diag 的 i32→i16 narrow(diag 内部算 i16 却输出 i32)同理 —— 没去碰,大概率也是中性。
