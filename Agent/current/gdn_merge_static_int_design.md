# GDN merge —「全程静态 int + 融合累加 + 布局直通」设计基线 (Step 2)

承接第1步(convhbh int16 matmul 已 sim bit-exact,见 `gdn_solve_NEXT_AGENT.md` 第1步)。
本文是 Step 2 (把 convhbh int16 接进 GDN merge) 的**权威设计基线**,由设计讨论拍定。

> 状态标注:**【已验证】** = sim/实测铁证;**【设计】** = 本文拍的方向,尚未在 GDN merge 上 bit-exact 验证。
> 第1步的「单个 convhbh 64³ matmul bit-exact」是【已验证】;本文的融合累加/布局直通在 GDN merge 上是【设计】,待 sim 钉。

---

## 0. 三条核心原则(HTP 最优形态)

1. **全程静态 int,单一全局标度 `s`,循环内无 quant/dequant,只有移位。**
   上游量化一次(A → int16),下游吃 int16(或边界外才 dequant)。solve op **int 进 int 出**。
   依据:`scripts/gdn_solve_static_quant_probe.py` 实测 int16-static off-diag oc **1.9e-4**、int32 累加器 **0 溢出**(不需 int64)。

2. **融合累加:内层 `Σ_k` 折进 HMX int32 累加器,不中间 drain,每输出块只 drain 一次。**
   省掉 HVX 加 + per-term rescale(动态 int8 的 51% merge-glue 主体)。

3. **布局直通:`output crouton16_row4` == `activation 输入 crouton16_row4`(同 contract,【已验证】)。**
   输出当下一个 matmul 的 **activation** 槽 → 零重排;只有当 **weight** 用时才需 k-major 重排。

---

## 1. 单 HEAD 全景  (C=256, BL=64, NB=4 块, T=(I−A)⁻¹)

```
 IN ▶ A : 256×256 严格下三角 — 上游量化一次 ⇒ STATIC int16, 单一全局标度 s
 OUT◀ T : 256×256 下三角     — STATIC int16, 同标度 s (下游直接吃 int)
        L = I − A  (单位下三角, 4×4 块, 每块 64×64)

           j→  0     1     2     3
        i↓  ┌─────┬─────┬─────┬─────┐    ① 对角块 T_ii = (I−A_ii)⁻¹  ×4
         0  │T00  │     │     │     │       HVX int16 前代回代, 纯 int MAC (无除法)
         1  │T10  │T11  │     │     │       ~17% 成本, 已在 HVX 下限 → 不动
         2  │T20  │T21  │T22  │     │
         3  │T30  │T31  │T32  │T33  │    ② 非对角块 T_ij=T_ii·(Σ_k A_ik·T_kj) ×6
            └─────┴─────┴─────┴─────┘       HMX int matmul (merge), ~83% 成本
            ╲对角╱      ╲──非对角──╱          依赖序(i−j): (1,0)(2,1)(3,2)→(2,0)(3,1)→(3,0)
```

---

## 2. 一个非对角块的流水  (例 T30 = T33·[A30·T00 + A31·T10 + A32·T20])  【设计】

```
操作数 A_ik, T_kj — 全 STATIC int16, 同标度 s

STEP A  内层 Σ = 一个【大 K HMX matmul】(融合累加)         K=(i−j)·64
   ┌ A_ij│…│A_i,i-1 ┐      ┌ T_jj  ┐
   └     64×K       ┘  ·   │  ⋮    │ K×64
        (activation)        └T_i-1,j┘ (weight)
              │
              ▼  HMX 累加器 int32 ── 所有 k 项累进去, 中间 NOT drain
        ┌──────────────────────────────────────────┐
        │ Σ = K-reduction = 累加器本性 = 免费         │
        │ ✗ 无 HVX 加   ✗ 无 per-term rescale        │
        └──────────────────────────────────────────┘
              │  drain 一次 (HMX cvt)
              ▼
        S(int16) = saturate( acc_int32 >> k )   ← 纯算术右移, 无浮点
                                  ▲ k = 静态标度指数 (control word 指数位 = 移几位)

STEP B  外层  T_ij = T_ii · S = 再一个 HMX matmul
        T_ii(i16) · S(i16) → acc(i32) → drain(>>k) → T_ij(int16)

每块共 2 次 drain (内层 Σ 后 1 次, 外层后 1 次). 全程 int, 每次缩窄 = 一个移位.
```

**为什么能融合累加**:静态量化 ⇒ 所有块同标度 ⇒ int32 积**直接可加** ⇒ 内层 Σ_k 写成
`[A_ij|…|A_i,i-1] @ [T_jj;…;T_i-1,j]`(K=(i−j)·64)一个大 K matmul,HMX 的 K-loop 天生沿 K 累加 = 求和。
动态 per-block 标度则每 term 要 fixed-point rescale → 就是 51% glue;静态把它清零。

**requant 真相(纠正早期 f16 叙事)**:循环内**没有浮点**。`int32→int16` = `saturate(acc >> k)`,
纯算术右移。HMX cvt 的 control word 里**指数位 = 移位量 k**(实测 `0x4040→0x4440` 增益×2 = 指数+1;
`word2=0x0000` 出 max_abs_diff=0 = 纯移位零舍入)。f16 只是硬件写"移几位"的格式,不是乘浮点。

---

## 3. 布局两条道(merge-glue 真正的杠杆)  【部分已验证】

```
HMX matmul:  out[m,n] = Σ_k act[m,k]·wt[k,n]
   activation 端: crouton16_row4   (read activation.ub)
   weight     端: k-major :deep    (read weight.b:deep)
   output     端: crouton16_row4   (cvt drain)        ← == activation 输入布局【已验证】

            ┌─ 下一个 matmul 的 ACTIVATION ─▶ 直接喂, 零重排 (crouton→crouton 一致)
out(int16) ─┤
crouton16   └─ 下一个 matmul 的 WEIGHT     ─▶ 必须重排 k-major :deep (HVX glue)
```

**落到 merge 结构**:
- `A_ik`(原始输入)、`T_ii`/`T_kj`(对角块/已算块)= **固定操作数**,整 solve 复用多次 →
  各打包一次 k-major,**常驻 VTCM,不重复重排**。
- 链里**真正每步要重排的,只有往前传的中间结果**(内层 `S` 当外层 weight 时,k-major 一次)。
- 代价 = **# 中间结果当 weight 的次数 × 一次 k-major pack**,丢给 HVX producer、**藏在 HMX compute 后面**(FEED_4P producer-consumer)。

**优化目标**:① 角色编排——尽量让往前传的输出落 activation 槽(免费直通),固定块放 weight 槽;
② 固定操作数预打包常驻 VTCM;③ 剩余中间 weight 重排丢 HVX worker,与 HMX 排空重叠。

---

## 4. 硬件编排  (HVX ∥ HMX producer–consumer, 单 HMX 不 thread → 否则 SSR)

```
 [HVX worker ×4]  对角块前代回代 + 固定操作数 k-major 预打包 + 中间 weight 重排 → VTCM ring
        │ (静态 int 算子, 预量化一次)
        ▼
 [HMX main 1 路]  纯排空: 大 K matmul 累 int32 → drain(>>k) → int16 → activation 直通/拼 T
```
数据类型一条线:`A int16 ─▶ HMX 累 int32 ─▶ (>>k) int16 ─▶ 下个 matmul int16 ─▶ … ─▶ T int16`

---

## 5. Step 2 待办(每步 sim bit-exact → 再上 device,不跳步)

1. **【设计→验】钉 control word 的"指数位=移位量 k"**:在 sim 扫死 convhbh 4-word const
   (`[0x4040,0x8040,0x0008,0x4000]`)各字段语义,确认 `int16=saturate(acc>>k)` 的 k 可精确设、零舍入。
2. **【设计→验】融合累加**:一个非对角块的内层 `Σ_k A_ik·T_kj` 拼成大 K matmul,HMX 累 int32 不中间 drain,
   一次 drain(>>k)→ int16,对 `gdn_matmul_i16` 参考 bit-exact + 测 int32 无溢出。
3. **【设计→验】布局直通**:把 STEP A 的输出(crouton)直接当 STEP B 的 activation,验证零重排正确;
   weight 端 k-major 预打包常驻。
4. **接 baremetal**:`gdnbm_imp.cpp` 加 convhbh kernel + `GDNBM_MM_I16_TEST`(nthreads=1 小心,SSR 风险),
   对 `gdn_matmul_i16` 验 + 测 cyc;再整块 producer-consumer,整 solve wall vs 基线(~122K)。

## 6. 待定的设计决策

- **int8-act 单 pass + int16-out(convhbh,我逆向过)vs w8a16/w16a16(int16 操作数)**:静态 probe 说要 int16
  操作数精度(int8-static oc 0.0545 太松)→ 倾向 int16 操作数;但 convhbh 单 pass int8-act 更便宜。
  需用真实 GDN 数据在 sim 比 oc + cyc 拍板(`prepare_owned_inputs.py --family w8a16/w16a16` 已 bit-exact,可直接对照)。
- **全局静态 `s` 的选取**:让 `int32→int16` 移位既不溢出又保精度(probe 已验 int32 不溢出);确认移位是否需 round(加 `1<<(k-1)`)。

## 7. 铁律(沿用)

验 HMX kernel 一律 hexagon-sim(零 SSR);HMX 绝不 thread;度量只信整 solve wall/单线程隔离/ablation;
改 .inc 后 `verify_hexagon_inline_asm.py` 复验;**做 HMX kernel 前先用现成手写实现**
(`example/qnn_hmx_matmul_*` + `run_handwritten_artifact_body_sim.py`,u8i8/w8a16/w16a16 全已 bit-exact),别从零逆向。
