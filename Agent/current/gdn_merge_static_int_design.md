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
        S(int16) = saturate_u16( zp + round_half_up( acc_int32 * 2^k ) )
                                  ▲ k = 静态标度指数 (control word word0 指数位[14:10])
                                  ▲ round_half_up: 下移 s 位 = (acc + 2^(s-1)) >> s

STEP B  外层  T_ij = T_ii · S = 再一个 HMX matmul
        T_ii(i16) · S(i16) → acc(i32) → drain(round>>k) → T_ij(int16)

每块共 2 次 drain (内层 Σ 后 1 次, 外层后 1 次). 全程 int, 每次缩窄 = 一个移位.
```

**为什么能融合累加**:静态量化 ⇒ 所有块同标度 ⇒ int32 积**直接可加** ⇒ 内层 Σ_k 写成
`[A_ij|…|A_i,i-1] @ [T_jj;…;T_i-1,j]`(K=(i−j)·64)一个大 K matmul,HMX 的 K-loop 天生沿 K 累加 = 求和。
动态 per-block 标度则每 term 要 fixed-point rescale → 就是 51% glue;静态把它清零。

**★cvt drain 真相(待办1 sim 实测,纠正早期"纯移位零舍入"叙事)**:
- **增益 = `2^(exp-16)`**,`exp` = word0 位[14:10](`word0 = 0x0040 | (exp<<10)`,mantissa 位固定 0x040)。
  实测 k∈[-5,+2] 增益逐档 ×2/÷2 干净(`0x4040`=×1、`0x4440`=×2、`0x3c40`=×½…)。**可任意设 k(含下移)。**
- **舍入 = round-half-up(四舍五入,非截断!)**:`word2=0x0000` 下 `int32→int16 = (acc + 2^(s-1)) >> s`,
  小幅度 sim 实测 **21/21 命中 round-half-up**(早期"word2=0 零舍入/纯移位"结论 **错**)。`word2=0x0008` 额外加偏置,别用。
- **✅ f16 精度坑已定性(待办2 消歧探针,sim):drain f16 化的是 NET 累加器(加完 bias 之后),不是 raw。**
  实测 net=1000 时 raw_acc 从 2280 扫到 129000(56×)、bias −1280→−128000 → **全精确**;net=30000 → 恒 +1(与 raw 无关)。
  → **bias 在 full int32 精度里加完才 f16 化**,零点偏置/大 raw 积**不伤**。f16 对 **整数 ≤2048 精确**,之上 ≈`2^-11` 相对误差。
  **对融合累加是好消息**:大 K 的 int32 求和精确,唯一损失 = 末次 drain 的 net→f16(典型 int16 输出幅度下 ≤~1 LSB,
  `|out|≤2048` 时 0 误差)。选 k 使排空 int16 不要太大即可;**不需为此用 zp=0**(zp=0 仍利于省 int16 量程)。
  脚本 `scripts/gdn_convhbh_drain_acc_probe.py`。
- 含义:merge drain 非"位精确纯移位",而是 **round-half-up + 大累加器 ≤2 LSB f16 噪声**。待办2 的门是
  **oc/max_abs ≤ 阈值**(参考用 round-half-up 整数模型),不是对截断整数参考 bit-exact。

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

1. **✅【已验】钉 control word 的"指数位=移位量 k"**(2026-06-06,sim,uniform-P 探针 零 depack):
   - 增益 = `2^(exp-16)`,exp = word0 位[14:10],逐档 ×2/÷2 干净(k∈[-5,+2] 实测,见 §2 cvt drain 真相)。
   - 舍入 = **round-half-up**(`word2=0x0000`,21/21),非截断;f16 路径在 raw_acc>~2^12 有 ≤2 LSB 噪声。
   - 脚本 `scripts/gdn_convhbh_control_word_sweep.py`(指数扫描)+ `scripts/gdn_convhbh_drain_round_probe.py`(舍入规则)。
2. **✅【已验,端到端 oc】融合累加 SOUND**(2026-06-06,真实 golden,`scripts/gdn_solve_fused_drain_probe.py`):
   - drain f16 化的是 **NET 累加器**(加完 bias 后),非 raw → 零点偏置/大 raw 积不伤(`gdn_convhbh_drain_acc_probe.py`)。
   - 端到端 oc:**w16a16 = 2.6e-4**(融合累加 + 真实 drain + power-of-2 requant),**int32 0 溢出**,f16 drain 可忽略。
   - → 见 §6 决策:**操作数必须双 int16(w16a16),int8 任一侧都不够**。
3. **【设计→验】布局直通**:把 STEP A 的输出(crouton)直接当 STEP B 的 activation,验证零重排正确;
   weight 端 k-major 预打包常驻。**(w16a16 输出仍 crouton16_row4,与 convhbh 同 → 直通契约不变。)**
4. **接 baremetal**:`gdnbm_imp.cpp` 加 **`hm_w16a16_v73_kernel`**(非 convhbh!) + `GDNBM_MM_I16_TEST`(nthreads=1 小心,
   SSR 风险),对 `gdn_matmul_i16` 验 + **测 w16a16 cyc**(决定是否仍 < vrmpy);再整块 producer-consumer,整 solve wall vs 基线(~122K)。

## 6. 设计决策

- **✅【已拍板,真实数据】操作数精度 = w16a16(int16×int16),不是 convhbh(w8a16)。**
  端到端 oc(真实 golden,融合累加 + 真实 drain 模型,`scripts/gdn_solve_fused_drain_probe.py`):
  | 变体 | oc(mean) | overflow | 结论 |
  |---|---|---|---|
  | **w16a16(act16×wt16)** | **2.6e-4** | 0 | ✅ 比出货 int8(1.2e-2)好 ~46× |
  | w8a16(act16×wt8) | 1.38e-2 | 0 | ❌ 比出货 int8 还**略差**(int8 权重不够) |
  | w8a16′(act8×wt16) | 3.8e-2 | 0 | ❌ |
  | int8(act8×wt8) | 4.1e-2 | 0 | ❌ |
  - **int8 操作数(任一侧)都不够**,且 **per-block-distance 分层静态标度也救不了**(w8a16 0.0138→0.0125、int8 0.041→0.040)
    —— 限制是**操作数位宽**(int8 127 级装不下 T 块内动态范围),非标度策略。**只有双 int16 达标。**
  - **★融合累加 ⟹ 必须全局标度**:一个内层 Σ_k 的 T_kj 跨不同 block-dist(0…d-1),分层标度会使各 term 积不可直接相加 →
    强制 per-term rescale(=要消的 51% glue)→ **分层与融合不兼容**。全局 int16 既达标又可融合 = 唯一解。
  - **f16 drain 可忽略**:w16a16 exact 2.2e-4 → f16 2.6e-4(+0.4e-4),其余变体 f16 列与 exact 几乎相同 → **drain 不是误差来源**。
  - **int32 累加器真实数据 0 溢出**(满量程 int16×int16 over K,maxcode~32767)→ **融合累加 int32 安全,不需 int64**。
  - **kernel 改用 `hm_w16a16_v73_kernel`**(已 byte-verified,`run_handwritten_artifact_body_sim.py --family w16a16` bit-exact)。
    输出 drain 仍是 `cvt.uh:2x2`(待办1/2 的 drain 结论照搬)。
  - **代价待测**:w16a16 比 convhbh 贵(int16 权重 2× 字节 + int16 激活 2-pass)。阶段3 的 509 cyc/64³(3.4×)是 convhbh 测的;
    **w16a16 需重测 cyc**——只要仍 < vrmpy merge(2081/64³ 4-路)就值;预计 ~1000–1300,仍 ~1.5–2× 便宜 + 46× 精度。
- **全局静态 `s`**:operand 标度是预处理(任意 float,off-device);只有 **drain gain 受 HW 限为 2^k**(power-of-2)。
  每次 drain 静态选 k 填满 int16 量程;power-of-2 snap 的 ≤1-bit 精度损失对 oc 可忽略(2.2e-4 已含)。round-half-up 自动(word2=0)。

## 7. 铁律(沿用)

验 HMX kernel 一律 hexagon-sim(零 SSR);HMX 绝不 thread;度量只信整 solve wall/单线程隔离/ablation;
改 .inc 后 `verify_hexagon_inline_asm.py` 复验;**做 HMX kernel 前先用现成手写实现**
(`example/qnn_hmx_matmul_*` + `run_handwritten_artifact_body_sim.py`,u8i8/w8a16/w16a16 全已 bit-exact),别从零逆向。
