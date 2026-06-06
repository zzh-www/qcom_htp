# GDN merge —「全程静态 int + 融合累加 + 布局直通」设计基线 (Step 2)

> ## ⛔ 最终结论(2026-06-06,实测推翻,READ FIRST)
> **HMX-merge 与静态-HVX-merge 在 GDN 所需的 int16 精度上都不提供有意义的加速。merge 已近"精度×吞吐地板"。**
> 实测推翻链(每步 committed,脚本在 `scripts/gdn_convhbh_*`、`gdn_solve_fused_drain_probe.py`):
> 1. **convhbh 是 8-bit/操作数 kernel**(`.ub` u8-act × `.b` i8-wt;单调用只用激活高字节,实测 out≈floor(P/256))。
>    → int16×int16 = **4 byte-pass ≈ 4×509 ≈ vrmpy 2081**,matmul 无提速。原"18× 便宜"是 int8-HMX vs int16-vrmpy 不等精度比较。
> 2. **静态-HVX 也不赢**:要"纯加 acc"(消 acc-rescale 36%)必须 int16 操作数(int8-static oc 0.0545 太松);
>    但 **HVX vrmpy int16 ≈ 4× int8**(`GdnSolveBROp.cpp:1164`)→ matmul 4× 吃掉 glue 省的 → 静态 int16 ≈ 现 int16(~125K),非赢。
> 3. **根因同一**:这颗硅 int16 精度结构性贵(HMX 只 int8 需 4× 字节分解;HVX vrmpy 只 int8 满速)。
> 4. **三角跳零**(roofline 点的"真杠杆")实查只 **~6% solve**(内层 Σ_k 仅 k=j 的 T_jj 块三角,6/10 term;k>j 块 dense)
>    且 vrmpy 内跳三角难实现 → 非银弹。
> - **现最优 = int8-dynamic + VEC_MM + PREQUANT_A = 89,880 cyc/head(已实测),已近最优。** w16a16 hand-written kernel 另外 UNSOLVED。
> - **可做但价值有限的剩余**:三角跳零(~6%)、扩展性微调(SMT 争用,难)。**别再投 HMX/静态-int 大改。**
> - 下方原"设计基线"是推翻前的内容,保留作过程记录(§6 决策表含逐步纠错)。

承接第1步(convhbh int16 matmul 已 sim bit-exact,见 `gdn_solve_NEXT_AGENT.md` 第1步)。
本文是 Step 2 (把 convhbh int16 接进 GDN merge) 的设计基线,**已被上方最终结论推翻**(过程记录见下)。

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
3. **✅【已验】布局直通**(2026-06-06,`scripts/gdn_merge_layout_directpipe_probe.py`,纯索引证明):
   C-harness `deblock_a16_crouton16_row4`(已 byte-verified 解 KERNEL 输出)== `pack_a16_crouton16_row4_surface`
   (激活打包)的**精确逆**——64³/256³/64×128/128×64/64×192 全 True。→ **kernel 的 int16 输出 surface 与激活 surface
   逐字节同格式 → STEP A 输出当 STEP B 激活零重排**(w16a16 是 int16→int16,2-pass lo/hi 自然读),只 weight 端需 k-major。
4. **接 baremetal**(唯一剩余,device + SSR 风险):
   - **★kernel = 2-pass convhbh(SOLVED),不是 w16a16(UNSOLVED)** —— 见 §6 修正。convhbh(w8a16)hand-written
     byte-exact;w16a16 hand-written 描述符未解(body-sim 18803/65536)。**用 convhbh 跑 2 遍(权重 hi/lo int8 拆)= int16 权重。**
     融合累加正确性已被 256³ convhbh body-sim 预验(K=256 单 drain bit-exact,merge K≤192 是子集)。
   - **步骤**:① `gdnbm_imp.cpp` 已有 `our_v73deep_kernel`(=convhbh,byte-verified)——直接复用,无需新 kernel。
     ② 写 `GDNBM_MM_I16_TEST`(仿 `GDNBM_MM_TEST` @ L354):确定性 int16 act/wt → 权重拆 hi/lo int8(`hi=(w+128)>>8, lo=w-hi*256`,
     qmax=127*256=32512)→ crouton16 act pack + k-major hi/lo 两套 weight → convhbh 跑 2 遍累 int32(`P=256*P_hi+P_lo`)
     → `deblock_a16_crouton16_row4` depack → 对 `gdn_matmul_i16`(int16×int16)验 bit-exact + 测 cyc。**nthreads=1 小心 SSR。**
     ③ 据微基准把整 solve 重构为 producer-consumer(4 HVX pack/diag + 1 main HMX),整 solve wall vs 基线 ~122K。
   - **度量**:只信整 solve wall(stats[0])/ ablation;2-pass convhbh 估 ~2×509=~1018 cyc/64³ < vrmpy 2081(~2× 便宜)+ glue 51% 消除。

## 6. 设计决策

- **✅【已拍板,真实数据】精度需 int16×int16(2.6e-4);实现 = 2-pass convhbh(SOLVED kernel),非 w16a16(UNSOLVED kernel)。**
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
  - **★★ kernel 修正(2026-06-06,实测纠错):w16a16 hand-written kernel UNSOLVED,不能用。**
    `run_handwritten_artifact_body_sim.py --family w16a16` 实测 **checksum_mismatch,18803/65536 exact**(非 bit-exact);
    `Agent/handoffs/w16a16_native_alignment_plan.md` 记录大量失败尝试 —— **w16a16 的 standalone 描述符契约未解**
    (kernel 字节对 native disasm byte-verified,但驱动它的描述符未解;只有 QNN QHPI "accepted" 路径对,baremetal 用不了)。
    对比:**convhbh(w8a16)= `byte_exact_checksum,65536/65536` ✅ SOLVED**。我之前 commit 误称 w16a16 bit-exact(没实跑就引用)。
  - **★解法 = 2-pass convhbh 模拟 int16 权重(用 SOLVED kernel)**:int16 权重码 `cT = hi*256 + lo`,`hi=(cT+128)>>8`、
    `lo=cT-hi*256`,二者 int8(权重量化 qmax=`127*256=32512`,保 hi≤127;0.8% 量程缩,可忽略)。convhbh 跑 2 遍
    (int16-act × int8-hi-weight、× int8-lo-weight)→ `P=256*P_hi+P_lo`(int32 精确重构,实测 `256*hi+lo==cT` True)。
    **= int16-act × int16-weight 精确,oc=2.6e-4(同 w16a16),但跑在 byte-exact 的 convhbh 上。** 融合累加:每 pass 沿 K 累 int32,
    2 个 fused matmul 后 combine `256*S_hi+S_lo`(每块 1 次 shift+add,小;非 per-term glue)。
  - **f16 drain 可忽略**;**int32 0 溢出**(融合累加安全,不需 int64)。
  - **★★★ 代价实测纠错(2026-06-06,`scripts/gdn_convhbh_i16_block_sim.py` 决定性):convhbh 是 8-bit/操作数 kernel
    (`.ub` u8-act × `.b` i8-wt),单次调用只用激活的高字节(实测 out≈floor(P/256))。**
    → **int16-act 需 2 pass(hi/lo)、int16-wt 需 2 pass → int16×int16 = 4 byte-pass ≈ 4×509 ≈ ~2036 cyc/64³ ≈ vrmpy 2081。**
    **matmul 的速度优势在所需的 int16 精度上消失**(HVX vrmpy 原生支持 int16;HMX 只 int8,int16 要 4× 字节分解)。
    原"convhbh 18× 便宜于 vrmpy"是 **int8-HMX vs int16-vrmpy 的不等精度比较**。
  - **★精度×速度不可兼得**:1-pass int8(509,oc 0.041)/ 2-pass act16×wt8(~1018,oc 0.0138≈出货)/ 4-pass int16²(~2036≈vrmpy,oc 2.6e-4)。
    **要 matmul 提速(≤2 pass)就精度退化;要精度(4 pass)matmul ≈ vrmpy 无提速。**
  - **★GO 被削弱/重定向**:HMX-merge 在 int16 精度上 matmul **≈ vrmpy,无提速**。仅剩的杠杆 = **静态全局标度消 per-term rescale glue
    (51%)——但这不需要 HMX,可直接施于现有 vrmpy int16 merge(在 HVX 上变静态即可)**。→ **实际可行的赢 = 把现 vrmpy merge 改静态,
    不是换 HMX。** 待办4 的 HMX 集成价值存疑,需用户定夺(见报告)。
- **全局静态 `s`**:operand 标度是预处理(任意 float,off-device);只有 **drain gain 受 HW 限为 2^k**(power-of-2)。
  每次 drain 静态选 k 填满 int16 量程;power-of-2 snap 的 ≤1-bit 精度损失对 oc 可忽略(2.2e-4 已含)。round-half-up 自动(word2=0)。

## 7. 铁律(沿用)

验 HMX kernel 一律 hexagon-sim(零 SSR);HMX 绝不 thread;度量只信整 solve wall/单线程隔离/ablation;
改 .inc 后 `verify_hexagon_inline_asm.py` 复验;**做 HMX kernel 前先用现成手写实现**
(`example/qnn_hmx_matmul_*` + `run_handwritten_artifact_body_sim.py`,u8i8/w8a16/w16a16 全已 bit-exact),别从零逆向。
