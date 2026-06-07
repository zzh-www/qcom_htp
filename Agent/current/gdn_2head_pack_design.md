# GDN 2-head 打包重构 — ❌ REFUTED(2026-06-08)

承接:`gdn_hmx_solve_plan.md`。当前最佳 = PURE-HMX PIPE P=4 **~2.50M**(32-head total wall,1.59× vs HVX)。
fine timeline 定位瓶颈 = **PREP(fold+quant+pack 喂 HMX)占 producer ~41%**。

## 试过什么
2-head 锁步 solve(`gdn_br_pair` + 融合 packer `gdn_pack_act_crouton8_2`/`gdn_pack_w8_kmajor_2`,
2 scratch g_scr/g_scrB,2 vt,0x100000/producer,顺序 2 dispatch)。`-DGDNBM_PIPE_2HEAD`(已 revert)。

## 结果:❌ 更慢 + 有 bug
- **更慢**:total wall 2.50M → ~3.5M。timeline 看穿:**PREP aggregate 6.34M(单 head 版 3.86M)—— 融合把 pack 搞慢了**,
  且每 DIAG 后有 DMA 空隙(2 head 的 A 用阻塞 udma_wait,无 ping-pong)。
- **不 bit-exact**(maxdiff 39178,correctness 也错)。

## 根因(关键教训,别再试同型融合)
我融合的是**同型流**:2 head 的 act-crouton 一起(都走 ALU)、2 head 的 wt-kmajor 一起(都走 permute)。
**同管道并发无 co-issue 收益,只增寄存器压力 → 更慢。**
memory [[project_gdn_solve_nhead_hmx_feed]] 的 1.30× 来自 **act-crouton(ALU)× wt-kmajor(permute) 异管道 co-issue**
(microbench `fp_pack_actwt2`),不是同型 2-head。要拿那个收益得把 **act+wt 一起打包**,但 act/wt 在 solve 里
**分开 cache(不同 key/生命周期)** → 无法直接融合。→ 此路在当前 cache 结构下不通。

## 结论
- **保持 PURE-HMX P=4 = 2.50M 为最佳**(default,已 commit,bit-exact)。
- 若将来还想攻 PREP:得先重构 operand cache 让 act+wt 同时可得,再做异管道 fused pack(act-crouton∥wt-kmajor)。
  风险/收益不明,非当前优先。
- 度量/铁律同 `gdn_hmx_solve_plan.md`:32-head total wall + 每步重画 timeline。
