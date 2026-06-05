# GDN solve — 下一个 Agent 交接（2026-06-06）

接手前先读：本文件 + `Agent/current/gdn_solve_hvxmixhmx.md`（权威设计/命名/铁律）。
目标不变：把 GDNSolveHVXMixHMX 重写成「静态对称 int16」版（merge 用 HMX + int16 输出）。

---

## 0. 一句话现状

**make-or-break 已实测拍板 = GO**：静态 int16 HMX-merge 的 matmul 比现 vrmpy merge 便宜 ~18×、精度比出货 int8 好
~60×、kernel 在 hexagon-sim 跑通。**当前唯一缺口 = 把它做 bit-exact 并集成**（纯 sim 可迭代，无 SSR 风险）。
顺带已交付一个 bit-exact、已设默认的 **~13% 多线程实测提速**（T 输出经 VTCM+DMA 写回）。

---

## 1. 已完成 + 已提交（不要重做）

设备 = `ssh oneplus`（用 `scripts/dssh.sh`，`pkill -9 gdnbm` between runs，线程 ≤4）。

| 项 | 结论（实测）| 状态 |
|---|---|---|
| 前代回代（对角块）| scalar-load-bound（6402=2016乘法下限的3.2×），但 4 线程 SMT 完全隐藏（cap 到 1/16 工作量 4 线程 solve 没变快）→ asm prefetch **~0% 真实收益,不做**。已是静态 int16 形态 | commit |
| 多-HVX roofline | P=1/2/3/4 = 414/214/157/140K → P=4 **74%**;损失在 DDR-写争用（DIAG 部分只扩 57%）| commit |
| **T 经 VTCM+DMA 写回（默认开）**| P=4 **140K→122K（~13%）**,扩展 74%→85%,**bit-exact**;旧路径 escape `-DGDN_BR_T_DDR_DIRECT` | commit |
| zero-fill 只写上三角块 | bit-exact,QNN oc 门过 | commit |
| **convhbh int16-out kernel sim 跑通（GO）**| **462 cyc ≈ u8i8 417（+11%）≪ vrmpy 8325（~18×）**| commit `1b74870` |
| 精度（设计层）| 静态 int16 oc **1.9e-4**,int32 累加器 **0 溢出**（不需 int64）,比出货 int8 好 ~60× | sim 已验 |

**度量铁律（血泪）**：① 只信整 solve wall（`stats[0]`）/ ablation（`-D<flag>` 消阶段测 wall 差）。② GLUE_BENCH 单阶段
REPS-loop 对带 DDR 写的阶段是**假象**（冷 DDR 写放大 ~18×）。③ const-input 循环被 -O2 hoist（要 CSE-proof:rep 间数据依赖）。
④ 验 HMX kernel 用 **hexagon-sim**（`gdn_hmx_matmul_sim.build_and_run`）= **零 SSR**,别上设备试错描述符（错→cDSP fault 需 reboot）。

---

## 2. 用的 kernel（已 byte-verified,别换）

**`hm_w8a16_v75deep_kernel`** = QNN 原生 `hmx_v75_convhbh1x1deep_stride1` @ `libQnnHtpV75Skel.so:0x2f5200`（1348B）的**逐字节复制品**。
- 文件:`example/handwritten_hmx_matmul/kernels/w8a16/v73deep_conv1x1_kernel.inc`（文件名 v73 是历史命名,字节是 v75 convhbh;
  另一份相同的在 `example/qnn_hmx_matmul_w8a16/src/`）。验证:`verify_hexagon_inline_asm.py --vma 0x2f5200 --size 1348`。
- 语义:`activation.ub`(u8) × `weight.b`(i8) → **`cvt.uh=acc(r25):2x2`(u16) + `mxmem=cvt`(密 drain)**。
- 与现 u8i8 kernel 唯一差异 = 输出 drain（int16 :2x2 4-tile/cvt vs int8 :cm 1-tile）。

---

## 3. 剩余工作（分步,纯 sim 起步,逐步上设备）

入口脚本:`scripts/gdn_hmx_convhbh_sim.py`（我已写好,在 hexagon-sim 跑 convhbh 64³,有 cyc/PASS）。
参考 u8i8 版:`scripts/gdn_hmx_matmul_sim.py`（bit-exact 4096/4096,可对照学正确的 pack/descriptor/depack）。

**第 1 步（当前卡点,纯 sim,先做这个）—— 让 convhbh 64³ bit-exact:**
- 现状:`gdn_hmx_convhbh_sim.py` 跑通但输出只 drain 1/4、且 depack 没对齐（输出成对 `o[2k]==o[2k+1]` = `:2x2` 签名）。
- **根因（已定位）**:convhbh 读 **plain `.ub`(u8 单 pass)**,我却用了 u16 的 `pack_a16_crouton16_row4_surface`。
  → 改用 **u8 act 的 plain 32×32 tile 布局**（试 `pack_a16_row32_tile_surface` 喂 u8,或对照 u8i8 的 crouton8 但去 :cm）。
- **还要敲定**:act 表是 2 项(同 u8i8)还是 32 项(w8a16 row4);`cvt.uh:2x2` row4-dense u16 输出的 depack→natural。
- 已解码的描述符(64³,M_t=N_t=K_t=2):mask `conv1x1_words(0x70b,0,0,0,0x20)`、extra `{1,1025,524}`、
  out_desc{table_stride=2,y_stride=2,n_tiles_pow2=8,m_total_minus_step=8,k_total_bytes=64}、act_desc{n_pairs=2,y_stride=2}。
  **注意**:w8a16 op `HmxU16I8ToU16MatMulOp.cpp` L1190-1320 是这些值的来源,但可能要按 u8 单 pass 调。
- packers 在 `example/handwritten_hmx_matmul/prepare_owned_inputs.py`:`pack_w8_kmajor`(权重)、
  `pack_native_a16_bias(8,w)`(bias=mxmem2 格式)、`pack_a16_*`(act 各布局)。
- **gate**:sim 输出 depack 后对 numpy ref `P[m,n]=Σ(act_u8-128)*w[k,n]` bit-exact（像 u8i8 版 max_abs_diff=0）。
- 迭代时 cyc 也会变准（当前 462 是 partial-drain;全输出后才是最终 int16 matmul cyc,预计 ~500–560）。

**第 2 步 —— baremetal 接 kernel + 单线程设备验证:**
- 把 convhbh kernel 加进 baremetal（`gdnbm_imp.cpp` 已 include `GdnSolveBROp.cpp` → 加第二个 naked 函数含 w8a16 .inc;
  标签 `L_hmx_w8a16_*` 与现 u8i8 不冲突）。
- 加 `GDNBM_MM_I16_TEST`（仿现有 `GDNBM_MM_TEST`）:确定性 int8 算子 → convhbh → u16 depack → 对 `gdn_matmul_i16` 验 + 测 cyc。
  **先 nthreads=1 小心跑（SSR 风险）,描述符确认对再多线程。**

**第 3 步 —— FEED_4P 集成（阶段3 微基准）:**
- 现 FEED_4P 单 pass = 509 cyc/matmul（u8i8,1 HMX run）。换成 convhbh:静态 int8 算子（预量化一次）+ 静态 int16 输出（`:2x2`）
  + 纯 int32 加（去 per-term rescale）+ 去 multipass。测 matmul-portion,apples-to-apples vs 基线（同线程,C=256）。
- 复现 FEED_4P:`cd example/gdn_native/baremetal; EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_FEED_PIPE -DGDNBM_FEED_4P -DGDNBM_FUSED_ACTWT" bash build.sh`。

**第 4 步 —— 整 solve 重构 producer-consumer:**
- 4 HVX worker 做 diag + pack 算子;1 main 线程纯 HMX 排空所有 64³ matmul（HMX 绝不 thread→SSR）。
- ⚠️ **不能跑现有 `-DGDNBM_HMX_MERGE_PATH`**（在 open 即 SSR,已 #error 禁）→ 是**据微基准重写**,不是修旧全 solve。
- 整 solve wall（stats[0])vs 基线 GDNSolveHVX（122K,默认含 T-VTCM-DMA）。

---

## 4. UDMA 坑（做 T 写回/DMA 时记住）

**两个并发 `Q6_dmstart_A` 会互相 clobber**（device 实证损坏 heads）→ 第二个 dmstart 冲掉在飞的第一个。
现 T-VTCM-DMA（`gdnbm_imp.cpp` solve_worker）的做法:T 写回在 A 预取的 `udma_wait` 之后才发,不并发。
完整 T-overlap（~119K,再 ~2%）需 UDMA 链/多流,卡在 `dmwait` 对链的完成语义,暂搁,不值得为 2% 冒险。

---

## 5. 关键数字速查

- 现基线全 solve（默认含 T-VTCM-DMA,4 线程,C=256,H=32）= **~122K cyc/head**;旧（DDR 直写）~140K。
- 64³ matmul:vrmpy(HVX) 8325 / u8i8-out HMX sim 417 / **convhbh int16-out HMX sim 462** / HMX-fed FEED_4P 509。
- 前代回代 floor 2016（实测 6402,SMT 隐藏,不优化）。
- 静态 int16 精度 oc **1.9e-4**,0 溢出（`scripts/gdn_solve_static_quant_probe.py`）。
- merge 占 solve **83%**（单线程 ablation:full 405912 / DIAG_ONLY 70671）。
