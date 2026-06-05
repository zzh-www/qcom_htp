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

**第 1 步（进行中 2026-06-06,纯 sim）—— 让 convhbh 64³ bit-exact:**
入口 `scripts/gdn_hmx_convhbh_sim.py` 已大改写,带探针模式(`--mode uni_coln|uni_p<N>|id_coln|
id_rowm|id_grid|random`)+ 隔离 out-slot 发现法。**已实测确立的事实(全部 sim,零 SSR):**

1. **输出 = 4 个 32×32 tile(m32×nt),不是 32 个 row4-tile。** 之前"只 drain 1/4"是 **out_off 重叠假象**
   (32 个 4×32 tile 放 64B 间距互相覆盖)。给每个 out-table 项**隔离 2KB slot**(`out_off=[i*2048]`)→ 全 4096
   u16 写满。kernel 实际只消耗 out_table[0..3]。slot 配对:n 变(uni_coln)slot0==slot2、slot1==slot3 → **slot = nt + 2*m32**。
2. **激活读 Crouton16 surface 的 HIGH BYTE**(★关键突破)。QNN u16 路径跑 lo+hi 两 pass;**静态 u8 单 pass = hi pass**。
   `pack_a16_crouton16_row4_surface(act_u8.astype(u16) << 8)` → `id_coln` 从 uniform(垃圾)变 **64 distinct 全列** ✓。
   (低字节填 → 输出与激活无关的常数 -258;高字节填 → 正确跟列变化。)
3. **act-table 物理契约(op L1254-1264 `crouton_row4_physical_ptr`)**:`act_tbl[row4*K_t+kt]=block_table[(row4&7)*K_t+kt]`
   **无 `(row4>>3)*256` m32 偏移**——kernel 用 `m_total_minus_step` 内部加 m32 半偏移;row4 与 row4+8 共享同一物理指针。
   (我曾错加 256 偏移,已去掉。)block_base=(phase*K_t+kt)*512,crouton16 sub-block(phase,kt,m32)=256B。
4. **P→u16 仿射(增益≈3,zp=0x8000)**:`uni_p<N>` 扫出 P=1→2,2→6,3→8,10→30,50→151,100→302
   (小 P 步进 +2/+4 交替=`6*⌊P/2⌋+2*(P&1)`)。增益由 bias const words `[0x4440,0x8040,0x0008,0x4000]`(`pack_native_a16_bias`)
   决定 → **GDN 静态版可改 const 设增益=1/任意标度**(待解码 const→gain 编码)。
5. **行别名 4:1 已解决 = `n_tiles_pow2` 太小**(不是 surface 配对)。op `M_t*4`(=8@64³)只迭代 16/64 行;
   **`M_t*16`(=32@64³)解全 64 行**(`id_rowm`=64,`id_coln`=64,`id_grid`=120 全 distinct,sweep 确认)。
   ⚠️ **`M_t*4` 是 op 默认且对真实 merge 尺寸(C=256,M_t=8→32)正确**;64³(M_t=2)撞小尺寸缩放边界。
6. **增益/标度**:effective bias = **-128*sum_w**(u8 zp;-32768 错,cvt 自己处理 <<8 的 ×256)。
   输出增益由 bias const **word0** 定:`0x4040`→×1、`0x4440`→×2。`out_u16=0x8000+round(P*gain)`,
   有 per-channel cvt 舍入(×1 时 ~±1 LSB)。**GDN 静态版按需设 word0 选标度。**
7. **OUTPUT depack(未完,但已定性)**:隔离-2KB-slot 的映射是**假象**(identity-weight 巧合;dense weight 下
   4 副本互不等)。真实输出 = **连续 out_raw**(op L1265:tile(row4,nt) 在 `(row4*4*N+nt*32)` u16),
   被 :2x2+crouton **置换**;64³ 下 32×32 storage tile 覆盖 4-行自然区→重叠,需真实尺寸(256)的 tiling
   或非重叠放置才能 depack。

**★净结论**:**激活布局(最难)+ 矩阵乘正确性已完全攻克并 sim 验证**(identity 探针证明 kernel 算对全 m+n)。
**剩余 = 真实尺寸下的 output depack + 精确 cvt 舍入**,建议直接在**真实 merge tile 尺寸**(非 64³)上做,绕开小尺寸缩放伪影。
- **gate(未达)**:`random` dense-weight matmul depack 后对 `P=Σ(act-128)*w` bit-exact —— 卡在 output depack。
- HMX kernel(mask 0x70b/extra{1,1025,524},eff=-128*sum_w)已 sim 跑通非 fault;identity 探针证明权重+激活打包对。
- 度量:当前 462 cyc(MAC 主导)→ 仍 ~15× < vrmpy 8325,GO 结论不变。
- 复现关键 config(`scripts/gdn_hmx_convhbh_sim.py`):act=`crouton16_row4(act_u8<<8)`、act_off 无 m32 偏移、
  `n_tiles_pow2=M_t*16`、eff=-128*sum_w、out_off 连续 `(row4*4*N+nt*32)*2`。探针:`--mode id_coln|id_rowm|id_grid`。

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
