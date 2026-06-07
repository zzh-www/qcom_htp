# GDN HMX-merge 全 solve — 后续计划(2026-06-08,审计后重写)

承接:`GDNBM_HMX_SOLVE` 单线程 HMX-merge 全 solve 已跑通 + 全静态量化栈已落地。
权威现状/命名/铁律:`gdn_solve_hvxmixhmx.md`、`gdn_solve_NEXT_AGENT.md`。

---

## 0. 现状(已 commit,working)

- **`GDNBM_HMX_SOLVE`**(`gdnbm_imp.cpp`):单线程、自取 0x60000 VTCM、HAP 锁 HMX、跑完整 solve 的 HMX merge。
  - load-SSR 根因已修(`qurt_hmx_lock` 未解析 → 改 HAP `compute_res_hmx_lock`)。
- **静态量化栈**(`-DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL`):单 pass 增益 + 固定操作数标度(去 maxabs/colabs/mx_out)
  + **A 的 fold→quant 融合**(`gdn_fold_quant_u8`,一遍 u16→u8 去 int32 往返)+ acc 纯加 + off-diag T drain@sTw → 权重纯 narrow。
- **精度**:off-diag oc ~8e-2(单一全局 int8 静态标度,精度先不管)。

### 度量方法(血泪,务必照做)
- **warmup 一次(丢弃)+ 一个 FastRPC session 跑 `GDNBM_REPS=8`,取 reps 2-4 稳态**(rep1 冷;rep≥5 设备节流变慢)。
- **绝不取 min**(min 抓快异常值,系统性高估——之前"310K/32%/86K"都是 min 假象,真实稳态 ~396K)。
- 干净的逐步实测(median reps 2-4,两 pass):**multipass ~421K → static-gain ~412K(−2%)→ static-full ~396K(−6%)**。

---

## 1. 审计结论:static-full **还不是**"全静态/全 VTCM/无可 skip"

| # | 问题 | 位置 | 性质 | 预期收益 |
|---|---|---|---|---|
| **A** | **A 从 uncached DDR 读,没驻 VTCM** | `gdnbm_imp.cpp:256` | **不是全 VTCM** | **大头**(diag 32%+fold 10% 是 A-bound;route doc:uncached DDR 让 diag 慢 7.8×) |
| B | requant 每块跑动态 maxabs 算 psh | `GdnSolveBROp.cpp:501` (`gdn_requant_block_out`) | 不是全静态(可纯删) | 小(~10块/head;静态下码 <2^15 → psh 恒 0) |
| C | Sacc 内层和的 quant 动态(mx=-1) | `GdnSolveBROp.cpp:1652` | 不是全静态 | 小(~6/head;静态化有精度代价) |

---

## 2. 计划(按收益排序;每步 warmup+reps2-4 干净测)

### 步骤 1 ★ A 驻 VTCM(最大,单线程 + 并行都受益)— ✅ 已做,实测 ~1.93×
- **实测结果**(2026-06-08,warmup+REPS=8 稳态 reps2-4):
  - DDR 读 A:**~371K cyc/head** → A 驻 VTCM ping-pong:**~192K cyc/head**(**1.93×**)。
  - 精度:输出与 DDR 路**逐字节相同**(maxdiff=0,off-diag relerr 1.093495e-01 一字不差)→ 纯数据搬运,零精度影响。
  - 验证:`-DGDNBM_HMX_A_DDR` 逃逸路径对照,bit-exact 确认。
  - VTCM 预算:acquire 0xA0000(vt surfaces<0x59000 @ +0 + A ping-pong 2×0x20000 @ +0x60000/+0x80000),设备容纳无误。
- **为什么最大**:diag(fold A_ii + 前向求逆)+ merge fold(A_ik)都在读/折叠 A;A 是 uncached DDR → 标量/向量读都慢。
  route doc 实测 A-resident 是量级提升(不是 ~6%)。**我之前的 HMX_SOLVE 偷懒从 DDR 读 A,这是真正被忽略的大杠杆。**
- **做法**:仿现成 `GDNBM_VTCM_RESIDENT` 路径(`gdnbm_imp.cpp:275+`)——每 head 的 A(256×256 u16=128KB)DMA DDR→VTCM,ping-pong 让 head h+1 的 A 在 head h 计算时载入。
- **VTCM 预算坑**:HMX_SOLVE 已占 0x60000(vt surfaces+cache)。A ping-pong 要 +0x40000(2×128KB)。
  确认总量 ≤ VTCM 容量(`HAP_compute_res_query_VTCM`);不够则缩 cache 或单缓冲。
- **gate**:relerr 不变(纯搬运)+ 稳态 cyc 降。**先单线程测,确认 diag/fold 大降。**

### 步骤 2 requant psh maxabs 删除(纯静态,免费,无精度损失)
- `gdn_requant_block_out` 在 STATIC_FULL 下 `psh=0`(码 <2^15),跳过 `gdn_maxabs_codes`。
- **gate**:relerr 完全不变(psh 本就是 0)+ 稳态 cyc 微降。

### 步骤 3 Sacc quant 静态化(精度先不管)
- 标定 Sacc 的固定标度(像 A/T 那样),`gdn_quant_i8_from_codes` 走固定标度(去 maxabs)。
- ⚠️ Sacc 幅度随内层和(d 项)变 → 固定标度会 underflow,relerr 会涨。先测 relerr 可接受再留。

### 步骤 4 ★ producer-consumer 并行(HVXMixHMX 本体)— ✅ 已搭起来 + 实测
- **已落地**(`-DGDNBM_HMX_PIPE`,2026-06-08):4 HVX producer(各跑整 per-head solve,diag/fold/quant/pack/acc/widen/requant 全 HVX)
  + 1 main-thread PURE-HMX consumer(单 HMX 不 SSR)。同步 hand-off:每 producer 一个 job slot,填好→自旋等 consumer 排空→depack。
  - hook 机制:`gdn_merge_packed` 单 pass HMX kernel 调用改走运行时 `g_hmx_dispatch`(null=本地单线程;设了=委托 consumer);
    `gdn_pipe_dispatch`(slot=`sc-g_scr`)填槽+自旋。A 每 producer 各自 VTCM 驻留 ping-pong(0xA0000/producer,VTCM 总 8MB 够)。
  - **坑(已修)**:`g_ops_u8/g_ops_i8/g_force_sP` 是 solve 内随阶段切值的 mutable 全局 → 多 producer 互相 clobber(relerr 1.09e-1→1.84e-1,bytes 差)。改 `__thread` 后**与单线程逐字节相同**(maxdiff=0)。
- **实测(指标统一为 32-head TOTAL wall,warmup+REPS=8 稳态 reps2-4,交替 A/B 控热)**:
  | 配置 | 总 wall(32 head) | vs HVX |
  |---|---|---|
  | 单线程 HMX,A-DDR | ~11.9M | 0.30× |
  | 单线程 HMX,A-VTCM(步骤1) | ~6.15M | 0.64× |
  | **PIPE 4-prod(步骤4)** | **~3.55M** | **1.10× ✓** |
  | GDNSolveHVX 4-thread 基线 | ~3.90M | 1.00× |
- **结论(换 total-wall 后翻转)**:**PIPE HVXMixHMX 已经打过纯 HVX 基线 ~1.09–1.13×**(还只是朴素同步 hand-off scaffold)。
  ⚠️ 之前 per-head 框架说"慢 1.26×"是**被 88K/head 假基线误导**——那是 tiler 低估 artifact(真实 HVX ~128–146K/head=3.9–4.7M total)。**只信 32-head total wall。**
- **剩余杠杆(把 3.55M 继续往下压)**:
  1. consumer 仍在 `gdn_hmx_run_only` 里做 bias-pack + out-zero(HVX 活),拖慢 consumer 吞吐 → 仿 FEED_4P 把 bias-pack 挪到 producer,consumer 纯 mxmem。
  2. 同步 hand-off → producer 等自己 matmul 时空转;可用 head-interleave 或多槽 ring 把延迟藏起来。
  3. consumer 主线程跑 HVX intrinsic(bias-pack)却没锁 HVX → 跟 producer 抢 HVX/SMT issue slot。

---

## 3. 铁律 / 风险
- 验 HMX kernel 一律 hexagon-sim(零 SSR);**HMX solve 单线程**(多 HMX 线程 SSR)。
- 度量:warmup + reps2-4 稳态,绝不 min;`pkill -9 gdnbm` between runs;设备 `ssh oneplus`(`source scripts/dssh.sh; dssh_open`)。
- UDMA 坑:两个并发 `Q6_dmstart_A` 互相 clobber(A 预取 wait 后再发 T 写回)。
- 标定常量(GDN_OPS_*)是 A_u16_h32 专用硬编码;通用化需传入。
