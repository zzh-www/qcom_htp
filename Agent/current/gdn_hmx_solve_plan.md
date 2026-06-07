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

### 步骤 1 ★ A 驻 VTCM(最大,单线程 + 并行都受益)
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

### 步骤 4 ★ producer-consumer 并行(HVXMixHMX 本体)
- 单线程 solve 优化到位后,重构成 **4 HVX producer + 1 main HMX consumer**(consumer 在 main 单 HMX,不 SSR;同 FEED_4P 结构)。
- producer 做 diag+fold+quant+pack+acc(并行),consumer 纯排空 HMX matmul。
- **A-VTCM(步骤1)对并行尤其重要**:producer 侧(diag/fold)是 producer-bound 的主体,A-bound 会拖慢 producer。
- **预期**(待步骤 1-3 把单线程压下来后重算):producer 侧 / 4 vs 4 线程 HVX 基线 88K。

---

## 3. 铁律 / 风险
- 验 HMX kernel 一律 hexagon-sim(零 SSR);**HMX solve 单线程**(多 HMX 线程 SSR)。
- 度量:warmup + reps2-4 稳态,绝不 min;`pkill -9 gdnbm` between runs;设备 `ssh oneplus`(`source scripts/dssh.sh; dssh_open`)。
- UDMA 坑:两个并发 `Q6_dmstart_A` 互相 clobber(A 预取 wait 后再发 T 写回)。
- 标定常量(GDN_OPS_*)是 A_u16_h32 专用硬编码;通用化需传入。
