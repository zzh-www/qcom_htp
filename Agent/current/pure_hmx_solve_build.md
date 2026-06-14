# Pure-HMX w16a16 三角求逆 — Loop Optimizer（入口文档）

> **ACTIVE LOOP**:`NEXT` 非空就直接做 `NEXT`,不下"已完成/闭环/收口/穷尽"结论。本文自洽自足,判本 loop 状态只看本文 STATE/NEXT。范围 = pure-HMX(`GDNSolveHMX`,全程 HMX matmul 三角求逆)。

## 口径（强制,违反 = 结论作废）

- **任何性能数走 skill `htp-cycle-metric`**;每数 = value + 口径 + context。四口径分清:**① graph-wall**(全程 span = 最终裁决)·**② op-latency**(`N_mm×256`,HMX 纯算力下限)·**③ unit-busy**(occupancy)·**④ per-call wall**(kernel-call,含字节 feed)。
- 全 PCYCLE(`C15:14`);**时钟自检 `wall/µs ≈ 1594`**(本设备 TURBO;≫此 = 读错计数器,作废重测)。
- **跨实现比 native 只用 graph-wall÷N(口径①)**;optrace per-op(②③)**绝不**对 baremetal per-call wall(④)比 —— 跨口径必造假差。native 锚:单 64³ ≈11,034 wall ≈ 我们;批 ~2,020 wall/matmul。
- **不把 per-call feed-inclusive(④)当 HMX 算力(②)**;`stats[5]` 含 ~10K mxmem feed 是正确 wall,不是 bug。

## OBJECTIVE

**最小化 32-head TOTAL wall(standalone metric),硬门 `oc < 4e-2`(每轮重测,绝对门)。**
- **本质目标(用户裁定 2026-06-15):wall → HMX 理论 MAC 峰值地板。** 即 wall 只剩"这组固定 matmul 在 HMX 上的纯 MAC 计算时间",其余一切——producer feed、握手/sync、**每 matmul 的 fill/drain**、DDR↔VTCM、startup——全算 **waste**,要清零或完全藏在 HMX 计算之下。**连"matmul 数本身的冗余 + fill/drain 不摊销"也算 waste**(地板锚在 MAC 峰值,不是当前实现的 HMX 占用)。
  - **地板值(推导,非猜测)≈ 284K cyc** = N_mm 768(24/head×32:diag 8 + merge 16)× native dominant-path ~370/64³-matmul(8 mxmem walk 完美流水、零 fill/drain 的 MAC-bound 关键链,`reference_64cube_conv_occupancy_vs_latency`)。单 HMX 串行 ⇒ 768×370≈284K。
  - **当前 wall 2.32M(cron#74)⇒ 距 MAC-峰值地板 ~8.2× 余量。** ⚠️ 注意 consumer-busy ~1.34M ≠ 地板:那是含每-matmul fill/drain 的 occupancy(1388/matmul 级),比 MAC 峰值(370/matmul)高 ~3.7×;**摊销 fill/drain(批处理)能把 HMX 时间本身从 1.34M 拉向 ~284K**,所以地板远低于 consumer-busy。
- oc = device 输出 vs fp64 真值 inv,32-head aggregate(`run_w16a16_head_phase4.py` 输出)。任何让 oc≥4e-2 的改动否决。
- wall 判定**一律同热窗 ACAC 交替环比配对中位**(`GDNBM_REPS=8` 取 reps2-4 median,绝不取 min,~6 配对轮),不跟固定常数比(绝对值随热漂 ±10%)。
- 不以"比别的实现快"为目标;目标 = 本实现做到设备极限 + 学清 w16a16 怎么用。真代码 / 真设备 / 真数据。

## STATE（live）

| 项 | 值 |
|---|---|
| **当前 best wall**(QNN graph wall = 32-head TOTAL) | **steady ~2.32M cyc**(≈1.46ms @1.59GHz;cron#74 = 全 VTCM 常驻,中间态 0 次 DDR)。**reps2-8 median 链:baseline 4.97M → cron#72 4.04M → cron#73 2.36M → cron#74 2.32M = 总 2.14×**,全程 bit-exact。cron#74 wall ~neutral vs cron#73 但 **架构铁律达成(中间态全 VTCM)+ slowest-prod-life 2.25M→1.72M**;瓶颈转单-HMX consumer + sync(见 ④)。 |
| **oc** | synthetic0.05 **4.238e-3**(cron#72/73 全程逐位不变)· 真 GDN(A_u16_h32)**1.107e-2**(均 < 4e-2) |
| **口径** | 全用 QNN 字段(`gdn_solve.md` §5.1):graph wall = `cycles_used`(单元) + `num_dominant_path`(关键链)。**32-head TOTAL graph wall = 唯一终指标。** wall 自测 = `GDNBM_REPS=8` reps2-8 median(高方差,单轮 ±15%)。 |
| **⚠️ 瓶颈定论(cron#72,实测推翻 cron#71 "同步握手" framing)** | wall = **HVX-bound,卡在 4-HVX-unit 天花板**,非"调度气泡/同步握手"。per-producer-life ≈ wall(4 线程并发);consumer HMX **已经**在 producer spin 里重叠掉(spin 仅 ~10%)。⇒ 唯一杠杆 = **砍 HVX feed 工作量** 或挪出 DSP。**测瓶颈用 FARF 全量 PROD-Σ 分解(actcopy/kmajor/scatter/renorm/spin/outcopy + lmax),别只信 GP_TRACE 的 traced stage(漏算 scatter/wtpack/renorm)。** |
| **feed 分解(cron#73-alias 实测,Σ over 4 threads / per-producer)** | **wt-pack 3.33M / 0.83M(#1,vgather+dilated pack+colsum)** · renorm/acc **1.64M / 0.41M** · scatter **~0(已消除)** · act/out copy ~0.5M · spin ~0.5M · **consumer HMX-busy 1.37M = wall 的 58%(单 HMX,已重叠)** ⇒ producer feed 与单-HMX consumer 渐趋平衡,继续砍 feed 边际递减(再砍 producer 后 wall 会塌到 max(producer-life, consumer-drain))。 |
| **matmul 现状(✅ NEXT① 已做 + cron#71 optrace 实证)** | solve 每 matmul = 单 64³ `crouton_pos` n_tiles=8。**HMX kernel = native parity:单 [1,1,64,64] 我们 `num_dominant_path`3789/`cycles_used`11152 vs native 3543/11176**。⚠️ consumer-busy 1.34M = occupancy(含每-matmul fill/drain ~1388/matmul),**不是地板**;MAC-峰值地板 = 768×370≈284K(摊销 fill/drain 后),见 OBJECTIVE。 |
| 唯一真路径 | `example/gdn_native/pure_hmx_solve/gdn_pure_solve.cpp`(`-DGDNBM_GDN_PURE_SOLVE`),改这里 |

## NEXT（后续方向）

**① ✅ DONE(cron#68):solve matmul 换最优 `crouton_pos` n_tiles=8 = consumer-busy 4.9M→1.5M(3.27×),graph-wall 6.61M→4.77M(1.39×),oc 不变。** `GP_CROUTON8`(默认 1)。改动闭式: g_lut=`crouton_pos` → cv 即 surface(`gp_cv_to_surf` 退化为连续 64-vector XOR 拷贝)+ native M=64 描述符;`g_hw/g_il/g_fl/diag-fix` 自动跟随 g_lut(PACKCHK=0);colsum 改无-gather(按 crouton_pos 结构按列归约,避免 feed 上多一次 64-vec gather)。**关键 bug 教训: `HVX_Vector`=64 u16,4096 codes=64 vec 不是 32**;`vsxt` lo/hi=偶/奇 lane 交织。

**② ✅ DONE(cron#72):gather 全流水 + memset 砍 = 1.23× wall(4.97M→4.04M),bit-exact。** 三处(`GP_GPIPE`=默认 1):
  - **scatter/wtpack 的 vgather 从"逐个 gather→立即读回同一 slot"(每 gather 全 latency stall ~184cyc)改成"64 个 gather 打到 64 个 distinct VTCM slot→批量读回"** —— gather 引擎流水起来。gp_perm/gp_unpack_blk/gp_pack_blk + gp_pack_wt_bias_hvx(后者原本只 2-deep 且 pack 在 gather 之间制造依赖)。scatter Σ 8.4M→~5.0M,wtpack Σ 5.5M→3.9M。
  - **memset(To,128KB) → 只清 6 个严格上三角 block**(lower-tri+diag 由 gp_pack_blk 全写;host 也只读 lower+diag)。sc_ms 0.88M→~0.4M。
  - **教训**:cron#71 "瓶颈=同步握手/dominant-path 缺口" 是错的(只看了 GP_TRACE 的 3 个 stage,漏了 scatter/wtpack/renorm 这些没被 trace 的 HVX 段)。FARF 的全量 PROD-Σ 分解(actcopy/kmajor/scatter/renorm/spin/outcopy + lmax)才是真账,spin 仅 10%。**测瓶颈用全量计数器分解,别只信 timeline 的 traced stage。**

**③ ✅ DONE(cron#73,用户裁定走 I/O 契约):scatter 彻底消除 = 1.45×(4.04M→2.36M)。** `GP_CVIO`(默认 1):
  - **A 以 cv-block 布局交付、T 以 cv-block 读回**(block (bi,bj) @ int16 offset (bi*4+bj)*4096,crouton_pos 序;exps 进 unused block(0,1))。host(`scripts/run_w16a16_head_phase4.py` + `_crouton_posf()`,numpy)做 linear↔cv 置换,相对 DSP wall 免费。
  - **零拷贝 alias**:solve_head 的 block 操作数走 `Ab[]/Tb[]` 指针,GP_CVIO 下直接指进 I/O 缓冲(Aq/To)——连 DSP 上的 8KB block-copy 都省了(copy 版 3.43M → alias 版 2.36M,再 1.41×)。前向回代顺序保证无 read-before-write,逐位不变。
  - 用户裁定依据(2026-06-15):生产中 solve 作 custom op,A 本就来自上游 crouton 布局,linear 接口是 standalone harness 验证产物 ⇒ on-DSP linear↔cv 转换是 artifact,挪出 DSP 合法。metric 现不含 layout 转换。
  - 副产:wt-pack bias 的 64-iter scalar `/2` 除法换成 `(-cs)>>1`(算术移位=floor,逐位等价),边际(~3%)。

**④ 全 VTCM 常驻(用户裁定 2026-06-15,内存铁律 [[feedback_vtcm_only_intermediates]]):DDR 只准在头部输入读入 + 尾部输出写出;中间所有过程态只准 VTCM。**

- **④a ✅ DONE(cron#74):`gp_acc_diag_add` 向量化**(64 次 scalar RMW → 对角 mask 向量 `acc += diagmask & splat(add)`,bit-exact)= 常驻 acc 的前置(否则常驻 acc 触发 scalar-VTCM 7× 慢)。
- **④b ✅ DONE(cron#74):全部中间态(A/T 16 block + AA/A3/M/Z/Tt/prod scratch + acc + lin)挪进 VTCM 常驻**(`GP_VSTRIDE` 0x30000→0x84000,4×=2.1MB<8MB;gp_ctx 数组→VTCM 指针,run() setup 赋址)。`c->A`/`c->T` 不再 alias rpcmem;头部 load A cv-block(rpcmem→VTCM)、尾部 store T(VTCM→rpcmem)各一次,中间 0 次 DDR。`gp_pack_wt_bias_hvx` 直接从常驻 cv gather(省 64-vec staging,wt-pack 2.72M→2.53M)。bit-exact + PACKCHK=0。
  - **结果:wall ~neutral(~2.32M ≈ cron#73 的 2.36M)**,但 **slowest-prod-life 2.25M→1.72M(producer feed 真降了)**。**架构铁律已满足(中间态全 VTCM)**。wall 没跟着降 = 瓶颈转移:producers 1.72M 完成但 wall 2.32M,**单-HMX consumer(1.34M)+ sync/drain 成了 limiter**(producer 做 head-load 时不 arm matmul → consumer starve → 串行尾巴,wall−lmax 缺口 0.18M→0.6M)。
  - **教训(写进 STATE)**:VTCM 常驻对 wall **不是**直接增益 —— scheduling-skill "VTCM 不比 DDR 快(单条顺序 HVX pass,L2 prefetch 已饱和)" 在此应验;常驻的价值 = 架构正确(铁律)+ 降 producer feed + 为 ④c 解锁(consumer 成瓶颈后,批处理才有意义)。

**cron#74 ④c 前基线 timeline**(`GP_TRACE` build,H=8 P=4,wall=736244 cyc / 110 列 ≈ 6.7K cyc/列;复现见 REPRODUCE):
```
P0   |                psxs  pxpxxssspspspxxspppxxxssssp           pss psspspppxssspppxxxsxssssp   | busy=38% [PREP=33% SPIN=23%]
P1   |                 pss  pxxsspspxsxsxsssxxxsssspsppx           pxxpsspxxxsspxxxssssppppxxxxs  | busy=40% [PREP=32% SPIN=27%]
P2   |                   px psxsspxxspxspppxsspppxxxxss           pss psxxsspxpsssssspppxxxsssssx | busy=39% [PREP=34% SPIN=25%]
P3   |                    psssspxxsxxsppxsspxsssssppxxxs            pxpsxssxxsssppxxxxsssssppxxxsx| busy=41% [PREP=33% SPIN=27%]
CONS |                  mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm           mmm mmmmmmmmmmmmmmmmmmmmmmmmmm  | busy=47% [MM=70%]
   p=PREP(wt-pack) m=matmul(HMX) s=SPIN(producer等consumer) x=DEPACK; Σ: MM=346K PREP=644K SPIN=501K DEPACK=28K
```
- **三段全员空闲 gap = 关键**:① 起始 ~16 列(spawn + 第1个 head-load,未 trace);② 中段全断 = head1→head2 边界的 head-store+head-load(VTCM↔DDR,未 trace),切 head 时不 arm → CONS 也饿死;③ 尾部 drain。
- **⚠️ 口径**:timeline 只画 traced stage;producer busy 仅 38-41% 是因 **head-load/store(scatter)+renorm/acc+act-format 没被 trace**(就是那些空白)。真账看 FARF PROD-Σ(Σ producer=4×wall,traced 仅 1.17M,未 trace ~1.77M)。**别把空白当"idle 可重叠",得 packet 核对**(feedback_perf_gap_must_check_trace)。
- **两个 wall 杀手 → 正是 ④c 的靶**:SPIN 23-27%(4 producer 抢 1 HMX,排队)+ head 边界 gap(head-load/store 串关键路、CONS 饿死)。

**④c ☞ 目标([[feedback_goal_driven_not_guessed_direction]]):wall 2.32M → HMX MAC-峰值地板 ~284K(见 OBJECTIVE),~8× 余量。** 一切非-MAC 时间(feed/sync/fill-drain/startup)= waste,逐项清零或藏到 HMX 之下。**方法 = 数据挑杠杆,不押手法。**
> **先 instrument 把 wall 的去向量化全**(当前未解释的 wall−lmax≈0.6M 拆成 startup/consumer-drain/join;producer 未 trace 段 ~1.77M = head-load/store+renorm+act-format 要 trace 进 timeline → 显真相,别拿空白当可重叠 idle,见 [[feedback_perf_gap_must_check_trace]])。再逐个实测候选、留有效砍无效(≤2-3 零增益就下结论)。
>
> **待实测候选假设(平行,数据决定,勿预设排序 / 勿当"计划步骤"执行):**
> - 摊销 fill/drain:地板 284K vs consumer-busy 1.34M 的 ~3.7× 差 = 每-matmul fill/drain 没摊销;批/连发 matmul 让 HMX 跑到 dominant-path(370)而非 occupancy(1388)级 —— **这是逼近 MAC 峰值的核心**。
> - matmul 数冗余:768 是否真最小?(算法锁定前提下)有无重复/可合并的 matmul。
> - feed 藏到 HMX 之下:producer feed(wt-pack/renorm/head-load)与 consumer matmul 重叠(预取下一独立链 / 消 head 边界 gap)。
> - sync:减/改握手(批多 matmul 一次 arm、双 job-slot、lock-free),砍 SPIN 23-27%。
> - rebalance:producer/consumer 配比;consumer 空闲时帮 feed。
> - feed 绝对量:wt-pack 剩余 scalar-VTCM bias 写向量化(注意 INVARIANT#6)、renorm/acc 工作量。
> - 判据:每候选 `GDNBM_REPS=8` reps2-8 median 配对 + oc(硬门 4e-2,P≥2);FARF PROD-Σ 全量分解 + 重渲 timeline 定位。

**禁**:❌ 降精度 / 低位 dtype(w8a16/u8i8,**需明确要求才加**)· ❌ 改数值算法(Newton 锁 0)· ❌ thread HMX(HMX=1,SSR)· ❌ 旧 compact 单-64³ 描述符猜测(死;真布局 = `crouton_pos`,已 dump 实证)· ❌ w16a16 fan-out 核(不存在)。

## 历史（压缩;细节见 git log + LEDGER 各 cron commit）

干净重写 `gdn_pure_solve.cpp`(单路径,替代旧 6 套迷宫)逐步优化:
**v1 1.354B → O1 4HVX+1HMX 线程 433M(3.13×)→ O2 cv 域+64³ 179.9M → O4a–d 全 feed 转 HVX**(copy/wt-pack/acc/renorm/scatter)**18.68M(72.5× vs v1,超旧参考 18.7M)→ Newton 2→0**(Taylor3 甜点,设备上 Newton 反效)**12.2M → cron#42 n_tiles 64→32**(切冗余 MAC walk,bit-exact,1.64×)**≈6.5M → cron#68 `GP_CROUTON8` n_tiles=8**(`crouton_pos` compact,consumer 4.9M→1.5M,1.39× wall)**≈4.79M → cron#72 `GP_GPIPE` gather 全流水 + memset 砍上三角**(scatter/wtpack 的 vgather 从串行读回改 64-distinct-slot 批读回,1.23× wall)**≈4.04M → cron#73 `GP_CVIO` cv-block I/O 契约 + 零拷贝 alias**(scatter 彻底消除,1.71× wall)**≈2.36M → cron#74 全 VTCM 常驻**(中间态 0 次 DDR,铁律;wall ~neutral 但 producer-feed lmax 2.25M→1.72M,瓶颈转 consumer)**≈2.32M**。全程 oc 不变(4.238e-3),PACKCHK=0。**本 session(cron#72-74)合计 4.97M→2.32M = 2.14×,全 bit-exact。**

**关键定论(别再翻烧饼):**
1. 瓶颈是 **producer HVX feed(字节流量)**,非 HMX 算力(HMX ~7% busy,真 matmul latency ~256/call)。
2. **Newton=0 是真最优**:每 Newton 步加的 w16a16 quant 噪声 > 截断收益,真 scale(‖A‖~0.6/max1.1)也成立(N0 1.1e-2 < N1 3.2e-2 < N2 3.35e-2)。Taylor 3→2 在真 scale 会爆(A³ 发散)。
3. **w16a16 无 descriptor-driven fan-out**(只 byte-weight 的 deep 核有;`convhhh`@0x2fdcc0 = 我们核逐字节同源,r17 软件 M-loop)。native 单 64³ wall≈我们;native 批的优势 = n_tiles 卡最小 + 批摊销,**我们 n_tiles=32 已达同 per-matmul 地板**(cron#67 实测,详见末节对比表)。
4. **matmul bit-exact to native + 输入构造闭式已掌握**(cron#66–67,末节)。

## INVARIANTS（grounded,禁止凭记忆改）

1. **w16a16 = 2× w8a16**:int16 权重拆 hi(int8)+lo(uint8),各 ×int16 act = 两遍 w8a16 分别 drain ×256 合并。`our_v73deep_kernel_i16`(dilated 权重 + 2×2 drain)内部即此两遍。权威 `docs/w16a16_is_two_w8a16.md`(CI gate)。
2. **drain 是 2 的幂,非 fp16**:增益 `2^(exp-16)`(bias 控制字 bits[14:10])。
3. **byte-exact 已 PROVEN**:standalone(Python/C pack + kernel + depack)hexagon-sim + 真 CDSP 双向 `diff=0`,M=256×任意 K,N,CI 守门。
4. **128 条独立求逆链 = 主结构**:C=256 每 head 切 4 个 64 对角块,每块一条 Newton 链(**链内严格依赖**);32 head × 4 块 = 128 条链彼此无依赖 ⇒ matmul 可批、feed 可流水。
5. **n_tiles 卡精确最小** = `ceil(M/32)×ceil(N/32)×2`(w16a16 byte_pass=2);过切 = 线性多做冗余 MAC。已设为 `w16a16_mm_init` 默认。吞吐成本模型见 `docs/w16a16_kernel_mechanism.md §5`。
6. **HVX lane 硬经验**:① `vsxt+vasr` 必须配对(序不配自检抓出 3968/4096 错);② **scalar 访 VTCM ≈4× 慢于 DDR-L2**(prep 走 DDR,VTCM 只给 HMX 面/gather);③ HVX bias lane-fold 两次都错,scalar LUT colsum 更划算;④ dense pack = `vadd128 + vasr8 + vpack_sat`。

## 死路（已证,别重试）

- **compact 单-64³ 描述符**(V1/V2/V3 全 DSP-fault/错):256 是 M=256 值不可搬给 64³。真路 = ramp-dump 真值(末节已解)。
- **对角预条件换便宜 dtype**(R1):math 必然 —— 远次对角 inv 压到廉价 dtype 分辨率下下溢成 0,还原救不回(`scripts/gdn_solve_precond_probe.py`)。
- **w16a16 fan-out 核**:不存在(只 byte-weight 有 deep 变体)。
- **BANNED 论断**:❌「drain 走 fp16 / 有损 / 是 blocker」(错,2-幂 + byte-exact;死因 = 用错配量化标定)· ❌「64³ 描述符永不可用」(错)。

## REPRODUCE / FILES

> ⚠️ **P=1(单线程)在 crouton8/cv-block 下崩(自 cron#68,从未在 P=1 复测过 —— metric 是 P=4)**:inline 路把整条
> solve_head→diag_inv→mm64 链跑在小的 main FastRPC 栈上,crouton8 的 4×128B colsum union 把它撑爆;threaded-P=1
> (1 producer)路也崩(未测路径)。**用 P≥2**(P=2/3/4 全 work + bit-exact)。`run_w16a16_head_phase4.py --threads 4`、
> `phase3` 已改 P=2。修它(非 metric,低优先):减 colsum 栈(4 union 挪进 gp_ctx scratch)或修 threaded-P=1。
> **I/O 默认 = cv-block 契约(`GP_CVIO=1`);host 脚本默认 cv-block,`--linear` 走旧 linear(需 `-DGP_CVIO=0` build)。**

```bash
# 当前 best(32-head wall + oc)。GP_CVIO=1(默认)= cv-block I/O 契约;脚本默认匹配。
uv run python scripts/run_w16a16_head_phase4.py --deploy --threads 4 --heads 32 --scale 0.05
# wall 稳态自测(reps2-8 median):GDNBM_REPS=8 直跑设备二进制(per-rep 打印 wall/scatter/wtpack/cons)
# QNN-对齐 perf 汇报:GP_TRACE → Perfetto chrometrace(HMX tid256/HVX tid512+,QNN 同 schema)
EXTRA_DEFS="-DGDNBM_GDN_PURE_SOLVE -DGP_TRACE" bash example/gdn_native/baremetal/build.sh
uv run python scripts/gdn_trace_to_chrometrace.py T.raw chrometrace.json   # decode_qnn_optrace.py 可直读
GDN_TL_COARSE=1 uv run python scripts/gdn_pipe_timeline.py T.raw 120        # ASCII timeline
```
- `pure_hmx_solve/gdn_pure_solve.cpp` — 唯一真 solve,改这里。`w16a16_mm.h`/`w16a16_pack.h` — 64³ 原语 + C 打包器(byte-exact,CI-gated)。
- `baremetal/src/gdnbm_imp.cpp` — FastRPC 驱动 + 4×producer∥1×HMX-consumer 管线(producer feed = 主战场)。
- `scripts/run_w16a16_*.py`(mm/diag/head phase1-4)· `scripts/gdn_solve_precond_probe.py`(R1 + ‖A‖ 分布)。

## 对齐 native 64³ matmul（cron#43–67 ✅ SOLVED + 闭式掌握）

> **✅ 目标达成(cron#66–67, 2026-06-14):dense n_tiles=8 64³ w16a16 matmul = QNN native `ConvLayer_s1.opt` BIT-EXACT(max|d|=0)+ per-call 1577 cyc(native parity)。** 全程 native-direct(零 CPU)。**输入构造 = 闭式 `crouton_pos(r,c)`(纯 bit-置换),无表;跨 4 个差异 case(3 大值 pattern + strictly-lower)全 max|d|=0。** probe = `-DGP_ALIGN -DGP_ALIGN_NCASE=4`(gdn_pure_solve.cpp,自含)。

### 🔑 输入构造闭式（掌握的核心方法）
native M=64 的 act/out crouton tile 布局 = **一个纯 bit-置换 `crouton_pos(r,c)`**(act pack 与 out untile **共用**):
```
crouton_pos(r,c) bits (lo→hi) = [ r0, c0,c1,c2,c3,c4, r1, r3,r4,r5, c5, r2 ]
  = r0<<0 | c0<<1|c1<<2|c2<<3|c3<<4|c4<<5 | r1<<6 | r3<<7|r4<<8|r5<<9 | c5<<10 | r2<<11
```
(ri=(r>>i)&1, ci=(c>>i)&1。act: `act[crouton_pos(r,c)]=Aref[r,c]`;untile: `outlin[r*64+c]=raw_out[crouton_pos(r,c)]`。
逆出自 ramp-dump,python 验逐位复现 dump 表。语义:bit0=r0=行对(row0/row1 相邻),bit1-5=col 低 5 位,bit10=c5=哪个 32 列半,其余=行高位散布。)

### ⚠️ 64³ conv 口径定论（cron#70,关键,别再翻烧饼）
**我们 n_tiles=8 64³ kernel 的 HMX occupancy 已 = native ConvLayer,持平(0.95×)。** 一度以为"3.6× 慢"是口径错配:
- native `ConvLayer_s1` 在 mm_cmp `[1,32,64,64]` 是**唯一**在 HMX(tid256)的 op(`htp_op_instances[].hmx=1` 仅它)⇒ HMX 单元 busy 44423/32 = **1388/conv 全是它的 occupancy**;它报的 **370 = dominant-path latency**(`num_dominant_path` 字段,非 occupancy)。
- 我们(NTSWEEP 自测,baremetal C15:14,无 QNN):`wall=255+n_tiles×165`,pure-mxmem(8 walk)= **1320**,per-call = 1576。
- **正确同口径对比 = occupancy:我们 1320 vs native 1388 = 0.95×(持平略优)。** `370` 谁都吃不到 occupancy 级(native 自己 HMX 也 busy 1388/conv);它是"8 walk 完美流水、零 fill/drain 开销"的理论关键路径。
- 旁证:256³ custom op 我们 kernel 74670 ≈ native 75432(大块上 latency≈occupancy,持平)。**别再拿 optrace 单 op 的 dominant-path(②)比我们的 per-call wall(④)——必造假差(skill `htp-cycle-metric` trap#4/cross-impl 禁令)。** DISTINCT-TILE 测:act/out tile 复用 vs 32-distinct per-walk 一样(1576),tile 布局非因。

### 📊 M=64 dense vs M=256 载体 对比（口径④ per-call,resident,本机实测）
| 实现 | M | 描述符 | act/out 布局 | per-call | per-64³ | bit-exact 怎么证 |
|---|---|---|---|---|---|---|
| 256 载体 原始 | 256 | n_tiles=**256**,out_y=256,m_total=1,act_y=128 | `pack_act_crouton16(256)` 8-row4 组+32 行 m32 块 | **42333** | 10583 | 传递性(== 之前验过 byte-exact==native 的 custom op) |
| 256 载体 cron#42 | 256 | n_tiles=**32** | 同上 | **5547** | **1387** | 同上(只在 M=256 验过) |
| **64 dense** | 64 | n_tiles=**8**,out_y=4,m_total=8,act_y=4 | **闭式 `crouton_pos`** | **1576** | **1576** | **这次直接对 native `ConvLayer_s1` 逐码(4 case)** |

**要点**:① 原始 n_tiles=256 是 8× 过切;cron#42 砍到 32 = 每块 8 tile × 4 块 = 已达 dense 地板。② **256 载体 1387/matmul 已是地板(批摊销略优于 64-dense 单块 1576)**;64 dense 不是更快,价值在 bit-exact-to-native + 单块简单。③ 旧 "10838 = 单 64³ 地板" 是 n_tiles=64 过切态,非真地板。

### 最终配方（dense n_tiles=8 bit-exact 到 native）
喂逐字节同源的 `our_v73deep_kernel_i16` @0x2fdcc0:
1. **act 布局 = `crouton_pos`**(闭式)。**不是** `pack_act_crouton16(.,64,64)`(那是 256-载体布局 → 周期-8 错位)。
2. **weight = `w16a16_pack_wt_kmajor`**(dilated 4-pass;低字节按 **signed int8** 重组 = 精确)。
3. **bias = `w16a16_pack_bias` 的 eff(`-colsum/2`,==native)+ control 字覆写 native `0x804035F3/0x4000023E`**(= `sA·sB/sC` drain;默认 `0x00404420/0x40000000` = `1/32767` drain,差 ~10×)。
4. **descriptor(native M=64)**:`out_tbl_stride=2, out_y=4, n_tiles=8, m_total=8, k_total=64, n_act=2, act_y=4`;atab/otab 4 tile @ `0/2048/4096/6144`(`i&3`)。
5. **mask**(MASK[16],mask0=0)+ **extra `{1,1536}`**(均 ==native)。
6. **out untile = `crouton_pos`**(同 act)。

### native ConvLayer_s1 I/O 契约 + 量化配置（lowered HTP 图,实测 scalar_params）
op 流:`act: InputSlice(uint16 zp32768)→ForceFormat_Crouton` ‖ `weight: InputSlice(SFIXED16 zp0)→Cast(→uint16 zp32768)→convert_weights_to_signed(→SFIXED8[64,128])→bias_weight_update→bias_scale_shuff` → `ConvLayer_s1.opt → OutputSlice`。
- **量化(真实 effective)**:act `uint16 zp32768 sA=7.3243e-6` · weight `signed int16 zp0 sB=9.1556e-6` · out `uint16 zp32768 sC=2.2522e-5`。`gain=sA·sB/sC=2.9774e-6`。
- ⚠️ **Cast 在 weight 侧,不是 act**:B 声明 int16→Cast 转 uint16(+32768)只是 byte-pass 喂数 plumbing,净效果权重仍 signed int16(`bias_weight_update` 把 act-zp colsum + weight cast 偏移都吸收进 bias)。act 本就 uint16,无 Cast。
- ConvLayer_s1 IN/OUT:act id16 `UFIXED16[1,8,8,64]` crouton · weight id13 `SFIXED8[1,1,64,128]`(int16 拆 2 byte-pass)· bias id14 `INT32[1,2,1,128]`(256 int32 = control+eff,**不是 u16**)· OUT id15 `UFIXED16[1,8,8,64]`。
- **语义**:`out_code = round(Σ_k(A_code−32768)·B_code · sA·sB/sC) + 32768` = 纯 int32 matmul + 线性 requant,**无 1.42×/byte-pass 异常**(之前的 "1.42×" 全是我们 act 布局+control 字+untile 三错的合成假象,不在 native)。

### RE 方法（ramp-dump,零猜测）
- **untile/actmap**:stub 写进 skel @0x2fdcc0(replace-kernel 跳 matmul),把位置 ramp(out_tile[i]=i)写进 out-tile → OutputSlice untile 后 host 读 `Cout[r,c]=落在(r,c)的 out-tile 原始索引` = 精确 untile(bijection)。act 同理:喂 `Aref=ramp(code=r*64+c)`,stub 拷 act-tile→out-tile,用已知 untile 反推。
- stub 配方:`hexagon-clang -O2 -mv75 -mhvx -mhvx-length=128B -mhmx -c` → `hexagon-llvm-objcopy -O binary --only-section=.text` → `dd seek=0x2fdcc0 conv=notrunc` splice 进 golden skel 拷贝(leaf,无重定位,`jumpr r31`)。

### 复现
- **bit-exact + cyc(自含,闭式)**:`EXTRA_DEFS="-DGDNBM_GDN_PURE_SOLVE -DGP_ALIGN -DGP_ALIGN_NATCTRL=1 -DGP_ALIGN_NCASE=4" bash example/gdn_native/baremetal/build.sh` → 部署 → `./gdnbm 1 probe_A.raw T.raw 1 256 ...` → `stats[0]=1577`(per-call);`T.raw[v*8192..]` = case v 输出码,python 对 native Cout = 4 case 全 max|d|=0。
- **native Cout**:`/tmp/align/dump_run.sh` 或 `qnn-net-run` mm1ex,A/B.raw fp32 =(case 公式码)×scale。ctx = `solve_op/standalone/mm_1x1x64x64/`。⚠️ **还原 skel 必须绝对路径** `tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so`(`$QNN_SDK_ROOT` 某些 shell 未设,曾把空文件写进 skel)。device skel md5 = **286cc5**。

### ⚠️ 仍 OPEN（matmul 本身已 SOLVED;以下属集成/旁支)
1. **control 字 `0x804035F3/0x4000023E` 是 mm1ex 这组 scale 专属**;solve 用自己 scale(0.05 regime,`1/32767` drain,默认 control),**bit-exact-to-native 只对 mm1ex 这组 scale 成立**;solve 判据是 oc 不是 bit-exact。control 字的 scale→编码 关系尚未逆(solve 不需要)。
2. ✅ **接 `crouton_pos` 进 solve(cron#68 DONE)**:`GP_CROUTON8`=默认,consumer-busy 4.9M→1.5M(3.27×),wall 6.61M→4.77M(1.39×),oc 不变。实测验证了 "对 solve 单 64³ 而言 crouton_pos 确是提速"。瓶颈随之转移到 producer feed(见 NEXT ②)。
- 方法论:correctness 用 *多个独立输入* 验(cron#47 单输入 "1577 bit-exact" 是 false-positive;cron#67 用 4 case 已排除)。
