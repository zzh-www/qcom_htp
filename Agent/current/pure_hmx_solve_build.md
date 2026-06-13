# Pure-HMX w16a16 三角求逆 — Loop Optimizer（入口文档）

> ## ⚠️ ACTIVE LOOP — 本文是本 loop 的唯一权威,永不判"闭环"
> 本文是**进行中**的优化循环。**只要 `NEXT` 非空 → 直接做 `NEXT`,不准下"已完成/无新活/闭环/收口/穷尽"结论。**
>
> ### 🔒 强制·性能口径(不可绕过)
> **任何性能统计 / 分析 / 报数 / wall 判定 / NEXT 决策,MUST 走 skill `htp-cycle-metric`。** 先 `Skill(htp-cycle-metric)` 读口径,再测、再报、再判。规则:
> - 每个数 = **value + 口径 + context**;四口径分清且分列:**① graph wall**(全程 span = 裁决)·**② op-latency**(`num_dominant_path_cycles` / `N_mm×256` = HMX 算力)·**③ unit-throughput/busy**(occupancy)·**④ per-call feed-inclusive**(kernel-call wall,含字节搬运)。
> - 全 PCYCLE(`C15:14`);**时钟自检 `wall/µs ≈ 1594`**(本设备 TURBO;≫此 = 读错计数器,作废重测)。
> - **硬门:单个 64³ 纯 HMX matmul 算力 = 200~300 cyc latency**(native int16 ~256)。任何把单 64³ 报成 >300 = 口径错(feed/padded 字节流量当算力)或实现没流水 → **先定位修正,该数不准用于判定**。
> - **不按此口径产出的任何性能结论 = 无效**,不得写进 STATE/LEDGER、不得用于 wall 环比或 NEXT 取舍。(本轮我违反过:把 `g_cbusy` feed-inclusive 当 HMX 算力 → 误判 HMX-bound;已纠为 feed-bound。)
>
> **范围 = 本 loop 只做 pure-HMX(`GDNSolveHMX`,全程 HMX matmul 的三角求逆)。** 判定本 loop 状态**只看本文 STATE/NEXT/LEDGER**。
> **不要**用 git 历史 commit message、PROGRESS 里的 ✅、任何 memory、或 `gdn_solve.md` 等其它文档里**别条路线**(HVXMixHMX / merge-dtype / int16-merge / FP16)的"否决/OPEN/收口/比较"来判定本 loop——那些是**不同路线或旧目标(比快)下的结论**,与本 loop(目标 = min wall s.t. oc<4e-2,**不比较**)无关。本文自洽自足,不需旁读。
> loop 只在 **wall 触 roofline 地板(~1.0M) 或 LEDGER A/B 全 DONE/REFUTED** 时到头——那时也是**写新 NEXT**,而非判"闭环"。
>
> ### 🔒 方向锁定(2026-06-13 用户裁定;后续 session 不准跑偏)
> **当前唯一活动方向 = O6b compact-crouton 描述符**(让 consumer 64³ 读 compact 8KB 而非 padded 32KB,per-mm 10.8K→~3.3K,wall→~4-5M)。**这是主线,只做这个。** 细节 = 下面 `NEXT` 块。
> **明令禁止跑偏(违反=作废):**
> - ❌ **不做 O6a w8a16 单 pass**——用户已砍,非根本突破。
> - ❌ **不动数值算法**(Taylor/Newton 次数、merge 顺序、scale)——O6b 纯布局,oc-中性。
> - ❌ **不开新支线 / 不建新文档**——方向就在本文,别另起 goal 文档、别去捡 LEDGER 里的 A2/SKIPFIN/fan-out 等旧条目当 NEXT。
> - ❌ **不重测已 REFUTED 的**(O6b 的 V1/V2 见 LEDGER;别再试"只挪 atab 间距")。
> **O6b 的 crux 已收敛到一件事:拿到 QNN/标准件读 512B-tile 的精确 STRIDE 描述符**(不是 atab 间距)。起手见 `NEXT`。拿到→scaffold(`-DGP_O6B_TEST`)验 maxdiff=0 + cyc~3.3K→港进 solve,oc<4e-2 + wall 环比。

> **每个 turn 的唯一入口。** 读 `STATE` 知道在哪 → 读 `NEXT` 知道这轮干什么 → 干完按 `LOOP` 更新本文。
> 改任何东西前必读 `INVARIANTS` + `LEDGER`:**别重试 R 的 2 条死路,别改 INVARIANTS(除非新设备证据)。其余(A/B/D)都是活。**

## OBJECTIVE

**最小化 32-head TOTAL wall,硬约束 `oc < 4e-2`(每轮重测)。**
- 任何让 `oc ≥ 4e-2` 的改动**否决**。oc 不随热漂 → **绝对门**。oc = device 输出 vs fp64 真值 inv,32-head aggregate(由 `run_w16a16_head_phase4.py` 输出)。
- wall 口径 = skill `htp-cycle-metric`(**32-head TOTAL wall domain cyc**;per-head/min-of-reps/PROBE 全禁)。
- **硬口径门(matmul sanity,每次报数自检):单个 64³ 纯 HMX matmul = 200~300 cyc latency**(native int16 ~256,dominant-path;或等效吞吐效率)。**任何把单 64³ matmul 报成 ≫300 cyc 的数 = 口径错(把 feed/padded 字节流量当算力,如 ~10.4K/call)或实现没流水(throughput/padding 主导)——必须先定位修正,不准当作"matmul 就这么贵"。** 见 INVARIANT 5/9 + skill `htp-cycle-metric`。本轮我违反过此门(g_cbusy 15.3M 当 HMX busy),已记录。
- **wall 判定一律环比:新旧 A/B 同热窗 ACAC 交替,看配对差中位,绝不跟固定常数比**(协议见 LOOP 步骤 2)。绝对 18.7M 随设备热态漂(±~10%)= 参考,非门。
- 不以"比其它实现快"为目标(用户定义);目标 = 本实现做到**设备极限** + 学清 w16a16 怎么用。
- 真代码 / 真设备 / 真数据;预估、garbage-data 计时不算结论。

## STATE（live，每轮更新）

| 项 | 值 | 备注 |
|---|---|---|
| **当前 wall** | **~10.0M cyc = 干净重写 全HVX + Newton0 + O5 scatter-kill**(v1 1.354B → 135×) | **O5 DONE(cron#19,−12%):** fuse linear↔block strided memcpy 进 gp_perm 的 HVX staging(`gp_unpack_blk`/`gp_pack_blk`)→ **scatter memcpy 13.98M→0**,wall 12M→~10M(同窗 REPS=8 reps2-4 median 12.71M→10.76M=−15.3%),oc 4.238e-3 不变。**瓶颈翻到 consumer**(busy ~9.1M ≈ wall)。**NEXT = O6b compact-crouton(唯一主线,见方向锁定+NEXT)** —— 砍 consumer per-mm 10.8K→~3.3K → wall ~4-5M。🟥 **O6b-compact 前提被证伪(cron#22 DESC_DUMP 真值)。** QNN op 给 64³ 用 `act_tbl[i]=i×2048`(padded 32KB)= 跟 solve 一模一样,**没有 compact-8KB 单-64³ 描述符**。per-call 差(10.8K vs 3.3K)是 **fill/drain 摊销(fan-out 批 128 tile)非布局**。**O6b-compact 死路;真杠杆 = descriptor fan-out 批。锁定方向的前提已废 → 待用户裁定下一方向(见本轮报告)。** |
| **oc** | **synthetic0.05 4.24e-3 · 真 GDN(A_u16_h32)1.107e-2** | 都 < 4e-2(真数据 2.6× 余量)。**Newton=0 在两 scale 都最优** |
| **关键发现** | **设备上 Newton 反效:oc 单调 ↑ Newton(真数据 N0=1.1e-2 < N1=3.2e-2 < N2=3.35e-2,max 3.69e-2 近门)** | w16a16 每 Newton 步加的 quant 噪声 > 截断收益(真 scale ‖A‖↑ 噪声更大)→ **Taylor(3) Newton=0 = 甜点;原 "Newton=2" 设计次优**。fp 截断分析(Newton 有益)被设备 quant **反转** |
| 配置 | Newton=2,**40 mm/head**(24 对角 + 16 merge)= 1280 mm/32head | scale = 精度旋钮(甜点 ~0.25) |
| 极致地板 | ~1.49M(全 w16a16)→ ~1.0M(merge u8i8)→ ~0.7M(+Newton1) | roofline,**留 ≥12× 在桌上,全系统侧 = 都还没做** |
| 参考实现 | `pure_hmx_solve.cpp::p4_threads`(run() H≥5&P≥2 dispatch)= 18.7M 真 solve,oc 9.66e-3 **复现实测** | **不是 garbage**;`pipe_producer` harness(PREP-proxy)是死代码,文件头注释 stale。**HVX-feed-bound(producer-bound)**:真 matmul latency ~256/call(native,见 `gdn_solve.md:103`),HMX 实占 ~7%(1792×256≈0.46M);我先前测的"g_cbusy 15.3M=82% HMX busy"是**口径错**——那是 kernel-call 的 padded-operand **字节流量**(~10.4K/call,`gdn_solve.md:117`),= feed 非 MAC。**A2(喂数流水/overlap)+ 砍 padded 字节流量 = 真杠杆**(doc 原 NEXT 方向对) |
| **干净重写(= 参考)** | `pure_hmx_solve/gdn_pure_solve.cpp`(`-DGDNBM_GDN_PURE_SOLVE`)= **单路径真 solve,oc 9.662e-3 bit-match,PACKCHK=0,全 feed HVX**;唯一真路径,改这里 | **v1** 1.354B → O1 433M → O2 cv+64³ 179.9M → O4a copy 107M → O4b wt-pack 48.3M → O4c acc/renorm 23.9M → **O4d HVX scatter 18.68M(累计 72.5×)**,oc 全程 9.662e-3 不变。**已达/超参考 18.7M(干净单实现替代迷宫)**。① 18.68M(1593.6 cyc/µs ✓)。**consumer-bound:kernel 15.2M=81% of wall(= 口径④ padded-64³ 字节流量,非 MAC)= 此路线地板;spin 59%。** 剩 3.5M = consumer idle+startup/tail → **NEXT=A2 async 收尾**(diminishing) |
| 精度 | oc 余量 ~4×;对角 dtype = w16a16,merge 可 u8i8+SBOOST | |

## NEXT（这轮单一最高优先）

> ## 🔧 NEXT = O6b compact-crouton（用户裁定 = 主线;O6a w8a16 砍掉,非根本突破）
> **用户裁定(2026-06-13):O6a w8a16 不做(不是根本优化);做 O6b compact-crouton —— 真·对齐 QNN,这才是主线。**
> **现状(O5 后):** wall ~10.0-10.6M,**consumer-busy ~9.1M ≈ wall = 瓶颈**。consumer = 768 mm × ~10.8K(口径④)。
> **compact 是真杠杆(已用 QNN trace 定性):** kernel = QNN op **byte-identical**;QNN instance 3,349 cyc、**25.76 cyc/packet → ~130 packet**,kernel 是 **packet-bound,packet ∝ 读的 tile 数**。我们 padded 读 4× 的 act tile(136KB 流量)→ ~4× packet = 那个 3.2-4×。**compact(少 tile)= 真降 per-mm,不是 pipeline。**
> **已挖到 ground truth(`handwritten_hmx_matmul/prepare_owned_inputs.py:generated_descriptor_tables`,M-general byte-exact):**
> - compact act packer 现成:`w16a16_pack_act_crouton16(A,out,64,64)` → **8KB(16 块×512B)**;depack `w16a16_depack_crouton16` blk=(M/32)*128 u16(M=256→2KB,M=64→512B)。
> - atab/otab offset = `((mt&7)*k_tiles+kt)*m_tiles*256` 字节(M=256 ×2048=我们现状;**M=64 ×512=compact**)。
> - 静态描述符(标准件,M-general):out_y_stride=256、n_tiles_pow2=256、k_total=128、**act_y_stride=512**、m_total_minus_step=1。
> **RE 实测(scaffold `-DGP_O6B_TEST`,设备;两轮 cron#20):全部猜测已 REFUTED → STEP 1a 是唯一路。**
>   - **V1**(标准件 strides + ×512 atab) → **DSP fault**。
>   - **V2**(solve 现工作 strides 64/64/64/128 + ×512 atab) → 跑通但 **maxdiff 25774(错)+ cyc 10880 ≈ padded(无加速)**。
>   - **V3(cron#20-2):offline 真值 strides** `prepare_owned_inputs.py:generated_descriptor_tables`(byte-exact 生成器)给的 w16a16 FIXED 描述符 = **out_y=256, n_tiles=256, k_total=128, act_y=512** + 同 ×512 compact 表 → **DSP fault 0x8000040d**(solve 在 O6b 块崩;之前 MM64 微基准 stats[5]=10838 正常)。那些 "256" 是 **M=256 值(256 literally = M)**,不能搬给真 compact 64³。同 V1 fault class。
>   - **⇒ 两个自然假设全死**:shape-scaled(V2,跑通但错)/ fixed-256(V3,崩)。**descriptor 猜测穷尽** = 与 cron#15「compact 越界设备证伪」+ cron#17「QNN 靠 batched 描述符 fan-out 拿 compact,非 per-call 单 64³ compact」一致。**单 64³ compact 描述符可能根本不存在于此 per-tile kernel;真路 = 抓 QNN 实际怎么做(STEP 1a/1b)。**
> **🔑 STEP 1a 已跑(cron#21)= 重大重定向:strides 不是问题,LAYOUT 才是。**
>   - **跑通了真 custom op 的 64³ device flow**(built 3 包 native_record_256;修 `dssh.sh` set -e bug + 装 lxml + CPU-op-pkg 注册失败→**HTP-only `--op_packages ...htp.so:...:HTP` 才行**;`run_w16a16_chain.sh M=64 K=64 N=64 CHAIN=1 MODE=direct`)。**custom op 注册+执行 64³ Crouton_16 图到完成**(ctx 含 `HmxU16I16ToU16MatMul` 节点)。
>   - **控制实验定性:M=256(已证 byte-exact 的形状)对 gen 的解析 ref `out_ref_u16` 的"对不上"模式与 M=64 完全相同**(都 dev 饱和 65535/30976 vs ref 小值 3079/70;256: 20770/65536)。256 是验证过的形状 → **是我的比对口径错(native_record_256 profile 验的是 native ORACLE sidecar,不是 gen 解析 ref),不是 kernel 错。** ⇒ **强证据:custom op 跑 compact 64³ 与 256 同样正确。**
>   - **⇒ 它给 64³ 用的描述符 = `FORMULA_DESC` = 64/64/64/128(从源码读),正好就是 V2 的 strides。可 V2 错了 → V2 的 bug 是 act 的 DATA/TABLE 布局(我的 `w16a16_pack_act_crouton16` + ×512 atab ≠ QNN 真 Crouton_16 块),不是 strides。** strides 一直是对的。**O6b 重定向:不再找 strides(已知=64/64/64/128),而是复刻 QNN 的 compact Crouton_16 act 块字节布局 + 表指针。**
> **STEP 1b/2(下轮起手,二选一或都做):**
>   - **(校验,严)用 `--activation-mode k_impulse --activation-k <k>`**(gen_quant_chain 有此 arg):act 单脉冲 → 输出 = 单条 weight 行,**不饱和、自证、且直接暴露 act/out 表的布局映射**。设备跑对 → 钉死 op 正确 + 读出布局。(避开 gen 解析 ref 口径坑;V3 已证 offline 生成器在 64³ 不可信。)
>   - **(拿布局)build `-DHMX_W16A16_DESC_DUMP`** 跑 64³ → 现有 DESC_DUMP 只 dump 表基址,**加一段 dump `act_tbl_all[0..31]` / `out_tbl_all` 相对块基址的 OFFSET**(= QNN 真 compact 表偏移模式)→ 对照我的 `w16a16_pack_act_crouton16` 找差 → 修包器/atab 让它 byte-match QNN Crouton_16 → 填 scaffold STEP 2 验 maxdiff=0+cyc~3.3K。
>   - 复现:`cd example/qnn_hmx_matmul_w16a16 && W16A16_KERNEL_PROFILE=native_record_256 bash build.sh && bash build_x86.sh`;`cd standard_flow/custom_w16a16 && M=64 K=64 N=64 CHAIN=1 MODE=direct W16A16_KERNEL_PROFILE=native_record_256 SKIP_DEVICE=1 GEN_EXTRA_ARGS="--weight-limit 1" bash run_w16a16_chain.sh`;device 手跑 HTP-only(CPU-pkg 注册会失败,只挂 htp.so)。需 `uv pip install lxml`。
>   - **1b(权威,1a 撞墙再上):vendor-dump native QNN** —— `[1,128,64,64]` 的 `q::ConvLayer` kernel 在 `libQnnHtpV75Skel.so`,打 vendor patch 在 conv slice 入口(`hmx_v73_convhhh1x1_stride1 @0x2fa740`)抓 r0..r5 + 指向字节;Skill `hmx-inline-asm` + 记忆 `reference_vendor_kernel_patch_dump`;复现 `GEN_BATCH=128 profile_all.sh`(已有 `output_b128/`)。
>   - **1c(兜底):trace asm** `v73deep_conv1x1_kernel_i16.inc` 的 `activation.ub=mxmem(...)` 地址算术,定位 `act_y_stride`/`k_total`/`n_act_pairs` 里哪个决定每 tile 读 2KB vs 512B。
> **STEP 2 验证:** 把精确描述符填进 `GP_O6B_TEST` scaffold(`gdn_pure_solve.cpp` 搜 `O6b COMPACT-64`)→ 构建 `-DGP_O6B_TEST` → 设备跑 → 门 **maxdiff=0 + cyc→~3.3K**(每版记进 LEDGER)。
> **STEP 3 港进 solve:** `g_lut`/`gp_cv_to_surf`/`gp_surf_to_cv` 改 512B 块 + compact 描述符 → 设备测整 solve **oc 不变(4.238e-3)+ wall 环比 ↓**(consumer 9.1M→~3.7M,wall→~4-5M)= O6b DONE。

**用户裁定(2026-06-13):旧实现是 6 套交叠迷宫 → 从头干净重写一套,然后逐步优化。** 干净重写 v1 已落地+验证正确(`gdn_pure_solve.cpp`,oc 9.662e-3 bit-match)。本 loop 转入**对 v1 逐步优化**,每步 oc<4e-2 + wall 环比门控。

**瓶颈 = HVX-feed(字节流量),不是 HMX 算力**(HMX ~7% busy,真 matmul latency ~256/call)。所以优化优先级 = **砍 feed 字节流量 + 让 feed overlap 到 matmul 下**,不是省 HMX。

**O1 ✅ DONE(2026-06-13):** 4×HVX producer + 1 HMX consumer 线程化,wall 1.354B→**433M(3.13×)**,oc 9.662e-3 不变。证实仍 feed-bound(consumer ④仅 14%,producer 96% 标量)。

**O2 cv-domain + 64³ ✅ DONE(cron#3,2.41×):** wall 433M→**179.9M**,oc 9.662e-3 不变。cv LUT `g_lut`(linear↔crouton)+ 直拷 act/out(无 depack/repack)+ 仅 weight 走 LUT-linearize+kmajor;64³ 描述符(`od.out_y_stride=64,n_tiles=64`)。depack 1.047B→266M、kernel 60M→15.3M。

**O4a HVX copy ✅ DONE(cron#4,1.68×):** out-copy 266M→0.26M(HVX vxor),wall 179.9M→**107M**,oc 不变。`gp_cv_to_surf`/`gp_surf_to_cv`,cv 数组 128-align,P=1 补 hvx-lock。

**O4c HVX acc/renorm ✅ DONE(cron#7,2.02×):** renorm/other 157M→41M,wall 48.3M→**23.9M**,oc 不变,PACKCHK=0。port 参考 `p4v_acc*`/`p4v_renorm` 全套(`gp_acc3/_negw/_zero/_addsh/_shr/_absmax/_to_cv/_renorm/_diag_add/_from_cv`),acc 换**交织布局**(vsxt lo/hi)+128-align。

**O4d HVX scatter ✅ DONE(cron#9,1.28×,过参考):** scatter 44.4M→18.3M,wall 23.9M→**18.68M < 参考 18.7M**,oc 不变。`gp_perm`(memcpy 连续化 + vgather)+ `g_il/g_fl` VTCM LUT。**至此全 feed(copy/pack/renorm/acc/scatter)皆 HVX。**

**里程碑:干净单路径 `gdn_pure_solve.cpp` 已达/超参考 18.7M(替代 6 套迷宫),72.5× vs v1,oc 9.662e-3 bit-exact,PACKCHK=0。**

**现状 = consumer-bound(此 64³ 路线地板)。** 口径④ consumer kernel 15.2M = 81% of wall(padded-64³ 字节流量,非 MAC,gdn_solve.md:117);spin 59%;剩 wall−kernel ≈ 3.5M = consumer idle + startup/tail。

**Newton 2→1→0 ✅ DONE(cron#10/11,各 ~1.2×):** @scale0.05 **oc 单调 ↑ Newton(0:4.24e-3 < 1:8.24e-3 < 2:9.66e-3)** —— Taylor(3) 已近精确,每 Newton 步只加 w16a16 quant 噪声,反而更差。Newton=0 默认:mm 1280→768,wall 18.68M→**12.21M**,oc **4.24e-3**(更快+更准)。

**DECISION 已解决(cron#11,真 scale 实测):** 真 GDN `A_u16_h32`(sA=2.77e-5)‖A_ii‖₂ median 0.600/max 1.099(比 synthetic 0.363 难)。设备真数据 oc 实测 **N0 1.107e-2 < N1 3.20e-2 < N2 3.35e-2**(N2 max 3.69e-2 近门)→ **Newton=0 在真 scale 也最优**,非 scale-fragile,非妥协。**根因:w16a16 每 Newton 步加的 quant 噪声 > 截断收益(真 scale ‖A‖↑ 噪声更大),fp 截断分析被设备 quant 反转。原 "Newton=2" 设计次优。** ⇒ Newton=0 锁定(默认)。

**结论:干净重写完成 + 真数据验证。** `gdn_pure_solve.cpp` Newton=0:**synthetic 12.2M/oc 4.24e-3,真 GDN 12.5M/oc 1.107e-2**,均 < 4e-2;111× vs v1,超参考 18.7M 1.5×;单路径全 HVX,oc-neutral 路径 + PACKCHK byte-exact。

**NEXT(可选,diminishing):** ① SKIPFIN 跳远 off-diag merge(真 scale 需测 oc,大 ‖A‖ 远块未必可跳);② A2 async 填 ~3M consumer idle(oc-中性,链内依赖限制);③ **收口出货**(已近 64³ 路线 consumer 字节地板 ~9M,真数据验证通过)。**Taylor 3→2 不可**(真 scale ‖A‖~1.1 → A³ 发散)。
- **旧 "1.49M roofline" 作废**(吞吐口径);真 consumer 地板 ~9-12M。**已近设备极限。**
- 文件:`example/gdn_native/pure_hmx_solve/gdn_pure_solve.cpp`(唯一真路径,改这里)。构建 `EXTRA_DEFS="-DGDNBM_GDN_PURE_SOLVE" bash build.sh`;复跑 `scripts/run_w16a16_head_phase4.py --deploy --threads 4 --heads 32 --scale 0.05`(oc 由它出)。
- **口径守则(吃过亏)**:kernel-call wall ≠ HMX 算力。w16a16 单调用 ~10.4K = padded-operand 字节流量(`gdn_solve.md:117`),真 matmul latency ~256。报 HMX busy% 必须用 latency 口径,别把 feed-inclusive kernel time 当算力(本轮我就错在这)。见 skill `htp-cycle-metric`。

## LOOP（每轮协议 + git 分支模型）

**基线分支 = `pure-hmx-opt`**(本 loop 的"主分支",= 当前态快照;repo 的 `main` 不被 loop 改动)。
每轮从基线切独立 feature 分支推进,按成败合回 —— "每轮从最新基线切"天然避开 stale-sandbox。

1. **切分支**:`git checkout pure-hmx-opt && git checkout -b loop/<lever>-<seq>`(从最新基线)。读 `STATE`+`NEXT`。
2. **推进**:在 feature 分支实现 `NEXT`(改代码)。
3. **验证(wall 环比,同 HVXMixHMX)**:A = 基线 best 构建,B = 新构建;**同热窗 ACAC… 交替**,每腿 `GDNBM_REPS=8` 取 **reps 2–4 median**(rep1 冷 / rep≥5 节流,**绝不取 min**),~6 个配对轮,记 **(B−A)/A 配对差中位**(消热漂)。oc 取 B 绝对值(不漂)。
   - **统一口径(skill `htp-cycle-metric`,每轮按模板报 value+口径+context):** 全部 PCYCLE(`C15:14`);时钟自检 `wall/µs ≈ 1594`(本设备 TURBO,≫此=读错计数器)。每轮 4 个口径分开列:**① graph wall**(`stats[0]` 全程 span = 最终裁决)·**② HMX 算力下限**(`N_mm × 256` latency口径,INVARIANT 9;= 不可约 MAC,判 HMX 是否瓶颈)·**③ kernel-call feed-inclusive**(`g_mmcyc`,含 padded 操作数字节搬运,>300/64³ = feed 非算力)·**④ feed/其余**(wall − ②,= pack/depack/renorm/glue = 当前主战场)。**绝不把 ③ 当 ②(算力)报**(本轮原罪)。
4. **判 + 合回**(门 = 环比 wall 配对差 **< 0** 且 `oc` **< 4e-2**;**单点降不算数**,只认配对中位):
   - **成功** → feature 上更新 `STATE`/`NEXT`/`LEDGER`(结论 = DONE + 环比 Δ%)**+ 保留代码** → `git checkout pure-hmx-opt && git merge --no-ff loop/<lever>-<seq>`(**码 + 文档一起进基线**)→ `git branch -d loop/<lever>-<seq>`。新 best = 下轮基线。
   - **失败** → feature 上**只把结论写进 `LEDGER`**(该杠杆 → REFUTED/数 + 为何)→ `git checkout pure-hmx-opt && git checkout loop/<lever>-<seq> -- Agent/current/pure_hmx_solve_build.md`(**只取文档,丢代码**)→ `git commit` → `git branch -D loop/<lever>-<seq>`。**码不进基线**。
5. 基线 `pure-hmx-opt` 始终 = 最优码 + 全部结论(成功档 + 失败教训)。回 1。
6. 守则:**HMX = 1 单元绝不 thread**(SSR;先读 skill `htp-hardware-scheduling`);基线 `GDNSolveHVX` 只测不改;不过先查实现不臆测(热循环 / 寄存器溢出 / 次优 intrinsic);`pkill -9 gdnbm` between runs;`INVARIANTS` 只在新设备证据下改。

## LEVER LEDGER

### O. 干净重写逐步优化（当前战场;`gdn_pure_solve.cpp`,全 oc-gated + wall 环比）
| 杠杆 | status | 结果 / 数（PCYCLE,统一口径) |
|---|---|---|
| **O0 干净 v1**（单线程正确性基线) | **DONE** | oc 9.662e-3 bit-match,sat=0;wall 1.354B(④ feed 96%) |
| **O1 4 producer + 1 HMX consumer 线程化** | **DONE(3.13×)** | wall 1.354B→**433M**,oc 不变;① 433M(1593.6 cyc/µs)② 0.33M(0.08%)④ 60M(14%,consumer 非瓶颈)·spin Σ61.9M;**仍 feed-bound**(producer 标量) |
| **probe: producer 阶段分解** | **DONE(cron#2)** | Σ4prod:**depack 1.047B(60.6%)**·pack 499M(28.9%)·renorm/other 122M(7.1%)·spin 61M(3.5%)·kernel 60M(隐藏)。**depack 是头号(散写),pack 次之** |
| **O2 cv-domain + O3 64³(中间留 crouton,杀 depack)** | **DONE(2.41×)** | wall 433M→**179.9M**,oc 不变;depack 1.047B→266M、kernel 60M→**15.3M**(64³)。cv LUT `g_lut` + 直拷 act/out + 弱-pack 仅 weight |
| **O4a HVX act/out copy(VTCM)** | **DONE(1.68×)** | out-copy 266M→**0.26M**(vxor);wall 179.9M→107M。`gp_cv_to_surf`/`gp_surf_to_cv` + cv 数组 128-align + P=1 hvx-lock |
| **O4b-probe pack 分解** | **DONE(cron#5)** | Σ4prod:renorm/other **159M(38.8%)** ≈ kmajor **154M(37.5%)** > gather 60M(14.5%) > bias 20M(5%) > spin 17M。**两个 co-bottleneck** |
| **O4b HVX wt-pack(vgather→VTCM)** | **DONE(2.22×)** | weight pack 234M→**8.2M(28×)**,wall 107M→**48.3M**,oc 不变,**PACKCHK=0**。`g_hw`+`gp_pack_wt_bias_hvx`(ported `p4v_pack_wt_bias`)+per-ctx VTCM stage |
| **O4c HVX acc/renorm(interleaved 布局)** | **DONE(2.02×)** | renorm/other 157M→**41M**,wall 48.3M→**23.9M**,oc 不变,PACKCHK=0。port `p4v_acc3/_negw/_zero/_addsh/_shr/_absmax/_to_cv/_renorm/_diag_add/_from_cv`(`gp_acc*`),acc 换交织布局+128-align |
| **probe scatter 分解** | **DONE(cron#8)** | renorm/other 41M **几乎全是 per-head 标量 LUT-scatter = 44.4M**;HVX renorm+acc 仅 3.4M。scatter 每 head 卡住 producer(unpack 时无 job armed)→ consumer idle gap |
| **O4d HVX A-unpack/T-pack scatter(vgather perm)** | **DONE(1.28×,过参考)** | scatter 44.4M→**18.3M**,wall 23.9M→**18.68M**(< 参考 18.7M),oc 不变。`gp_perm`(memcpy 连续化 + vgather)+`g_il/g_fl` LUT。全 feed 现 HVX |
| **Newton 2→1(精度换速)** | **DONE(1.22×)** | @scale0.05 oc **8.24e-3**(≈2 的 9.66e-3,quant-dominated)< 4e-2;mm 1280→1024,wall 18.68M→**15.23M**。默认改 1(`GP_NEWTON`)。⚠️ 大 ‖A‖ 需复核 |
| **Newton 1→0 Taylor-only(精度换速)** | **DONE(1.25×)** | mm 1024→768,wall 15.23M→**12.21M**,oc **4.24e-3**(更优,Newton 单调加 quant 噪声)。默认改 0 + `#if GP_NEWTON>0` guard。⚠️ scale0.05-specific |
| **O5 scatter-kill: fuse strided memcpy 进 gp_perm staging** | **DONE(cron#19,−12%)** | probe 拆 scatter 17.45M → **memcpy 13.98M(78.5%)** / gp_perm 3.39M / memset 0.39M。**memcpy 是全部瓶颈**(~1 byte/cyc,未向量化的 linear↔block strided 重排)。fix = `gp_unpack_blk`/`gp_pack_blk`:把 Aq 的 strided 64×64 块直接 HVX-load 进 gp_perm 的 VTCM staging(skip c->lin)、vgather 输出直接 strided-store 进 To → **memcpy 13.98M→0**。wall 12M→**~10.6M**(同窗 REPS=8 reps2-4 median 12.71M→10.76M=**−15.3%**;all-8 11.78M→10.62M=−9.8%),oc **4.238e-3 不变**(纯重排,oc-中性,128-align 安全)。**瓶颈翻到 consumer**(stats[3]~9.1M≈wall;producer-life 12.2M→9.9M)→ NEXT=O6 砍 consumer |
| **DECISION 解决(真 scale 实测)** | **DONE(cron#11)** | 真 GDN ‖A‖ median 0.600/max 1.099(> synthetic 0.363)。设备真数据 oc:**N0 1.107e-2 < N1 3.2e-2 < N2 3.35e-2** → **Newton=0 在真 scale 也最优**(非 scale-fragile)。Newton=0 是真·最优,非精度换速妥协。**Taylor 3→2 在真 scale 会爆**(‖A‖~1.1,A³ 发散)→ 不可再降 diag |
| **consumer idle probe + dynhead** | **DONE(cron#13)** | 实测 consumer 分解:busy 9.15M(78%)· **inter-job idle 2.04M(17%=A2 天花板)**· startup 0.33M · tail 0.21M。**dynhead 动态调度 REFUTED**(idle 2.04→2.08M 不变 → idle 非 head-对齐,是 1-job/producer 结构性 starve) |
| **NOSCATTER 诊断** | **DONE(cron#14)** | 关 scatter:wall 11.73M→**9.18M**、consumer idle 2.04M→**0.02M** → **2.04M idle = 100% scatter 造成**(非 between-matmul)→ **double-buffer/async 无用**(scatter 时无 matmul 可 prep) |
| **scatter-interleave(队列 drain 进 spin)** | **REFUTED(中性)** | 把 A-unpack 队列化、每 matmul spin drain 一块。oc 4.238e-3 ✓ 但 **wall/idle 不变**:**单块 scatter ~27K > 单 matmul 11.9K** → drain 溢出 spin,藏不住 |
| **单 64³ matmul 实测(回应硬门质疑)** | **DONE(cron#15)** | 微基准(back-to-back resident)= **10,844 cyc/call** ≈ consumer 聚合 11.8K(非 contention)。**单 64³ 降不到 300**:300 是 MAC latency(data-resident,饱和拿不到);饱和 throughput = 字节流量 136KB = 4× padding(2KB HW tile,**compact 越界设备证伪** gdn_solve.md:145)× 2× double-pass(w16a16=2×w8a16,INVARIANT 1)。**= pure-HMX 64³ 根本地板,输 HVXMixHMX 的根因** |
| **口径再纠(用户)** | **DONE(cron#16)** | 我的 10,844 是 **per-call feed-inclusive(口径④)非 HMX 计算**。consumer 9.1M = HMX 算 + feed |
| **QNN-native 对齐(optrace+timeline,用户导)** | **DONE(cron#17)** | **重大纠错**:测 `[1,128,64,64]`(128 独立 distinct-weight 64³ = 128 链)→ **QNN hmx_used 1,430/matmul,ConvLayer 1,504/matmul,wall 2,020/matmul = 比我孤立 10,844 少 ~8×**。我微基准已证 back-to-back 同描述符仍 10,844/call → **8× 不是循环,是 descriptor fan-out**(QNN 一个 batched op 走 128 tile 保持 HMX 流水满,摊销 ~9K/call 的 fill/drain)。**我做 768 次 per-tile mxmem = 错模式**(`reference_hmx_dsp_vs_descriptor`:efficient HMX=fan-out 非 per-tile)。**12M 不是 feed 地板,是 per-call fill 重复 768 次** |
| **stats 分解纠错(cron#18)= wall 不是纯 consumer-bound** | **DONE = 重定向 NEXT** | 实测 stats:wall 11.87M,**consumer-busy(stats[3])9.10M + 2.77M idle;slowest-producer-life(stats[11])11.8M ≈ wall;scatter Σ(stats[9])17.45M(/4=4.36M/prod)= producer 最大单项;spin Σ(stats[4])21.7M**。⇒ **producer/consumer 平衡流水**,非"768×fill 单一 consumer 地板"(cron#17 半对)。**两杠杆:O5 scatter-kill(producer 标量 repack,低险先)+ compact-crouton(consumer,后)**。详见 NEXT |
| **O6b compact-64³ 描述符猜测 V1/V2/V3** | **全 REFUTED(cron#20)** | scaffold `-DGP_O6B_TEST` 设备实测三个描述符假设全死:**V1**(standalone strides+×512 表)= DSP fault;**V2**(shape-scaled 64/64/64/128 + ×512 compact 表/数据)= 跑通但 **maxdiff 25774 错**、cyc 10880 无加速;**V3**(offline byte-exact 生成器 `prepare_owned_inputs.py` 的 w16a16 FIXED 256/256/128/512 + ×512 表)= **DSP fault 0x8000040d**。⇒ 两自然假设(shape-scaled / fixed-256)全废,**256 是 M=256 值不可搬给 64³;offline 生成器在 64³ 不可信**。与 cron#15「compact 越界证伪」+ cron#17「QNN 靠 batched fan-out 非 per-call compact」一致。**单 64³ compact 描述符大概率不存在于 per-tile kernel** → NEXT 唯一路 = STEP 1a 抓真 op dump(别再猜 strides) |
| **✅ 等价性验证(用户要求,cron#22):our custom op ≡ native QNN matmul,64³ bit-exact** | **DONE = 4096/4096 maxdiff=0** | 用户问"建 1×128×64×64 test 验证你当前用的实现和 native 等价"。做了:① native 64³ oracle(`profile_all.sh --configs w16a16 --shape 64,64,64` → `output_native64/device_out/Y.raw`);② 抽 native 权重 sidecar 喂我们的 custom op(`run_w16a16_chain.sh ... W16A16_NATIVE_ORACLE_DIR=output_native64`,同 act 同 wt);③ 设备 HTP-only 跑 → **out_native vs native Y.raw = 4096/4096 bit-exact,maxdiff=0**。**∴ `our_v73deep_kernel_i16`(solve 用的核)= native QNN matmul,设备实证字节级等价(非仅 sha256 slice)。** native `[1,128,64,64]`(`output_b128`)= timeline 258,544/128 = **2,020 cyc/matmul wall、hmx_used 183,009/128 = 1,430/matmul**(128-tile fan-out)。**∴ gap(我们 per-call 10,844 vs native 2,020)100% 是 dispatch/fan-out,不是 kernel 不是布局**(核 bit-exact + 布局同 ×2048 padded)。fan-out 是唯一真杠杆,已三方坐实 |
| **🟥 STEP 1a DESC_DUMP: 抓到 QNN op 64³ 真表偏移(cron#22)= O6b-compact 前提被证伪** | **DONE = DISPOSITIVE REFUTE** | 改 op 的 DESC_DUMP 把 act/out 表偏移写进每个 out 块(robust 于块序;crouton-row4 un-depack 读回)。**64³ 实测真值:descriptor 64/64/64/128(= FORMULA_DESC 确认);act_entries=32;`act_tbl[i]-a0 = i×2048`(= **PADDED 2KB 步长**),out 表同 ×2048。** ⇒ **QNN 自己的 op 给 64³ 用的就是 ×2048 padded 32KB 布局 = 跟 solve 现状一模一样;不存在 compact-8KB 单-64³ 描述符**(= INVARIANT 7「QHPI 给 padded 2048B 块」)。**O6b 整个前提("QNN 读 compact 8KB,我们读 padded 32KB = 那个 3.2× 差")被 op 自己的真值证伪。** V2 的 ×512 表错正因 kernel 要 ×2048。**per-call 差(我们 ~10.8K vs QNN ~3.3K,cron#17)不是布局,是 fill/drain 摊销(QNN `[1,128,64,64]` fan-out 一个描述符驱 128 tile 保 HMX 流水满)→ compact 单-64³ 治不了。** 复现:op build `-DHMX_W16A16_DESC_DUMP` + `run_w16a16_chain.sh M=64`,device HTP-only,读 out_native.raw un-depack rows0-3。⇒ **真杠杆 = fan-out 批(reference_hmx_dsp_vs_descriptor),非 compact;需用户裁定(锁定方向前提已废)** |
| **STEP 1a: 跑真 custom op 64³ + M=256 控制(cron#21)** | **DONE = 重定向(strides 不是问题)** | 跑通真 custom op 的 64³ QNN device flow(built 3 包;修 `dssh.sh` set -e bug、装 lxml、CPU-op-pkg 注册失败→**HTP-only 才行**)。op 注册+执行 64³ Crouton_16 图到完成。**控制实验:M=256(已证 byte-exact)对 gen 解析 ref 的失配模式 = M=64 完全相同(都 dev 饱和 vs ref 小值)→ 是比对口径错(profile 验 native oracle 非解析 ref),非 kernel 错。** ⇒ **op 跑 compact 64³ 与 256 同样正确;它用的描述符=FORMULA_DESC 64/64/64/128=V2 的 strides。** **∴ V2 的 bug 是 act DATA/TABLE 布局(我的 `w16a16_pack_act_crouton16`+×512 atab ≠ QNN 真 Crouton_16 块),不是 strides。** O6b 重定向:复刻 QNN compact 块字节布局+表,别再调 strides。下轮 = k_impulse 自证 + DESC_DUMP 抓真表偏移 |
| **起手:compact-crouton(consumer 第二杠杆,cron#18)** | **OPEN(scatter-kill 后)** | kernel `our_v73deep_kernel_i16` = QNN op **byte-identical**;描述符 FORMULA_DESC(M_t*32=M=64)= 我 override 一致。**我 M=64 单 mm 10,844(act 16×2KB padded span 32KB)vs QNN M=64 compact 3,349(读 18KB)= 3.2×,差在 act 的 row4-crouton 块格式(2KB-padded vs compact 512B-live)**。stride-hack 测无效(kernel 要 2KB-align,stride512/0→退化 4.9M/0)。要真港 `HmxU16I16ToU16MatMulOp.cpp` 的 row4 block table。**w16a16_mm.h "64³ 不 byte-exact" = 另一种手写;QNN M=64 compact 是 byte-exact(accepted 65536/65536)**。port 后 consumer 9.1M→~3.7M |
| **w8a16 单 pass(砍 feed 2×,真 NEXT)** | **OPEN = NEXT** | w16a16=2×w8a16 双 pass 流 act/out 两遍。单 pass w8a16(int8 权重×int16 act)**腰斩 feed** → 5.2K/matmul → consumer ~4.5M → **wall ~7M**。sim:int8 权重 real-GDN oc 1.27e-3(< 4e-2,30× 余量,6-bit 都过)。需 int8 权重 pack + 单 pass kernel,设备测 oc |
| M=256 批 / scatter-vshuff | 次要 | M=256 共享权重模式不配;scatter 填 2.04M idle 是次要(consumer feed 才是大头) |
| A2 async dispatch overlap | OPEN | 去同步 spin(4.2%,小) |
| O5 fan-out 批 | OPEN-评估 | HMX 已非瓶颈(④ 14%),收益存疑 |

### A. 系统级（旧架构,历史;非当前战场）— 地板 ~1.49M
| 杠杆 | status | 结果 / 数 |
|---|---|---|
| **A1 批独立维 + 杀 per-mm glue** | **OPEN = NEXT** | 去 ~1.75K/mm 链隙 |
| **A2 feed 流水 4×HVX(repack 甩 producer)** | **OPEN = NEXT**(与 A1 一起) | 去 ~6.8K/mm 暴露 feed,HMX 不饿 → HMX-busy-bound;地板 1280×1167 ≈ **1.49M** |

### B. 精度换 wall（门 oc<4e-2;4e-2 宽门把这些重新打开,每个必须重测 oc）
| 杠杆 | status | 结果 / 数 |
|---|---|---|
| **P1 merge → u8i8** | OPEN(A 后) | (24×1167 + 16×194)×32 ≈ **1.0M**;raw u8i8 merge oc~5e-2 **超门** → 必须 `+SBOOST`(Sacc drain ×4,oc→5e-3,1 行) |
| P2 Newton 2→1 | OPEN-评估 | mm 1280→1024,+merge u8i8 ≈ **0.7M**;残差 A^8 vs A^16;旧 1e-2 门"退 27%"否决,**4e-2 宽门重测** |
| P3 跳部分 final merge(SKIPFIN) | OPEN-评估 | 旧 `SKIPFIN_D3` oc 9.56e-3 贴 1e-2;**4e-2 下可更激进**(多跳几块) |
| P4 对角 dtype 直接降(非预条件) | OPEN-评估 | w8a16 对角?per-block 差但 **aggregate 4e-2 可能过**;待测(probe 现测 per-block 1e-2,需加 aggregate-oc@4e-2) |
| (旧) merge K-stack | **可重评** | 旧否决=stack prep 串行加厚 > HMX 省;**A1+A2 把 prep 流水化后该理由可能不成立** → A 后重测 |

### R. 跳过这 2 条（已证死路,别重试;**不影响 A/B/D 的活,A1+A2 照做**）
| 杠杆 | 为何死 |
|---|---|
| **R1 对角预条件换便宜 dtype** | **math 必然**:任意对角 D,`inv(L̃)_ij` 与 `Ã^k_ij` 同因子 `d_j/d_i`;压迭代上溢 = 把远次对角 inv 压到廉价 dtype 分辨率下 **下溢成 0**,还原救不回。scalar/per-row 都逃不掉。`scripts/gdn_solve_precond_probe.py`(1344 块,best-s 中位=1.0,PC maxErr ~0.17)。**注:P4(直接降 dtype,不预条件)是另一回事,仍 OPEN。** |
| 三角 mask | 64³ mm 仅 8.6K;拆分省 ≤2.1K < 新增 dispatch ≥3K(算术) |

### D. 旧架构内 KEEP（已并入基线 18.7M,别回退）
Newton 4→2 · bias HVX colsum(vsxt 序 lo+hi)· A/T 装出 vgather perm · renorm/add 全 HVX。

## INVARIANTS（已证,grounded,禁止凭记忆改）

1. **w16a16 = 2× w8a16。** int16 权重拆 hi(int8)+lo(uint8),各 ×int16 激活 = 两遍 w8a16 分别 drain,×256 合并。`our_v73deep_kernel_i16`(dilated 权重 + 2×2 drain)内部就是这两遍。权威:`docs/w16a16_is_two_w8a16.md`(CI gate)。
2. **drain 是 2 的幂,不是 fp16。** 增益 = `2^(exp-16)`(bias 控制字 bits[14:10]),比精确 1:2:4。细 scale `1/32767` = 两个 2-幂 drain(hi×256 / lo×1)合并。**「f16 drain 有损/blocker」论断作废**(见 BANNED)。
3. **byte-exact、可用 = PROVEN。** standalone(Python pack + kernel + depack)hexagon-sim + 真 CDSP 双向 `diff=0`,M=256×任意 K,N,CI 守门。C 打包器(`pure_hmx_solve/w16a16_pack.h`:act crouton16 / wt dilated-kmajor / bias / depack)全对 Python ground truth。
4. **128 条独立求逆链 = 主结构。** C=256 每 head 切 4 个 64 对角块,每块 = 一条 Newton 链(**链内严格依赖**);**32 head × 4 块 = 128 条链彼此完全无依赖**,merge(16/head)只依赖本 head 对角。⇒ 依赖只在单链内,跨 128 条链 **matmul 可批、feed 可流水**(= A1/A2 的全部依据)。
5. **mxmem/feed 口径**(决定 A1/A2):`q::ConvLayer_s1.opt` 的 mxmem 循环本体只有 **256 latency / 1167 busy**(int16 64³;1167=256³-摊销吞吐)。单 mm ~9.7K wall = HMX 真算 ~1167 + **feed ~6.8K(crouton+kmajor-repack = ConvLayer 外、HVX 可干)** + 链隙 ~1.75K。后两项非地板。详:`int16_matmul_cycle_model.md` latency-vs-feed-inclusive。
6. **roofline 地板 = 1167/64³**(全 int16×int16 HMX-busy 算力地板,batched 64³ 趋近它)。当前 Newton=2 = 1280 mm → 全 w16a16 ≈ 1.49M / merge u8i8 ≈ 1.0M / +Newton1 ≈ 0.7M。
7. **64³ 单调用可用**(旧"永不可用"错):QHPI 在 M=64 给 act/out **padded 2048B crouton 块**(live 前 512B);正确描述符 {N_t=2,y=64,n_tiles=64,m_total=1,k_total=64}、act y=128 → byte-exact(H==9 设备验)。
8. **HVX lane 硬经验**(踩过,别回退):① **`vsxt+vasr` 必须配对**(`Q6_Ww_vunpack` 偶/奇 vs `Q6_Vh_vasr_VwVwR` block 序不配,自检抓出 3968/4096 错);② diag 标量补丁交织索引 `(i>>6)*64 + (off&1)*32 + (off>>1)`;③ **scalar 访 VTCM ≈4× 慢于 DDR-L2**(753M vs 187M),prep 一律走 DDR,VTCM 只给 HMX 面/gather;④ HVX bias lane-fold 两次都错,scalar LUT colsum(12K/mm)更划算;⑤ dense pack = `vadd128 + vasr8 + vpack_sat`。
9. **单 64³ 纯 HMX matmul 的算力 = 200~300 cyc latency(native int16 ~256,dominant-path)= 硬指标**(口径门见 OBJECTIVE)。这是 **matmul 本身**;与之分开的是:**feed**(crouton/kmajor pack + padded-operand 字节流量,单调用 ~6.8–10.4K = HVX/字节带宽,**非 HMX 算力**)、**链隙/glue**、**吞吐口径** 1167(2×w8a16 软分解的 HMX-busy 摊销,≠ latency)。**报任何"单 matmul N cyc"前先判 N 落哪个口径:>300 必是含 feed/glue 或 throughput,不是算力**(`gdn_solve.md:103/117`、`int16_matmul_cycle_model.md`)。inverse 是 **feed/producer-bound**,HMX 实占 ~7%。

## 精度预算（当前不是约束,别当 blocker）

aggregate oc **9.66e-3 ≪ 门 4e-2 = ~4× 余量**,精度宽松 → B 组可放心拿精度换 wall。精度旋钮(与速度正交,需要时再用):A-scale(甜点 ~0.25 → oc 4.5e-3)+ Newton 步数。对角 dtype 固定 w16a16(更便宜的预条件路 = R1,LEDGER 里);余量这么大,**精度不是停下来的理由**。probe:`scripts/gdn_solve_precond_probe.py`。

## 地基（已验证可复用的件 = NEXT 的起点,不是终点）

下面都跑通过、A1+A2 直接复用;**18.7M 是从这里继续往下降的起点,不是终态**。
- **w16a16 64³ 原语** `pure_hmx_solve/w16a16_mm.h`(C 打包 + convhhh + depack),对配量化 byte-exact、oc 1.4e-5。打包器 `w16a16_pack.h` 也 byte-exact(host `w16a16_pack_test.c`)。`run_w16a16_mm_phase1.py --deploy`(H=1)。
- **对角 Newton** 单块 = 6 mm(Newton=2)/ 10 mm(Newton=4),X0=I+A+A²+A³;**指数双向归一必须**(只右移翻倍报废)。`run_w16a16_diag_phase2.py`(H=2)。
- **全 head 组装** 4×diag + 16 merge,块指数表随 T 回传。`run_w16a16_head_phase3.py`(H=3)。
- **4HVX∥1HMX 管线** = 当前基线骨架。⚠️ **它的 consumer 是旧 per-mm-dispatch 设计 = A1+A2 要替换的对象,本身不是终态**。`run_w16a16_head_phase4.py --deploy --threads 4 --heads 32 --scale 0.05`。
- **18.7M 怎么来的**:101.3M → 36.5M → **18.7M**,全是**旧架构内的微杠杆**(Newton4→2→40 mm/head、bias HVX colsum、A/T vgather、renorm 全 HVX)。**系统级 A1+A2 从未做** = 当前主战场,~12× headroom 就在这里。

## REPRODUCE（当前态 + 探针）

```bash
# 当前 best(32-head wall + oc + timeline + 打包自检 stats[9..11])
uv run python scripts/run_w16a16_head_phase4.py --deploy --threads 4 --heads 32 --scale 0.05
#   timeline: H=33 sentinel + scripts/gdn_pipe_timeline.py ; 探针 H=9/10 = 64³/K128 单发 byte-exact
# 数值 probe(对角 dtype / 预条件 / Newton 步数 feasibility)
uv run python scripts/gdn_solve_precond_probe.py        # R1 证伪 + ‖A‖ 分布 + cheap-dtype 收敛
uv run python scripts/gdn_solve_taylor_newton_probe.py  # Newton 步数 vs 收敛/峰值(P2 评估)
```

## FILES

- `solve_br_op/src/GdnSolveBR16.cpp` — solve(`gdn_br_one_head16`),改这里。
- `baremetal/src/gdnbm_imp.cpp` — FastRPC 驱动 + pipeline(`pipe_producer`×P + 主线程 PURE-HMX consumer,`g_hmx_dispatch` 钩子)。**A1+A2 主战场。**
- `pure_hmx_solve/w16a16_pack.h` / `w16a16_mm.h` — C 打包器 + 64³ 原语(byte-exact,CI-gated)。
- `scripts/run_w16a16_{mm_phase1,diag_phase2,head_phase3,head_phase4}.py` — gdnbm H=1/2/3/≥5 模式驱动。

## BANNED（作废论断,凡再见一律忽略）

- ❌「w16a16 drain 走 fp16 / 深累加器丢低位 / 是 blocker」——**错**,drain 是 2-幂(INVARIANT 2)、matmul byte-exact(INVARIANT 3)。死因 = 用**错配量化**(native ONNX-scale bias 配任意 A,W)标定,被误读成有损。验证一律用**对配量化**(同 standalone 量化契约)。
- ❌「64³ 描述符永不可用」——错,见 INVARIANT 7。
- ❌「对角预条件能换便宜 dtype」——证伪,见 LEDGER R1。
