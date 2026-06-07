# GDN Solve 优化 Harness + 流程(权威,2026-06-08 沉淀)

这是 GDN C=256 三角求逆 solve 在 v75 HTP 上做性能优化的**正确 harness、可靠工具、流程循环**。
后续任何 GDN solve 优化**先读这篇**,照流程做,别重新发明轮子、别重踩已记的坑。

---

## 0. 当前状态(权威)

- **最佳 = `GdnSolveBR16.cpp` int16 静态 solve,producer-consumer pipeline,PURE-HMX consumer,P=4**。
  32-head total wall **~1.93M domain cyc ≈ 1.28 ms ≈ 2.06× vs GDNSolveHVX 4-thread 基线(~3.97M)**,全程 bit-exact。
- **架构三文件**:
  - `solve_br_op/src/GdnSolveBR16.cpp` — **新基线**,干净 int16-only 静态 solve(`gdn_br_one_head16`)。后续在这改。
  - `baremetal/src/gdnbm_imp.cpp` — FastRPC 驱动 + pipeline(`pipe_producer` ×P + 主线程 PURE-HMX consumer,`g_hmx_dispatch` 钩子)。
  - `solve_br_op/src/GdnSolveBROp.cpp` — 共享 helper(pack/effective/merge_packed/diag/HMX kernel)+ 旧 int32 solve。
- **构建**(默认即最佳;`GDN_BR_I16`+`GDNBM_PIPE_PURE_HMX` 在 STATIC_FULL 下自动开):
  ```
  cd example/gdn_native/baremetal
  EXTRA_DEFS="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL" bash build.sh
  ```
- **运行**:`./gdnbm 4 A_u16_h32.raw T.raw 32 256 32768 32768 2.770166930875267e-05 6.103701895199438e-05`,`GDNBM_REPS=8`。
- **数据**:`A_u16_h32.raw`(u16 输入,zpA=32768)/ `T_ref_h32.raw`(fp32 golden)/ `/tmp/T_avtcm.raw`(int32-solve 设备输出,bit-exact 基准)。
- **PCYCLE turbo = 1.594 GHz**(实测 domain÷1594=ms,两 PIPE 行互验)。

---

## 1. 流程循环(经验提炼,照这个走)

```
  ┌─> [1] 测量(4列 + A/B 控热 + 稳态)
  │      └ 不准就停,先把基线测干净
  │   [2] 画 timeline(每次改完都画)
  │      └ ⚠️ span 会高估 SMT 隐藏的 load-bound 阶段
  │   [3] 选候选瓶颈(从 timeline + 4列)
  │   [4] ★ ablation 验证(cap-test 砍工作量看 4线程 wall 是否降)
  │      └ 这步是真相;别只信 timeline span
  │   [5] 上杠杆(优先级见 §3)
  │   [6] 重测:bit-exact 门 + 4列 + 重画 timeline
  └──[7] 变快→留+commit;变慢/无收益→revert+把负结论落盘(别重试)
```

**铁律**:
- **每改一次代码都重画 timeline**(`feedback_retimeline_after_every_change`),别只看总 cycle。
- **bit-exact 门**:每次改完跟 `/tmp/T_avtcm.raw` 比 `maxdiff==0`(int16 静态下应无损)。
- **A/B 控热**:两个 build 交替跑(`git stash` 切),被测的那个常占热劣势→若它还赢就是真赢。
- **负结论必须落盘**(`feedback_dont_chain_no_op_experiments`):2-head fusion / round-trip 消除 / diag 都试过更慢或无效,别重试。

---

## 2. 工具箱(可靠性已标注,直接复用)

| 工具 | 是什么 / 怎么用 | 可靠性 |
|---|---|---|
| **32-head total wall (domain)** | `gdnbm_solve` 报 `wall=t1-t0`(主线程 PCYCLE,整个并行 solve makespan) | ✅ 唯一权威终指标 |
| **4列 instrumentation** | PIPE 块内:stats[3]=总HMX(`g_pipe_cbusy`)/[4]=总HVX(Σ`life-spin`)/[5]=domain/[6]=真µs(`HAP_perf_get_time_us`) | ✅ 看机制(HMX vs HVX vs 时间)。commit 9a40752 |
| **A/B 控热** | `git stash`→build A→run→`git stash pop`→build B→run,各 REPS=8 取 median | ✅ 抗热噪声 |
| **bit-exact 门** | dump 设备 T uint16,跟 `/tmp/T_avtcm.raw` `np.array_equal` | ✅ 正确性 |
| **timeline 渲染** | `-DGDN_BR_TRACE` build → 设备跑 REPS=4 → dump T → `scripts/gdn_pipe_timeline.py T.raw [W]`(`GDN_TL_COARSE=1` 看粗粒度) | ✅ 看分布,⚠️ span 高估 SMT 隐藏的 load-bound 阶段(见 diag) |
| **cap-test ablation** | 加 flag 砍某阶段工作量(如 `GDN_BR_FWD_CAP` 砍 diag 到 1/16),看 4线程 wall 是否降 | ✅ **SMT 隐藏阶段的唯一真相**;timeline span 不行时用它 |
| **dssh.sh** | `source scripts/dssh.sh; dssh_open oneplus; dssh "cmd"`(ControlMaster 复用,抗 ssh 假死) | ✅ 设备必用 |
| **PROBE_CYCLES** | `-DGDN_BR_PROBE_CYCLES` 每阶段 C15:14 桶 | ❌ **不可信**(嵌套 C15:14 求和 = 3.5× wall);只用 timeline disjoint span |
| **per-head metric** | — | ❌ tiler 低估 artifact(88K 假基线),**禁用** |
| **min-of-reps** | — | ❌ 抓快异常值,系统性高估;用 median 稳态 reps2-4 |

---

## 3. 杠杆优先级(v75 实测教训)

**核心硬件事实:v75 瓶颈 = HVX issue/compute,不是 VTCM 带宽。** 由此:

| 杠杆 | 有效? | 证据 |
|---|---|---|
| **更少 HVX op / 更宽 lane**(int16=64 vs int32=32) | ✅ **主杠杆** | int16 砍总HVX −19%→wall −17%;Q15/widening 乘 64-lane |
| **A 驻 VTCM**(避免 uncached DDR 标量读) | ✅ 大(单线程 1.93×) | A-bound 的 diag/fold |
| **PURE-HMX consumer + P=匹配单元数** | ✅ | timeline 看穿 P=4 starvation(5 线程挤 4 HVX 单元) |
| 减内存搬运 / 去 int32 round-trip | ❌ **无效甚至更慢** | VTCM 带宽充足,round-trip 几乎免费;fuse 反伤调度(实测 +3%) |
| 同型 2-head pack fusion | ❌ 更慢 | 同管道无 co-issue;异管道(act∥wt)才有但 cache 分开 |
| 攻 diag(前向求逆) | ❌ ~0% | load-bound 但被 SMT 藏住(cap-test 砍 1/16,4线程 wall 不变) |
| effective(零点税) | ⚠️ 只 3-4% | 不值精度;留着 |
| HMX 侧(matmul/descriptor) | ❌ 无关 | HMX 仅占 domain 7.6%,94% 空,~13× 余量,开销全被覆盖 |

**下一步候选**(未验证):int16-lane 化 fold(`gdn_fold_quant_u8`/`gdn_fold_block_hvx` 的 u16×M 现走 32-lane,改 widening 乘 64-lane)——**但先 cap-test 确认它在关键路径**(别像 diag 那样被藏)。

---

## 4. 已确证的关键结论(别重新质疑)

- **指标**:只信 32-head total wall(domain);终汇报必带 4 列(总HMX/总HVX/总domain/ms)。
- **精度**:int16 在静态下无损(diag 码 −7716..16384 全在 int16;`GDN_BR_TI=2/32767` 专为此设计;Sacc pure-add ±381)。
- **HMX**:做 8×256³ MAC(512 个 64³),304 cyc/64³,健康;1.42× 是分块 descriptor 摊销,因 HMX 空转而对 wall 零影响。
- **producer-bound**:consumer/HMX 6-7% 忙;瓶颈全在 producer 的 HVX glue。

---

## 5. 速查命令

```bash
# 设备
source scripts/dssh.sh; dssh_open oneplus
# build + 部署 + 跑(4列)
cd example/gdn_native/baremetal
EXTRA_DEFS="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL" bash build.sh
W=$(dssh 'echo $HOME/gdnbm_run'); dssh "pkill -9 gdnbm; true"
dssh "cat > $W/libgdnbm_skel.so" < build/libgdnbm_skel.so
dssh "cat > $W/gdnbm" < build/gdnbm; dssh "chmod +x $W/gdnbm"
dssh "cd $W && GDNBM_REPS=8 LD_LIBRARY_PATH=$W:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH='$W;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' ./gdnbm 4 A_u16_h32.raw T.raw 32 256 32768 32768 2.770166930875267e-05 6.103701895199438e-05"
# timeline
EXTRA_DEFS="... -DGDN_BR_TRACE" bash build.sh   # 部署同上, REPS=4
dssh "cat $W/T.raw" > /tmp/T_tl.raw; GDN_TL_COARSE=1 python3 scripts/gdn_pipe_timeline.py /tmp/T_tl.raw 110
# bit-exact
dssh "cat $W/T.raw" > /tmp/T_x.raw
python3 -c "import numpy as np;a=np.fromfile('/tmp/T_avtcm.raw',dtype=np.uint16);b=np.fromfile('/tmp/T_x.raw',dtype=np.uint16);print('maxdiff',int(np.abs(a.astype(int)-b.astype(int)).max()))"
```

相关:`gdn_int16_plan.md`(int16 细节+负结论)、`gdn_hmx_solve_plan.md`(pipeline 历程)、`gdn_2head_pack_design.md`(REFUTED)。
memory:`[[project_gdn_pipe_beats_hvx_2026-06-08]]`、`[[feedback_gdn_metric_32head_total_wall]]`、`[[feedback_retimeline_after_every_change]]`、`[[reference_htp_smt_pmu_hardware]]`。
